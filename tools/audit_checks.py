"""Checks behind the claims in the paper, run against the released dataset and baseline.

    python tools/audit_checks.py --out audit [--device cuda:0] [--seeds 20]

1. Calibration: the raw 99.5th percentile of the difference map before the floor,
   and the raw healthy flagged fraction of each component before the offset.
2. Exact pixel metrics per scenario, where the false-positive pixels lie relative
   to the true change, and recall by true contrast.
3. The antenna Gaussians flagged on the healthy satellite: where they sit on the
   antenna, which scenarios flag the same ones, and what causes them (pose error,
   sensor noise or reconstruction error).
4. Antenna-missing scenario: whether the flagged bus Gaussians lay in the
   antenna's shadow on the healthy satellite.
5. Sensitivity of the verdicts to the hand-set detection and classification values.
6. Repeat draws of pose error and sensor noise (new seeds, same baseline).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from larissa.dataset import LarissaSynth, camera_from_json  # noqa: E402
from larissa.detect import (add_sensor_noise, calibrate_threshold, classify_components, detect,  # noqa: E402
                            difference_map, dilate, verdict)
from larissa.gaussians import GaussianModel, render  # noqa: E402
from larissa.scene import (_intersect, build_satellite, inspection_cameras, perturb,  # noqa: E402
                           render_scene, sun_direction, _rot)

EXPECTED = {"healthy_control": "NO CHANGE", "panel_hinge": "GEOMETRIC", "antenna_missing": "GEOMETRIC",
            "mli_discoloration": "APPEARANCE", "sun_shift_control": "NO FAULT"}
FAULTS = {"healthy_control": (None, 68.0), "panel_hinge": ("panel_hinge", 68.0),
          "antenna_missing": ("antenna_missing", 68.0), "mli_discoloration": ("mli_discoloration", 68.0),
          "sun_shift_control": (None, 98.0)}
QUIET = lambda *a, **k: None  # noqa: E731


def correct(v: str, exp: str) -> bool:
    if exp == "NO FAULT":
        return (not v.startswith("GEOMETRIC") and "illumination" in v) or v == "NO CHANGE"
    return v.startswith(exp)


def comp_levels(det, factor=3.0, offset=0.02):
    return {n: max(factor * c["fraction"], c["fraction"] + offset) for n, c in det.components.items()}


def relabel(det, thr, comp_min=60):
    """Re-apply component alarm levels to an existing detection (no re-render)."""
    for n, c in det.components.items():
        frac = c["flagged"] / max(1, c["evaluated"])
        c["threshold"] = thr[n]
        c["changed"] = bool(frac >= thr[n] and c["flagged"] >= comp_min)
    return det


def short(v):
    for k, s in (("GEOMETRIC", "geo"), ("APPEARANCE CHANGE ON MANY", "illum"), ("APPEARANCE", "app"),
                 ("NO CHANGE", "none")):
        if v.startswith(k):
            return s + ((":" + v.split(":", 1)[1].strip()) if ":" in v else "")
    return v


def pixel_metrics(masks, gts):
    pred, gt = torch.stack(masks), torch.stack(gts)
    inter, union = int((pred & gt).sum()), int((pred | gt).sum())
    r = {"pred_px": int(pred.sum()), "true_px": int(gt.sum()), "inter": inter, "union": union,
         "fp_px": int((pred & ~gt).sum()), "fp_rate": float((pred & ~gt).float().mean())}
    if gt.any():
        r["iou"] = inter / union
        r["recall"] = inter / int(gt.sum())
        r["precision"] = inter / max(1, int(pred.sum()))
        dists = []
        for p, g in zip(masks, gts):
            fp = (p & ~g).numpy()
            if fp.any() and g.any():
                dists.append(distance_transform_edt(~g.numpy())[fp])
        if dists:
            dd = np.concatenate(dists)
            r["fp_dist_to_truth_px"] = {"median": float(np.median(dd)), "within_3px": float((dd <= 3).mean()),
                                        "within_6px": float((dd <= 6).mean()), "max": float(dd.max())}
    return r


def antenna_geometry(means):
    """Where each antenna Gaussian lies: dish rim, dish faces or mast."""
    c = torch.tensor([0.05, 0.0, 1.1], device=means.device)
    Rd = _rot("y", 25.0).to(means.device)
    loc = (means - c) @ Rd  # world -> dish-local
    radial = loc[:, :2].norm(dim=1)
    on_dish = (loc[:, 2].abs() <= 0.08) & (radial <= 0.5)
    part = torch.full((len(means),), 0, dtype=torch.long)  # 0 mast, 1 dish face, 2 dish rim
    part[on_dish.cpu() & (radial.cpu() <= 0.34)] = 1
    part[on_dish.cpu() & (radial.cpu() > 0.34)] = 2
    face = torch.sign(loc[:, 2]).cpu()
    return part, face, radial.cpu()


def regen(seed, hold_cams, S, dev, train_views=96):
    """Holdout and inspection images for a new draw of pose error and noise, with the
    random sequence of tools/export_dataset.py (seed 0 reproduces the released images)."""
    gen = torch.Generator().manual_seed(seed)
    zero = torch.zeros(S, S, 3)
    for _ in range(train_views):  # training-view noise comes first in the original sequence
        add_sensor_noise(zero, gen)
    healthy = build_satellite(None, 68.0)
    hold = [add_sensor_noise(render_scene(healthy, perturb(c, 0.15, 0.01, gen), device=dev)["rgb"].cpu(), gen).to(dev)
            for c in hold_cams]
    insp = inspection_cameras(S, S)
    scen = {}
    for name, (fault, az) in FAULTS.items():
        scene, ref = build_satellite(fault, az), build_satellite(None, az)
        imgs, gts = [], []
        for c in insp:
            true = perturb(c, 0.15, 0.01, gen)
            r = render_scene(scene, true, device=dev)
            imgs.append(add_sensor_noise(r["rgb"].cpu(), gen).to(dev))
            if fault is None:
                gts.append(torch.zeros(S, S, dtype=torch.bool))
            else:
                h = render_scene(ref, true, device=dev)
                gts.append(((r["comp"] != h["comp"]) | (r["material"] != h["material"])).cpu())
        scen[name] = (imgs, insp, gts)
    return hold, scen


def run_all(model, hold, hold_cams, scen, tau=None, vote=0.5, factor=3.0, offset=0.02, comp_min=60, cls_kw=None):
    tau = calibrate_threshold(model, hold, hold_cams) if tau is None else tau
    deth = detect(model, hold, hold_cams, tau, vote_thresh=vote)
    thr = comp_levels(deth, factor, offset)
    out = {}
    for name, (imgs, cams, gts) in scen.items():
        det = detect(model, imgs, cams, tau, vote_thresh=vote, comp_thresholds=thr, comp_min=comp_min)
        cl = classify_components(model, imgs, cams, det, log=QUIET, **(cls_kw or {}))
        v = verdict(cl)
        out[name] = {"verdict": short(v), "correct": correct(v, EXPECTED[name]),
                     "flagged": {n: c["flagged"] for n, c in det.components.items()},
                     "changed": [n for n, c in det.components.items() if c["changed"]],
                     "classes": {n: {k: c[k] for k in ("type", "vanished_share", "inconsistent_share",
                                                       "new_matter_fraction")} for n, c in cl.items()},
                     **pixel_metrics(det.masks, gts)}
    return tau, thr, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/larissa_synth_v1")
    ap.add_argument("--out", default="audit")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = LarissaSynth(args.data)
    S = data.meta["resolution"]
    model = GaussianModel.from_state_dict(torch.load(Path(args.data) / "baseline_gaussians.pt", weights_only=True), dev)
    comps = model.comp.cpu()
    R = {}
    t_start = time.time()

    def save():
        (out / "audit.json").write_text(json.dumps(R, indent=1))

    # ------------------------------------------------------------------ 1. calibration
    hold, hold_cams = data.holdout()
    hold = [h.to(dev) for h in hold]
    ds, fg = [], []
    with torch.no_grad():
        for img, cam in zip(hold, hold_cams):
            o = render(model, cam)
            d = difference_map(img, o["rgb"])
            ds.append(d.flatten())
            fg.append((o["alpha"] > 0.5).flatten())
    d_all, fg_all = torch.cat(ds), torch.cat(fg)
    R["calibration"] = {
        "pixels": int(d_all.numel()), "foreground_share": float(fg_all.float().mean()),
        "q995_all": float(torch.quantile(d_all, 0.995)), "q999_all": float(torch.quantile(d_all, 0.999)),
        "q995_foreground": float(torch.quantile(d_all[fg_all], 0.995)),
        "q995_background": float(torch.quantile(d_all[~fg_all], 0.995)),
        "max": float(d_all.max()), "share_above_0.04": float((d_all > 0.04).float().mean()),
        "share_above_0.04_foreground": float((d_all[fg_all] > 0.04).float().mean()),
    }
    tau = calibrate_threshold(model, hold, hold_cams)
    deth = detect(model, hold, hold_cams, tau)
    R["calibration"].update({"tau": tau,
                             "healthy_fraction": {n: c["flagged"] / max(1, c["evaluated"]) for n, c in deth.components.items()},
                             "healthy_flagged": {n: c["flagged"] for n, c in deth.components.items()},
                             "levels": comp_levels(deth)})
    thr = comp_levels(deth)
    print("calibration", json.dumps(R["calibration"], indent=1), flush=True)
    save()

    # ------------------------------------------------------------------ 2. scenarios (released images)
    insp_true = {s: [camera_from_json(e["camera_true"]) for e in data.meta["inspection"][s]["views"]] for s in data.scenarios}
    R["scenarios"] = {}
    dets, scen0 = {}, {}
    for s in data.scenarios:
        imgs, cams, gts = data.inspection(s)
        imgs = [im.to(dev) for im in imgs]
        scen0[s] = (imgs, cams, gts)
        det = detect(model, imgs, cams, tau, comp_thresholds=thr)
        cl = classify_components(model, imgs, cams, det, log=QUIET)
        v = verdict(cl)
        dets[s] = det
        r = {"verdict": v, "correct": correct(v, EXPECTED[s]), **pixel_metrics(det.masks, gts),
             "flagged": {n: c["flagged"] for n, c in det.components.items()},
             "classes": cl}
        fault, az = FAULTS[s]
        if fault is not None:
            # recall by true contrast between faulty and healthy scene at the true pose (noise free)
            scene, ref = build_satellite(fault, az), build_satellite(None, az)
            bins = {"<0.02": [0, 0], "0.02-0.04": [0, 0], "0.04-0.08": [0, 0], ">=0.08": [0, 0]}
            for m, g, ct in zip(det.masks, gts, insp_true[s]):
                a = render_scene(scene, ct, device=dev)["rgb"].cpu()
                b = render_scene(ref, ct, device=dev)["rgb"].cpu()
                c = (a - b).abs().mean(-1)[g]
                hit = m[g]
                for key, lo, hi in (("<0.02", 0, 0.02), ("0.02-0.04", 0.02, 0.04), ("0.04-0.08", 0.04, 0.08),
                                    (">=0.08", 0.08, 9)):
                    sel = (c >= lo) & (c < hi)
                    bins[key][0] += int(sel.sum())
                    bins[key][1] += int((sel & hit).sum())
            r["recall_by_contrast"] = {k: {"true_px": n, "recall": h / max(1, n)} for k, (n, h) in bins.items()}
        R["scenarios"][s] = r
        print(s, v, {k: r[k] for k in ("pred_px", "true_px", "inter", "union", "fp_rate")}, flush=True)
    save()

    # ------------------------------------------------------------------ 3. antenna flags on the healthy satellite
    ant = comps == 4
    A0 = dets["healthy_control"].flagged & ant
    part, face, radial = antenna_geometry(model.means)
    names = {0: "mast", 1: "dish_face", 2: "dish_rim"}
    R["antenna_flags"] = {
        "healthy_flagged": int(A0.sum()),
        "all_antenna_by_part": {names[k]: int((ant & (part == k)).sum()) for k in names},
        "healthy_flagged_by_part": {names[k]: int((A0 & (part == k)).sum()) for k in names},
        "healthy_flagged_face_sign": {"+": int((A0 & (face > 0)).sum()), "-": int((A0 & (face < 0)).sum())},
        "overlap_with_healthy": {s: {"flagged": int((d.flagged & ant).sum()),
                                     "same_as_healthy": int((d.flagged & A0).sum())} for s, d in dets.items()},
    }
    # which inspection views vote for them
    dh = dets["healthy_control"]
    per_view = []
    for (u, v, on), m in zip(dh.proj, dh.masks):
        mk = dilate(m, 5)[v, u]
        per_view.append({"on_surface": int((on & A0).sum()), "voted_changed": int((on & mk & A0).sum())})
    R["antenna_flags"]["per_view"] = per_view
    # cause: rerun the healthy scenario with pose error and noise switched off one at a time
    healthy = build_satellite(None, 68.0)
    nominal = inspection_cameras(S, S)
    gen = torch.Generator().manual_seed(12345)
    variants = {
        "nominal_pose_no_noise": [render_scene(healthy, c, device=dev)["rgb"] for c in nominal],
        "true_pose_no_noise": [render_scene(healthy, c, device=dev)["rgb"] for c in insp_true["healthy_control"]],
        "nominal_pose_with_noise": [add_sensor_noise(render_scene(healthy, c, device=dev)["rgb"].cpu(), gen).to(dev)
                                    for c in nominal],
        "released_images": scen0["healthy_control"][0],
    }
    R["antenna_flags"]["cause"] = {}
    for k, imgs in variants.items():
        d = detect(model, imgs, nominal, tau, comp_thresholds=thr)
        R["antenna_flags"]["cause"][k] = {"pred_px": int(torch.stack(d.masks).sum()),
                                         "flagged": {n: c["flagged"] for n, c in d.components.items()},
                                         "antenna_same_as_released": int((d.flagged & A0).sum())}
    # reconstruction error on the antenna: difference map, baseline vs noise-free truth at nominal poses
    ant_px, ant_hi, bus_px, bus_hi = 0, 0, 0, 0
    for c, img in zip(nominal, variants["nominal_pose_no_noise"]):
        lab = render_scene(healthy, c, device=dev)["comp"].cpu()
        d = difference_map(img, render(model, c)["rgb"]).cpu()
        ant_px += int((lab == 4).sum()); ant_hi += int(((lab == 4) & (d > tau)).sum())
        bus_px += int((lab == 1).sum()); bus_hi += int(((lab == 1) & (d > tau)).sum())
    R["antenna_flags"]["noise_free_share_above_tau"] = {"antenna": ant_hi / max(1, ant_px), "bus": bus_hi / max(1, bus_px)}
    print("antenna", json.dumps(R["antenna_flags"], indent=1), flush=True)
    save()

    # ------------------------------------------------------------------ 4. antenna-missing: bus flags in the antenna shadow?
    da = dets["antenna_missing"]
    bus = comps == 1
    B = da.flagged & bus
    sun = sun_direction(68.0).to(dev)
    shadowed = torch.zeros(len(model), dtype=torch.bool)
    o = model.means + 0.005 * sun
    dd = sun.expand_as(o).contiguous()
    for p in healthy.prims:
        if p.comp == 4:
            t, _, _ = _intersect(o, dd, p)
            shadowed |= torch.isfinite(t).cpu()
    z = model.means[:, 2].cpu()
    # brightness change at the flagged bus Gaussians (observed minus baseline, over on-surface views)
    imgs = scen0["antenna_missing"][0]
    diff_sum = torch.zeros(len(model)); cnt = torch.zeros(len(model))
    for (u, v, on), img, rend in zip(da.proj, imgs, da.renders):
        lum = lambda x: x @ torch.tensor([0.299, 0.587, 0.114])  # noqa: E731
        dl = lum(img.cpu())[v, u] - lum(rend)[v, u]
        diff_sum += torch.where(on, dl, torch.zeros_like(dl)); cnt += on.float()
    dlum = diff_sum / cnt.clamp(min=1)
    R["antenna_missing_bus"] = {
        "flagged": int(B.sum()), "flagged_in_antenna_shadow": int((B & shadowed).sum()),
        "all_bus_in_antenna_shadow": int((bus & shadowed).sum()), "all_bus": int(bus.sum()),
        "flagged_top_face": int((B & (z > 0.70)).sum()),
        "flagged_mean_lum_change": float(dlum[B].mean()),
        "flagged_share_brighter": float((dlum[B] > 0).float().mean()),
    }
    print("antenna_missing bus", R["antenna_missing_bus"], flush=True)
    save()

    # ------------------------------------------------------------------ 5. sensitivity
    sens = []
    q = R["calibration"]["q995_all"]
    taus = sorted({round(q, 4), 0.03, 0.04, 0.05, 0.06})
    for t_, vote in itertools.product(taus, (0.4, 0.5, 0.6)):
        deth_ = detect(model, hold, hold_cams, t_, vote_thresh=vote)
        ds_ = {s: detect(model, im, ca, t_, vote_thresh=vote) for s, (im, ca, gt) in scen0.items()}
        for off, fac in itertools.product((0.01, 0.02, 0.04), (2.0, 3.0, 4.0)):
            thr_ = comp_levels(deth_, fac, off)
            row = {"tau": t_, "vote": vote, "offset": off, "factor": fac, "verdicts": {}}
            for s, (im, ca, gt) in scen0.items():
                d_ = relabel(ds_[s], thr_)
                v = verdict(classify_components(model, im, ca, d_, log=QUIET))
                row["verdicts"][s] = short(v)
                row.setdefault("ok", 0)
                row["ok"] += int(correct(v, EXPECTED[s]))
            sens.append(row)
        print(f"sens tau {t_} vote {vote}: " + " ".join(str(r["ok"]) for r in sens[-9:]), flush=True)
    R["sensitivity_detection"] = sens
    save()
    # classifier thresholds, one at a time around the defaults
    cls_grid = {"black": (0.02, 0.035, 0.05), "spread_margin": (0.06, 0.12, 0.18), "vanish_share": (0.2, 0.4, 0.6),
                "incons_share": (0.15, 0.3, 0.45), "new_matter": (0.05, 0.10, 0.15, 0.20)}
    cs = []
    for key, vals in cls_grid.items():
        for val in vals:
            row = {"param": key, "value": val, "verdicts": {}, "ok": 0}
            for s, (im, ca, gt) in scen0.items():
                d_ = relabel(dets[s], thr)
                v = verdict(classify_components(model, im, ca, d_, log=QUIET, **{key: val}))
                row["verdicts"][s] = short(v)
                row["ok"] += int(correct(v, EXPECTED[s]))
            cs.append(row)
            print("cls", key, val, row["ok"], row["verdicts"], flush=True)
    # component minimum count
    for cm in (30, 60, 120, 240):
        row = {"param": "comp_min", "value": cm, "verdicts": {}, "ok": 0}
        for s, (im, ca, gt) in scen0.items():
            d_ = relabel(dets[s], thr, comp_min=cm)
            v = verdict(classify_components(model, im, ca, d_, log=QUIET))
            row["verdicts"][s] = short(v)
            row["ok"] += int(correct(v, EXPECTED[s]))
        cs.append(row)
        print("comp_min", cm, row["ok"], row["verdicts"], flush=True)
    R["sensitivity_classifier"] = cs
    save()

    # ------------------------------------------------------------------ 6. repeat draws
    reps = []
    for seed in range(0, args.seeds + 1):
        t0 = time.time()
        hold_s, scen_s = regen(seed, hold_cams, S, dev)
        tau_s, thr_s, res = run_all(model, hold_s, hold_cams, scen_s)
        reps.append({"seed": seed, "tau": tau_s, "levels": thr_s, "results": res})
        print(f"seed {seed}: tau {tau_s:.4f} ok {sum(r['correct'] for r in res.values())}/5 "
              + " ".join(f"{k[:6]}={r['verdict']}" for k, r in res.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        R["repeats"] = reps
        save()
    R["seconds"] = time.time() - t_start
    save()
    print("done in", R["seconds"], flush=True)


if __name__ == "__main__":
    main()
