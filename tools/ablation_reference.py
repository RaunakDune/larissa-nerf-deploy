"""Is the 3DGS baseline needed? Comparison with a stored reference image per camera.

    python tools/ablation_reference.py --out audit [--device cuda:0] [--seeds 10]

Two detectors see the same inspection images:

  3dgs  the paper's detector: render the 3DGS baseline at the (known) pose of
        each inspection camera, compare, vote on Gaussians, classify.
  ref2d no 3D model: each camera keeps one healthy reference image taken at its
        ORIGINAL pose; new images are compared with it using the same
        shift-tolerant difference map, threshold rule, component alarm rule and
        (where possible in 2D) the same classification tests. Component labels
        come from the label image of the reference view.

  warp2d as ref2d, but the reference image and its labels are first warped to
        the current camera pose with the homography induced by a plane through
        the spacecraft centre, facing the reference camera (poses are known).

All are calibrated on the same K healthy passes of the cameras at their current
poses (threshold from the pooled passes; each component's healthy flag rate is
the largest over the passes). "3dgs_k1" is the 3DGS detector with component
levels from the first pass only. The cameras are then moved along their rings by `delta` degrees of
azimuth away from the poses at which the reference images were taken
(delta = 0: fixed cameras, the setting of the paper).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_checks import EXPECTED, FAULTS, QUIET, comp_levels, correct, pixel_metrics, relabel, short  # noqa: E402
from larissa.dataset import LarissaSynth  # noqa: E402
from larissa.detect import (_lum, add_sensor_noise, binary_open, calibrate_threshold, classify_components,  # noqa: E402
                            detect, difference_map, dilate, erode, verdict)
from larissa.gaussians import GaussianModel, render  # noqa: E402
from larissa.scene import COMPONENTS, build_satellite, look_at, perturb, render_scene  # noqa: E402
import torch.nn.functional as F  # noqa: E402

TARGET = {"panel_hinge": "solar_array_port", "antenna_missing": "antenna", "mli_discoloration": "bus"}


def cams_at(delta_deg, S, radius=8.5):
    cams = []
    for k in range(8):
        az = math.radians(22.5 + 45 * k + delta_deg)
        el = math.radians(28 if k % 2 == 0 else -12)
        eye = (radius * math.cos(el) * math.cos(az), radius * math.cos(el) * math.sin(az), radius * math.sin(el))
        cams.append(look_at(eye, W=S, H=S))
    return cams


def warp_to(img, cam0, cam1, nearest=False):
    """Warp an image (or label map) taken by cam0 into the view of cam1, assuming the
    scene lies on the plane through the world origin that faces cam0."""
    dev = img.device
    K = torch.tensor([[cam0.fx, 0, cam0.cx], [0, cam0.fy, cam0.cy], [0, 0, 1.0]])
    Rr = cam1.R @ cam0.R.T
    tr = cam1.t - Rr @ cam0.t
    n = torch.tensor([0.0, 0.0, 1.0])
    Hm = K @ (Rr + torch.outer(tr, n) / float(cam0.t[2])) @ torch.linalg.inv(K)   # cam0 pixels -> cam1 pixels
    Hi = torch.linalg.inv(Hm).to(dev)
    ys, xs = torch.meshgrid(torch.arange(cam1.H, device=dev, dtype=torch.float32) + 0.5,
                            torch.arange(cam1.W, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
    p = torch.stack([xs, ys, torch.ones_like(xs)], -1) @ Hi.T
    gx, gy = p[..., 0] / p[..., 2], p[..., 1] / p[..., 2]
    grid = torch.stack([2 * gx / cam0.W - 1, 2 * gy / cam0.H - 1], -1)[None]
    x = img.float()
    x = x.permute(2, 0, 1)[None] if x.dim() == 3 else x[None, None]
    o = F.grid_sample(x, grid, mode="nearest" if nearest else "bilinear", padding_mode="zeros", align_corners=False)[0]
    return o.permute(1, 2, 0) if img.dim() == 3 else o[0].to(img.dtype)


def blur3(x):
    return F.avg_pool2d(x.permute(2, 0, 1)[None], 3, 1, 1, count_include_pad=False)[0].permute(1, 2, 0)


@torch.no_grad()
def tau2d(imgs, refs, quantile=0.995, floor=0.04):
    v = torch.cat([difference_map(i, r).flatten() for i, r in zip(imgs, refs)])
    return max(floor, torch.quantile(v, quantile).item())


@torch.no_grad()
def detect2d(imgs, refs, ref_comp, tau):
    masks, voted, comps = [], [], {n: [0, 0] for c, n in COMPONENTS.items() if c}
    for img, ref, comp in zip(imgs, refs, ref_comp):
        mask = binary_open(difference_map(img, ref) > tau)
        mk = dilate(mask, 5)
        masks.append(mask)
        voted.append(mk)
        for cid, name in COMPONENTS.items():
            if cid:
                sel = comp == cid
                comps[name][0] += int(sel.sum())
                comps[name][1] += int((sel & mk).sum())
    return {"masks": masks, "voted": voted,
            "components": {n: {"evaluated": e, "flagged": f, "fraction": f / max(1, e)} for n, (e, f) in comps.items()}}


@torch.no_grad()
def classify2d(imgs, refs, ref_comp, det, thr, comp_min=60, black=0.035, lit=0.08, vanish_share=0.4, new_matter=0.10):
    """Vanished and new-matter tests in image space. The multi-view consistency
    test has no 2D equivalent (no correspondence between cameras)."""
    ids = {n: c for c, n in COMPONENTS.items()}
    out = {}
    for name, c in det["components"].items():
        c["changed"] = bool(c["fraction"] >= thr[name] and c["flagged"] >= comp_min)
        if not c["changed"]:
            continue
        van = n = new = raw = 0
        for img, ref, comp, mask, mk in zip(imgs, refs, ref_comp, det["masks"], det["voted"]):
            sel = (comp == ids[name]) & mk
            if not sel.any():
                continue
            lo, lb = _lum(blur3(img))[sel], _lum(blur3(ref))[sel]
            van += int(((lo < black) & (lb > lit)).sum())
            n += int(sel.sum())
            r = mask & dilate(sel, 21)
            new += int((r & erode(comp == 0, 5)).sum())
            raw += int(r.sum())
        v_share, new_frac = van / max(1, n), new / max(1, raw)
        out[name] = {"type": "geometric" if (v_share >= vanish_share or new_frac >= new_matter) else "appearance",
                     "vanished_share": round(v_share, 3), "new_matter_fraction": round(new_frac, 3)}
    return out


def summarize(name, v, changed, masks, gts):
    r = {"verdict": short(v), "correct": correct(v, EXPECTED[name]), "changed": changed}
    m = pixel_metrics([x.cpu() for x in masks], gts)
    r.update({k: m[k] for k in ("fp_rate", "iou", "recall", "precision") if k in m})
    if name in TARGET:
        r["target_reported"] = TARGET[name] in changed
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/larissa_synth_v1")
    ap.add_argument("--out", default="audit")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--passes", type=int, default=5, help="healthy calibration passes")
    ap.add_argument("--tag", default="")
    ap.add_argument("--deltas", default="0,1,2,5,10,20,45")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = LarissaSynth(args.data)
    S = data.meta["resolution"]
    model = GaussianModel.from_state_dict(torch.load(Path(args.data) / "baseline_gaussians.pt", weights_only=True), dev)
    healthy = build_satellite(None, 68.0)
    deltas = [float(x) for x in args.deltas.split(",")]
    runs = []

    def capture(scene, cam, gen):
        true = perturb(cam, 0.15, 0.01, gen)
        return add_sensor_noise(render_scene(scene, true, device=dev)["rgb"].cpu(), gen).to(dev), true

    M2 = ("ref2d", "warp2d")
    diag = {}
    fname = f"ablation_reference{args.tag}.json"

    def levels(fracs_per_pass):
        return {n: max(3 * f, f + 0.02) for n, f in
                {n: max(p[n] for p in fracs_per_pass) for n in fracs_per_pass[0]}.items()}

    for delta in deltas:
        ref_cams, cams = cams_at(0.0, S), cams_at(delta, S)
        comp0 = [render_scene(healthy, c, device=dev)["comp"] for c in ref_cams]
        ref_comp = {"ref2d": comp0, "warp2d": [warp_to(c, a, b, nearest=True) for c, a, b in zip(comp0, ref_cams, cams)]}
        # how well the baseline renders these views (noise free, nominal poses)
        ps, hi, fg = [], 0, 0
        with torch.no_grad():
            for c in cams:
                tr = render_scene(healthy, c, device=dev)
                o = render(model, c)["rgb"].clamp(0, 1)
                ps.append(float(-10 * torch.log10(((o - tr["rgb"]) ** 2).mean())))
                dm = difference_map(tr["rgb"], o)
                hi += int(((tr["comp"] > 0) & (dm > 0.04)).sum()); fg += int((tr["comp"] > 0).sum())
        diag[delta] = {"psnr": st.mean(ps), "foreground_share_above_0.04": hi / fg}
        print(f"delta {delta:g}: baseline PSNR {diag[delta]['psnr']:.2f} dB, foreground above 0.04: {100 * hi / fg:.2f}%", flush=True)
        for seed in range(args.seeds):
            t0 = time.time()
            gen = torch.Generator().manual_seed(200_000 + int(round(delta * 1000)) * 100 + seed)
            refs0 = [capture(healthy, c, gen)[0] for c in ref_cams]         # stored reference images (original poses)
            refs = {"ref2d": refs0, "warp2d": [warp_to(r, a, b) for r, a, b in zip(refs0, ref_cams, cams)]}
            cal = [[capture(healthy, c, gen)[0] for c in cams] for _ in range(args.passes)]   # healthy passes, current poses
            flat = [i for p in cal for i in p]
            tau_g = calibrate_threshold(model, flat, cams * args.passes)
            fr = [{n: c["fraction"] for n, c in detect(model, p, cams, tau_g).components.items()} for p in cal]
            thr_g, thr_g1 = levels(fr), levels(fr[:1])
            run = {"delta": delta, "seed": seed, "tau_3dgs": tau_g, "tau_3dgs_k1": tau_g,
                   "levels_3dgs": thr_g, "levels_3dgs_k1": thr_g1, "3dgs": {}, "3dgs_k1": {}}
            tau_r, thr_r = {}, {}
            for m in M2:
                tau_r[m] = tau2d(flat, refs[m] * args.passes)
                thr_r[m] = levels([{n: c["fraction"] for n, c in detect2d(p, refs[m], ref_comp[m], tau_r[m])["components"].items()}
                                   for p in cal])
                run.update({"tau_" + m: tau_r[m], "levels_" + m: thr_r[m], m: {}})
            for name, (fault, az) in FAULTS.items():
                scene, href = build_satellite(fault, az), build_satellite(None, az)
                imgs, gts = [], []
                for c in cams:
                    img, true = capture(scene, c, gen)
                    imgs.append(img)
                    if fault is None:
                        gts.append(torch.zeros(S, S, dtype=torch.bool))
                    else:
                        a, b = render_scene(scene, true, device=dev), render_scene(href, true, device=dev)
                        gts.append(((a["comp"] != b["comp"]) | (a["material"] != b["material"])).cpu())
                det = detect(model, imgs, cams, tau_g, comp_thresholds=thr_g)
                for key, th in (("3dgs", thr_g), ("3dgs_k1", thr_g1)):
                    relabel(det, th)
                    cl = classify_components(model, imgs, cams, det, log=QUIET)
                    run[key][name] = summarize(name, verdict(cl), [n for n, c in det.components.items() if c["changed"]],
                                               det.masks, gts)
                    run[key][name]["types"] = {n: c["type"] for n, c in cl.items()}
                for m in M2:
                    d2 = detect2d(imgs, refs[m], ref_comp[m], tau_r[m])
                    cl = classify2d(imgs, refs[m], ref_comp[m], d2, thr_r[m])
                    run[m][name] = summarize(name, verdict(cl), [n for n, c in d2["components"].items() if c["changed"]],
                                             d2["masks"], gts)
                    run[m][name]["types"] = {n: c["type"] for n, c in cl.items()}
            runs.append(run)
            print(f"delta {delta:g} seed {seed}: tau 3dgs {tau_g:.3f} ref2d {tau_r['ref2d']:.3f} warp2d {tau_r['warp2d']:.3f} | "
                  + " ".join(f"{n[:5]} " + "/".join(run[m][n]["verdict"][:5] for m in ("3dgs", "3dgs_k1") + M2) for n in FAULTS)
                  + f" ({time.time() - t0:.0f}s)", flush=True)
            (out / fname).write_text(json.dumps({"diag": diag, "runs": runs}, indent=1))

    # ---------------------------------------------------------------- summary
    summary = []
    print("\ndelta method   tau    correct verdicts by scenario (healthy hinge antenna blanket sun) | total | "
          "IoU hinge/antenna/blanket | healthy FP px", flush=True)
    for delta in deltas:
        rs = [r for r in runs if r["delta"] == delta]
        for m in ("3dgs", "3dgs_k1", "ref2d", "warp2d"):
            row = {"delta": delta, "method": m, "n": len(rs), "tau_mean": st.mean(r["tau_" + m] for r in rs)}
            for name in FAULTS:
                row[name] = {"correct": sum(r[m][name]["correct"] for r in rs),
                             "fp_rate": st.mean(r[m][name]["fp_rate"] for r in rs)}
                if name in TARGET:
                    row[name].update({"iou": st.mean(r[m][name]["iou"] for r in rs),
                                      "recall": st.mean(r[m][name]["recall"] for r in rs),
                                      "target_reported": sum(r[m][name]["target_reported"] for r in rs),
                                      "extra_parts": sum(len(set(r[m][name]["changed"]) - {TARGET[name], "bus"}) > 0 for r in rs)})
            row["total"] = sum(row[n]["correct"] for n in FAULTS)
            summary.append(row)
            print(f"{delta:5g} {m:8s} {row['tau_mean']:.3f}  " + " ".join(f"{row[n]['correct']:2d}" for n in FAULTS)
                  + f" | {row['total']:3d}/{5 * len(rs)} | "
                  + "/".join(f"{row[n]['iou']:.2f}" for n in TARGET)
                  + f" | {100 * row['healthy_control']['fp_rate']:.2f}%", flush=True)
    (out / fname).write_text(json.dumps({"summary": summary, "diag": diag, "runs": runs}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
