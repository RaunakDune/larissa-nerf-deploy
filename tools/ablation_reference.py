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

Both are calibrated on the same healthy pass of the cameras at their current
poses. The cameras are then moved along their rings by `delta` degrees of
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
from audit_checks import EXPECTED, FAULTS, QUIET, comp_levels, correct, pixel_metrics, short  # noqa: E402
from larissa.dataset import LarissaSynth  # noqa: E402
from larissa.detect import (_lum, add_sensor_noise, binary_open, calibrate_threshold, classify_components,  # noqa: E402
                            detect, difference_map, dilate, erode, verdict)
from larissa.gaussians import GaussianModel  # noqa: E402
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

    for di, delta in enumerate(deltas):
        ref_cams, cams = cams_at(0.0, S), cams_at(delta, S)
        ref_comp = [render_scene(healthy, c, device=dev)["comp"] for c in ref_cams]
        for seed in range(args.seeds):
            t0 = time.time()
            gen = torch.Generator().manual_seed(100_000 + 1000 * di + seed)
            refs = [capture(healthy, c, gen)[0] for c in ref_cams]          # stored reference images (original poses)
            cal = [capture(healthy, c, gen)[0] for c in cams]               # healthy calibration pass (current poses)
            # --- calibration, same rules for both detectors
            tau_g = calibrate_threshold(model, cal, cams)
            thr_g = comp_levels(detect(model, cal, cams, tau_g))
            tau_r = tau2d(cal, refs)
            dcal = detect2d(cal, refs, ref_comp, tau_r)
            thr_r = {n: max(3 * c["fraction"], c["fraction"] + 0.02) for n, c in dcal["components"].items()}
            run = {"delta": delta, "seed": seed, "tau_3dgs": tau_g, "tau_ref2d": tau_r,
                   "levels_3dgs": thr_g, "levels_ref2d": thr_r, "3dgs": {}, "ref2d": {}}
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
                v = verdict(classify_components(model, imgs, cams, det, log=QUIET))
                run["3dgs"][name] = summarize(name, v, [n for n, c in det.components.items() if c["changed"]], det.masks, gts)
                d2 = detect2d(imgs, refs, ref_comp, tau_r)
                v2 = verdict(classify2d(imgs, refs, ref_comp, d2, thr_r))
                run["ref2d"][name] = summarize(name, v2, [n for n, c in d2["components"].items() if c["changed"]],
                                               d2["masks"], gts)
            runs.append(run)
            print(f"delta {delta:g} seed {seed}: tau 3dgs {tau_g:.3f} ref2d {tau_r:.3f} | "
                  + " ".join(f"{n[:5]} {run['3dgs'][n]['verdict'][:5]}/{run['ref2d'][n]['verdict'][:5]}" for n in FAULTS)
                  + f" ({time.time() - t0:.0f}s)", flush=True)
            (out / "ablation_reference.json").write_text(json.dumps({"runs": runs}, indent=1))

    # ---------------------------------------------------------------- summary
    summary = []
    print("\ndelta method   tau    correct verdicts by scenario (healthy hinge antenna blanket sun) | total | "
          "IoU hinge/antenna/blanket | healthy FP px", flush=True)
    for delta in deltas:
        rs = [r for r in runs if r["delta"] == delta]
        for m in ("3dgs", "ref2d"):
            row = {"delta": delta, "method": m, "n": len(rs), "tau_mean": st.mean(r["tau_" + m] for r in rs)}
            for name in FAULTS:
                row[name] = {"correct": sum(r[m][name]["correct"] for r in rs),
                             "fp_rate": st.mean(r[m][name]["fp_rate"] for r in rs)}
                if name in TARGET:
                    row[name].update({"iou": st.mean(r[m][name]["iou"] for r in rs),
                                      "recall": st.mean(r[m][name]["recall"] for r in rs),
                                      "target_reported": sum(r[m][name]["target_reported"] for r in rs)})
            row["total"] = sum(row[n]["correct"] for n in FAULTS)
            summary.append(row)
            print(f"{delta:5g} {m:6s} {row['tau_mean']:.3f}  " + " ".join(f"{row[n]['correct']:2d}" for n in FAULTS)
                  + f" | {row['total']:3d}/{5 * len(rs)} | "
                  + "/".join(f"{row[n]['iou']:.2f}" for n in TARGET)
                  + f" | {100 * row['healthy_control']['fp_rate']:.2f}%", flush=True)
    (out / "ablation_reference.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
