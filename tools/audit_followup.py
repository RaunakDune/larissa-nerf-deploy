"""Follow-up checks to tools/audit_checks.py (post hoc, after its results were seen).

    python tools/audit_followup.py --out audit [--device cuda:0] [--seeds 20]

1. Why the sensitivity sweep fails where it does: the healthy antenna fraction on
   the held-out views at tau 0.05, and the classification evidence for the
   starboard array in the Sun-shift pass when the component offset is 0.01.
2. Component levels calibrated on a separate healthy pass of the inspection
   cameras themselves (instead of the held-out poses), evaluated on the same
   repeat draws as audit_checks.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_checks import EXPECTED, FAULTS, QUIET, comp_levels, correct, regen, relabel, short  # noqa: E402
from larissa.dataset import LarissaSynth  # noqa: E402
from larissa.detect import add_sensor_noise, calibrate_threshold, classify_components, detect, verdict  # noqa: E402
from larissa.gaussians import GaussianModel  # noqa: E402
from larissa.scene import build_satellite, inspection_cameras, perturb, render_scene  # noqa: E402


def healthy_pass(seed, S, dev):
    gen = torch.Generator().manual_seed(seed)
    healthy = build_satellite(None, 68.0)
    cams = inspection_cameras(S, S)
    imgs = [add_sensor_noise(render_scene(healthy, perturb(c, 0.15, 0.01, gen), device=dev)["rgb"].cpu(), gen).to(dev)
            for c in cams]
    return imgs, cams


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
    hold, hold_cams = data.holdout()
    hold = [h.to(dev) for h in hold]
    R = {}

    # ---------------------------------------------------------------- 1. sweep failure modes
    R["holdout_fraction_by_tau"] = {}
    for t in (0.03, 0.04, 0.05, 0.06):
        d = detect(model, hold, hold_cams, t)
        R["holdout_fraction_by_tau"][t] = {n: {"flagged": c["flagged"], "fraction": c["fraction"]} for n, c in d.components.items()}
    imgs, cams, _ = data.inspection("sun_shift_control")
    imgs = [i.to(dev) for i in imgs]
    deth = detect(model, hold, hold_cams, 0.04)
    thr = comp_levels(deth, 3.0, 0.01)
    det = detect(model, imgs, cams, 0.04, comp_thresholds=thr)
    cl = classify_components(model, imgs, cams, det, log=QUIET)
    R["sun_shift_offset_0.01"] = {"levels": thr, "components": det.components, "classes": cl, "verdict": verdict(cl)}
    # inspection-camera healthy pass: what the healthy dish error looks like from the inspection cameras
    imgs0, cams0, _ = data.inspection("healthy_control")
    d0 = detect(model, [i.to(dev) for i in imgs0], cams0, 0.04)
    R["healthy_inspection_fraction"] = {n: c["fraction"] for n, c in d0.components.items()}
    print(json.dumps(R, indent=1, default=str)[:4000], flush=True)

    # ---------------------------------------------------------------- 2. calibrate on the inspection cameras
    reps = []
    for seed in range(1, args.seeds + 1):
        t0 = time.time()
        hold_s, scen_s = regen(seed, hold_cams, S, dev)
        tau = calibrate_threshold(model, hold_s, hold_cams)
        cal_imgs, cal_cams = healthy_pass(10_000 + seed, S, dev)
        thr_s = comp_levels(detect(model, cal_imgs, cal_cams, tau))
        res = {}
        for name, (im, ca, gt) in scen_s.items():
            d = detect(model, im, ca, tau, comp_thresholds=thr_s)
            v = verdict(classify_components(model, im, ca, d, log=QUIET))
            res[name] = {"verdict": short(v), "correct": correct(v, EXPECTED[name]),
                         "changed": [n for n, c in d.components.items() if c["changed"]]}
        reps.append({"seed": seed, "tau": tau, "levels": thr_s, "results": res})
        print(f"seed {seed}: levels " + " ".join(f"{k[:4]}={v:.3f}" for k, v in thr_s.items())
              + f" ok {sum(r['correct'] for r in res.values())}/5 "
              + " ".join(f"{k[:6]}={r['verdict']}" for k, r in res.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    R["inspection_calibrated_repeats"] = reps
    (out / "audit_followup.json").write_text(json.dumps(R, indent=1, default=str))
    print("done", flush=True)


if __name__ == "__main__":
    main()
