"""Project Larissa demo: Gaussian Splatting change detection on a synthetic satellite.

    python run_demo.py --device cuda:0            # full run (P40: about 10 to 20 minutes)
    python run_demo.py --device cpu --quick       # small smoke test
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import torch

from larissa.scene import (COMPONENTS, build_satellite, fibonacci_cameras, inspection_cameras, look_at,
                           perturb, render_scene)
from larissa.gaussians import GaussianModel, init_from_views, psnr, render, train
from larissa.detect import (add_sensor_noise, calibrate_threshold, classify_components, detect, verdict)
from larissa import report

SCENARIOS = [
    # name, fault, sun azimuth, description, expected verdict prefix
    ("healthy_control", None, 68.0, "No fault; same Sun angle; pose error and sensor noise only", "NO CHANGE"),
    ("panel_hinge", "panel_hinge", 68.0, "Port solar array hinge stuck 22 degrees short of deployment", "GEOMETRIC"),
    ("antenna_missing", "antenna_missing", 68.0, "Antenna dish and mast detached", "GEOMETRIC"),
    ("mli_discoloration", "mli_discoloration", 68.0, "Thermal blanket patch degraded (darkened) on the bus", "APPEARANCE"),
    ("sun_shift_control", None, 98.0, "No fault; Sun azimuth moved 30 degrees", "NO CHANGE"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--train-views", type=int, default=96)
    ap.add_argument("--holdout-views", type=int, default=12)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--refit-iters", type=int, default=200)
    ap.add_argument("--pose-noise-deg", type=float, default=0.15)
    ap.add_argument("--pose-noise-m", type=float, default=0.01)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a CPU smoke test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        args.res, args.train_views, args.holdout_views, args.iters, args.refit_iters = 96, 24, 4, 150, 20

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "w")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    gen = torch.Generator().manual_seed(args.seed)
    S = args.res
    timings = {}
    log(f"Project Larissa demo | device {dev} "
        f"({torch.cuda.get_device_name(dev) if dev.type == 'cuda' else platform.processor() or 'cpu'}) | res {S}")

    # ------------------------------------------------------------------ capture
    t0 = time.time()
    healthy = build_satellite(None, 68.0)
    cams_all = fibonacci_cameras(args.train_views + args.holdout_views, 9.0, S, S)
    hold_idx = set(range(0, len(cams_all), max(1, len(cams_all) // args.holdout_views))[:args.holdout_views])
    train_cams = [c for i, c in enumerate(cams_all) if i not in hold_idx]
    hold_cams = [c for i, c in enumerate(cams_all) if i in hold_idx]
    train_views = [render_scene(healthy, c, device=dev) for c in train_cams]
    for v in train_views:
        v["rgb"] = add_sensor_noise(v["rgb"].cpu(), gen).to(dev)
        for k in ("depth", "comp"):
            v[k] = v[k].cpu()
    # held-out healthy captures with the same pose error model as inspections
    hold_imgs = []
    for c in hold_cams:
        true = perturb(c, args.pose_noise_deg, args.pose_noise_m, gen)
        hold_imgs.append(add_sensor_noise(render_scene(healthy, true, device=dev)["rgb"].cpu(), gen).to(dev))
    timings["capture_s"] = time.time() - t0
    log(f"captured {len(train_cams)} training and {len(hold_cams)} held-out views in {timings['capture_s']:.1f}s")

    # ------------------------------------------------------------------ baseline
    t0 = time.time()
    views_cpu = [{"rgb": v["rgb"].cpu(), "depth": v["depth"], "comp": v["comp"]} for v in train_views]
    model = init_from_views(views_cpu, [c for c in train_cams], stride=2 if S <= 128 else 3,
                            voxel=0.035 if S <= 128 else 0.02, max_points=6000 if args.quick else 60000,
                            device=dev)
    log(f"initialized {len(model)} Gaussians (labelled by component)")
    model = train(model, [v["rgb"] for v in train_views], train_cams, iters=args.iters,
                  log_every=max(1, args.iters // 10),
                  prune_at=tuple(int(args.iters * f) for f in (0.3, 0.6)), log=log)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    timings["baseline_train_s"] = time.time() - t0
    with torch.no_grad():
        clean_hold = [render_scene(healthy, c, device=dev)["rgb"] for c in hold_cams]
        hold_psnr = sum(psnr(render(model, c)["rgb"].clamp(0, 1), g) for c, g in zip(hold_cams, clean_hold)) / len(hold_cams)
    log(f"baseline trained in {timings['baseline_train_s']:.1f}s | {len(model)} Gaussians | held-out PSNR {hold_psnr:.2f} dB")
    torch.save(model.state_dict(), out / "baseline_gaussians.pt")

    # ------------------------------------------------------------------ threshold
    tau = calibrate_threshold(model, hold_imgs, hold_cams)
    log(f"change threshold calibrated on healthy held-out views: tau = {tau:.4f}")

    # ------------------------------------------------------------------ inspections
    insp_cams = inspection_cameras(S, S)
    overview = look_at((6.5, 5.0, 4.5), W=S * 2, H=S * 2, fov_scale=1.25)
    results = []
    panels = []
    for name, fault, sun_az, desc, expected in SCENARIOS:
        log(f"\n=== scenario: {name} | {desc}")
        scene = build_satellite(fault, sun_az)
        ref = build_satellite(None, sun_az)
        imgs, gt_masks = [], []
        for c in insp_cams:
            true = perturb(c, args.pose_noise_deg, args.pose_noise_m, gen)
            r = render_scene(scene, true, device=dev)
            imgs.append(add_sensor_noise(r["rgb"].cpu(), gen).to(dev))
            # ground truth: pixels whose component or material differ from the healthy satellite
            if fault is None:
                gt_masks.append(torch.zeros(S, S, dtype=torch.bool))
            else:
                h = render_scene(ref, true, device=dev)
                gt_masks.append(((r["comp"] != h["comp"]) | (r["material"] != h["material"])).cpu())
        if dev.type == "cuda":
            torch.cuda.synchronize()
        det = detect(model, imgs, insp_cams, tau)
        log(f"check over {len(insp_cams)} cameras took {det.seconds:.2f}s")
        for cname, info in det.components.items():
            if info["flagged"]:
                log(f"  {cname:22s} flagged {info['flagged']:5d}/{info['evaluated']:5d} "
                    f"({100 * info['fraction']:.1f}%) {'CHANGED' if info['changed'] else ''}")
        classes = classify_components(model, imgs, insp_cams, det, iters=args.refit_iters, log=log)
        v = verdict(classes)
        log(f"VERDICT: {v}")
        pred = torch.stack(det.masks)
        gt = torch.stack(gt_masks)
        inter, union = int((pred & gt).sum()), int((pred | gt).sum())
        fp_rate = float((pred & ~gt).float().mean())
        res = {
            "scenario": name, "description": desc, "expected": expected, "verdict": v,
            "correct": v.startswith(expected),
            "pixel_iou": round(inter / union, 3) if gt.any() else None,
            "pixel_recall": round(inter / max(1, int(gt.sum())), 3) if gt.any() else None,
            "false_positive_pixel_rate": round(fp_rate, 5),
            "changed_pixels_predicted": int(pred.sum()), "changed_pixels_true": int(gt.sum()),
            "unexplained_pixels": det.unexplained_px,
            "check_seconds": round(det.seconds, 2),
            "components": det.components, "classification": classes,
        }
        results.append(res)
        # overview render with flagged Gaussians in red
        with torch.no_grad():
            dirs = model.means - overview.center.to(dev)
            dirs = dirs / dirs.norm(dim=1, keepdim=True)
            from larissa.gaussians import sh_to_rgb
            base = sh_to_rgb(model.sh, dirs).clamp(0, 1)
            grey = (0.35 + 0.65 * base.mean(1, keepdim=True)).expand(-1, 3) * 0.75
            fl = det.flagged.to(dev)[:, None]
            col = torch.where(fl, torch.tensor([1.0, 0.18, 0.12], device=dev), grey)
            loc = render(model, overview, colors_override=col)["rgb"].clamp(0, 1).cpu()
        best = int(torch.tensor([int(m.sum()) for m in det.masks]).argmax()) if any(m.any() for m in det.masks) else 0
        panels.append({"name": name, "desc": desc, "verdict": v, "obs": imgs[best].cpu(), "render": det.renders[best],
                       "dmap": det.dmaps[best], "mask": det.masks[best], "gt": gt_masks[best], "loc": loc, "tau": tau})

    # ------------------------------------------------------------------ outputs
    summary = {
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "resolution": S, "training_views": len(train_cams), "inspection_cameras": len(insp_cams),
        "gaussians": len(model), "baseline_heldout_psnr_db": round(hold_psnr, 2),
        "threshold_tau": round(tau, 4), "timings_s": {k: round(v, 1) for k, v in timings.items()},
        "pose_noise": {"rotation_deg": args.pose_noise_deg, "translation_m": args.pose_noise_m},
        "scenarios": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    report.write_markdown(summary, out / "summary.md")
    report.scenario_figure(panels, out / "scenarios.png")
    report.capture_figure([v["rgb"].cpu() for v in train_views[:8]], out / "training_views.png")
    log(f"\nwrote {out}/summary.json, summary.md, scenarios.png")
    n_ok = sum(r["correct"] for r in results)
    log(f"{n_ok}/{len(results)} scenarios matched the expected verdict")


if __name__ == "__main__":
    main()
