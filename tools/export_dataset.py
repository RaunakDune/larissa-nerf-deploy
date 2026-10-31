"""Export the Larissa-Synth dataset: every image, label and camera used by run_demo.py.

The script replays the random sequence of run_demo.py (same seed, same order),
so the exported images are exactly those behind results/summary.json.

    python tools/export_dataset.py --device cuda:0 --out data/larissa_synth_v1 \
        --baseline results/baseline_gaussians.pt

Formats
  rgb_*.png    16-bit RGB, linear [0,1] scaled to 0..65535 (as fed to the method)
  depth_*.png  16-bit camera z-depth in millimetres (0 = background)
  comp_*.png   8-bit component id (see dataset.json)
  gt_*.png     8-bit ground-truth change mask (255 = changed)
  dataset.json cameras (intrinsics, world-to-camera R and t; nominal and true poses)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from larissa.scene import (COMPONENTS, MATERIALS, build_satellite, fibonacci_cameras, inspection_cameras,  # noqa: E402
                           perturb, render_scene)
from larissa.detect import add_sensor_noise  # noqa: E402

SCENARIOS = [
    ("healthy_control", None, 68.0),
    ("panel_hinge", "panel_hinge", 68.0),
    ("antenna_missing", "antenna_missing", 68.0),
    ("mli_discoloration", "mli_discoloration", 68.0),
    ("sun_shift_control", None, 98.0),
]


def cam_json(c):
    return {"fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy, "width": c.W, "height": c.H,
            "R_world_to_cam": c.R.tolist(), "t_world_to_cam": c.t.tolist()}


def save_rgb16(img: torch.Tensor, path: Path):
    """16-bit RGB PNG (Pillow cannot write these, so pypng is used)."""
    import png
    a = (img.clamp(0, 1).cpu().numpy() * 65535 + 0.5).astype(np.uint16)
    with open(path, "wb") as f:
        png.Writer(a.shape[1], a.shape[0], greyscale=False, bitdepth=16).write(f, a.reshape(a.shape[0], -1).tolist())


def save_u16(x: np.ndarray, path: Path):
    Image.fromarray(x.astype(np.uint16)).save(path)


def save_u8(x: np.ndarray, path: Path):
    Image.fromarray(x.astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="data/larissa_synth_v1")
    ap.add_argument("--baseline", default=None, help="trained baseline_gaussians.pt to include")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--train-views", type=int, default=96)
    ap.add_argument("--holdout-views", type=int, default=12)
    ap.add_argument("--pose-noise-deg", type=float, default=0.15)
    ap.add_argument("--pose-noise-m", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    S = args.res
    gen = torch.Generator().manual_seed(args.seed)
    meta = {"name": "Larissa-Synth v1", "resolution": S, "seed": args.seed,
            "components": COMPONENTS, "materials": {i + 1: n for i, n in enumerate(MATERIALS)},
            "sun_azimuth_deg_default": 68.0, "sun_elevation_deg": 30.0,
            "pose_noise": {"rotation_deg": args.pose_noise_deg, "translation_m": args.pose_noise_m},
            "sensor_noise": {"sigma": 0.01, "gain_jitter": 0.03},
            "train": [], "holdout": [], "inspection": {}}

    # ---- same order as run_demo.py
    healthy = build_satellite(None, 68.0)
    cams_all = fibonacci_cameras(args.train_views + args.holdout_views, 9.0, S, S)
    hold_idx = set(range(0, len(cams_all), max(1, len(cams_all) // args.holdout_views))[:args.holdout_views])
    train_cams = [c for i, c in enumerate(cams_all) if i not in hold_idx]
    hold_cams = [c for i, c in enumerate(cams_all) if i in hold_idx]

    d = out / "train"
    d.mkdir(exist_ok=True)
    for i, c in enumerate(train_cams):
        r = render_scene(healthy, c, device=dev)
        rgb = add_sensor_noise(r["rgb"].cpu(), gen)
        save_rgb16(rgb, d / f"rgb_{i:03d}.png")
        save_u16((r["depth"].cpu().numpy() * 1000).round(), d / f"depth_{i:03d}.png")
        save_u8(r["comp"].cpu().numpy(), d / f"comp_{i:03d}.png")
        meta["train"].append({"id": i, "camera": cam_json(c)})

    d = out / "holdout"
    d.mkdir(exist_ok=True)
    for i, c in enumerate(hold_cams):
        true = perturb(c, args.pose_noise_deg, args.pose_noise_m, gen)
        rgb = add_sensor_noise(render_scene(healthy, true, device=dev)["rgb"].cpu(), gen)
        save_rgb16(rgb, d / f"rgb_{i:03d}.png")
        meta["holdout"].append({"id": i, "camera_nominal": cam_json(c), "camera_true": cam_json(true)})

    insp = inspection_cameras(S, S)
    for name, fault, az in SCENARIOS:
        d = out / "inspection" / name
        d.mkdir(parents=True, exist_ok=True)
        scene, ref = build_satellite(fault, az), build_satellite(None, az)
        entries = []
        for k, c in enumerate(insp):
            true = perturb(c, args.pose_noise_deg, args.pose_noise_m, gen)
            r = render_scene(scene, true, device=dev)
            rgb = add_sensor_noise(r["rgb"].cpu(), gen)
            if fault is None:
                gt = np.zeros((S, S), np.uint8)
            else:
                h = render_scene(ref, true, device=dev)
                gt = (((r["comp"] != h["comp"]) | (r["material"] != h["material"])).cpu().numpy() * 255)
            save_rgb16(rgb, d / f"rgb_{k}.png")
            save_u8(gt, d / f"gt_{k}.png")
            save_u8(r["comp"].cpu().numpy(), d / f"comp_{k}.png")
            entries.append({"id": k, "camera_nominal": cam_json(c), "camera_true": cam_json(true)})
        meta["inspection"][name] = {"fault": fault, "sun_azimuth_deg": az, "views": entries}

    if args.baseline:
        shutil.copy(args.baseline, out / "baseline_gaussians.pt")
        mp = Path(args.baseline).with_name("baseline_meta.json")
        if mp.exists():
            shutil.copy(mp, out / "baseline_meta.json")
    (out / "dataset.json").write_text(json.dumps(meta, indent=1))
    print("exported to", out)


if __name__ == "__main__":
    main()
