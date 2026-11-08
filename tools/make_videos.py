"""Orbit video of the satellite, the 3DGS baseline and the detection results.

    python tools/make_videos.py --out videos [--device cuda:0] [--preview]

Writes videos/larissa_orbit.mp4 (1280x720, 30 fps, H.264). With --preview it
writes a few full-size JPEG frames instead, for checking the layout.

Scenes: title; healthy satellite (ray-traced, 3DGS render, Gaussians by part);
inside the model (individual splats shrunk to show their layout); one orbit per
scenario with the faulty spacecraft and the Gaussians the detector flagged.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from larissa.dataset import LarissaSynth  # noqa: E402
from larissa.detect import calibrate_components, calibrate_threshold, classify_components, detect, verdict  # noqa: E402
from larissa.gaussians import GaussianModel, render, sh_to_rgb  # noqa: E402
from larissa.scene import build_satellite, look_at, render_scene  # noqa: E402

W, H, FPS, P = 1280, 720, 30, 400
XS = (20, 440, 860)
PY = 92
BG = (11, 14, 22)
FG = (232, 236, 242)
DIM = (150, 160, 175)
ACC = (255, 120, 90)
COMP_COLORS = {1: (0.85, 0.55, 0.15), 2: (0.20, 0.45, 0.85), 3: (0.30, 0.75, 0.95), 4: (0.85, 0.25, 0.30), 5: (0.55, 0.55, 0.60)}
SCEN = [
    ("panel_hinge", "panel_hinge", 68.0, "Fault: port solar array hinge stuck 22° short", "geometric"),
    ("antenna_missing", "antenna_missing", 68.0, "Fault: antenna dish and mast detached", "geometric"),
    ("mli_discoloration", "mli_discoloration", 68.0, "Fault: darkened thermal blanket patch on the bus", "appearance"),
    ("sun_shift_control", None, 98.0, "No fault: Sun moved 30° in azimuth", "no fault"),
]
PRETTY = {"solar_array_port": "port solar array", "solar_array_starboard": "starboard solar array"}


def font(size, bold=False):
    import matplotlib.font_manager as fm
    return ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")), size)


F_TITLE, F_SUB, F_LAB, F_CAP = font(40, True), font(24), font(20), font(21)


def orbit_cam(t, az0=35.0, r=9.0):
    az = math.radians(az0 + 360.0 * t)
    el = math.radians(24.0 + 10.0 * math.sin(2 * math.pi * t))
    eye = (r * math.cos(el) * math.cos(az), r * math.cos(el) * math.sin(az), r * math.sin(el) + 0.15)
    return look_at(eye, target=(0.0, 0.0, 0.15), W=P, H=P, fov_scale=1.25)


def sweep_cam(t, r=9.0):
    """Swing back and forth across the sunlit side (Sun azimuth 68 to 98 degrees)."""
    az = math.radians(75.0 + 65.0 * math.sin(2 * math.pi * t))
    el = math.radians(26.0 + 8.0 * math.sin(4 * math.pi * t))
    eye = (r * math.cos(el) * math.cos(az), r * math.cos(el) * math.sin(az), r * math.sin(el) + 0.15)
    return look_at(eye, target=(0.0, 0.0, 0.15), W=P, H=P, fov_scale=1.25)


def to_u8(x):
    return (np.clip(x.detach().float().cpu().numpy(), 0, 1) ** (1 / 1.25) * 255 + 0.5).astype(np.uint8)


def frame(panels, labels, title, caption=None, caption_color=FG, fade=1.0):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((W // 2, 46), title, font=F_SUB, fill=FG, anchor="mm")
    for x, p, lab in zip(XS, panels, labels):
        im.paste(Image.fromarray(p), (x, PY))
        d.rectangle((x - 1, PY - 1, x + P, PY + P), outline=(45, 52, 66))
        d.text((x + P // 2, PY + P + 24), lab, font=F_LAB, fill=DIM, anchor="mm")
    if caption:
        for k, line in enumerate(caption if isinstance(caption, list) else [caption]):
            col = caption_color if k == 0 else DIM
            d.text((W // 2, 590 + 38 * k), line, font=F_CAP if k else F_SUB, fill=col, anchor="mm")
    a = np.asarray(im)
    return a if fade >= 1 else (a * fade).astype(np.uint8)


def fade_of(i, n, k=15):
    return min(1.0, (i + 1) / k, (n - i) / k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/larissa_synth_v1")
    ap.add_argument("--out", default="videos")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = LarissaSynth(args.data)
    model = GaussianModel.from_state_dict(torch.load(Path(args.data) / "baseline_gaussians.pt", weights_only=True), dev)
    comp_col = torch.tensor([COMP_COLORS.get(int(c), (0, 0, 0)) for c in model.comp.cpu()], dtype=torch.float32, device=dev)
    keep = torch.sigmoid(model.opacity_logit).flatten() > 0.5  # opaque surface Gaussians only (no floaters)
    small = GaussianModel(model.means[keep], model.log_scales[keep] + math.log(0.22), model.quats[keep],
                          model.opacity_logit[keep].clamp(min=3.0), model.sh[keep], model.comp[keep])
    small_col = comp_col[keep]
    n_keep = int(keep.sum())

    # detection results on the released inspection images
    hold, hold_cams = data.holdout()
    hold = [h.to(dev) for h in hold]
    tau = calibrate_threshold(model, hold, hold_cams)
    thr = calibrate_components(model, hold, hold_cams, tau)
    flagged, verdicts = {}, {}
    for s, *_ in SCEN:
        imgs, cams, _ = data.inspection(s)
        imgs = [i.to(dev) for i in imgs]
        det = detect(model, imgs, cams, tau, comp_thresholds=thr)
        v = verdict(classify_components(model, imgs, cams, det, log=lambda *a: None))
        flagged[s], verdicts[s] = det.flagged.to(dev), v
        print(s, v, int(det.flagged.sum()), flush=True)

    def gs(m, cam, colors=None):
        with torch.no_grad():
            return to_u8(render(m, cam, colors_override=colors)["rgb"])

    def flag_colors(cam, fl):
        with torch.no_grad():
            dirs = model.means - cam.center.to(dev)
            dirs = dirs / dirs.norm(dim=1, keepdim=True)
            base = sh_to_rgb(model.sh, dirs).clamp(0, 1)
            grey = (0.35 + 0.65 * base.mean(1, keepdim=True)).expand(-1, 3) * 0.7
            return torch.where(fl[:, None], torch.tensor([1.0, 0.22, 0.12], device=dev), grey)

    healthy = build_satellite(None, 68.0)
    title = "Project Larissa: spacecraft change detection with 3D Gaussian Splatting"

    def scene_title():
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((W // 2, 300), "Project Larissa", font=F_TITLE, fill=FG, anchor="mm")
        d.text((W // 2, 360), "Spacecraft change detection with 3D Gaussian Splatting", font=F_SUB, fill=FG, anchor="mm")
        d.text((W // 2, 420), "Synthetic satellite, Larissa-Synth v1. Rendered on a Tesla P40.", font=F_CAP, fill=DIM, anchor="mm")
        a = np.asarray(im)
        for i in range(90):
            yield (a * fade_of(i, 90)).astype(np.uint8)

    def scene_healthy(n=240):
        for i in range(n):
            cam = orbit_cam(i / n)
            rt = to_u8(render_scene(healthy, cam, device=dev)["rgb"])
            yield frame([rt, gs(model, cam), gs(model, cam, comp_col)],
                        ["Ground truth (ray-traced)", "3DGS baseline (43,929 Gaussians)", "Gaussians by component"],
                        title, ["The healthy spacecraft and its learned baseline",
                                "bus (orange), port array (blue), starboard array (light blue), antenna (red), thruster (grey)"],
                        fade=fade_of(i, n))

    def scene_inside(n=180):
        for i in range(n):
            cam = orbit_cam(0.02 + 0.45 * i / n, r=7.3)
            yield frame([gs(model, cam), gs(small, cam), gs(small, cam, small_col)],
                        ["Rendered splats", f"{n_keep:,} opaque Gaussians, shrunk", "Shrunk, by component"],
                        title, ["Inside the model: overlapping 3D Gaussians",
                                "Every Gaussian keeps the label of the part it was initialized on"],
                        fade=fade_of(i, n))

    def scene_fault(s, fault, az, desc, expected, n=180):
        scene = build_satellite(fault, az)
        v = verdicts[s]
        kind, _, parts = v.partition(":")
        parts = ", ".join(PRETTY.get(p.strip(), p.strip().replace("_", " ")) for p in parts.split(",")) if parts else ""
        short = {"GEOMETRIC CHANGE": "Geometric change", "APPEARANCE CHANGE": "Appearance change",
                 "NO CHANGE": "No change"}.get(kind.strip())
        if short is None:
            short, parts = "likely illumination change on 3 parts, no fault", ""
        nfl = int(flagged[s].sum())
        for i in range(n):
            cam = sweep_cam(i / n)
            rt = to_u8(render_scene(scene, cam, device=dev)["rgb"])
            yield frame([rt, gs(model, cam), gs(model, cam, flag_colors(cam, flagged[s]))],
                        ["Inspection: current spacecraft", "Healthy 3DGS baseline", f"Flagged Gaussians ({nfl:,})"],
                        title, [desc, f"Verdict from 8 inspection cameras: {short}" + (f" ({parts})" if parts else "")],
                        caption_color=FG, fade=fade_of(i, n))

    scenes = [scene_title(), scene_healthy(), scene_inside()] + [scene_fault(*a) for a in SCEN]

    if args.preview:
        picks = [scene_healthy(240), scene_inside(180), scene_fault(*SCEN[0]), scene_fault(*SCEN[3])]
        for k, g in enumerate(picks):
            for j, f in enumerate(g):
                if j == 40:
                    Image.fromarray(f).save(out / f"preview_{k}.jpg", quality=92)
                    break
        print("previews written", flush=True)
        return

    import imageio_ffmpeg
    tmp, final = out / "larissa_orbit.tmp.mp4", out / "larissa_orbit.mp4"
    w = imageio_ffmpeg.write_frames(str(tmp), (W, H), fps=FPS, codec="libx264", quality=None,
                                    output_params=["-crf", "18", "-preset", "veryfast", "-movflags", "+faststart"],
                                    pix_fmt_out="yuv420p", macro_block_size=16)
    w.send(None)
    n = 0
    for g in scenes:
        for f in g:
            w.send(np.ascontiguousarray(f))
            n += 1
            if n % 60 == 0:
                print("frames", n, flush=True)
    w.close()
    os.replace(tmp, final)
    print("wrote", final, n, "frames", flush=True)


if __name__ == "__main__":
    main()
