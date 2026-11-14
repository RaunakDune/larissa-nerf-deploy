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
from larissa.scene import Camera, build_satellite, look_at, render_scene  # noqa: E402

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
F_CREDIT, F_NUM = font(19), font(15, True)
CREDIT = "Raunak Sarbajna, University of Houston"
G_W, G_H, P2 = 600, 450, 450        # camera scenes: overview panel and image panel
TRAIN_C, HOLD_C, INSP_C, ACT_C = (70, 200, 160), (170, 120, 255), (255, 160, 60), (255, 230, 80)


def credit(d):
    d.text((W - 22, H - 16), CREDIT, font=F_CREDIT, fill=(120, 128, 142), anchor="rs")


def god_cam(t):
    """Wide overview that shows the whole camera sphere."""
    az, el, r = math.radians(30.0 + 100.0 * t), math.radians(20.0), 20.0
    eye = (r * math.cos(el) * math.cos(az), r * math.cos(el) * math.sin(az), r * math.sin(el))
    return look_at(eye, target=(0.0, 0.0, 0.0), W=G_W, H=G_H, fov_scale=0.6)


def project(cam, X):
    Xc = X @ cam.R.T + cam.t
    return cam.fx * Xc[:, 0] / Xc[:, 2] + cam.cx, cam.fy * Xc[:, 1] / Xc[:, 2] + cam.cy, Xc[:, 2]


def mix(c, a):
    return tuple(int(BG[k] + (c[k] - BG[k]) * a) for k in range(3))


def draw_frustum(d, god, c, color, width=1, size=1.0, sight=False, dim=True):
    """Camera as a small pyramid pointing at what it sees; dimmed when on the far side."""
    half = size * 0.5 * c.W / c.fx
    loc = torch.tensor([[0.0, 0.0, 0.0], [-half, -half, size], [half, -half, size], [half, half, size], [-half, half, size]])
    pts = (loc - c.t) @ c.R  # camera -> world
    u, v, z = project(god, pts)
    zc = float(project(god, torch.zeros(1, 3))[2][0])
    col = color if (not dim or float(z[0]) < zc + 2.0) else mix(color, 0.45)
    P_ = [(float(a), float(b)) for a, b in zip(u, v)]
    if sight:
        uo, vo, _ = project(god, torch.zeros(1, 3))
        d.line([P_[0], (float(uo[0]), float(vo[0]))], fill=mix(color, 0.55), width=1)
    for k in range(1, 5):
        d.line([P_[0], P_[k]], fill=col, width=width)
    d.line(P_[1:] + [P_[1]], fill=col, width=width)
    return P_[0]


def draw_rings(d, god, radius, els, color):
    for el in els:
        e = math.radians(el)
        a = torch.linspace(0, 2 * math.pi, 181)
        X = torch.stack([radius * math.cos(e) * torch.cos(a), radius * math.cos(e) * torch.sin(a),
                         torch.full_like(a, radius * math.sin(e))], 1)
        u, v, _ = project(god, X)
        d.line([(float(x), float(y)) for x, y in zip(u, v)], fill=color, width=1)


def scaled(c, n):
    s = n / c.W
    return Camera(c.R, c.t, c.fx * s, c.fy * s, n / 2, n / 2, n, n)


def cam_frame(god_img, right, right_label, god_label, caption, fade=1.0):
    im = Image.new("RGB", (W, H), BG)
    im.paste(god_img, (20, 92))
    im.paste(right, (650 + (610 - right.size[0]) // 2, 92))
    d = ImageDraw.Draw(im)
    d.text((W // 2, 46), "Project Larissa: spacecraft change detection with 3D Gaussian Splatting", font=F_SUB, fill=FG, anchor="mm")
    d.rectangle((19, 91, 20 + G_W, 92 + G_H), outline=(45, 52, 66))
    d.text((20 + G_W // 2, 92 + G_H + 22), god_label, font=F_LAB, fill=DIM, anchor="mm")
    d.text((955, 92 + G_H + 22), right_label, font=F_LAB, fill=DIM, anchor="mm")
    for k, line in enumerate(caption):
        d.text((W // 2, 606 + 36 * k), line, font=F_CAP if k else F_SUB, fill=FG if k == 0 else DIM, anchor="mm")
    credit(d)
    a = np.asarray(im)
    return a if fade >= 1 else (a * fade).astype(np.uint8)


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
    credit(d)
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
        d.text((W // 2, 500), CREDIT, font=F_SUB, fill=FG, anchor="mm")
        d.text((W // 2, 534), "Department of Computer Science", font=F_CAP, fill=DIM, anchor="mm")
        a = np.asarray(im)
        for i in range(90):
            yield (a * fade_of(i, 90)).astype(np.uint8)

    train_cams = data.train_cameras()
    hold_nom = hold_cams
    insp_imgs, insp_cams, _ = data.inspection("healthy_control")

    def img_u8(x, n):
        return Image.fromarray(to_u8(x)).resize((n, n), Image.BILINEAR)

    def god_base(gc):
        return Image.fromarray(to_u8(render_scene(healthy, gc, device=dev)["rgb"]))

    def scene_training(n=390):
        a, b = int(0.62 * n), int(0.80 * n)
        for i in range(n):
            gc = god_cam(i / n * 0.6)
            g = god_base(gc)
            d = ImageDraw.Draw(g)
            draw_rings(d, gc, 9.0, (-60, -30, 0, 30, 60), (32, 40, 54))
            if i < a:
                k = min(95, int(96 * i / a))
                for j in range(k):
                    draw_frustum(d, gc, train_cams[j], TRAIN_C)
                draw_frustum(d, gc, train_cams[k], ACT_C, width=3, size=1.4, sight=True, dim=False)
                right = img_u8(data.train_view(k)["rgb"], P2)
                rl, gl = f"What training camera {k + 1} of 96 sees", f"Training cameras: {k + 1}"
                cap = ["Step 1: 96 cameras on a 9 m sphere photograph the healthy spacecraft",
                       "Their images train the 3D Gaussian Splatting baseline (15 minutes on one GPU)"]
            elif i < b:
                k = min(11, int(12 * (i - a) / (b - a)))
                for c in train_cams:
                    draw_frustum(d, gc, c, mix(TRAIN_C, 0.5))
                for j in range(k):
                    draw_frustum(d, gc, hold_nom[j], HOLD_C, width=2)
                draw_frustum(d, gc, hold_nom[k], ACT_C, width=3, size=1.4, sight=True, dim=False)
                right = img_u8(hold[k], P2)
                rl, gl = f"Held-out camera {k + 1} of 12", "Training (green) and held-out (purple) cameras"
                cap = ["Step 2: 12 held-out views of the healthy spacecraft, never used for training",
                       "They show how much the baseline differs from a healthy image, which sets the alarm levels"]
            else:
                k = (int((i - b) / 12) * 23) % 96
                for c in train_cams:
                    draw_frustum(d, gc, c, mix(TRAIN_C, 0.5))
                draw_frustum(d, gc, train_cams[k], ACT_C, width=3, size=1.4, sight=True, dim=False)
                right = Image.fromarray(gs(model, scaled(train_cams[k], P2)))
                rl, gl = f"3DGS baseline rendered from camera {k + 1}", "Training cameras"
                cap = ["Result: a 3D model of the healthy spacecraft that renders from any pose",
                       "43,929 Gaussians, each labeled with the part it belongs to"]
            yield cam_frame(g, right, rl, gl, cap, fade_of(i, n))

    def scene_inspection(n=300):
        tile, gap = 146, 6
        for i in range(n):
            gc = god_cam(0.6 + 0.25 * i / n)
            g = god_base(gc)
            d = ImageDraw.Draw(g)
            draw_rings(d, gc, 8.5, (28, -12), (70, 60, 45))
            k = int(8 * i / n * 1.5) % 8
            for j, c in enumerate(insp_cams):
                p0 = draw_frustum(d, gc, c, ACT_C if j == k else INSP_C, width=3 if j == k else 2,
                                  size=1.4, sight=True, dim=j != k)
                d.text((p0[0], p0[1] - 12), str(j + 1), font=F_NUM, fill=ACT_C if j == k else INSP_C, anchor="mm")
            right = Image.new("RGB", (4 * tile + 3 * gap, 2 * tile + gap + 150), BG)
            rd = ImageDraw.Draw(right)
            for j, im in enumerate(insp_imgs):
                x, y = (j % 4) * (tile + gap), (j // 4) * (tile + gap)
                right.paste(img_u8(im, tile), (x, y))
                rd.rectangle((x, y, x + tile - 1, y + tile - 1), outline=ACT_C if j == k else (45, 52, 66), width=3 if j == k else 1)
                rd.text((x + 8, y + 6), str(j + 1), font=F_NUM, fill=ACT_C if j == k else INSP_C)
            y0 = 2 * tile + gap + 18
            for m, line in enumerate(["Every inspection pass:",
                                      "1. render the baseline at each camera's pose",
                                      "2. compare it with the new image",
                                      "3. vote across cameras on each Gaussian"]):
                rd.text((4, y0 + 30 * m), line, font=F_CAP, fill=FG if m == 0 else DIM)
            yield cam_frame(g, right, "What the 8 inspection cameras see (healthy)", "Fixed inspection cameras (orange)",
                            ["Step 3: 8 fixed inspection cameras watch the spacecraft",
                             "8.5 m away on two rings, 28\u00b0 above and 12\u00b0 below, at poses not used for training"],
                            fade_of(i, n))

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

    scenes = [scene_title(), scene_training(), scene_healthy(), scene_inside(), scene_inspection()] + \
        [scene_fault(*a) for a in SCEN]

    if args.preview:
        picks = [(scene_training(), 100), (scene_training(), 270), (scene_training(), 350),
                 (scene_inspection(), 60), (scene_fault(*SCEN[3]), 40)]
        for k, (g, at) in enumerate(picks):
            for j, f in enumerate(g):
                if j == at:
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
