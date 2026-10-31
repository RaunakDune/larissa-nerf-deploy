"""Figures for the IEEE Aerospace paper, made from the released dataset and baseline.

    python tools/paper_figures.py --out paper_figures [--device cuda:0]

Writes fig_scene.png (satellite, component labels, labelled baseline) and
fig_results.png (one column per scenario). Sized for print with >= 10 pt text.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from larissa.dataset import LarissaSynth  # noqa: E402
from larissa.detect import calibrate_components, calibrate_threshold, classify_components, detect, verdict  # noqa: E402
from larissa.gaussians import GaussianModel, render, sh_to_rgb  # noqa: E402
from larissa.scene import build_satellite, look_at, render_scene  # noqa: E402

plt.rcParams.update({"font.size": 10, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})
COMP_COLORS = {0: (0, 0, 0), 1: (0.85, 0.55, 0.15), 2: (0.20, 0.45, 0.85), 3: (0.30, 0.75, 0.95),
               4: (0.85, 0.25, 0.30), 5: (0.55, 0.55, 0.60)}
SHORT = {"healthy_control": "Healthy\ncontrol", "panel_hinge": "Panel\nhinge", "antenna_missing": "Antenna\nmissing",
         "mli_discoloration": "Blanket\ndiscoloration", "sun_shift_control": "Sun shift\n(no fault)"}
VERD = {"NO CHANGE": "No change", "GEOMETRIC CHANGE": "Geometric",
        "APPEARANCE CHANGE ON MANY PARTS": "Illumination notice", "APPEARANCE CHANGE": "Appearance"}
EXPECTED = {"healthy_control": "No change", "panel_hinge": "Geometric", "antenna_missing": "Geometric",
            "mli_discoloration": "Appearance", "sun_shift_control": ("No change", "Illumination notice")}


def crop(img, box):
    y0, y1, x0, x1 = box
    return img[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/larissa_synth_v1")
    ap.add_argument("--out", default="paper_figures")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = LarissaSynth(args.data)
    model = GaussianModel.from_state_dict(torch.load(Path(args.data) / "baseline_gaussians.pt", weights_only=True), dev)
    S = data.meta["resolution"]

    # ---------------------------------------------------------------- scene figure
    ov = look_at((6.5, 5.0, 4.5), W=320, H=320, fov_scale=1.25)
    r = render_scene(build_satellite(None), ov, device=dev)
    comp_rgb = np.zeros((320, 320, 3))
    for k, c in COMP_COLORS.items():
        comp_rgb[r["comp"].cpu().numpy() == k] = c
    with torch.no_grad():
        col = torch.tensor([COMP_COLORS[int(c)] for c in model.comp.cpu()], dtype=torch.float32, device=dev)
        gs = render(model, ov, colors_override=col)["rgb"].clamp(0, 1).cpu().numpy()
        dirs = model.means - ov.center.to(dev)
        rgb_gs = render(model, ov)["rgb"].clamp(0, 1).cpu().numpy()
    box = (40, 300, 20, 300)
    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.25), dpi=300)
    panels = [(r["rgb"].cpu().numpy(), "(a) Ray-traced scene"), (comp_rgb, "(b) Component labels"),
              (rgb_gs, "(c) 3DGS baseline"), (gs, "(d) Gaussians by part")]
    for a, (im, t) in zip(ax, panels):
        a.imshow(np.clip(crop(im, box), 0, 1)); a.set_title(t, fontsize=10); a.axis("off")
    names = ["bus", "port array", "stbd array", "antenna", "thruster"]
    handles = [plt.Line2D([], [], marker="s", ls="", color=COMP_COLORS[i + 1], markersize=8, label=n) for i, n in enumerate(names)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
    plt.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "fig_scene.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---------------------------------------------------------------- results figure
    hold_imgs, hold_cams = data.holdout()
    hold_imgs = [im.to(dev) for im in hold_imgs]
    tau = calibrate_threshold(model, hold_imgs, hold_cams)
    thr = calibrate_components(model, hold_imgs, hold_cams, tau)
    ov2 = look_at((6.5, 5.0, 4.5), W=S, H=S, fov_scale=1.25)
    cols = []
    for s in data.scenarios:
        imgs, cams, gts = data.inspection(s)
        imgs = [im.to(dev) for im in imgs]
        det = detect(model, imgs, cams, tau, comp_thresholds=thr)
        v = verdict(classify_components(model, imgs, cams, det, log=lambda *a: None))
        i = int(np.argmax([int(m.sum()) for m in det.masks])) if any(m.any() for m in det.masks) else 2
        obs = imgs[i].cpu().numpy()
        over = obs * 0.55
        m, g = det.masks[i].numpy(), gts[i].numpy()
        over[m] = over[m] * 0.3 + np.array([1.0, 0.15, 0.1]) * 0.7
        over[binary_dilation(g) & ~g] = [0.2, 0.95, 1.0]
        with torch.no_grad():
            dirs = model.means - ov2.center.to(dev)
            dirs = dirs / dirs.norm(dim=1, keepdim=True)
            base = sh_to_rgb(model.sh, dirs).clamp(0, 1)
            grey = (0.35 + 0.65 * base.mean(1, keepdim=True)).expand(-1, 3) * 0.75
            colr = torch.where(det.flagged.to(dev)[:, None], torch.tensor([1.0, 0.18, 0.12], device=dev), grey)
            loc = render(model, ov2, colors_override=colr)["rgb"].clamp(0, 1).cpu().numpy()
        short = next((VERD[k] for k in VERD if v.startswith(k)), v)
        parts = v.split(":", 1)[1].strip().replace("solar_array_", "").replace("_", " ") if ":" in v else ""
        cols.append((s, obs, over, loc, short, parts))
        print(s, v, flush=True)
    fig, ax = plt.subplots(3, 5, figsize=(7.0, 5.0), dpi=300)
    b = (30, 226, 10, 246)
    for k, (s, obs, over, loc, short, parts) in enumerate(cols):
        for rr, im in enumerate([obs, over, loc]):
            ax[rr, k].imshow(np.clip(crop(im, b), 0, 1)); ax[rr, k].set_xticks([]); ax[rr, k].set_yticks([])
            for sp in ax[rr, k].spines.values():
                sp.set_visible(False)
        ax[0, k].set_title(SHORT[s], fontsize=10)
        exp = EXPECTED[s]
        ok = short in exp if isinstance(exp, tuple) else short == exp
        colr = "#B22222" if not ok else ("#9A5B00" if short == "Illumination notice" else "#1B5E20")
        ax[2, k].set_xlabel(short + ("\n" + parts if parts else "\n "), fontsize=10, color=colr)
    for rr, t in enumerate(["Inspection\nimage", "Pixel change\nvs truth", "Flagged\nGaussians"]):
        ax[rr, 0].set_ylabel(t, fontsize=10)
    plt.tight_layout(w_pad=0.3, h_pad=0.4)
    fig.savefig(out / "fig_results.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
