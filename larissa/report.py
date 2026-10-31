"""Figures and tables for the demo output."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY = "#1F3A5F"


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def _crop_box(panels):
    """Common crop around the satellite so details are visible."""
    return None


def scenario_figure(panels, path: Path):
    n = len(panels)
    cols = ["Inspection image", "Baseline rendered\nat same pose", "Detected change\n(red) vs truth (cyan outline)",
            "Localized on 3D model\n(flagged Gaussians red)"]
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.25 * n), dpi=140)
    fig.patch.set_facecolor("white")
    for r, p in enumerate(panels):
        obs, ren = _np(p["obs"]), _np(p["render"])
        mask, gt = _np(p["mask"]), _np(p["gt"])
        over = obs * 0.55
        over[mask] = over[mask] * 0.3 + np.array([1.0, 0.15, 0.1]) * 0.7
        if gt.any():
            from scipy.ndimage import binary_dilation
            edge = binary_dilation(gt, iterations=1) & ~gt
            over[edge] = np.array([0.2, 0.95, 1.0])
        for c, img in enumerate([obs, ren, over, _np(p["loc"])]):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=10, color=NAVY, fontweight="bold")
        axes[r, 0].set_ylabel(p["name"].replace("_", " "), fontsize=10, color=NAVY, fontweight="bold")
        axes[r, 3].text(0.02, -0.06, p["verdict"], transform=axes[r, 3].transAxes, fontsize=8, va="top",
                        color="#B22222" if not p["verdict"].startswith("NO") else "#2E7D32", wrap=True)
        axes[r, 0].text(0.02, -0.06, p["desc"], transform=axes[r, 0].transAxes, fontsize=7.5, va="top", color="#555")
    plt.tight_layout(h_pad=2.2)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def capture_figure(imgs, path: Path):
    fig, axes = plt.subplots(1, len(imgs), figsize=(2 * len(imgs), 2.2), dpi=120)
    for ax, im in zip(axes, imgs):
        ax.imshow(np.clip(_np(im), 0, 1)); ax.axis("off")
    fig.suptitle("Sample baseline capture views (healthy satellite)", color=NAVY, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_markdown(summary: dict, path: Path):
    L = []
    L.append("# Project Larissa demo results\n")
    gpu = summary.get("gpu") or summary["device"]
    L.append(f"- Hardware: {gpu}")
    L.append(f"- Resolution: {summary['resolution']} x {summary['resolution']} px; "
             f"{summary['training_views']} baseline views; {summary['inspection_cameras']} inspection cameras")
    L.append(f"- Baseline: {summary['gaussians']} Gaussians, held-out PSNR {summary['baseline_heldout_psnr_db']} dB, "
             f"trained in {summary['timings_s']['baseline_train_s']} s")
    L.append(f"- Change threshold (calibrated on healthy held-out views): {summary['threshold_tau']}")
    pn = summary["pose_noise"]
    L.append(f"- Simulated camera pose error: {pn['rotation_deg']} deg rotation, {pn['translation_m']} m translation; sensor noise on all images\n")
    L.append("| Scenario | Expected | Verdict | Correct | Pixel IoU | False-positive px rate | Check time (s) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in summary["scenarios"]:
        iou = "" if r["pixel_iou"] is None else f"{r['pixel_iou']:.2f}"
        L.append(f"| {r['scenario']} | {r['expected']} | {r['verdict']} | {'yes' if r['correct'] else 'no'} | "
                 f"{iou} | {r['false_positive_pixel_rate']:.4f} | {r['check_seconds']} |")
    L.append("")
    for r in summary["scenarios"]:
        if not r["classification"]:
            continue
        L.append(f"**{r['scenario']}** component classification:\n")
        L.append("| Component | Flagged Gaussians | Type | Residual reduction after colour-only refit | Note |")
        L.append("|---|---|---|---|---|")
        for k, c in r["classification"].items():
            ci = r["components"][k]
            L.append(f"| {k} | {ci['flagged']}/{ci['evaluated']} | {c['type']} | {c['residual_reduction']:.2f} | {c['note']} |")
        L.append("")
    path.write_text("\n".join(L))
