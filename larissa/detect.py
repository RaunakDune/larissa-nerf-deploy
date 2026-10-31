"""Change detection against a Gaussian Splatting baseline.

Stages (matching the Project Larissa pipeline):
  1. render the baseline at each inspection camera pose
  2. compare with the new image (colour difference + structural difference)
  3. vote per Gaussian across views, so a change must be seen consistently
  4. report per component
  5. classify each flagged component as a geometric or an appearance change
     by testing whether a colour-only refit of the baseline can explain it
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .gaussians import GaussianModel, render, ssim_map, SH_C0
from .scene import COMPONENTS, Camera


def add_sensor_noise(img: torch.Tensor, gen: torch.Generator, sigma: float = 0.01, gain: float = 0.03) -> torch.Tensor:
    g = 1 + (torch.rand(1, generator=gen).item() * 2 - 1) * gain
    n = torch.randn(img.shape, generator=gen) * sigma
    return (img * g + n.to(img.device)).clamp(0, 1)


def _box_blur(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    return F.avg_pool2d(x[None, None], k, stride=1, padding=k // 2, count_include_pad=False)[0, 0]


def difference_map(obs: torch.Tensor, rend: torch.Tensor) -> torch.Tensor:
    """Per-pixel change score in [0, ~1]: blurred colour difference plus structural (SSIM) difference."""
    l1 = _box_blur((obs - rend).abs().mean(-1))
    dssim = ((1 - ssim_map(obs, rend)) / 2).clamp(0, 1)
    return 0.5 * l1 + 0.5 * dssim


def binary_open(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    m = mask.float()[None, None]
    er = -F.max_pool2d(-m, k, 1, k // 2)
    return F.max_pool2d(er, k, 1, k // 2)[0, 0] > 0.5


def dilate(mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    return F.max_pool2d(mask.float()[None, None], k, 1, k // 2)[0, 0] > 0.5


@torch.no_grad()
def calibrate_threshold(model: GaussianModel, images, cams, quantile: float = 0.999, margin: float = 1.25,
                        floor: float = 0.04) -> float:
    """Threshold set from healthy held-out captures, so it reflects the
    reconstruction error, sensor noise and pose error of the system itself."""
    vals = []
    for img, cam in zip(images, cams):
        d = difference_map(img, render(model, cam)["rgb"])
        vals.append(d.flatten())
    v = torch.cat(vals)
    q = torch.quantile(v[torch.randperm(len(v))[:2_000_000]] if len(v) > 2_000_000 else v, quantile).item()
    return max(floor, q * margin)


@dataclass
class Detection:
    tau: float
    masks: list = field(default_factory=list)
    dmaps: list = field(default_factory=list)
    renders: list = field(default_factory=list)
    score: torch.Tensor | None = None
    n_vis: torch.Tensor | None = None
    flagged: torch.Tensor | None = None
    components: dict = field(default_factory=dict)
    unexplained_px: int = 0
    seconds: float = 0.0


@torch.no_grad()
def detect(model: GaussianModel, images, cams: list[Camera], tau: float, min_views: int = 2,
           vote_thresh: float = 0.5, comp_frac: float = 0.15, comp_min: int = 25,
           depth_tol: float = 0.08) -> Detection:
    t0 = time.time()
    dev = model.means.device
    N = len(model)
    changed = torch.zeros(N, device=dev)
    visible = torch.zeros(N, device=dev)
    det = Detection(tau=tau)
    unexplained = 0
    for img, cam in zip(images, cams):
        out = render(model, cam)
        d = difference_map(img, out["rgb"])
        mask = binary_open(d > tau)
        det.masks.append(mask.cpu())
        det.dmaps.append(d.cpu())
        det.renders.append(out["rgb"].clamp(0, 1).cpu())
        # change where the baseline has nothing (new geometry over empty space)
        unexplained += int((mask & (out["alpha"] < 0.5)).sum())
        # vote: is each Gaussian on the visible surface, and is its pixel changed?
        u = out["u"].round().long()
        v = out["v"].round().long()
        inside = out["visible"] & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
        uu, vv = u.clamp(0, cam.W - 1), v.clamp(0, cam.H - 1)
        surf_depth = out["depth"][vv, uu]
        cover = out["alpha"][vv, uu]
        on_surface = inside & (out["z"] <= surf_depth + depth_tol) & (cover > 0.5)
        mk = dilate(mask, 5)[vv, uu]
        visible += on_surface.float()
        changed += (on_surface & mk).float()
    det.n_vis = visible.cpu()
    det.score = (changed / visible.clamp(min=1)).cpu()
    det.flagged = (visible >= min_views).cpu() & (det.score >= vote_thresh)
    det.unexplained_px = unexplained
    comps = model.comp.cpu()
    for cid, name in COMPONENTS.items():
        if cid == 0:
            continue
        sel = (comps == cid) & (det.n_vis >= min_views)
        n_eval = int(sel.sum())
        n_flag = int((det.flagged & (comps == cid)).sum())
        frac = n_flag / max(1, n_eval)
        det.components[name] = {"id": cid, "evaluated": n_eval, "flagged": n_flag, "fraction": round(frac, 4),
                                "changed": bool(frac >= comp_frac and n_flag >= comp_min)}
    det.seconds = time.time() - t0
    return det


def _masked_residual(model, images, cams, masks):
    tot, n = 0.0, 0
    with torch.no_grad():
        for img, cam, m in zip(images, cams, masks):
            if m.sum() == 0:
                continue
            r = render(model, cam)["rgb"]
            mm = m.to(img.device)
            tot += float((r - img).abs().mean(-1)[mm].sum())
            n += int(mm.sum())
    return tot / max(n, 1)


def classify_components(model: GaussianModel, images, cams, det: Detection, iters: int = 150,
                        recover: float = 0.6, dark: float = 0.05, seed: int = 0, log=print) -> dict:
    """For each changed component, refit only the colours of its flagged
    Gaussians (geometry frozen). If colour alone explains the change, it is an
    appearance change; if it cannot, or the part must turn black to match
    (i.e. it has vanished against empty space), it is geometric."""
    results = {}
    comps = model.comp.cpu()
    masks = [dilate(m, 3) for m in det.masks]
    base_res = _masked_residual(model, images, cams, masks)
    for name, info in det.components.items():
        if not info["changed"]:
            continue
        t0 = time.time()
        sel = (det.flagged & (comps == info["id"])).to(model.means.device)
        m2 = model.clone()
        sh = m2.sh.clone().requires_grad_(True)
        m2.sh = sh
        opt = torch.optim.Adam([sh], lr=0.02)
        g = torch.Generator().manual_seed(seed)
        for _ in range(iters):
            i = int(torch.randint(len(images), (1,), generator=g))
            out = render(m2, cams[i])
            loss = (out["rgb"] - images[i]).abs().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            sh.grad[~sel] = 0
            opt.step()
            with torch.no_grad():
                sh[~sel] = model.sh[~sel]
        m2.sh = sh.detach()
        after = _masked_residual(m2, images, cams, masks)
        lum_before = float((SH_C0 * model.sh.detach()[sel, 0] + 0.5).clamp(min=0).mean())
        lum_after = float((SH_C0 * m2.sh[sel, 0] + 0.5).clamp(min=0).mean())
        reduction = 1 - after / max(base_res, 1e-6)
        if lum_after < dark and lum_before > 2 * dark:
            kind, note = "geometric", "part missing or displaced (colour refit drives it to empty space)"
        elif reduction >= recover:
            kind, note = "appearance", "explained by a change of surface colour"
        else:
            kind, note = "geometric", "shape or position change (colour refit cannot explain it)"
        results[name] = {"type": kind, "note": note, "residual_before": round(base_res, 4),
                         "residual_after": round(after, 4), "residual_reduction": round(reduction, 3),
                         "luminance_before": round(lum_before, 3), "luminance_after": round(lum_after, 3),
                         "seconds": round(time.time() - t0, 1)}
        log(f"  {name}: {kind} (residual reduction {reduction:.2f}, luminance {lum_before:.2f}->{lum_after:.2f})")
    return results


def verdict(classes: dict) -> str:
    geo = [k for k, v in classes.items() if v["type"] == "geometric"]
    app = [k for k, v in classes.items() if v["type"] == "appearance"]
    if geo:
        return "GEOMETRIC CHANGE: " + ", ".join(geo)
    if len(app) >= 3:
        return "APPEARANCE CHANGE ON MANY PARTS (likely illumination): " + ", ".join(app)
    if app:
        return "APPEARANCE CHANGE: " + ", ".join(app)
    return "NO CHANGE"
