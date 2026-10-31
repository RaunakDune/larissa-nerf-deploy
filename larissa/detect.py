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

from .gaussians import GaussianModel, render, SH_C0
from .scene import COMPONENTS, Camera


def add_sensor_noise(img: torch.Tensor, gen: torch.Generator, sigma: float = 0.01, gain: float = 0.03) -> torch.Tensor:
    g = 1 + (torch.rand(1, generator=gen).item() * 2 - 1) * gain
    n = torch.randn(img.shape, generator=gen) * sigma
    return (img * g + n.to(img.device)).clamp(0, 1)


def _box_blur(x: torch.Tensor, k: int = 5) -> torch.Tensor:
    return F.avg_pool2d(x[None, None], k, stride=1, padding=k // 2, count_include_pad=False)[0, 0]


def _shift_min(a: torch.Tensor, b: torch.Tensor, r: int) -> torch.Tensor:
    """For each pixel of a, the smallest colour distance to b within a (2r+1)^2 window."""
    H, W, C = b.shape
    bp = F.pad(b.permute(2, 0, 1)[None], (r, r, r, r), mode="replicate")[0]
    best = None
    for dy in range(2 * r + 1):
        for dx in range(2 * r + 1):
            d = (a - bp[:, dy:dy + H, dx:dx + W].permute(1, 2, 0)).abs().mean(-1)
            best = d if best is None else torch.minimum(best, d)
    return best


def difference_map(obs: torch.Tensor, rend: torch.Tensor, shift: int = 2) -> torch.Tensor:
    """Per-pixel change score, tolerant to small misregistration.

    Each pixel is compared with the best-matching pixel of the other image
    within +/- `shift` pixels, in both directions, after a light blur. A camera
    pose error of a pixel or two therefore does not register as change along
    the sharp silhouette of the spacecraft, while real changes (which persist
    under any small shift) do."""
    k = lambda x: F.avg_pool2d(x.permute(2, 0, 1)[None], 3, 1, 1, count_include_pad=False)[0].permute(1, 2, 0)
    o, r = k(obs), k(rend)
    d = torch.maximum(_shift_min(o, r, shift), _shift_min(r, o, shift))
    return _box_blur(d, 5)


def binary_open(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    m = mask.float()[None, None]
    er = -F.max_pool2d(-m, k, 1, k // 2)
    return F.max_pool2d(er, k, 1, k // 2)[0, 0] > 0.5


def erode(mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    return -F.max_pool2d(-mask.float()[None, None], k, 1, k // 2)[0, 0] > 0.5


def dilate(mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    return F.max_pool2d(mask.float()[None, None], k, 1, k // 2)[0, 0] > 0.5


@torch.no_grad()
def calibrate_threshold(model: GaussianModel, images, cams, quantile: float = 0.995, margin: float = 1.0,
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
    proj: list = field(default_factory=list)    # per view: (u, v, on_surface) of every Gaussian
    empty: list = field(default_factory=list)   # per view: pixels where the baseline has no surface
    seconds: float = 0.0


@torch.no_grad()
def detect(model: GaussianModel, images, cams: list[Camera], tau: float, min_views: int = 2,
           vote_thresh: float = 0.5, comp_thresholds: dict | None = None, comp_min: int = 60,
           depth_tol: float = 0.08) -> Detection:
    """`comp_thresholds` maps component name to the flagged fraction that
    counts as a change (see calibrate_components); without it, 15% is used."""
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
        # pixels clearly away from any baseline surface (eroded so silhouette edges do not count)
        empty = erode(out["alpha"] < 0.5, 5)
        unexplained += int((mask & empty).sum())
        det.empty.append(empty.cpu())
        # vote: is each Gaussian on the visible surface, and is its pixel changed?
        u = out["u"].round().long()
        v = out["v"].round().long()
        inside = out["visible"] & (u >= 0) & (u < cam.W) & (v >= 0) & (v < cam.H)
        uu, vv = u.clamp(0, cam.W - 1), v.clamp(0, cam.H - 1)
        surf_depth = out["depth"][vv, uu]
        cover = out["alpha"][vv, uu]
        on_surface = inside & (out["z"] <= surf_depth + depth_tol) & (cover > 0.5)
        det.proj.append((uu.cpu(), vv.cpu(), on_surface.cpu()))
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
        thr = (comp_thresholds or {}).get(name, 0.15)
        det.components[name] = {"id": cid, "evaluated": n_eval, "flagged": n_flag, "fraction": round(frac, 4),
                                "threshold": round(thr, 4),
                                "changed": bool(frac >= thr and n_flag >= comp_min)}
    det.seconds = time.time() - t0
    return det


@torch.no_grad()
def calibrate_components(model: GaussianModel, images, cams, tau: float, factor: float = 3.0,
                         offset: float = 0.04) -> dict:
    """Per-component alarm level from healthy held-out captures: a component
    counts as changed only if its flagged fraction clearly exceeds what the
    healthy satellite produces (rendering error around glints, fine texture)."""
    det = detect(model, images, cams, tau)
    return {name: max(factor * c["fraction"], c["fraction"] + offset) for name, c in det.components.items()}


def _residual(model, images, cams, regions):
    tot, n = 0.0, 0
    with torch.no_grad():
        for img, cam, m in zip(images, cams, regions):
            if m.sum() == 0:
                continue
            r = render(model, cam)["rgb"]
            mm = m.to(img.device)
            tot += float((r - img).abs().mean(-1)[mm].sum())
            n += int(mm.sum())
    return tot / max(n, 1)


def _footprints(det: Detection, sel: torch.Tensor, H: int, W: int, grow: int = 21):
    """Per view, the image region covered by the selected Gaussians, grown by `grow` px."""
    fps = []
    for (u, v, on) in det.proj:
        m = torch.zeros(H, W, dtype=torch.bool)
        k = sel & on
        m[v[k], u[k]] = True
        fps.append(dilate(m, grow) if k.any() else m)
    return fps


def classify_components(model: GaussianModel, images, cams, det: Detection, iters: int = 150,
                        recover: float = 0.6, new_matter: float = 0.10, dark: float = 0.05,
                        seed: int = 0, log=print) -> dict:
    """Decide, for each changed component, whether the change is geometric or
    an appearance change. Three tests, in order:

    1. New matter in empty space: changed pixels near the component where the
       baseline has no surface at all. Repainting the baseline cannot create
       material, so a sizeable share of these means a geometric change.
    2. Vanishing: refitting only the colours of the flagged Gaussians (geometry
       frozen) drives them to black, i.e. the part has gone.
    3. Colour-only refit: if colour alone removes most of the residual in the
       changed region, it is an appearance change; otherwise geometric."""
    results = {}
    comps = model.comp.cpu()
    H, W = det.masks[0].shape
    masks = [dilate(m, 3) for m in det.masks]
    for name, info in det.components.items():
        if not info["changed"]:
            continue
        t0 = time.time()
        sel_cpu = det.flagged & (comps == info["id"])
        sel = sel_cpu.to(model.means.device)
        fps = _footprints(det, sel_cpu, H, W)
        regions = [m & f for m, f in zip(masks, fps)]
        raw = [m & f for m, f in zip(det.masks, fps)]
        n_region = sum(int(r.sum()) for r in regions)
        n_new = sum(int((r & e).sum()) for r, e in zip(raw, det.empty))
        new_frac = n_new / max(1, sum(int(r.sum()) for r in raw))
        base_res = _residual(model, images, cams, regions)
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
        after = _residual(m2, images, cams, regions)
        lum_before = float((SH_C0 * model.sh.detach()[sel, 0] + 0.5).clamp(min=0).mean())
        lum_after = float((SH_C0 * m2.sh[sel, 0] + 0.5).clamp(min=0).mean())
        reduction = 1 - after / max(base_res, 1e-6)
        if new_frac >= new_matter:
            kind, note = "geometric", "material appears where the baseline is empty (moved or deformed part)"
        elif lum_after < dark and lum_before > 2 * dark:
            kind, note = "geometric", "part has vanished (colour refit drives it to empty space)"
        elif reduction >= recover:
            kind, note = "appearance", "explained by a change of surface colour"
        else:
            kind, note = "geometric", "shape or position change (colour refit cannot explain it)"
        results[name] = {"type": kind, "note": note, "new_matter_fraction": round(new_frac, 3),
                         "residual_before": round(base_res, 4), "residual_after": round(after, 4),
                         "residual_reduction": round(reduction, 3),
                         "luminance_before": round(lum_before, 3), "luminance_after": round(lum_after, 3),
                         "seconds": round(time.time() - t0, 1)}
        log(f"  {name}: {kind} (new-matter {new_frac:.2f}, residual reduction {reduction:.2f}, "
            f"luminance {lum_before:.2f}->{lum_after:.2f})")
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
