"""Change detection against a Gaussian Splatting baseline.

Stages (matching the Project Larissa pipeline):
  1. render the baseline at each inspection camera pose
  2. compare with the new image (colour difference + structural difference)
  3. vote per Gaussian across views, so a change must be seen consistently
  4. report per component
  5. classify each flagged component as a geometric or an appearance change
     from what every camera now sees where the component's Gaussians project
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


def _footprints(det: Detection, sel: torch.Tensor, H: int, W: int, grow: int = 21):
    """Per view, the image region covered by the selected Gaussians, grown by `grow` px."""
    fps = []
    for (u, v, on) in det.proj:
        m = torch.zeros(H, W, dtype=torch.bool)
        k = sel & on
        m[v[k], u[k]] = True
        fps.append(dilate(m, grow) if k.any() else m)
    return fps


def _lum(x: torch.Tensor) -> torch.Tensor:
    return x @ torch.tensor([0.299, 0.587, 0.114], device=x.device)


def classify_components(model: GaussianModel, images, cams, det: Detection, black: float = 0.035,
                        lit: float = 0.08, spread_margin: float = 0.12, vanish_share: float = 0.4,
                        incons_share: float = 0.3, new_matter: float = 0.10, log=print) -> dict:
    """Decide, for each changed component, whether the change is geometric or
    an appearance change, from what the inspection cameras now see at the
    location of each flagged Gaussian.

    The baseline knows where every Gaussian projects in every camera. For each
    flagged Gaussian we sample the new image there in every camera that sees it:

    * appearance change (discoloration, shadow): the surface is still there, so
      all cameras see one consistent new colour (sunlit diffuse shading does not
      depend on viewpoint);
    * vanished part: cameras see black space where a lit surface used to be;
    * moved or deformed part: cameras see whatever lies behind, which differs
      from camera to camera, so the colours are inconsistent.

    Inconsistency is measured relative to the baseline's own rendered colours at
    the same pixels, which absorbs normal view-dependent effects such as glints.
    A large share of changed pixels where the baseline has no surface at all
    (new material in empty space) is also geometric."""
    results = {}
    comps = model.comp.cpu()
    H, W = det.masks[0].shape
    blur = lambda x: F.avg_pool2d(x.permute(2, 0, 1)[None], 3, 1, 1, count_include_pad=False)[0].permute(1, 2, 0)
    obs_imgs = [blur(im.float()).cpu() for im in images]
    base_imgs = [blur(r.float()) for r in det.renders]
    for name, info in det.components.items():
        if not info["changed"]:
            continue
        t0 = time.time()
        fl = det.flagged & (comps == info["id"])
        idx = torch.nonzero(fl).squeeze(1)
        obs = torch.full((len(idx), len(images), 3), float("nan"))
        base = torch.full_like(obs, float("nan"))
        for vi, (u, v, on) in enumerate(det.proj):
            vis = on[idx]
            if not vis.any():
                continue
            uu, vv = u[idx][vis], v[idx][vis]
            obs[vis, vi] = obs_imgs[vi][vv, uu]
            base[vis, vi] = base_imgs[vi][vv, uu]
        nvis = (~torch.isnan(obs[..., 0])).sum(1)
        ok = nvis >= 2
        obs, base = obs[ok], base[ok]
        lo, lb = _lum(torch.nan_to_num(obs, nan=0.0)), _lum(torch.nan_to_num(base, nan=0.0))
        valid = ~torch.isnan(obs[..., 0])
        big = torch.tensor(9.0)
        def spread(l):
            return torch.where(valid, l, -big).max(1).values - torch.where(valid, l, big).min(1).values
        med = lambda l: torch.nanmedian(torch.where(valid, l, torch.tensor(float("nan"))), dim=1).values
        vanished = (med(lo) < black) & (med(lb) > lit)
        inconsistent = spread(lo) > spread(lb) + spread_margin
        n = max(1, int(ok.sum()))
        v_share = float(vanished.float().sum()) / n
        i_share = float((inconsistent & ~vanished).float().sum()) / n
        # new material in empty space near the component
        fps = _footprints(det, fl, H, W)
        raw = [m & f for m, f in zip(det.masks, fps)]
        n_new = sum(int((r & e).sum()) for r, e in zip(raw, det.empty))
        new_frac = n_new / max(1, sum(int(r.sum()) for r in raw))
        if v_share >= vanish_share:
            kind, note = "geometric", "part has vanished: cameras now see empty space where a lit surface was"
        elif i_share >= incons_share or new_frac >= new_matter:
            kind, note = "geometric", "part moved or deformed: cameras disagree on what is now at its location"
        else:
            kind, note = "appearance", "surface still in place with a consistent new colour"
        results[name] = {"type": kind, "note": note, "gaussians_tested": n,
                         "vanished_share": round(v_share, 3), "inconsistent_share": round(i_share, 3),
                         "new_matter_fraction": round(new_frac, 3), "seconds": round(time.time() - t0, 2)}
        log(f"  {name}: {kind} (vanished {v_share:.2f}, inconsistent {i_share:.2f}, new-matter {new_frac:.2f})")
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
