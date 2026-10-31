"""3D Gaussian Splatting in plain PyTorch.

A tile-based differentiable rasterizer following Kerbl et al. (2023), written
with ordinary tensor operations so it needs no CUDA compilation and runs on
older GPUs such as the Tesla P40. It is slower than the reference CUDA kernels
but exact enough for a demonstration.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .scene import Camera

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


@dataclass
class GaussianModel:
    means: torch.Tensor       # (N,3)
    log_scales: torch.Tensor  # (N,3)
    quats: torch.Tensor       # (N,4) wxyz, unnormalized
    opacity_logit: torch.Tensor  # (N,)
    sh: torch.Tensor          # (N,4,3) degree-1 spherical harmonics
    comp: torch.Tensor        # (N,) component label (not optimized)

    def params(self):
        return [self.means, self.log_scales, self.quats, self.opacity_logit, self.sh]

    def __len__(self):
        return self.means.shape[0]

    def subset(self, keep: torch.Tensor) -> "GaussianModel":
        return GaussianModel(*(p.detach()[keep].clone() for p in self.params()), self.comp[keep].clone())

    def clone(self) -> "GaussianModel":
        return GaussianModel(*(p.detach().clone() for p in self.params()), self.comp.clone())

    def state_dict(self):
        return {k: getattr(self, k).detach().cpu() for k in ("means", "log_scales", "quats", "opacity_logit", "sh", "comp")}

    @staticmethod
    def from_state_dict(sd, device):
        return GaussianModel(*(sd[k].to(device) for k in ("means", "log_scales", "quats", "opacity_logit", "sh", "comp")))


def quat_to_rot(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=1, keepdim=True)
    w, x, y, z = q.unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], 1).reshape(-1, 3, 3)


def sh_to_rgb(sh: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """sh (N,4,3), dirs (N,3) unit vectors from camera to Gaussian."""
    x, y, z = dirs.unbind(1)
    c = SH_C0 * sh[:, 0] + SH_C1 * (-y[:, None] * sh[:, 1] + z[:, None] * sh[:, 2] - x[:, None] * sh[:, 3])
    return (c + 0.5).clamp(min=0.0)


def rgb_to_sh_dc(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def render(model: GaussianModel, cam: Camera, bg: float = 0.0, tile: int = 16,
           colors_override: torch.Tensor | None = None, chunk_elems: int = 24_000_000):
    """Render a model. Returns dict rgb (H,W,3), depth (H,W), alpha (H,W),
    and per-Gaussian camera-space depth and pixel coordinates.

    Exact up to the standard 1/255 alpha cut-off: every Gaussian overlapping a
    tile is composited. Tiles are processed in groups of similar occupancy,
    each padded only to its own maximum, which bounds memory without
    truncating crowded tiles."""
    dev = model.means.device
    W, H = cam.W, cam.H
    R = cam.R.to(dev)
    t = cam.t.to(dev)

    p = model.means @ R.T + t
    x, y, z = p.unbind(1)
    valid = z > 0.2
    zc = z.clamp(min=0.2)
    u = cam.fx * x / zc + cam.cx
    v = cam.fy * y / zc + cam.cy

    Rg = quat_to_rot(model.quats)
    S = torch.exp(model.log_scales)
    M = Rg * S[:, None, :]
    cov3 = M @ M.transpose(1, 2)
    covc = R @ cov3 @ R.T
    J = torch.zeros(len(model), 2, 3, device=dev)
    J[:, 0, 0] = cam.fx / zc
    J[:, 0, 2] = -cam.fx * x / zc ** 2
    J[:, 1, 1] = cam.fy / zc
    J[:, 1, 2] = -cam.fy * y / zc ** 2
    cov2 = J @ covc @ J.transpose(1, 2)
    a = cov2[:, 0, 0] + 0.3
    b = cov2[:, 0, 1]
    c = cov2[:, 1, 1] + 0.3
    det = a * c - b * b
    valid = valid & (det > 1e-8)
    det = det.clamp(min=1e-8)
    conic = torch.stack([c / det, -b / det, a / det], 1)
    mid = 0.5 * (a + c)
    lam = mid + torch.sqrt((mid * mid - det).clamp(min=0.1))
    opac = torch.sigmoid(model.opacity_logit)
    with torch.no_grad():
        # radius beyond which alpha < 1/255 (and is cut off anyway)
        k_sigma = torch.sqrt((2 * torch.log((255 * opac).clamp(min=1.0))).clamp(min=0))
        radius = torch.ceil(k_sigma * torch.sqrt(lam))
        valid = valid & (radius > 0)

    TX, TY = math.ceil(W / tile), math.ceil(H / tile)
    T = TX * TY
    P = tile * tile
    with torch.no_grad():
        tx0 = torch.floor((u - radius) / tile).clamp(0, TX - 1).long()
        tx1 = torch.floor((u + radius) / tile).clamp(0, TX - 1).long()
        ty0 = torch.floor((v - radius) / tile).clamp(0, TY - 1).long()
        ty1 = torch.floor((v + radius) / tile).clamp(0, TY - 1).long()
        onscreen = (u + radius >= 0) & (u - radius < W) & (v + radius >= 0) & (v - radius < H)
        keep = valid & onscreen
        gid = torch.nonzero(keep).squeeze(1)
        wt = (tx1 - tx0 + 1)[gid]
        nt = wt * (ty1 - ty0 + 1)[gid]
        total = int(nt.sum())
        pair_g = torch.repeat_interleave(gid, nt)
        starts = torch.cumsum(nt, 0) - nt
        k = torch.arange(total, device=dev) - torch.repeat_interleave(starts, nt)
        wrep = torch.repeat_interleave(wt, nt)
        ptx = tx0[pair_g] + k % wrep
        pty = ty0[pair_g] + k // wrep
        tile_id = pty * TX + ptx
        depth_rank = torch.empty(len(model), dtype=torch.long, device=dev)
        depth_rank[torch.argsort(z.detach())] = torch.arange(len(model), device=dev)
        key = tile_id * len(model) + depth_rank[pair_g]
        order = torch.argsort(key)
        pair_g, tile_id = pair_g[order], tile_id[order]
        counts = torch.bincount(tile_id, minlength=T)
        tstart = torch.cumsum(counts, 0) - counts
        tile_order = torch.argsort(counts, descending=True)
        sorted_counts = counts[tile_order].tolist()
        # pixel coordinates per tile
        ly, lx = torch.meshgrid(torch.arange(tile, device=dev), torch.arange(tile, device=dev), indexing="ij")
        tyy, txx = torch.meshgrid(torch.arange(TY, device=dev), torch.arange(TX, device=dev), indexing="ij")
        px = (txx.reshape(-1, 1) * tile + lx.reshape(1, -1)).float() + 0.5   # (T,P)
        py = (tyy.reshape(-1, 1) * tile + ly.reshape(1, -1)).float() + 0.5

    if colors_override is None:
        dirs = model.means - cam.center.to(dev)
        dirs = dirs / dirs.norm(dim=1, keepdim=True)
        colors = sh_to_rgb(model.sh, dirs)
    else:
        colors = colors_override
    feats = torch.cat([colors, z[:, None]], 1)  # rgb + depth

    done_tiles, out_rgbd, out_a = [], [], []
    s = 0
    while s < T and sorted_counts[s] > 0:
        Mg = sorted_counts[s]                       # largest occupancy in this group
        n = max(1, chunk_elems // (P * Mg))
        grp = tile_order[s:s + n]
        cnt = counts[grp]
        grp = grp[cnt > 0]
        cnt = cnt[cnt > 0]
        s += n
        ar = torch.arange(Mg, device=dev)
        m = ar[None, :] < cnt[:, None]              # (t,M)
        ic = pair_g[(tstart[grp][:, None] + ar[None, :]).clamp(max=max(total - 1, 0))]
        mu_u, mu_v = u[ic], v[ic]
        cn = conic[ic]
        op = opac[ic] * m
        dx = px[grp][:, :, None] - mu_u[:, None, :]  # (t,P,M)
        dy = py[grp][:, :, None] - mu_v[:, None, :]
        power = -0.5 * (cn[:, None, :, 0] * dx * dx + cn[:, None, :, 2] * dy * dy) - cn[:, None, :, 1] * dx * dy
        alpha = (op[:, None, :] * torch.exp(power.clamp(max=0))).clamp(max=0.99)
        alpha = alpha * (alpha > 1.0 / 255).float() * (power <= 0).float()
        trans = torch.cumprod(1 - alpha, dim=2)
        trans = torch.cat([torch.ones_like(trans[..., :1]), trans[..., :-1]], 2)
        w = alpha * trans
        out_rgbd.append(torch.einsum("tpm,tmc->tpc", w, feats[ic]))
        out_a.append(w.sum(2))
        done_tiles.append(grp)
    rgbd = torch.zeros(T, P, 4, device=dev, dtype=feats.dtype)
    acc = torch.zeros(T, P, device=dev, dtype=feats.dtype)
    if done_tiles:
        idx_all = torch.cat(done_tiles)
        rgbd = rgbd.index_copy(0, idx_all, torch.cat(out_rgbd, 0))
        acc = acc.index_copy(0, idx_all, torch.cat(out_a, 0))

    def untile(x):  # (T,P,C) -> (H,W,C)
        C = x.shape[-1]
        x = x.reshape(TY, TX, tile, tile, C).permute(0, 2, 1, 3, 4).reshape(TY * tile, TX * tile, C)
        return x[:H, :W]

    rgbd = untile(rgbd)
    acc = untile(acc[..., None])[..., 0]
    rgb = rgbd[..., :3] + (1 - acc[..., None]) * bg
    depth = rgbd[..., 3] / acc.clamp(min=1e-6)
    return {"rgb": rgb, "depth": depth, "alpha": acc, "z": z.detach(), "u": u.detach(), "v": v.detach(),
            "visible": (valid & onscreen).detach()}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def _gauss_window(size=11, sigma=1.5, device="cpu"):
    g = torch.exp(-((torch.arange(size, device=device) - size // 2) ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: (H,W,C) in [0,1]. Returns per-pixel SSIM averaged over channels (H,W)."""
    a = a.permute(2, 0, 1)[None]
    b = b.permute(2, 0, 1)[None]
    C = a.shape[1]
    w = _gauss_window(device=a.device).expand(C, 1, -1, -1)
    mu_a = F.conv2d(a, w, padding=5, groups=C)
    mu_b = F.conv2d(b, w, padding=5, groups=C)
    sa = F.conv2d(a * a, w, padding=5, groups=C) - mu_a ** 2
    sb = F.conv2d(b * b, w, padding=5, groups=C) - mu_b ** 2
    sab = F.conv2d(a * b, w, padding=5, groups=C) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return s[0].mean(0)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return 10 * math.log10(1.0 / max(mse, 1e-10))


# ---------------------------------------------------------------------------
# Initialization and training
# ---------------------------------------------------------------------------
def init_from_views(views: list[dict], cams: list[Camera], stride: int = 3, voxel: float = 0.02,
                    max_points: int = 60_000, seed: int = 0, device="cpu") -> GaussianModel:
    """Initialize Gaussians from back-projected depth of the capture views.

    This plays the role of the structure-from-motion point cloud in a real
    pipeline. Each point inherits the component label of its pixel, which is
    how the baseline becomes 'labelled by component'."""
    pts, cols, comps = [], [], []
    for vw, cam in zip(views, cams):
        d = vw["depth"][::stride, ::stride]
        cmap = vw["comp"][::stride, ::stride]
        rgb = vw["rgb"][::stride, ::stride]
        ys, xs = torch.meshgrid(torch.arange(0, cam.H, stride, dtype=torch.float32) + 0.5,
                                torch.arange(0, cam.W, stride, dtype=torch.float32) + 0.5, indexing="ij")
        m = cmap > 0
        zc = d[m]
        xc = (xs[m] - cam.cx) / cam.fx * zc
        yc = (ys[m] - cam.cy) / cam.fy * zc
        pc = torch.stack([xc, yc, zc], 1)
        pw = (pc - cam.t) @ cam.R  # camera -> world
        pts.append(pw)
        cols.append(rgb[m])
        comps.append(cmap[m])
    pts = torch.cat(pts)
    cols = torch.cat(cols)
    comps = torch.cat(comps)
    # voxel downsample
    key = torch.floor(pts / voxel).long()
    key = key[:, 0] * 1_000_003 + key[:, 1] * 1_009 + key[:, 2]
    _, inv = torch.unique(key, return_inverse=True)
    first = torch.full((int(inv.max()) + 1,), len(pts), dtype=torch.long)
    first.scatter_reduce_(0, inv, torch.arange(len(pts)), reduce="amin")
    pts, cols, comps = pts[first], cols[first], comps[first]
    if len(pts) > max_points:
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(len(pts), generator=g)[:max_points]
        pts, cols, comps = pts[sel], cols[sel], comps[sel]
    n = len(pts)
    sh = torch.zeros(n, 4, 3)
    sh[:, 0] = rgb_to_sh_dc(cols.clamp(0.02, 0.98))
    model = GaussianModel(
        means=pts.clone(),
        log_scales=torch.full((n, 3), math.log(voxel * 0.8)),
        quats=torch.cat([torch.ones(n, 1), torch.zeros(n, 3)], 1),
        opacity_logit=torch.full((n,), 0.0),
        sh=sh,
        comp=comps.clone(),
    )
    return GaussianModel(*(p.to(device).requires_grad_(True) for p in model.params()), model.comp.to(device))


def train(model: GaussianModel, images: list[torch.Tensor], cams: list[Camera], iters: int = 3000,
          lr_scale: float = 1.0, seed: int = 0, log_every: int = 250, prune_at=(1000, 2000), log=print):
    g = torch.Generator().manual_seed(seed)
    groups = [
        {"params": [model.means], "lr": 4e-4 * lr_scale},
        {"params": [model.log_scales], "lr": 5e-3 * lr_scale},
        {"params": [model.quats], "lr": 1e-3 * lr_scale},
        {"params": [model.opacity_logit], "lr": 5e-2 * lr_scale},
        {"params": [model.sh], "lr": 2.5e-3 * lr_scale},
    ]
    opt = torch.optim.Adam(groups, eps=1e-15)
    for it in range(1, iters + 1):
        frac = it / iters
        groups[0]["lr"] = 4e-4 * lr_scale * (0.02 ** frac)  # exponential decay of position lr
        i = int(torch.randint(len(images), (1,), generator=g))
        out = render(model, cams[i])
        gt = images[i]
        l1 = (out["rgb"] - gt).abs().mean()
        dssim = (1 - ssim_map(out["rgb"], gt).mean()) / 2
        loss = 0.8 * l1 + 0.2 * dssim
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it in prune_at:
            with torch.no_grad():
                keep = torch.sigmoid(model.opacity_logit) > 0.02
            if (~keep).any():
                model = _prune(model, keep, opt, groups)
        if log_every and it % log_every == 0:
            log(f"iter {it:5d}  loss {loss.item():.4f}  gaussians {len(model)}")
    return model


def _prune(model, keep, opt, groups):
    new = []
    for gi, p in enumerate(model.params()):
        q = p.detach()[keep].clone().requires_grad_(True)
        st = opt.state.pop(p, None)
        if st:
            st = {k: (v[keep] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in st.items()}
            opt.state[q] = st
        groups[gi]["params"] = [q]
        new.append(q)
    return GaussianModel(*new, model.comp[keep])
