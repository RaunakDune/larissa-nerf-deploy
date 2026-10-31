"""Procedural satellite scene and a GPU ray-caster.

The scene stands in for a real spacecraft with labelled components. Every
pixel the ray-caster produces carries a component id and a material id, which
gives exact ground truth for change detection.

Coordinates: metres, satellite at the origin, +z up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch

# ---------------------------------------------------------------------------
# Component and material tables
# ---------------------------------------------------------------------------
COMPONENTS = {
    0: "background",
    1: "bus",
    2: "solar_array_port",
    3: "solar_array_starboard",
    4: "antenna",
    5: "thruster",
}

# name: (albedo rgb, diffuse k, specular k, shininess, texture)
MATERIALS = {
    "mli": ((0.85, 0.64, 0.26), 0.85, 0.35, 24.0, "crinkle"),
    "mli_degraded": ((0.30, 0.20, 0.13), 0.85, 0.10, 8.0, "crinkle"),
    "radiator": ((0.86, 0.86, 0.88), 0.90, 0.10, 16.0, "none"),
    "solar": ((0.07, 0.11, 0.33), 0.80, 0.55, 60.0, "cells"),
    "boom": ((0.55, 0.55, 0.57), 0.85, 0.30, 30.0, "none"),
    "dish": ((0.90, 0.90, 0.90), 0.90, 0.15, 20.0, "none"),
    "metal": ((0.32, 0.32, 0.35), 0.80, 0.45, 40.0, "none"),
}
MATERIAL_IDS = {name: i + 1 for i, name in enumerate(MATERIALS)}

AMBIENT = 0.09  # Earth albedo and scattered light


def _rot(axis: str, deg: float) -> torch.Tensor:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
    if axis == "y":
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)


@dataclass
class Primitive:
    kind: str                 # "box" or "cyl" (cylinder along local z)
    center: torch.Tensor      # (3,)
    R: torch.Tensor           # (3,3) local -> world
    size: torch.Tensor        # box: half extents (3,); cyl: (radius, half height, 0)
    comp: int
    material: str
    # optional per-face material override for boxes: {face_key: material}
    face_materials: dict = field(default_factory=dict)
    # optional rectangular patch on one box face: (face_key, (u0,u1,v0,v1), material)
    patch: tuple | None = None


@dataclass
class Scene:
    prims: list[Primitive]
    sun_dir: torch.Tensor     # unit vector pointing from the scene toward the Sun


def sun_direction(azimuth_deg: float = 68.0, elevation_deg: float = 30.0) -> torch.Tensor:
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return torch.tensor([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])


def build_satellite(fault: str | None = None, sun_azimuth_deg: float = 68.0) -> Scene:
    """Build the satellite. `fault` is one of None, 'panel_hinge',
    'antenna_missing', 'mli_discoloration'."""
    I = torch.eye(3)
    prims: list[Primitive] = []

    # Bus: gold MLI with a white radiator on -y.
    bus = Primitive("box", torch.tensor([0.0, 0.0, 0.0]), I, torch.tensor([0.6, 0.6, 0.75]), 1, "mli",
                    face_materials={"-y": "radiator"})
    if fault == "mli_discoloration":
        # Degraded blanket patch on the +y face (u = local x, v = local z).
        bus.patch = ("+y", (-0.38, 0.30, -0.55, 0.15), "mli_degraded")
    prims.append(bus)

    # Solar arrays: boom + panel on each side. Panels lie in the x-z plane.
    for comp, sign in ((2, 1.0), (3, -1.0)):
        prims.append(Primitive("box", torch.tensor([sign * 0.9, 0.0, 0.0]), I,
                               torch.tensor([0.3, 0.035, 0.035]), comp, "boom"))
        hinge = torch.tensor([sign * 1.2, 0.0, 0.0])
        R = I
        if fault == "panel_hinge" and comp == 2:
            R = _rot("z", 22.0)  # partially deployed hinge, rotated about the vertical axis
        offset = torch.tensor([sign * 0.92, 0.0, 0.0])
        prims.append(Primitive("box", hinge + R @ offset, R, torch.tensor([0.9, 0.02, 0.45]), comp, "solar"))

    # Antenna: mast + tilted dish on top.
    if fault != "antenna_missing":
        prims.append(Primitive("cyl", torch.tensor([0.0, 0.0, 0.9]), I, torch.tensor([0.04, 0.15, 0.0]), 4, "boom"))
        Rd = _rot("y", 25.0)
        prims.append(Primitive("cyl", torch.tensor([0.05, 0.0, 1.1]), Rd, torch.tensor([0.42, 0.025, 0.0]), 4, "dish"))

    # Thruster nozzle underneath.
    prims.append(Primitive("cyl", torch.tensor([0.0, 0.0, -0.88]), I, torch.tensor([0.16, 0.13, 0.0]), 5, "metal"))

    return Scene(prims, sun_direction(sun_azimuth_deg))


# ---------------------------------------------------------------------------
# Ray intersection
# ---------------------------------------------------------------------------
def _intersect_box(o, d, p: Primitive):
    """Returns t (N,), local normal (N,3), local hit (N,3)."""
    R, c, h = p.R.to(o), p.center.to(o), p.size.to(o)
    ol = (o - c) @ R
    dl = d @ R
    inv = 1.0 / torch.where(dl.abs() < 1e-9, torch.full_like(dl, 1e-9), dl)
    t0 = (-h - ol) * inv
    t1 = (h - ol) * inv
    tmin = torch.minimum(t0, t1)
    tmax = torch.maximum(t0, t1)
    tnear, axis = tmin.max(dim=1)
    tfar = tmax.min(dim=1).values
    hit = (tnear <= tfar) & (tfar > 1e-4)
    t = torch.where(tnear > 1e-4, tnear, tfar)
    t = torch.where(hit, t, torch.full_like(t, float("inf")))
    pl = ol + t[:, None].clamp(max=1e6) * dl
    nl = torch.zeros_like(ol)
    sgn = torch.sign(pl.gather(1, axis[:, None]).squeeze(1))
    nl.scatter_(1, axis[:, None], sgn[:, None])
    return t, nl, pl


def _intersect_cyl(o, d, p: Primitive):
    R, c = p.R.to(o), p.center.to(o)
    r, hh = float(p.size[0]), float(p.size[1])
    ol = (o - c) @ R
    dl = d @ R
    inf = torch.full((o.shape[0],), float("inf"), device=o.device)
    # side
    a = dl[:, 0] ** 2 + dl[:, 1] ** 2
    b = 2 * (ol[:, 0] * dl[:, 0] + ol[:, 1] * dl[:, 1])
    cc = ol[:, 0] ** 2 + ol[:, 1] ** 2 - r * r
    disc = b * b - 4 * a * cc
    sq = torch.sqrt(disc.clamp(min=0))
    a_s = torch.where(a.abs() < 1e-12, torch.full_like(a, 1e-12), a)
    ts = []
    for tc in ((-b - sq) / (2 * a_s), (-b + sq) / (2 * a_s)):
        z = ol[:, 2] + tc * dl[:, 2]
        ok = (disc >= 0) & (tc > 1e-4) & (z.abs() <= hh) & (a > 1e-12)
        ts.append(torch.where(ok, tc, inf))
    t_side = torch.minimum(ts[0], ts[1])
    # caps
    dz = torch.where(dl[:, 2].abs() < 1e-12, torch.full_like(dl[:, 2], 1e-12), dl[:, 2])
    t_caps = []
    for zc in (hh, -hh):
        tc = (zc - ol[:, 2]) / dz
        pc = ol + tc[:, None] * dl
        ok = (tc > 1e-4) & (pc[:, 0] ** 2 + pc[:, 1] ** 2 <= r * r)
        t_caps.append(torch.where(ok, tc, inf))
    t_cap = torch.minimum(t_caps[0], t_caps[1])
    t = torch.minimum(t_side, t_cap)
    pl = ol + t.clamp(max=1e6)[:, None] * dl
    n_side = torch.stack([pl[:, 0], pl[:, 1], torch.zeros_like(pl[:, 0])], 1) / r
    n_cap = torch.zeros_like(pl)
    n_cap[:, 2] = torch.sign(pl[:, 2])
    nl = torch.where((t_side <= t_cap)[:, None], n_side, n_cap)
    return t, nl, pl


def _intersect(o, d, p: Primitive):
    return _intersect_box(o, d, p) if p.kind == "box" else _intersect_cyl(o, d, p)


# ---------------------------------------------------------------------------
# Procedural textures
# ---------------------------------------------------------------------------
def _hash3(p: torch.Tensor) -> torch.Tensor:
    return torch.frac(torch.sin(p @ torch.tensor([12.9898, 78.233, 37.719], device=p.device)) * 43758.5453)


def _value_noise(x: torch.Tensor) -> torch.Tensor:
    i = torch.floor(x)
    f = x - i
    u = f * f * (3 - 2 * f)
    out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                corner = i + torch.tensor([dx, dy, dz], device=x.device, dtype=x.dtype)
                w = (u[:, 0] if dx else 1 - u[:, 0]) * (u[:, 1] if dy else 1 - u[:, 1]) * (u[:, 2] if dz else 1 - u[:, 2])
                out = out + w * _hash3(corner)
    return out


def _texture(kind: str, pl: torch.Tensor, nl: torch.Tensor) -> torch.Tensor:
    """Multiplicative albedo modulation (N,1) or additive line mask for cells."""
    if kind == "crinkle":
        n = 0.6 * _value_noise(pl * 9.0) + 0.4 * _value_noise(pl * 23.0)
        return (0.72 + 0.45 * n)[:, None]
    if kind == "cells":
        cell = 0.15
        gx = torch.frac(pl[:, 0] / cell)
        gz = torch.frac(pl[:, 2] / cell)
        line = ((gx < 0.06) | (gz < 0.06)).float()
        return line[:, None]  # handled specially (silver grid lines)
    return torch.ones(pl.shape[0], 1, device=pl.device)


def _face_key(nl: torch.Tensor) -> list:
    keys = []
    for ax, name in enumerate("xyz"):
        keys.append((nl[:, ax] > 0.5, "+" + name))
        keys.append((nl[:, ax] < -0.5, "-" + name))
    return keys


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
@dataclass
class Camera:
    R: torch.Tensor   # world -> camera rotation (3,3); camera looks along +z, y down
    t: torch.Tensor   # world -> camera translation (3,)
    fx: float
    fy: float
    cx: float
    cy: float
    W: int
    H: int

    @property
    def center(self) -> torch.Tensor:
        return -self.R.T @ self.t

    def to(self, device):
        return replace(self, R=self.R.to(device), t=self.t.to(device))


def look_at(eye, target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), W=256, H=256, fov_scale=1.3) -> Camera:
    eye = torch.as_tensor(eye, dtype=torch.float32)
    target = torch.as_tensor(target, dtype=torch.float32)
    up = torch.as_tensor(up, dtype=torch.float32)
    fwd = target - eye
    fwd = fwd / fwd.norm()
    if abs(float(fwd @ up)) > 0.98:
        up = torch.tensor([0.0, 1.0, 0.0])
    right = torch.linalg.cross(fwd, up)
    right = right / right.norm()
    down = torch.linalg.cross(fwd, right)
    R = torch.stack([right, down, fwd], 0)
    t = -R @ eye
    f = fov_scale * W
    return Camera(R, t, f, f, W / 2, H / 2, W, H)


def perturb(cam: Camera, rot_deg: float, trans_m: float, gen: torch.Generator) -> Camera:
    """Small random pose error, standing in for calibration drift."""
    axis = torch.randn(3, generator=gen)
    axis = axis / axis.norm()
    a = math.radians(rot_deg)
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    dR = torch.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    dt = torch.randn(3, generator=gen)
    dt = dt / dt.norm() * trans_m
    return replace(cam, R=dR @ cam.R, t=dR @ cam.t + dt)


def _camera_rays(cam: Camera, device):
    ys, xs = torch.meshgrid(torch.arange(cam.H, device=device, dtype=torch.float32) + 0.5,
                            torch.arange(cam.W, device=device, dtype=torch.float32) + 0.5, indexing="ij")
    dc = torch.stack([(xs - cam.cx) / cam.fx, (ys - cam.cy) / cam.fy, torch.ones_like(xs)], -1).reshape(-1, 3)
    R = cam.R.to(device)
    dw = dc @ R  # camera -> world (R^T applied to column vectors)
    dw = dw / dw.norm(dim=1, keepdim=True)
    o = cam.center.to(device).expand_as(dw).contiguous()
    return o, dw


def _trace(o, d, scene: Scene):
    n = o.shape[0]
    best_t = torch.full((n,), float("inf"), device=o.device)
    best_i = torch.full((n,), -1, dtype=torch.long, device=o.device)
    best_nl = torch.zeros(n, 3, device=o.device)
    best_pl = torch.zeros(n, 3, device=o.device)
    for i, p in enumerate(scene.prims):
        t, nl, pl = _intersect(o, d, p)
        closer = t < best_t
        best_t = torch.where(closer, t, best_t)
        best_i = torch.where(closer, torch.full_like(best_i, i), best_i)
        best_nl = torch.where(closer[:, None], nl, best_nl)
        best_pl = torch.where(closer[:, None], pl, best_pl)
    return best_t, best_i, best_nl, best_pl


@torch.no_grad()
def render_scene(scene: Scene, cam: Camera, device="cpu", supersample: int = 2):
    """Returns dict with rgb (H,W,3), depth (H,W), comp (H,W), material (H,W).

    rgb is anti-aliased by supersampling; the label maps use the centre sample."""
    hi = replace(cam, fx=cam.fx * supersample, fy=cam.fy * supersample, cx=cam.cx * supersample,
                 cy=cam.cy * supersample, W=cam.W * supersample, H=cam.H * supersample)
    o, d = _camera_rays(hi, device)
    t, idx, nl, pl = _trace(o, d, scene)
    hitmask = idx >= 0
    rgb = torch.zeros(o.shape[0], 3, device=device)
    comp = torch.zeros(o.shape[0], dtype=torch.long, device=device)
    mat = torch.zeros(o.shape[0], dtype=torch.long, device=device)
    sun = scene.sun_dir.to(device)
    sun = sun / sun.norm()

    for i, p in enumerate(scene.prims):
        m = hitmask & (idx == i)
        if not m.any():
            continue
        R = p.R.to(device)
        nl_i, pl_i = nl[m], pl[m]
        n_w = nl_i @ R.T
        view = -d[m]
        flip = (n_w * view).sum(1) < 0  # thin parts are two-sided
        n_w = torch.where(flip[:, None], -n_w, n_w)
        # material per pixel
        names = [p.material] * int(m.sum())
        mat_ids = torch.full((int(m.sum()),), MATERIAL_IDS[p.material], device=device)
        if p.kind == "box":
            for fm, key in _face_key(nl_i):
                if key in p.face_materials:
                    mat_ids[fm] = MATERIAL_IDS[p.face_materials[key]]
            if p.patch is not None:
                key, (u0, u1, v0, v1), pm = p.patch
                axis = "xyz".index(key[1])
                sgn = 1.0 if key[0] == "+" else -1.0
                uax, vax = [a for a in range(3) if a != axis]
                inpatch = (nl_i[:, axis] * sgn > 0.5) & (pl_i[:, uax] >= u0) & (pl_i[:, uax] <= u1) & \
                          (pl_i[:, vax] >= v0) & (pl_i[:, vax] <= v1)
                mat_ids[inpatch] = MATERIAL_IDS[pm]
        out = torch.zeros(int(m.sum()), 3, device=device)
        # shadow test
        hit_pts = o[m] + t[m][:, None] * d[m] + n_w * 2e-3
        st, _, _, _ = _trace(hit_pts, sun.expand_as(hit_pts).contiguous(), scene)
        lit = torch.isinf(st).float()[:, None]
        for name, mid in MATERIAL_IDS.items():
            mm = mat_ids == mid
            if not mm.any():
                continue
            alb, kd, ks, shin, tex = MATERIALS[name]
            alb = torch.tensor(alb, device=device).expand(int(mm.sum()), 3)
            tx = _texture(tex, pl_i[mm], nl_i[mm])
            if tex == "cells":
                alb = alb * (1 - tx) + torch.tensor([0.75, 0.75, 0.78], device=device) * tx
            else:
                alb = alb * tx
            nn, vv = n_w[mm], view[mm]
            ndl = (nn @ sun).clamp(min=0)[:, None]
            h = sun[None] + vv
            h = h / h.norm(dim=1, keepdim=True)
            spec = ((nn * h).sum(1).clamp(min=0) ** shin)[:, None]
            out[mm] = alb * (AMBIENT + kd * ndl * lit[mm]) + ks * spec * lit[mm] * ndl.gt(0).float()
        rgb[m] = out
        comp[m] = p.comp
        mat[m] = mat_ids
        del names

    depth_world = torch.where(hitmask, t, torch.zeros_like(t))
    # convert ray distance to camera z-depth
    zdir = cam.R.to(device)[2]
    depth = depth_world * (d @ zdir)
    H, W, s = cam.H, cam.W, supersample
    rgb = rgb.clamp(0, 1).reshape(H, s, W, s, 3).mean((1, 3))
    centre = s // 2
    depth = depth.reshape(H, s, W, s)[:, centre, :, centre]
    comp = comp.reshape(H, s, W, s)[:, centre, :, centre]
    mat = mat.reshape(H, s, W, s)[:, centre, :, centre]
    return {"rgb": rgb, "depth": depth, "comp": comp, "material": mat}


# ---------------------------------------------------------------------------
# Camera sets
# ---------------------------------------------------------------------------
def fibonacci_cameras(n: int, radius: float, W: int, H: int, offset: float = 0.0) -> list[Camera]:
    cams = []
    golden = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1 - z * z))
        th = golden * i + offset
        eye = (radius * r * math.cos(th), radius * r * math.sin(th), radius * z)
        cams.append(look_at(eye, W=W, H=H))
    return cams


def inspection_cameras(W: int, H: int, radius: float = 8.5) -> list[Camera]:
    """Eight fixed 'station' cameras on two rings, at poses not used for training."""
    cams = []
    for k in range(8):
        az = math.radians(22.5 + 45 * k)
        el = math.radians(28 if k % 2 == 0 else -12)
        eye = (radius * math.cos(el) * math.cos(az), radius * math.cos(el) * math.sin(az), radius * math.sin(el))
        cams.append(look_at(eye, W=W, H=H))
    return cams
