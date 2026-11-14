import sys, torch, json
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from ablation_reference import cams_at
from larissa.detect import difference_map
from larissa.gaussians import GaussianModel, render
from larissa.scene import build_satellite, render_scene, COMPONENTS, MATERIALS
dev = torch.device("cuda:0")
model = GaussianModel.from_state_dict(torch.load("data/larissa_synth_v1/baseline_gaussians.pt", weights_only=True), dev)
h = build_satellite(None, 68.0)
for delta in (0, 5, 10, 20, 45):
    comp_hi = {}; comp_n = {}; mat_hi = {}; mat_n = {}; percam = []
    for k, c in enumerate(cams_at(delta, 256)):
        tr = render_scene(h, c, device=dev)
        with torch.no_grad():
            d = difference_map(tr["rgb"], render(model, c)["rgb"].clamp(0, 1))
        hi = d > 0.04
        percam.append(round(100 * float((hi & (tr["comp"] > 0)).sum()) / max(1, int((tr["comp"] > 0).sum())), 1))
        for cid, name in COMPONENTS.items():
            if cid:
                s = tr["comp"] == cid
                comp_n[name] = comp_n.get(name, 0) + int(s.sum()); comp_hi[name] = comp_hi.get(name, 0) + int((s & hi).sum())
        for mid in tr["material"].unique().tolist():
            if mid > 0:
                s = tr["material"] == mid
                mat_n[mid] = mat_n.get(mid, 0) + int(s.sum()); mat_hi[mid] = mat_hi.get(mid, 0) + int((s & hi).sum())
    print("delta", delta, "per camera %", percam)
    print("  by component %", {n: round(100 * comp_hi[n] / max(1, comp_n[n]), 1) for n in comp_n})
    print("  by material %", {MATERIALS[m - 1] if isinstance(MATERIALS, (list, tuple)) else m: round(100 * mat_hi[m] / mat_n[m], 1) for m in mat_n})
