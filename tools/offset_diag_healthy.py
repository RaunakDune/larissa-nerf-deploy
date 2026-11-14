import sys, torch
sys.path.insert(0, "tools"); sys.path.insert(0, ".")
from ablation_reference import cams_at
from larissa.detect import add_sensor_noise, detect
from larissa.gaussians import GaussianModel
from larissa.scene import build_satellite, perturb, render_scene
dev = torch.device("cuda:0")
model = GaussianModel.from_state_dict(torch.load("data/larissa_synth_v1/baseline_gaussians.pt", weights_only=True), dev)
h = build_satellite(None, 68.0)
for delta in (10.0, 5.0, 0.0):
    cams = cams_at(delta, 256)
    gen = torch.Generator().manual_seed(777 + int(delta))
    seq = []
    for _ in range(60):
        p = [add_sensor_noise(render_scene(h, perturb(c, 0.15, 0.01, gen), device=dev)["rgb"].cpu(), gen).to(dev) for c in cams]
        seq.append(detect(model, p, cams, 0.04).components["antenna"]["flagged"])
    print("delta", delta, "antenna flagged over 60 healthy passes:", seq, "zero share", sum(s == 0 for s in seq) / 60, flush=True)
