"""Re-run detection and classification on the fault scenarios with a saved baseline.

Usage: python tools/classifier_check.py SCOPE ITERS LR   (SCOPE: component | flagged)
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, warnings; warnings.filterwarnings("ignore")
from larissa.scene import *
from larissa.gaussians import *
from larissa.detect import *
dev = torch.device("cuda:0"); S = 256
m = GaussianModel.from_state_dict(torch.load("baseline/baseline_gaussians.pt"), dev)
g = torch.Generator().manual_seed(11)
ic = inspection_cameras(S, S)
hold = fibonacci_cameras(108, 9.0, S, S)[::9]
def imgs(sc, cams):
    return [add_sensor_noise(render_scene(sc, perturb(c, 0.15, 0.01, g), device=dev)["rgb"].cpu(), g).to(dev) for c in cams]
hi = imgs(build_satellite(None), hold)
tau = calibrate_threshold(m, hi, hold); thr = calibrate_components(m, hi, hold, tau)
scope = sys.argv[1]; iters = int(sys.argv[2]); lr = float(sys.argv[3])
for name, f, az in [("mli", "mli_discoloration", 68.0), ("hinge", "panel_hinge", 68.0), ("antenna", "antenna_missing", 68.0), ("sun", None, 98.0)]:
    im = imgs(build_satellite(f, az), ic)
    d = detect(m, im, ic, tau, comp_thresholds=thr)
    cl = classify_components(m, im, ic, d, iters=iters, lr=lr, scope=scope, log=lambda *a: None)
    print(name, verdict(cl), {k: (v["type"][:3], v["residual_reduction"], v["new_matter_fraction"], v["luminance_before"], v["luminance_after"], v["seconds"]) for k, v in cl.items()}, flush=True)
