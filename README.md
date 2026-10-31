# Project Larissa: change-detection demo

A small, self-contained demonstration of the core idea behind Project Larissa:
**detect mechanical faults on a spacecraft by comparing new camera images
against a 3D Gaussian Splatting (3DGS) model of the spacecraft in a known-good
state.**

It is a proof of concept on a synthetic satellite, not the full research system.

![Scenario results](results/scenarios.png)

## What the demo does

1. **Builds a test spacecraft.** A procedural satellite (bus with gold thermal
   blanket and a radiator, two solar arrays, an antenna, a thruster) is
   ray-traced under hard directional sunlight with shadows and specular
   glints. Every pixel carries a component label, which gives exact ground
   truth.
2. **Builds the baseline.** A 3DGS model is trained from 96 views of the healthy
   satellite. Each Gaussian inherits the component label of the surface it was
   initialized from, so the baseline is *labelled by component*.
3. **Calibrates a change threshold** on 12 held-out healthy views that include
   the same camera pose error and sensor noise as the inspections. No fault
   data are used to tune it.
4. **Inspects.** Eight fixed "station" cameras, at poses never used for
   training, image the satellite in five scenarios (below). Every inspection
   image has sensor noise, an exposure jitter and a small camera pose error.
5. **Detects and localizes.** The baseline is rendered at each camera's nominal
   pose and compared with the new image (blurred colour difference plus
   structural (SSIM) difference). Each Gaussian then receives a vote from every
   camera that sees it; it is flagged only if most of those cameras see a
   change. Flagged Gaussians are summed per component.
6. **Classifies.** For each changed component, the colours of its flagged
   Gaussians are refit to the new images with geometry frozen. If colour alone
   explains the change, it is an *appearance* change (discoloration,
   contamination); if it cannot, or the part has to turn black to match, it is
   a *geometric* change (moved, bent or missing hardware).

## Scenarios

| Scenario | What changed | Expected verdict |
|---|---|---|
| `healthy_control` | Nothing; pose error and sensor noise only | No change |
| `panel_hinge` | Port solar array hinge stuck 22° short of deployment | Geometric |
| `antenna_missing` | Antenna dish and mast detached | Geometric |
| `mli_discoloration` | Thermal blanket patch on the bus darkened | Appearance |
| `sun_shift_control` | Nothing; Sun azimuth moved 30° | No change |

## Results

RESULTS_PLACEHOLDER

## Running it

```bash
git clone https://github.com/RaunakDune/larissa-nerf-deploy.git
cd larissa-nerf-deploy
bash run_demo.sh 0            # GPU index; creates ./venv on first use
# or, without a GPU, a small smoke test:
pip install -r requirements.txt && python run_demo.py --device cpu --quick
```

Outputs go to `results/`: `summary.md`, `summary.json`, `scenarios.png`,
`training_views.png`, `log.txt` and the trained baseline (`baseline_gaussians.pt`,
not committed).

## Code layout

| File | Purpose |
|---|---|
| `larissa/scene.py` | Procedural satellite, fault variants, GPU ray-caster with shadows and procedural textures, camera rigs |
| `larissa/gaussians.py` | 3D Gaussian Splatting: model, tile-based differentiable rasterizer in plain PyTorch, training |
| `larissa/detect.py` | Threshold calibration, render-and-compare, multi-view voting, per-component report, geometric versus appearance test |
| `larissa/report.py` | Figures and the markdown summary |
| `run_demo.py` | End-to-end script |

The rasterizer is written with ordinary PyTorch operations, without custom
CUDA kernels. That makes it portable (it runs on the Tesla P40 cards in
`joshi.hpcc.uh.edu`, which recent compiled 3DGS libraries no longer target)
but several times slower than the reference implementation. Swapping in a
compiled rasterizer such as `gsplat` is the obvious first speed-up.

## Limitations

These are deliberate simplifications for a demo; each is a work item for the
funded project.

- **Synthetic data.** The spacecraft, lighting and faults are simulated. Real
  imagery (for example the open Astrobee ISS datasets) is the next step.
- **Known camera poses.** Cameras are assumed calibrated, with a small simulated
  error. Pose estimation from images (PnP against the baseline) is not
  implemented.
- **Initialization from depth.** The baseline is seeded from rendered depth,
  standing in for a structure-from-motion point cloud.
- **Fixed illumination model.** The baseline has no notion of Sun angle. The
  Sun-shift control shows what that costs; a Sun-aware relightable baseline
  is the planned fix.
- **Simple detector.** Render-and-compare with voting, not a reimplementation of
  published methods such as MV3DCD or GS-Diff.
- **Heuristic rules.** Thresholds for voting, component flagging and the
  geometric/appearance test were set by hand and checked on these five
  scenarios only.
