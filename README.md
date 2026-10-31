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
3. **Calibrates its alarm levels on healthy data.** Twelve held-out views of the
   healthy satellite, with the same camera pose error and sensor noise as the
   inspections, set (a) the per-pixel change threshold and (b) for each
   component, how much flagging counts as a real change.
4. **Inspects.** Eight fixed "station" cameras, at poses never used for
   training, image the satellite in five scenarios (below). Every inspection
   image has sensor noise, an exposure jitter and a small camera pose error.
5. **Detects and localizes.** The baseline is rendered at each camera's nominal
   pose and compared with the new image. The comparison tolerates
   misregistration of up to 2 pixels, so small pose errors along the sharp
   silhouette against black space do not register as change. Each Gaussian
   then receives a vote from every camera that sees it, and is flagged only if
   most of those cameras see a change. Flagged Gaussians are counted per
   component.
6. **Classifies.** For each changed component, the system samples what every
   camera now sees where that component's flagged Gaussians project:
   - **appearance change** (discoloration, contamination): the surface is still
     there, so the cameras agree on one new colour;
   - **geometric change, vanished:** the cameras see black space where a sunlit
     surface used to be;
   - **geometric change, moved or deformed:** the cameras see whatever is now
     behind, so they disagree.

## Scenarios

| Scenario | What changed | Expected verdict |
|---|---|---|
| `healthy_control` | Nothing; pose error and sensor noise only | No change |
| `panel_hinge` | Port solar array hinge stuck 22° short of deployment | Geometric |
| `antenna_missing` | Antenna dish and mast detached | Geometric |
| `mli_discoloration` | Thermal blanket patch on the bus darkened | Appearance |
| `sun_shift_control` | Nothing; Sun azimuth moved 30° | No fault (an "illumination" notice would be acceptable) |

## Results

Single end-to-end run of `run_demo.py` (default settings, training from scratch)
on one Tesla P40 in `joshi.hpcc.uh.edu`, at commit `4fe95a3`. Full output: [`results/summary.md`](results/summary.md),
[`results/summary.json`](results/summary.json), [`results/log.txt`](results/log.txt).

**4 of 5 scenarios gave the expected verdict.** All three faults were detected,
attributed to the right component and given the right type; the healthy control
raised no alarm; the Sun-shift control raised false alarms.

| Scenario | Verdict | Correct | Flagged Gaussians on the faulty part | Pixel recall | Pixel IoU |
|---|---|---|---|---|---|
| Healthy control | No change | yes | none above alarm level | | |
| Panel hinge | Geometric: port solar array | yes | 1981 / 2191 (90%) | 0.49 | 0.29 |
| Antenna missing | Geometric: antenna | yes | 420 / 462 (91%) | 0.62 | 0.27 |
| Blanket discoloration | Appearance: bus | yes | 305 / 4277 (7%; the patch is about 5% of the bus) | 0.93 | 0.40 |
| Sun shift (no fault) | Geometric: port solar array, antenna | **no** | (false alarm) | | |

**Speed.** Once the baseline exists, one inspection pass (rendering the
baseline at 8 camera poses, comparing, voting and reporting) takes **0.46 s**
on the P40. Building the baseline took 36 minutes (5000 iterations, 43,390
Gaussians, held-out PSNR 32.5 dB) because the rasterizer is plain PyTorch;
compiled 3DGS libraries report minutes or less for comparable scenes.

**How to read the figure.** The third column shows pixel-level changes
*before* voting. Even the healthy control has a few scattered red pixels (about
0.5% of the image) where the baseline renders glints and fine blanket texture
imperfectly. These do not raise an alarm because a change has to be seen
consistently by most cameras viewing a Gaussian, and a component has to exceed
the level its healthy views produce. The fourth column shows the Gaussians that
survived voting, on a 3D overview of the satellite.

**Pixel IoU is modest by design.** The decision is made per component, not per
pixel: the masks are deliberately smoothed and partly miss thin or grazing
regions, while the component-level verdicts are right.

**Why the Sun-shift control fails.** Moving the Sun by 30° moves every shadow
and glint. The detector correctly sees that the sunlit side changed (48% of the
bus flagged), but the classifier cannot tell moved glints from moved hardware,
because the baseline was captured under a single Sun angle. This is the main
open problem identified in the literature review, and a natural first work
package for the funded project: a baseline that can be re-rendered for the current Sun
direction (for example Park and D'Amico, iSpaRo 2025), or comparison only
against captures at matching Sun geometry.

## Dataset and notebook

- **[`data/larissa_synth_v1`](data/larissa_synth_v1)**: Larissa-Synth v1, every
  image, depth map, component label, camera pose and ground-truth change mask
  used in the results above, plus the trained baseline (38 MB). See its
  [README](data/larissa_synth_v1/README.md).
- **[`notebooks/larissa_walkthrough.ipynb`](notebooks/larissa_walkthrough.ipynb)**:
  loads the dataset and baseline, calibrates on the healthy held-out views, runs
  all five inspection scenarios and checks the verdicts against
  `results/summary.json`. Executed on a Tesla P40; it reproduces the original
  run exactly (same threshold, same flagged-Gaussian counts) in a few seconds.

## Running it

```bash
git clone https://github.com/RaunakDune/larissa-nerf-deploy.git
cd larissa-nerf-deploy
bash run_demo.sh 0            # GPU index; creates ./venv on first use
# or, without a GPU, a small smoke test:
pip install -r requirements.txt && python run_demo.py --device cpu --quick
```

Expect about 40 minutes on a Tesla P40, almost all of it baseline training. To
re-run only the detection on a saved baseline:
`python run_demo.py --load-baseline results/baseline_gaussians.pt --out results_rerun`.

Outputs go to `results/`: `summary.md`, `summary.json`, `scenarios.png`,
`training_views.png`, `log.txt` and the trained baseline (`baseline_gaussians.pt`,
not committed).

## Code layout

| File | Purpose |
|---|---|
| `larissa/scene.py` | Procedural satellite, fault variants, GPU ray-caster with shadows and procedural textures, camera rigs |
| `larissa/gaussians.py` | 3D Gaussian Splatting: model, tile-based differentiable rasterizer in plain PyTorch, training |
| `larissa/detect.py` | Calibration on healthy views, shift-tolerant render-and-compare, multi-view voting, per-component report, geometric versus appearance test |
| `larissa/dataset.py` | Loader for the released dataset |
| `tools/export_dataset.py` | Regenerates the dataset (replays the random sequence of `run_demo.py`) |
| `tools/classifier_check.py` | Re-runs detection and classification on all scenarios using a saved baseline (minutes, not the full training time) |
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
- **Fixed illumination model.** The baseline has no notion of Sun angle, and
  the Sun-shift control fails because of it (see Results). A Sun-aware
  relightable baseline is the planned fix.
- **Simple detector.** Render-and-compare with voting, not a reimplementation of
  published methods such as MV3DCD or GS-Diff.
- **Rules tuned on the scenarios they are tested on.** The detector went
  through several iterations against these same five scenarios (git history
  shows each change). The pixel and component thresholds are calibrated on
  healthy data, but the voting, classification thresholds and component alarm
  offset were chosen by hand while looking at these results. Independent fault
  scenarios are needed before any accuracy claim.
