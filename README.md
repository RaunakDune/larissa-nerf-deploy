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
on one Tesla P40 in `joshi.hpcc.uh.edu`, at commit `4e61b23`. Full output:
[`results/summary.md`](results/summary.md), [`results/summary.json`](results/summary.json),
[`results/log.txt`](results/log.txt).

**All 5 scenarios gave the expected verdict in this run.** The three faults
were detected, attributed to the right component and given the right type; the
healthy control raised no alarm; the Sun-shift control raised no fault, only an
illumination notice. Over 20 further random draws of pose error and noise the
faults were always right, but the healthy control raised a false antenna alarm
in 6 of 20 draws (94 of 100 verdicts correct); see [Audit](#audit-repeat-draws-parameter-sweep-and-diagnostics).

| Scenario | Verdict | Flagged Gaussians on the changed part | Pixel recall | Pixel IoU |
|---|---|---|---|---|
| Healthy control | No change | none above alarm level | | |
| Panel hinge | Geometric: port solar array | 8114 / 8977 (90%) | 0.74 | 0.43 |
| Antenna missing | Geometric: antenna | 1956 / 1956 (100%) | 0.95 | 0.54 |
| Blanket discoloration | Appearance: bus | 2182 / 19943 (11%; the patch is about 5% of the bus surface, alarm level 2%) | 1.00 | 0.67 |
| Sun shift (no fault) | Illumination notice: bus, port array, antenna (all appearance) | (no fault reported) | | |

**Baseline.** 43,929 Gaussians, held-out PSNR 39.3 dB, trained in 15 minutes.

**Speed.** Once the baseline exists, detection over 8 cameras (render, compare,
vote, report) takes **0.52 s** on the P40; with classification a complete pass
takes at most 1.3 s.

**How to read the figure.** The third column shows pixel-level changes
*before* voting; the fourth shows the Gaussians that survive voting on a 3D
overview. The same 253 of 1,956 antenna Gaussians are flagged in every
scenario, including the healthy control. They stay flagged with pose error and
noise switched off, so they come from a reconstruction error of the baseline on
the dish, which one inspection camera sees. In this run calibration raised the
antenna's alarm level to 22%, which absorbs them; in other draws it does not
(see the audit below).

**The illumination control.** Moving the Sun 30° moves every shadow and glint,
and 45% of the bus, 8% of the port array and 75% of the antenna are flagged.
Most flagged Gaussians still show one consistent new colour across cameras
(at most 14% inconsistent), so all three are classified as appearance changes, and because three parts change at once
the pass is reported as a likely illumination change, not a fault. This is
acceptable here, but it means a real appearance fault (such as the blanket
discoloration) occurring together with a change in Sun angle would be hidden in
that notice. A Sun-aware baseline, or comparison only against captures at
matching Sun geometry, remains the key next step.

**A bug found and fixed.** An earlier version of the rasterizer kept at most
768 Gaussians per 16 x 16 pixel tile. Crowded tiles hold up to about 9,800,
so renders (training and detection) were silently truncated wherever a tile
was crowded. With that bug the baseline reached only 32.5 dB, the calibrated
threshold was 0.077 instead of 0.040,
and the Sun-shift control was misclassified as a geometric fault (4/5 correct).
The bug was found while preparing paper figures and fixed in commit `4e61b23`;
the renderer now matches a brute-force reference to within 4e-7. The detection
rules were not changed after the fix. The git history keeps the earlier results.

## Audit: repeat draws, parameter sweep and diagnostics

`tools/audit_checks.py` and `tools/audit_followup.py` re-examine the claims
above from the released dataset and baseline (about 17 minutes on a P40);
outputs are in [`audit/`](audit). Main findings:

- **Calibration is mostly the hand-set floors.** The raw 99.5th percentile of
  the difference map on the held-out views is 0.034 (90% of their pixels are
  black space), so tau = 0.040 is the floor. Four of five component levels are
  the 0.02 offset because the held-out views flag (almost) no Gaussians on them;
  only the antenna level (0.217) comes from the data.
- **Repeat draws (seeds 1 to 20, same baseline): 94 of 100 verdicts correct.**
  Hinge, antenna and blanket were correct in all 20 draws; the Sun-shift pass
  was an illumination notice in all 20. The healthy control raised a false
  antenna alarm in 6 draws: in those draws the held-out views flagged no antenna
  Gaussian, so the antenna level fell to 0.02, while the inspection cameras
  still saw the dish error (172 to 254 antenna Gaussians flagged).
- **Calibrating on the inspection cameras fixes it (post hoc).** With component
  levels set from a separate healthy pass of the eight inspection cameras, all
  100 verdicts were correct. This was designed after seeing the failures.
- **Parameter sweep (135 settings of tau, vote score, offset and factor):** 87
  give 5/5 and 48 give 4/5. The three faults are right in every setting; the
  errors are a false geometric Sun-shift verdict from glints on the starboard
  array (offset 0.01), the antenna false alarm (tau 0.05), and Sun-shift passes
  where only two parts are reported so no illumination note is given.
- **Thin margins in the classifier.** The hinge is typed geometric by the
  new-matter test alone (0.13 to 0.15 against a level of 0.10); a level of 0.15
  types it as appearance. A black level of 0.05 types the shadowed bus in the
  Sun-shift pass as vanished.
- **Pixel metrics.** Hinge recall is limited by contrast: 18% of its truly
  changed pixels differ from the healthy image by less than tau (dark panel
  against black space). 68% of the hinge's false-positive pixels are on the
  rotated panel where its shading really changed but the ground truth (component
  and material changes only) does not mark it.

## Is the 3D model needed? Camera-offset comparison

`tools/ablation_reference.py` compares the 3DGS detector with two detectors that use no 3D model: a stored
healthy reference image per camera, and the same image warped to the new pose with a plane-induced homography.
All share the same difference map, threshold rule and alarm rule, and are calibrated on five healthy passes.
The eight inspection cameras are moved 0 to 45 degrees along their rings away from the poses of the reference
images; 20 random draws per offset (results in `audit/ablation_offsets.json`, figure from `tools/offset_figure.py`).

| Offset | 3DGS: faults /60, healthy /20, Sun /20 | Reference image | Warped reference image |
|---|---|---|---|
| 0° | 60, 20, 20 | 60, 20, 20 | 60, 20, 20 |
| 1° | 60, 20, 20 | 60, 20, 20 | 60, 19, 20 |
| 2° | 60, 20, 20 | 60, 20, 9 | 60, 20, 15 |
| 3° | 60, 20, 19 | 45, 20, 7 | 59, 20, 8 |
| 5° | 60, 20, 16 | 20, 20, 20 | 37, 20, 3 |
| 10° | 60, 14, 7 | 20, 20, 20 | 20, 20, 20 |
| 20° | 60, 20, 1 | 20, 20, 20 | 20, 20, 20 |
| 45° | 40, 20, 0 | 20, 20, 20 | 18, 20, 20 |

- **Fixed cameras: the 3D model adds nothing.** All three give 100 of 100 verdicts and the reference image has
  cleaner masks.
- **Moving cameras: only 3DGS still finds the faults** from 5 degrees. The image-based detectors' thresholds rise
  until they report only the missing antenna (their control scores at large offsets are empty: they report nothing).
- **The cost of the 3D model is false alarms on the controls.** The antenna-dish error is flagged intermittently
  (`tools/offset_diag_healthy.py`), so even five calibration passes miss it at 10 degrees (6 of 20 false alarms).
  At 20 and 45 degrees one camera nears a specular reflection the baseline cannot reproduce
  (`tools/offset_diag_render.py`), and the Sun-shift control fails almost always.

`audit/ablation_offsets_singlepass_v1.json` is an earlier run of the same experiment with single-pass calibration
and without the warped detector.

## Dataset and notebook

- **[`data/larissa_synth_v1`](data/larissa_synth_v1)**: Larissa-Synth v1, every
  image, depth map, component label, camera pose and ground-truth change mask
  used in the results above, plus the trained baseline (about 38 MB). See its
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

Expect about 16 minutes on a Tesla P40, almost all of it baseline training. To
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
| `tools/paper_figures.py` | Figures 2 and 3 of the paper, from the released dataset |
| `tools/audit_checks.py` | Calibration diagnostics, exact pixel metrics, antenna and shadow checks, parameter sweep, repeat draws |
| `tools/audit_followup.py` | Failure modes of the sweep; calibration on a healthy pass of the inspection cameras |
| `larissa/report.py` | Figures and the markdown summary |
| `run_demo.py` | End-to-end script |

The rasterizer is written with ordinary PyTorch operations, without custom
CUDA kernels. That makes it portable (it runs on the Tesla P40 cards in
`joshi.hpcc.uh.edu` without compiling custom extensions) but slower than the
reference implementation. Swapping in a
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
  Sun-shift control is handled only by a heuristic (three or more parts with
  appearance changes at once is reported as illumination), which would mask a
  real appearance fault under a new Sun angle. A Sun-aware relightable baseline
  is the planned fix.
- **Simple detector.** Render-and-compare with voting, not a reimplementation of
  published methods such as MV3DCD or GS-Diff.
- **Rules tuned on the scenarios they are tested on.** The detector went
  through several iterations against these same five scenarios (git history
  shows each change). The threshold floor, the component alarm offset (which
  set four of the five levels), the voting and the classification thresholds
  were chosen by hand while looking at these results. Independent fault
  scenarios are needed before any accuracy claim.
- **Calibration views.** The held-out calibration views do not see the
  baseline's dish error that one inspection camera sees, which causes false
  antenna alarms in some draws. Calibrating on healthy passes of the inspection
  cameras themselves is the planned fix.
