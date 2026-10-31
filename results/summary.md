# Project Larissa demo results

- Hardware: Tesla P40
- Resolution: 256 x 256 px; 96 baseline views; 8 inspection cameras
- Baseline: 43390 Gaussians, held-out PSNR 32.45 dB, trained in 2161.6 s
- Change threshold (calibrated on healthy held-out views): 0.077
- Simulated camera pose error: 0.15 deg rotation, 0.01 m translation; sensor noise on all images

| Scenario | Expected | Verdict | Correct | Pixel IoU | False-positive px rate | Check time (s) |
|---|---|---|---|---|---|---|
| healthy_control | NO CHANGE | NO CHANGE | yes |  | 0.0046 | 0.46 |
| panel_hinge | GEOMETRIC | GEOMETRIC CHANGE: solar_array_port | yes | 0.29 | 0.0114 | 0.46 |
| antenna_missing | GEOMETRIC | GEOMETRIC CHANGE: antenna | yes | 0.27 | 0.0067 | 0.46 |
| mli_discoloration | APPEARANCE | APPEARANCE CHANGE: bus | yes | 0.40 | 0.0048 | 0.46 |
| sun_shift_control | NO FAULT | GEOMETRIC CHANGE: solar_array_port, antenna | no |  | 0.0262 | 0.46 |

**panel_hinge** component classification:

| Component | Flagged Gaussians | Type | Vanished share | Inconsistent share | New-matter share | Note |
|---|---|---|---|---|---|---|
| solar_array_port | 1981/2191 | geometric | 0.60 | 0.01 | 0.04 | part has vanished: cameras now see empty space where a lit surface was |

**antenna_missing** component classification:

| Component | Flagged Gaussians | Type | Vanished share | Inconsistent share | New-matter share | Note |
|---|---|---|---|---|---|---|
| bus | 125/4277 | appearance | 0.00 | 0.16 | 0.04 | surface still in place with a consistent new colour |
| antenna | 420/462 | geometric | 0.44 | 0.04 | 0.03 | part has vanished: cameras now see empty space where a lit surface was |

**mli_discoloration** component classification:

| Component | Flagged Gaussians | Type | Vanished share | Inconsistent share | New-matter share | Note |
|---|---|---|---|---|---|---|
| bus | 305/4277 | appearance | 0.00 | 0.18 | 0.05 | surface still in place with a consistent new colour |

**sun_shift_control** component classification:

| Component | Flagged Gaussians | Type | Vanished share | Inconsistent share | New-matter share | Note |
|---|---|---|---|---|---|---|
| bus | 2055/4277 | appearance | 0.06 | 0.30 | 0.02 | surface still in place with a consistent new colour |
| solar_array_port | 691/2191 | geometric | 0.03 | 0.59 | 0.03 | part moved or deformed: cameras disagree on what is now at its location |
| antenna | 230/462 | geometric | 0.00 | 0.30 | 0.09 | part moved or deformed: cameras disagree on what is now at its location |
