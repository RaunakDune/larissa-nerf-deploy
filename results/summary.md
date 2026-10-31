# Project Larissa demo results

- Hardware: Tesla P40
- Resolution: 256 x 256 px; 96 baseline views; 8 inspection cameras
- Baseline: 43929 Gaussians, held-out PSNR 39.31 dB, trained in 914.0 s
- Change threshold (calibrated on healthy held-out views): 0.04
- Simulated camera pose error: 0.15 deg rotation, 0.01 m translation; sensor noise on all images

| Scenario | Expected | Verdict | Correct | Pixel IoU | False-positive px rate | Check time (s) |
|---|---|---|---|---|---|---|
| healthy_control | NO CHANGE | NO CHANGE | yes |  | 0.0006 | 0.52 |
| panel_hinge | GEOMETRIC | GEOMETRIC CHANGE: solar_array_port | yes | 0.42 | 0.0120 | 0.52 |
| antenna_missing | GEOMETRIC | GEOMETRIC CHANGE: antenna | yes | 0.54 | 0.0040 | 0.52 |
| mli_discoloration | APPEARANCE | APPEARANCE CHANGE: bus | yes | 0.67 | 0.0017 | 0.52 |
| sun_shift_control | NO FAULT | APPEARANCE CHANGE ON MANY PARTS (likely illumination): bus, solar_array_port, antenna | yes |  | 0.0317 | 0.52 |

**panel_hinge** component classification:

| Component | Flagged Gaussians | Type | Vanished share | Inconsistent share | New-matter share | Note |
|---|---|---|---|---|---|---|
| solar_array_port | 8114/8977 | geometric | 0.21 | 0.05 | 0.14 | part moved or deformed: cameras disagree on what is now at its location |

**antenna_missing** component classification:

| Component | Flagged Gaussians | Type | Vanished share | Inconsistent share | New-matter share | Note |
|---|---|---|---|---|---|---|
| bus | 927/19943 | appearance | 0.00 | 0.03 | 0.00 | surface still in place with a consistent new colour |
| antenna | 1956/1956 | geometric | 0.68 | 0.03 | 0.00 | part has vanished: cameras now see empty space where a lit surface was |

**mli_discoloration** component classification:

| Component | Flagged Gaussians | Type | Vanished share | Inconsistent share | New-matter share | Note |
|---|---|---|---|---|---|---|
| bus | 2182/19943 | appearance | 0.00 | 0.01 | 0.00 | surface still in place with a consistent new colour |

**sun_shift_control** component classification:

| Component | Flagged Gaussians | Type | Vanished share | Inconsistent share | New-matter share | Note |
|---|---|---|---|---|---|---|
| bus | 8990/19943 | appearance | 0.05 | 0.14 | 0.00 | surface still in place with a consistent new colour |
| solar_array_port | 717/8977 | appearance | 0.00 | 0.08 | 0.00 | surface still in place with a consistent new colour |
| antenna | 1458/1956 | appearance | 0.00 | 0.07 | 0.00 | surface still in place with a consistent new colour |
