# Atrium capture validation

Measured on September 7, 2026. The dense capture passes independent checks for **132 native 512×288 RGB/EXR frame pairs**, with **66 observations and 65 adjacent transitions per door state**. These are rendered training data, not outputs from a learned world model. Machine-readable evidence is in [dense-pair-validation.json](results/dense-pair-validation.json).

The capture used Blender 5.1.2, Cycles on Metal, 32 samples and scene seed 51000. Its manifest reports 199.608 seconds for scene creation, rendering and saving. This timing is an offline data-generation measurement. It is not model inference speed.

## What passed

The validator checked every RGB and EXR file against its recorded SHA-256, the original scene hash, image dimensions, native multipart EXR channels, finite foreground depth, object IDs, normal lengths, camera matrices and action-before-frame alignment. `--require-dense` checks all observations are actually present. The earlier [12-frame preview](results/wide-pair-validation.json) has only one adjacent RGB transition per state and fails that dense requirement.

Camera intrinsics were checked against both the dense manifest's explicit camera model and Blender's native projection: 18 mm lens, 36 mm horizontal sensor, square pixels, `fx=fy=256`, `cx=256`, `cy=144`. Older preview manifests lack `camera_model`; the validator only labels their intrinsics physically verified after the independent native-camera check.

A separate background CPU process cast 576 geometric rays across frames 0, 1, 25, 41, 53 and 65 in both states. All **534 foreground hits** matched the native object IDs. Native EXR depth agrees with **axial camera-Z in meters**: median relative error `2.1901e-7`, maximum `1.2111e-5`. The Euclidean ray-distance hypothesis has median relative error `0.171119`. This geometric check covers 12 selected frames; file and channel checks cover all 132.

Foreground coverage varies from 71.45% to 100%. Sky/background depth uses the native `1e10` sentinel and must be masked. Cycles' filtered shading normals can be shorter than one at edges or material details. Their measured lengths are reported, with finite-value, length-bound and bulk-degeneration checks. Training code must not assume each native vector is an exact unit normal.

## The turned-away views reveal the door state indirectly

The door has no directly visible pixels in either state at frames **8 through 58**, inclusive. Nevertheless, the paired RGB views differ throughout that interval. Mean absolute RGB differences range from `0.00104658` to `0.01501851` on a 0–1 scale. Global illumination supplies an indirect cue, so this sequence does not isolate memory of an invisible state.

| Frame | Closed/open door pixels | Paired RGB MAE | Changed pixels |
|---|---:|---:|---:|
| 1, immediately after interaction | 22,082 / 521 | 0.08054896 | 81.14% |
| 25, facing away | 0 / 0 | 0.01501845 | 100% |
| 41, after waiting | 0 / 0 | 0.01501840 | 100% |
| 53, turning back | 0 / 0 | 0.00323704 | 55.11% |
| 65, returned | 22,082 / 521 | 0.08054902 | 81.14% |

The common initial state also has tiny numerical rendering differences: 18 pixels differ, with RGB MAE `1.5957e-7`. Exact byte equality is therefore not a reproducibility claim for this renderer run.

There is one fixed layout, one camera position, yaw rotations and a remote scripted door toggle. The toggle directly changes a hinge angle; it does not establish proximity-valid manipulation or learned physical dynamics. These two sequences cannot establish unseen-scene generalization, long-term learned memory, broad visual quality or Genie 3 parity. Both states must stay in the same dataset split.

## Reproduce the independent check

From the repository root, using Blender and a Python environment with NumPy, Pillow and OpenEXR:

```sh
blender --background --factory-startup --disable-autoexec --python-exit-code 1 \
  --python experiments/atrium_data/query_rays.py -- \
  /path/to/capture --frames 0 1 25 41 53 65 --output /path/to/rays.json

python -m experiments.atrium_data.validate /path/to/capture \
  --require-dense --raycast-json /path/to/rays.json \
  --output /path/to/validation.json
```

Output files must be new. Omit `--frames` to query all captured observations; omit `--require-dense` only when intentionally inspecting a sparse preview. The query loads the capture's saved original scene with script execution disabled and performs CPU ray casts without rendering. Query, renderer, scene, manifest and validator hashes are recorded in the reports. Public JSON copies replace the local directory with a portable capture label; measured values are unchanged.

The validator uses Apache-2.0; the Blender query uses GPL-3.0-or-later. Targeted validation tests: **20 passed**, using Python 3.11.9, NumPy 2.4.2, Pillow 12.1.1 and OpenEXR 3.4.4. Test fixtures check file/schema failures; the real capture and geometric measurements above provide separate evidence.
