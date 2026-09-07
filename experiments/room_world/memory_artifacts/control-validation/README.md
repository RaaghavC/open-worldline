# Visibility audit for failed-interaction controls

Eight already captured development controls passed an independent CPU audit. At action index 6, after six backward moves, each camera is outside interaction range while the closed door remains visible. The reconstructed RGB image equals captured observation 6 byte for byte, and the interact command changes neither the teacher state nor the captured next image.

The [audit script](audit_visibility.py) renders two teacher copies with only the door color changed to black and white. The original renderer uses direct lighting, so changed pixels demonstrate visible door surfaces rather than reflected lighting. All eight cases contain between 781 and 965 such pixels out of 4,096.

The [full report](visibility-report.json) records capture/source hashes, exact reconstructed-image hashes, distances, facing values, visible-pixel counts and render/mask hashes. Neither these metadata nor the diagnostic renders are model inputs. Existing source/capture files were not changed, no neural model ran, and reserved test scenes were not accessed.

From the repository root, reproduce into a new report path:

```sh
python experiments/room_world/memory_artifacts/control-validation/audit_visibility.py \
  --output /path/to/new-visibility-report.json
```

This verifies the original programmed control data. It does not establish a trained model's ability to recognize failed interactions or remember door state. Audit source uses Apache-2.0; the captured data retain their documented CC0 dedication.
