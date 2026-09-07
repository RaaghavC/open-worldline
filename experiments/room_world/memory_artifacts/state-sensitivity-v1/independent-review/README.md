# Independent state sensitivity audit

All 48 cases and 384 saved tensors passed file, source, checkpoint, data, alignment and numerical checks. Independent NumPy scoring matched 3,104 reported values/checks, with a maximum numerical difference of 8.88e-16. The audit executes no model and loads no checkpoint tensor values. All 24 reset cases remained bit-exact under state swapping and zeroing.

Across the 24 carry cases, exchanging the two branch states changed the paired region by a mean RGB MAE of 0.00002816. Zeroing the complete state changed it by 0.02218. Neither intervention achieved both-branches-correct on any case. State sensitivity does not prove correct semantic memory. The fixed study's acceptance results remain unchanged.

`audit-original.py` and `report-original.json` are the first exact audit source and report. The separate `audit-portable.py` revision adds only a checkpoint-directory override. Its separate report repeats the same saved-value audit and passes. Both original records are preserved. The script hashes every final checkpoint file without loading its tensor values.

Run the portable revision with the original diagnostic metadata and tensors, existing validation captures, original scalar summary and a copy of the completed training study:

```sh
python audit-portable.py --run /path/to/original-diagnostic-run \
  --validation /path/to/validation-capture \
  --summary /path/to/original-state-summary.json \
  --checkpoint-root /path/to/training-512-v2 \
  --output /path/to/new-audit-report.json
```

The retained runtime plan contained absolute checkpoint paths. `--checkpoint-root` ignores those locations and resolves each seed/mode under the supplied study, retaining exact checkpoint-hash checks. If using separately path-relocated public metadata, reconstruct the original JSON bytes using its explicit relocation mapping first. This audit does not silently accept changed metadata hashes. The supplied report contains hashes of every measured artifact and all source snapshots.

Dependencies for the executed audit: Python 3.11.9, PyTorch 2.5.1 and NumPy 2.4.2. No GPU, renderer, training, inference rerun or reserved test access occurred.
