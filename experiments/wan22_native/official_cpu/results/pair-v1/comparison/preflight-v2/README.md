# Comparator preflight v2

The comparator requires a completed official CPU reference run directory. It verifies the parent, worker and terminal records, exactly two predictions, pinned inputs, saved output hashes, and both retained and current official source files. It imports no model code.

The independent checks passed against this exact comparator. They use analytical arrays, the existing pinned portable CPU/MPS outputs, and explicitly synthetic run metadata. They do not execute or certify an official reference model run.

Both existing portable baselines exactly satisfy the recorded eager FP32 guidance equation. The official reference must also satisfy that equation exactly. The report retains descriptive residuals for all three sources without changing any output tensor.

Run from the workspace root after the actual reference run completes:

```sh
work/wan-adapter-env/bin/python work/compare-wan22-official-reference.py \
  --repo outputs/open-worldline \
  --reference-run work/ACTUAL_COMPLETED_REFERENCE_RUN \
  --output work/NEW_COMPARISON_REPORT.json
```

Use a fresh output filename. The previous comparator and independent report are preserved in `../wan22-reference-comparator-preflight-v1/`.
