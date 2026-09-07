# Independent check of the saved gradient diagnostic

The [independent report](report.json) verifies all 39 saved tensor hashes, exact next-frame targets from training scene 5023, the frozen predictor and memory checkpoint, the executed source snapshots, and the unchanged failed probe. It recomputes the prediction differences, loss-gradient signs, every parameter and RGB gradient comparison, and the complete gradient decomposition on CPU. It performs no model forward, optimizer update, rendering, or reserved-test access.

Exactly three residuals change between zero and ±5.96046448e-8. No residual changes between opposite nonzero signs. The shared output gradient is exactly the reference loss gradient retained by the diagnostic. With that same gradient supplied to both implementations, the largest RGB gradient difference is 2.04636e-12 and its relative L2 difference is 2.57168e-7. Both meet the original bounds. The full decomposition has an exactly zero residual when evaluated in double precision. The direct target derivative explains part of the change; the prediction derivative also carries changed loss gradients into earlier input frames.

The ordinary L1 RGB gradient check remains **failed** under its original bounds. This audit supports a separately recorded 50-update-per-arm batched timing and integrity profile. It does not establish identical optimization trajectories, MPS agreement in reset mode, generation quality, or learned memory.

Run the portable audit from the repository root after reconstructing the exact published tensor file:

```sh
gzip -dk experiments/room_world/memory_artifacts/sequence-gradient-diagnostic-v1/diagnostic-tensors.pt.gz
mkdir -p work
python experiments/room_world/memory_artifacts/sequence-gradient-diagnostic-v1/independent-review/audit_gradient_diagnostic.py \
  --tensors experiments/room_world/memory_artifacts/sequence-gradient-diagnostic-v1/diagnostic-tensors.pt \
  --output work/independent-gradient-audit.json
```

Use a Python environment with NumPy and PyTorch. The output JSON path must be new. The script accepts `--repo`, `--diagnostic`, `--failed-probe`, `--profile`, and `--train` overrides for a relocated artifact checkout. All model and measurement sources are verified against the retained `measured-source` files. The current trainer need not equal the executed trainer: current source hashes are recorded separately as information and its code is never imported or executed. The audit reads only the declared training scene, with no model access to validation metadata.

The [original audit source](original-audit.py.txt) and [original report](original-report.json) are exact copies of the first local review. The portable script changes path arguments only. The reproduced report has the same numerical checks; the audit script hash and informational current-source hashes can differ after later code edits. [Checksums](checksums.json) bind these files and the compressed tensor publication. The original compressed tensors and executed diagnostic files are retained unchanged.
