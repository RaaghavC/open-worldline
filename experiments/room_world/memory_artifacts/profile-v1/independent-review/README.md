# Independent profile artifact audit

The CPU audit passed for both completed 50-update arms. It verifies the exact source snapshots, shared initialization and scene schedule, all saved checkpoint hashes, finite optimizer and memory tensors, optimizer step counts, and unchanged predictor tensors against the published base checkpoint. It also checks the recorded process limits and summarizes timing. It loads weights with `weights_only=True` and does not run a neural forward pass, training, rendering or GPU operation.

Carry took 72.99 s and reset 73.99 s within their training intervals. These are one-seed runtime measurements. No memory-quality or control-retention evaluation was performed. Memory measurements were sampled and can miss short peaks; RSS and Metal allocation are overlapping counts.

[The report](report.json) was generated against the published profile files. The [original audit source](original-audit.py.txt) and [original report](original-report.json) are exact copies from the first audit of the local run. The executable [audit_profile.py](audit_profile.py) adds explicit path arguments and refuses an existing report output. Its validation calculations are unchanged. The two reports differ only in the recorded audit-source hash.

From the repository root, using the Python environment with PyTorch and NumPy:

```sh
python experiments/room_world/memory_artifacts/profile-v1/independent-review/audit_profile.py \
  --profile experiments/room_world/memory_artifacts/profile-v1 \
  --repo . \
  --output /tmp/worldline-memory-profile-audit.json
```

The output path must not already exist. The audit intentionally requires current model/trainer sources to match the measured snapshots. The profile, executable audit and matching sources are available together at repository revision `00c03bf5a1d33b08eb48894259c1c32840d5b70b`; use that revision for this command after later trainer changes. The original measured files remain unchanged. [Checksums](checksums.json) record the current package files. This README gained the explicit repository revision after the audit; its executed source and reports are unchanged. Code is Apache-2.0.
