# Independent batched-profile audit

The [CPU artifact audit](report.json) passed for both completed 50-update arms. It verifies the source snapshots, explicit batched-path labels, all final and recovery tensor hashes, optimizer steps and moment shapes, finite values, unchanged frozen predictor, and equal realized budgets. The starting memory file is byte-identical to the earlier stepwise profile. Both profiles use the same scene schedule, data, and optimizer settings.

| Arm | Training time | Supervised elapsed time | Linear estimate for 1,024 updates |
|---|---:|---:|---:|
| Carry | 15.9274 s | 17.7756 s | 326.19 s |
| Reset | 21.9479 s | 24.3413 s | 449.49 s |

The largest recorded Metal driver allocation is 2.0900 GiB. Memory values are sampled, and overlapping Metal and process memory values must not be added. The longer-run estimates exclude startup and evaluation. They support retaining the existing 600 s limit per arm, with incomplete runs excluded from matched comparisons.

This is one timing profile per arm. The faster observed execution does not establish identical optimizer trajectories or any improvement in generated images or memory. The original L1 input-gradient comparison remains failed under its original bounds. No quality evaluation or reserved-test access occurred in this profile or audit.

To repeat the read-only audit from the repository root, use a Python environment with PyTorch and a new output path:

```sh
mkdir -p work
python experiments/room_world/memory_artifacts/profile-batched-v1/independent-review/audit_profile.py \
  --profile experiments/room_world/memory_artifacts/profile-batched-v1 \
  --stepwise experiments/room_world/memory_artifacts/profile-v1 \
  --repo . \
  --output work/batched-profile-independent-audit.json
```

The audit reads retained source snapshots, checkpoints, metrics, and the published base weights. It does not import a trainer, run a model, or require current trainer files to match an older executed snapshot. The [original local report](original-report.json) and public-artifact [reproduced report](report.json) are byte-identical. [Checksums](checksums.json) bind the audit files. Existing measured profile files are unchanged.
