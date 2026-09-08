# Saved command-attention resource-profile audit

This auditor uses NumPy and the original frozen transfer. It does not import Torch, the producer, or the model, and makes no provider or network calls. It does not load foundation weight files.

For a completed profile it checks the exact launch protocol, the 16 recorded executable/configuration source hashes against the original transfer, all 858 original non-input source files, and the consumed initial controller, six windows, two text contexts, first two noise/RNG draws and fixed evaluation-noise input file. The original input manifest remains pinned to `4f37fa59791d7b0645529e4b97978f08d928baf8987e79a9aad6da1b42f3046a`; the transfer manifest is pinned to `4dfb2da33228ce58dd56646bb818fd93e6bf9455b685986d7bfc294e3e7cb932`.

Saved numerical checks cover two original native velocities, the 24 recorded native-versus-controller hash comparisons, four main-flow predictions, four auxiliary predictions, both complete post-clip gradient bundles, and initial/final controller files. Main loss uses the target-corrupted flow objective and excludes the observed latent. The auxiliary loss uses the future-only ideal `s=1` clean difference `-(G_b-G_a)` against `z_b-z_a`, with `G=N+5(P-N)`. Independent CPU loss reduction is compared with recorded FP32 GPU scalars using relative `1e-5` and absolute `1e-7` rounding tolerance. FP64 gradient summaries use relative `1e-10` and absolute `1e-12`. These are record-consistency checks, not output-equivalence or quality thresholds.

The producer discarded the 24 full/cached controller outputs after comparison. The auditor verifies their recorded hashes against the two retained native arrays; it cannot compare the discarded arrays again. Optimizer moments are not retained by this profile, and no backward or AdamW replay is claimed. All three saved maps of 825 current foundation-value hashes are compared with the pinned original weight records. Their underlying foundation tensors are not available in this recovery for another hash computation. Rotary preservation and absence of foundation gradients are producer assertions, supported by the frozen source and saved records rather than a new model inspection.

The auditor also checks worker and parent completion, complete worker output hashes, sampled host/CUDA limits, hardware/runtime metadata, and the optional external dispatch request/PID/terminal/source and one-hour lease bindings. When a verified recovery receipt is provided, the run's exact file map, original transfer identity and unchanged runtime input comparison must agree. Memory counters were read sequentially; `allocated <= reserved` is not an added gate. A sampled resource pass does not independently approve a later 512-update run or establish visible control.

Failed or missing producer output yields `status: failed`. Available metadata and tensor-file identities are still reported, including incomplete tensor files. The auditor preserves the source evidence and creates a fresh output directory. It writes no claim of successful completion for an interrupted profile. Errors are retained in the audit report; do not change a failed auditor's source in place to reinterpret an actual result.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 work/wan-adapter-env/bin/python \
  work/command-attention-profile-audit-v1/audit.py \
  --recovered-root work/command-attention-recovered-v1/recovered \
  --recovery-verified work/command-attention-recovered-v1/recovery-verified.json \
  --original-transfer work/command-attention-transfer-v1/local-extraction-v1 \
  --output work/command-attention-profile-actual-audit-v1
```

Use the actual new recovery/output paths when root supplies them. Omit `--recovery-verified` only for an explicitly identified partial local recovery; the report then makes no verified full-recovery claim. Exit status is zero only for a completed, checked profile. A failed scientific profile may still have its retained failure bytes checked.

The two small tests use exactly representable scalar velocities with independently known losses, zero/positive gradient groups and complete parity records. They also reject changed scalar and parity records and changed retained-file bytes. They do not execute a native model or claim coverage of unexecuted CUDA behavior.
