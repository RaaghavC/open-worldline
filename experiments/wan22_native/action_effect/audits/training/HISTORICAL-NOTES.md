# Independent action-effect training artifact audit

This work-only auditor is prepared for the new 128-update action-effect experiment. No actual effect-training recovery has been audited. The old fixed128 auditor remains unchanged; its exact source is retained in `reference/fixed128-audit.py`. The new auditor requires the final prepared-v3 plan and all 88 frozen producer dependency hashes in `pins.json`.

The result schema is `worldline-action-effect128-actual-independent-v1`. A passed report exposes `completed_updates=128`, `foundation_values_unchanged=true`, `final_checkpoint_only=true`, `main_predictions=256`, `auxiliary_updates=32`, `auxiliary_feature_extracts=64`, and `auxiliary_head_predictions=128`. Its `identity` binds parent, worker, training, terminal and plan file SHA256 values, final checkpoint manifest/adapter SHA256, and the executed source map. These fields support the separate final-checkpoint evaluators; a passed training audit does not admit image generation by itself.

The auditor checks every selected run file against the original verified recovery index, source/CPU/admission and input hashes, all 128 original main noise/timestep/RNG records, 256 main predictions, two native/zero-adapter comparisons, 825 original foundation value identities before and after training, 128 combined post-clip gradient bundles, all nine checkpoint/optimizer/RNG bundles, and sampled resource limits. It never regenerates Gaussian inputs across architectures. Main predictions may change after an auxiliary update; the original main inputs, equation and schedule must remain exact.

At updates 1,5,...125, it requires all four auxiliary head files and independently reconstructs separate FP32 operations `G=N+5*(P-N)`, then `-(Gopen-Gclosed)`. The target is the untouched FP32 open-minus-closed target encoding. Only the four future latent frames enter the auxiliary MSE. Both commands must share the saved pure-noise input, independent observation and exact zero/999 token times; only the initial interact component may differ. The scale is the declared ideal endpoint s=1, not a claim about the literal solver's first sigma. Other updates must have no auxiliary calls or weighted loss.

The retained main and auxiliary scalar losses are independently reduced in FP64. Recorded scalar consistency uses the existing diagnostic allowance of relative `2e-6` and absolute `1e-8`, fixed before a real result. Combined post-clip norm and clip-scaling consistency use the same numerical allowance, with the original one-unit clip. These checks detect inconsistent records and are unrelated to action-quality thresholds. The exact saved gradients also feed the prior descriptive FP64 AdamW moment/parameter equations between consecutive 16-step checkpoints; their roundoff residuals are reported without claiming GPU replay.

Only combined post-clip parameter gradients are retained. Separate main and auxiliary parameter gradients, native feature tensors and their Jacobians are not available. The audit therefore verifies checked source order, retained combined gradients and optimizer evidence, but cannot independently reconstruct the two backward contributions. Feature-call counts are source-bound runtime records, not retained hidden-state reconstruction. Sampled resource checks cover recorded instants. No unsupported instantaneous relation is imposed between counters read sequentially.

The safe NumPy tensor reader bounds file/header sizes before materialization, requires exact requested keys, validates dtype/shape/storage coverage, rejects nonfinite values and malformed metadata, and does not load model weights. Optimizer recovery uses the existing CPU `weights_only=True` loader with only the installed `TorchVersion` string subclass allowed for metadata. CUDA must remain uninitialized.

Use the existing local environment after root provides an original verified recovery and run path:

```sh
work/wan-adapter-env/bin/python work/wan22-action-effect-audit-prep-v1/audit.py \
  --repo outputs/open-worldline \
  --runner-source work/wan22-action-effect-prep-v1 \
  --recovery /absolute/original-verified-recovery \
  --run-root /absolute/original-verified-recovery/recovered/action-results/effect128-spatial-v1 \
  --probe-root /absolute/completed-original-probe \
  --cache-root /absolute/completed-original-cache \
  --diagnostic-root /absolute/completed-original14-diagnostic \
  --output work/wan22-action-effect-actual-audit-v1
```

The required prior inputs are the same completed probe, cache and original14 diagnostic used by the frozen producer. Output must be fresh and outside the repository and recovered evidence. Failures retain a report and checked source copies; an existing output is never overwritten. Public derivatives with redacted operational audit paths must be handled explicitly rather than substituted for the original hash-bound recovery.

The small CPU fixture exercises exact four-path arithmetic and signed derivatives, future-only loss, target isolation in the common input, full schedule/counter failures, corrupted scalar/gradient records, malformed/nonfinite/trailing tensor storage, and retained preflight failure/no-overwrite behavior. These are analytic fixtures, not a real model, training or backward replay.

Final readiness: all 10 tiny CPU fixtures passed in 0.589 seconds with exact before/after source hashes. `cpu-v1/report.json` and its source copies retain the checked bytes. An earlier review-wrapper SyntaxError stopped before any fixture ran; its exact wrapper source and failure record remain separate. The auditor itself and producer sources were unchanged by that correction. Actual training audit execution remains pending the original verified recovery.
