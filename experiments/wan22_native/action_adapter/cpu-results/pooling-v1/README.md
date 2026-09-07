# Adaptive-pooling portability: CPU evidence

All 45 bounded CPU tests passed in 1.722 s: 15 new pooling checks, 24 original adapter checks and six previously independent adapter checks. No official weight values were loaded and no GPU was used. The report and measured sources are copied byte-for-byte from the local run. The previous v1 adapter evidence remains unchanged.

For the actual observation shape, `[1,48,18,32]` to `[1,48,4,8]`, the explicit-bin helper differs from the existing CPU kernel by at most 1.49e-7 in output values and has exactly equal input gradients. Cases with uneven width, both dimensions uneven, overlapping bins and output sizes larger than input also passed. Predeclared comparison limits are absolute tolerance 2e-7 and relative tolerance 3e-6. These limits compare equivalent FP32 reductions; they do not assert bit-exact output for every shape.

The original CPU model still calls PyTorch adaptive pooling. The new helper is selected only for MPS tensors. The report checks that the model edit consists only of the new import and pooling dispatch, and that native foundation source hashes are unchanged.

From the repository root, using the pinned Wan environment plus the already installed pytest package:

```sh
python -m pytest experiments/wan22_native/action_adapter/test_pooling.py \
  experiments/wan22_native/action_adapter/test_cpu.py \
  experiments/wan22_native/action_adapter/test_independent.py -q
```

The exact local report-building script is retained as `audit-source.py.txt`. Its original invocation was `work/wan-adapter-env/bin/python work/report-wan22-action-pooling-cpu.py` from the workspace root. That script appends existing base-site packages after the pinned environment to access pytest without replacing the pinned numerical libraries.

A separate actual MPS operation check is required before retrying the full model. This CPU record is not an optimizer-feasibility or image-quality result.
