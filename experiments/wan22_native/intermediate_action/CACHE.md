# Frozen prefix cache for the intermediate action bridge

This implementation adds `extract_features` and `predict_from_features` to the published `IntermediateActionBridge`. It preserves the inherited full-forward implementation and the unchanged `PostBlockActionAdapter`. It supports the previous auxiliary objective's two feature extractions and four command/text prediction paths. The [subsequent CUDA profile](results/a100-profile-v1/README.md) measures its exact native parity, gradients and resource use with real weights. It contains no generated-video quality result.

The default is after block 28, before the final native block 29. The final-block control uses index 29. Both are zero-based. The existing baseline/spatial shapes, input validation, FP32 stored parameters, BF16 CUDA autocast, original complex RoPE and FA2-only production requirements remain unchanged.

```python
from experiments.wan22_native.intermediate_action.cached_intermediate import CachedIntermediateActionBridge

bridge = CachedIntermediateActionBridge(
    core, adapter, block_index=28, profile="spatial",
)
features = bridge.extract_features(noisy, times, [positive_context])
closed = bridge.predict_from_features(features, closed_commands, observation)
opened = bridge.predict_from_features(features, open_commands, observation)
# An auxiliary loss may combine these with a separate negative-context pair.
# Each prediction has its own adapter/suffix graph. No target is a cache input.
```

Run from the repository root. `bridge(noisy, times, contexts, commands=..., observation=...)` still uses the previously tested full-forward hook implementation. The new methods serve callers that explicitly reuse a prefix. They neither cache nor reuse command-dependent suffix outputs.

## What is retained

Extraction calls the literal native forward under `inference_mode(False)` and `no_grad`. One temporary output hook records the original `time_embedding` result. A second hook records the selected block's output and its exact six keyword arguments, then raises a private per-call sentinel. Both hooks are removed in `finally`, including a failure while installing the second hook. Only that exact sentinel object is caught, and its traceback is cleared to release the native forward's unused intermediates.

The returned `FrozenIntermediateFeatures` holds:

- Normal frozen FP32 hidden tokens and the raw FP32 time embedding used by the native head.
- The native block's projected FP32 `e` with shape `[1,L,6,dim]`, sequence lengths, grid, original rotary frequencies, projected text context and `context_lens=None`.
- A separate copy of the exact observed input prefix, plus owner/profile/block identities and tensor version records.

Projected text retains its native dtype: FP32 in the small CPU fixture and BF16 under production CUDA autocast. The implementation never recomputes time embeddings, context projection, modulation or native block equations. It accounts for native forward's existing first-call transfer of the rotary tensor to the model device, then binds the resulting live tensor.

Prediction validates the bundle, applies the unchanged adapter, calls each literal remaining block with the captured keyword arguments, then calls the original head and unpatchify. `track_grad=True` explicitly enables the new adapter/suffix graph; `False` makes an inference result. Commands and the independent observation must be non-gradient inputs, and the observation must equal the cached clean prefix exactly.

The bundle can be reused after adapter updates because the prefix did not depend on adapter parameters. Core tensor changes, replaced cache tensors and ordinary in-place cache mutations invalidate it. The bundle is owner-specific, read-only evidence for one live wrapper, not a serialized dataset or a replacement for full foundation value-hash checks. Separate wrappers still require external exclusive ownership of a shared core.

The unchanged adapter masks its direct initial/padded-token residual. This wrapper accepts the declared unpadded profiles. Later native attention can alter the predicted velocity at observed tokens; the clean input itself is never changed. An external sampler must retain its explicitly specified observed-prefix clamp.

## CPU checks

The cached path passed 29 preparation checks. The [published-copy integration log](cpu-results/native-integration-v3.txt) records 58 passing checks in a fresh CPU process: 25 full-forward bridge tests, 29 cache tests and four saved-response diagnostic tests. The [source identities](cpu-results/integration-v3.json) accompany the result. A separate fresh process passed all 12 data-reader and validator checks. Their isolated-process requirement is preserved.

The fixture uses two small literal native blocks with a scoped CPU attention equation. It tests exact zero-adapter equality, output and every adapter-gradient comparison across four command/text paths, reuse across three adapter updates, prefix-only execution counts, inference contexts, sentinel cleanup, partial hook-registration failure, foreign errors, suffix errors and malformed or mutated feature rejection. No CUDA initialization or pretrained weight load occurred. Independent review found no material capture, graph or cleanup defect.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  experiments/wan22_native/intermediate_action/test_cached.py
```

## Memory and execution limits

The cache makes prefix reuse possible. At the spatial profile, hidden plus raw time plus the six-way projected time tensor alone occupy 421,724,160 bytes, or 402.19 MiB, per feature bundle. This excludes projected context, the observed prefix, shared rotary storage, metadata and all suffix backward activations. Two text-context bundles therefore retain more than 804 MiB before the command-dependent graphs. The original final-head cache did not retain the six-way projected time tensor.

The completed bounded CUDA profile passed all 16 native/full/cached initial comparisons across both placements and both text/command branches. Four training updates retained finite gradients and unchanged foundation value records. After-block-28 training peaked at 25.012 GiB allocated and 25.432 GiB reserved. Frozen suffix weights continue to require backward activations. These are measurements from two updates per placement in one seen room, with no quality promotion or relaxed tolerance.

This prototype is Apache 2.0 code using the unchanged repository and upstream Wan components under their existing licenses. It makes no novelty or action-quality claim.
