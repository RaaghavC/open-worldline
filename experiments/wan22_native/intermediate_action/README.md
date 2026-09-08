# Experimental intermediate-block action bridge

This experimental implementation moves the unchanged `PostBlockActionAdapter` to the output of a selected native Wan transformer block. It preserves the published adapter, core, precision policy, weight loader and solver. The [actual A100 profile](results/a100-profile-v1/README.md) passed 16 exact initial comparisons and two training updates at each of two placements. It establishes execution correctness and resource use; visible action quality remains unresolved.

The default is `block_index=28`, after block 28 and before block 29 in the 30-block native core. Exactly one frozen transformer block then follows the adapter. `block_index=29` is the matched final-block control. Earlier indices are configurable but have not been admitted or profiled. A placement comparison must keep the same data, initialization, noise, losses, schedule and optimizer; new factorial data would be a separate experimental change.

```python
from experiments.wan22_native.intermediate_action.bridge import IntermediateActionBridge

# `core` is an externally verified original FP32 CUDA WanModel.
# `adapter` is the existing FP32 PostBlockActionAdapter.
bridge = IntermediateActionBridge(core, adapter, block_index=28, profile="spatial")
prediction = bridge(
    noisy, times, [context], commands=commands,
    observation=observation, track_grad=True,
)
loss = (prediction[:, :, 1:] - target_velocity[:, :, 1:]).square().mean()
loss.backward()
# An external trainer would optimize bridge.adapter.parameters() only.
```

The API retains the existing `baseline` and `spatial` B=1 profiles. Production shapes are respectively `[1,48,5,18,32]` and `[1,48,5,44,78]`, with 720 or 4,290 unpadded tokens. It requires matching integer token times, one native text context, FP32 `[1,16,6]` commands and the exact independently encoded first observation in the noisy prefix. All inputs must have `requires_grad=False`, no prior graph and normal tensor storage. The existing native checks still require FP32 frozen foundation parameters, original pinned model types, BF16 CUDA autocast and upstream FlashAttention 2 only. There is no CPU fallback.

The implementation makes one unchanged native forward call with one temporary selected-block output hook. Ordinary autograd remains enabled when training. Because all native parameters, buffers and incoming tensors are constants, it creates no graph before insertion. The hook explicitly checks that its input hidden tokens have neither `requires_grad` nor a `grad_fn`. The adapter output starts the graph. Subsequent frozen blocks, the original head and original unpatchify retain the operations needed to differentiate the loss with respect to the adapter. `track_grad=False` creates no graph. Outer `no_grad` and inference contexts are restored after the call.

The adapter still owns the initial-token and padding masks. This bridge accepts the declared unpadded profiles and passes the exact grid to the adapter. It never edits or clamps the noisy input. Later native self-attention can carry a future-token action residual into the observed-token output velocity, even though the adapter's direct observed-token residual is zero. That is not input mutation. A sampler must continue to apply its separately specified clean-prefix clamp; this prototype adds no sampler or velocity clamp.

The core must be used exclusively during each call. Existing hooks or instance forward overrides are rejected and remain untouched. The one new hook is removed in `finally`, including errors before the selected block, inside the adapter, in subsequent blocks, in the head or in unpatchify. There is no sentinel, retained feature cache or replay of command-dependent suffixes. The per-instance lock prevents recursive or concurrent calls to the same bridge; it cannot coordinate separate owners of the same core.

## CPU evidence

The original work-copy fixture passed 25 checks in 4.67 seconds. After publication-path integration, [the local log](cpu-results/publication-v1.txt) records 29 passing checks in 5.95 seconds: 25 bridge checks and 4 saved-response diagnostic checks. Python 3.11.9 and Torch 2.5.1 were used. CUDA remained uninitialized. The log retains expected warnings from vendor CUDA-autocast contexts being disabled on CPU.

The fixture uses a literal two-block native `WanModel` with dimensions 8 and two latent channels, a deliberately nonzero literal head, original time/RoPE/block/head/unpatchify code and small random parameters. Only the test fixture temporarily substitutes a written FP32 CPU attention equation for the vendor's CUDA-only FlashAttention function. The implementation never performs that substitution. These checks establish CPU placement and gradient mechanics, not numerical equality to the actual CUDA attention kernel or the 30-block pretrained model.

The tests verify:

- Zero-adapter full-output equality with the literal native forward at both possible fixture placements.
- Nonzero-adapter output and every adapter gradient against an explicit native suffix calculation, including paired accumulation.
- Exact final-block output and gradient agreement with the existing final-head bridge.
- No graph before insertion, a graph through the later frozen block, and zero first-update then positive second-update GRU gradients.
- Frozen core values and absent core gradients, unchanged input tensors, direct initial/padded-token masks and the separate output-prefix caveat.
- Cleanup at six exception locations, invalid indices, gradient-bearing inputs, changed observations, foreign hooks and invalid frozen parameter state.

With the repository dependencies and pytest installed, run from the repository root in a fresh process:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  experiments/wan22_native/intermediate_action/test_bridge.py
```

The original report was produced in the existing Wan Python environment with the already installed system pytest package appended to its module search path. No dependency installation, model download, provider operation or real training run occurred.

The separate [frozen-prefix cache](CACHE.md) provides the paired objective's `extract_features` and `predict_from_features` methods. Its 29 additional CPU checks passed. The subsequent CUDA profile measures this path with the actual pretrained core and unchanged native attention.

## Actual CUDA measurements and remaining limits

The completed bounded profile used identical saved tensors and unchanged precision for both placements. All full and cached initial outputs matched native outputs bit for bit; all 825 retained foundation value records remained unchanged. Recurrent adapter gradients became positive on the second update. The combined parent took 353.189 seconds. These checks retained the original thresholds.

Freezing the suffix weights does not remove its backward activation cost. For the spatial profile, one FP32 hidden tensor alone contains `1 * 4290 * 3072` values, or 52,715,520 bytes (50.27 MiB). The actual training allocation peaked at 25.012 GiB after block 28 and 20.093 GiB after block 29. Two updates took 4.089 and 3.873 seconds respectively, excluding load and validation. The retained command-response diagnostic remains poorly aligned with the target difference. Longer training and generated-video evaluation remain necessary to measure usable control.

This new wrapper is Apache 2.0 experiment code. The unchanged adapter and literal Wan dependencies retain their existing repository and upstream licenses and notices. The wrapper makes no novelty claim about intermediate action conditioning.
