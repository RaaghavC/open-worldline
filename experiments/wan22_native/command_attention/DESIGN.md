# Command-conditioned native attention controller

This work-only implementation replaces the earlier 947,712-parameter residual adapter. It adds trainable rank-32 factors to the native self-attention Q/K/V/O projections in zero-based Wan blocks 24–29. All original foundation parameters remain frozen FP32 tensors. No original module, method, weight loader, Q/K normalization, RoPE, FlashAttention call, head, sampler or codec is replaced.

For each selected projection, the added term is `B((A(x)) * g(command_history))`. Factors and command computation use FP32; the added term is converted to the original projection's output dtype before addition. Native BF16 autocast remains active for the original CUDA operations. The gate is `1 + tanh(linear(state))`; there is no additional alpha/rank multiplier. Every B matrix starts at zero. A uses PyTorch's default Linear initialization. Four destination-aligned six-channel transitions feed one future latent group's width-128 GRU step, with a zero initial state on each call. No state persists between clips.

The default has **4,936,448 trainable parameters**:

- 4,718,592 low-rank matrix parameters, across 24 projection sites.
- 19,712 command embedding parameters.
- 99,072 GRU parameters.
- 99,072 gate parameters.

Gates and direct residuals are exactly zero for the observed latent group and padded tokens. Later native attention can still change their output velocities. The sampler must restore the clean observed latent as before. Commands never replace the noisy input or supply realized camera matrices, door states or future targets. Main flow-matching inputs may still contain target-corrupted futures; that is the trainer's declared objective.

## Interface

Put this directory and the repository on `PYTHONPATH`. Supply an already loaded and externally verified original FP32 CUDA core, and a controller prepared once from saved FP32 bytes on the same device:

```python
from controller import CommandAttentionController
from bridge import NativeCommandAttentionBridge

# core: pinned literal original Wan2.2 5B, eval mode, all parameters frozen.
# controller: instantiate and restore the prepared controller checkpoint before use.
controller = CommandAttentionController().to(core.patch_embedding.weight.device)
bridge = NativeCommandAttentionBridge(core, controller, profile="spatial")

# noisy [1,48,5,44,78], times int64 [1,4290], one text [L,4096].
# commands [1,16,6], observation [1,48,1,44,78] == noisy[:,:,:1].
velocity = bridge(noisy, times, [text], commands=commands,
                  observation=observation, track_grad=True)

# Reuse the unchanged native preprocessing and blocks 0–23 for both commands.
features = bridge.extract_features(noisy, times, [text])
velocity_a = bridge.predict_from_features(features, commands_a, observation)
velocity_b = bridge.predict_from_features(features, commands_b, observation)
```

The cache captures literal native input tokens immediately before block 24, plus the original raw/projected time embeddings, text and block keyword arguments. The suffix executes original blocks 24–29, head and unpatchify. A fresh controller/suffix graph is built per prediction, so the same cache can be reused after controller optimizer updates. Cache tensors are non-gradient constants bound to one live bridge and core version. Do not write through `.data`, mutate shared storage or change the foundation. Version checks are not a replacement for original value hashes.

`track_grad=False` uses no backward graph. Both paths explicitly disable outer inference mode while preserving the caller's context afterward. Inputs must have been created as ordinary non-gradient tensors, not inference tensors. In particular, the first native reference forward must use `inference_mode(False)` plus `no_grad` so native Wan's lazy RoPE transfer produces a normal constant.

`baseline` accepts [1,48,5,18,32]. Optional `padding_tokens` is explicit, bounded to 128, with zero padding times. The default spatial execution uses no padding. Malformed shapes, dtypes, times, interaction labels, wrong observation prefixes, preexisting hooks, model overrides, trainable foundation weights, wrong cache ownership, ordinary cache/core mutations and reentrant calls reject. Every temporary hook is removed after success or failure, including partial registration failure. The caller must own the core exclusively across wrappers and threads.

## CPU verification

The 30 tests execute small literal Wan blocks, with a scoped FP32 CPU attention fixture. They cover native zero identity, all-projection independent matrix-oracle output and gradients, full/cache four-CFG-path output and all-parameter gradient equality, two AdamW updates, command history causality, observed/padded masks, malformed inputs, failure cleanup and cache mutation. Production CUDA/FA2/BF16 identity and backward memory are **unmeasured** by these tests.

With Torch, NumPy, Diffusers and pytest already installed:

```sh
PYTHONPATH="$REPO:$CONTROLLER" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  python "$CONTROLLER/run_cpu_review.py" --output "$FRESH_CPU_REPORT"
```

`cpu-v1/report.json` retains exact local/native/fixture source hashes, test names, Python/package versions and a CUDA-uninitialized statement. The native source's CUDA-autocast decorators issue warnings on a CPU-only host; the test fixture itself never initializes CUDA.

## Resource profile integration

The root-owned profiler is being prepared separately at `work/command-attention-profile-v1`. It can call the interface above against the pinned loaded CUDA core and saved conditions. Before fitting, compare native versus zero-controller full/cached velocities; then measure two paired optimizer updates and the worst-case four-CFG-branch endpoint backward from two shared cached prefixes. Keep all four auxiliary graphs live until their joint loss backward, accumulate the earlier main gradients, and perform one final clip/AdamW step. Preserve raw predictions, parameter gradients, before/after foundation hashes and measured resource samples in that explicit profiler. This module performs no model loading, provider operation, deadline admission or automatic profile run.

The current selected fitting proposal combines this architecture with seven predeclared auxiliary edges: three within-motion door contrasts and four camera-versus-stationary contrasts within each door condition. That package is not an architecture-only ablation. The bridge contains no loss or edge selection, so the trainer must bind the objective and saved draws explicitly.

## Attribution and limits

The bridge calls the byte-exact Apache-2.0 Wan implementation in `experiments/wan22_native/cuda_reference/vendor/`. Temporary-hook/cache contracts follow the project's prior intermediate bridge. Low-rank updates are based on [LoRA](https://arxiv.org/abs/2106.09685); distributed action conditioning is informed by [Matrix-Game2.0](https://arxiv.org/html/2508.13009v1). The conditional gate, rank and six-block placement here are proposed engineering choices. This implementation does not establish control quality, generalization, persistent memory, real-time generation or novelty.
