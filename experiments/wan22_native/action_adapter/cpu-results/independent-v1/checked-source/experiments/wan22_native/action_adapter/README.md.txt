# Post-block29 action adapter

This is original Worldline adapter code for the frozen, externally pretrained Wan2.2 TI2V-5B model. It has **947,712 FP32 trainable parameters**. Its checks use random tiny CPU models; no real native 5B adapter training has run. The separate native full-clip experiment produced severe visual artifacts, so this package does not establish a working visual foundation, action control, memory, or improved image quality.

The adapter runs after the last transformer block and before the unchanged frozen output head. All transformer blocks run without an autograd graph. The head remains in the gradient path to the adapter even though its weights are frozen. There are no forward hooks, replacements of core methods, persistent session state, multi-site adapters, feature-cache trainer or optimizer in this package.

## Inputs and computation

The production wrapper accepts exactly:

| Input | Shape and meaning |
| --- | --- |
| `noisy` | FP32 `[B,48,5,18,32]`, clean initial latent plus noisy future latents |
| `times` | int64 `[B,720]`, first 144 values zero and one shared future time per branch, between 0 and 1000 |
| `contexts` | List of B FP32 `[L,4096]` contexts, `1 <= L <= 512` |
| `commands` | FP32 `[B,16,6]`, ordered actual commands causing each RGB transition |
| `observation` | FP32 `[B,48,1,18,32]`, independently encoded initial RGB; must equal `noisy[:,:,:1]` |

The command channels retain their existing order and units: local right/up/forward meters, yaw-left and pitch-up radians, and a zero-or-one interaction pulse. The current Atrium data exercises only yaw and interaction. The caller must verify the genuine text cache, new native 48-channel data cache, complete provenance and action alignment. Shape validation inside this model cannot prove where a tensor came from.

The action MLP receives four commands concatenated in chronological order. A width-128 GRU then produces a causal prefix feature for each of the four future latent groups, starting from zero on every call. The initial group has an exact zero command feature and does not run a GRU step. The first observation is pooled into 32 tokens and projected to width 128. A single-head FP32 attention operation uses the native hidden token as query and those observation tokens as keys and values, with dropout zero. The attended value plus command-prefix feature passes through SiLU and a zero-initialized projection to width 3,072. The residual is masked to future valid tokens only.

The GRU stores requested command history within this clip. It does not know whether a toggle succeeded, receive a teacher door-state label, or remember visually observed changes across clips. A causal command encoder does not make the bidirectional native clip generator causal: later supplied commands can affect earlier generated positions through attention during later denoising calls.

## Explicit feature/head API

Importing the package does not load weights. An external guarded runner must load and verify the core outside `torch.inference_mode()`, keep its parameters frozen, move the small adapter to the core device and use genuine verified inputs:

```python
from experiments.wan22_native.action_adapter.model import PostBlockActionAdapter
from experiments.wan22_native.action_adapter.wrapper import NativeActionWrapper

adapter = PostBlockActionAdapter().to(core.patch_embedding.weight.device)
model = NativeActionWrapper(core, adapter)

# This is a finite, normal tensor bundle with no gradient graph through the core.
features = model.extract_features(noisy, times, contexts, seq_len=720)

# Call outside no_grad/inference_mode when training the adapter.
prediction = model.predict_from_features(features, commands, observation)

# Equivalent convenience call; it validates commands/observation before extraction.
prediction = model(noisy, times, contexts,
                   commands=commands, observation=observation, seq_len=720)
```

The output is FP32 `[B,48,5,18,32]` velocity. `FrozenFeatures` contains hidden features, head time embeddings, token-grid metadata and a copied observed prefix. It belongs to one wrapper instance and is not a supported serialization format. No clean target, future RGB, pose, depth or object-state argument exists. For a paired update, a runner can calculate the future loss divided by two and call backward for each B=1 branch sequentially, then perform one optimizer update. It must compute a fresh adapter graph for each branch and optimize `model.adapter.parameters()` only.

`test_only=True` permits explicit small CPU fixtures, not an alternative production model configuration. The production path requires the native 30-block core and exact 947,712-parameter adapter. All input and adapter computation casts remain explicit. The measured native core files are imported unchanged.

## CPU evidence and limits

The source-bound report and checked source snapshots are in `cpu-results/v1/`. The 24 implementation tests cover both FP32 and selective-BF16 tiny core paths, a nonzero frozen head, exact zero-adapter identity, chronological command encoding, causal command-prefix gradients, session isolation, observed/padding masks, invalid-input rejection before feature extraction, frozen core weights, differentiable head, second-backward GRU gradients, sequential paired-loss agreement and disabled external autocast.

The first run passed 23 tests and failed a byte-exact comparison between one-example and two-example GRU calls. The maximum difference was `2.9802322387695312e-08`. The corrected test keeps exact isolation when another session changes in the same batch and uses `atol=1e-7, rtol=1e-6` only for the different-batch-size comparison. The failed report and source are retained separately. Zero-adapter identity and same-shape causal isolation still require exact equality. No product arithmetic was changed to force batch-shape equality.

Using the existing pinned core dependencies plus pytest, from the repository root:

```sh
python -m pytest experiments/wan22_native/action_adapter/test_cpu.py -q
```

The local measured invocation used `work/wan-adapter-env` with Python 3.11.9, Torch 2.5.1, Diffusers 0.34.0, NumPy 1.26.4 and safetensors 0.5.3, appending the already installed pytest location after the environment's existing search paths. The report preserves the exact command and versions. Tests use one CPU thread and load no official parameter values.

These are software correctness checks. They do not measure real native 5B backward memory or speed. Full native training remains unexecuted, and the visual foundation gate remains failed. Any later guarded run must preserve native solver/text/prefix conventions, a verified canonical initial observation for both start-0 interventions, independently encoded 48-channel targets, exact source/data/model hashes and the existing memory limits.

## Attribution

The new adapter and wrapper are Apache-2.0 Worldline code. The wrapper repeats the measured native feature-extraction equations using the existing core modules. Wan source, parameters and architecture remain external and attributed in the parent [NOTICE](../NOTICE), [provenance](../provenance.json) and [Apache license](../LICENSE-APACHE-2.0.txt). The original renderer data is separately CC0. No model weights, trained adapter checkpoint, or external assets are bundled here.
