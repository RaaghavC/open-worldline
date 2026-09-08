# Original action adapter through the native CUDA forward

The existing **947,712-parameter FP32 Worldline action adapter** completed a [two-update A100 numerical probe](results/a100-v1/README.md) on the literal, externally pretrained Wan2.2 TI2V-5B CUDA model. Native versus zero-adapter predictions were bit-exact on both tested inputs, the second update reached the GRU, and all 825 foundation values remained unchanged. **That probe did not execute fixed16 or evaluate rendered action quality.**

The later [sixteen-update checkpoint](results/fixed16-a100-v1/README.md) has now undergone a [matched visual evaluation](results/visual-a100-v1/README.md). The room remains recognizable, but both the door interaction and camera turn fail. This numerical training path is working; useful action control is not yet demonstrated. The complete clips and captured-frame comparison are retained.

The separate high-resolution native clip passed its severe-distortion visual check: it showed a recognizable room with natural colors and little motion. That is an external foundation diagnostic. It did not execute this adapter or test action control. The earlier MPS visual experiment remains failed; its [two-update numerical probe](../action_training/results/probe-v2/README.md) is separate evidence for the previous MPS wrapper.

## Boundary and inputs

`bridge.py` calls the original `WanModel.forward` under `inference_mode(False)`, `no_grad()` and native CUDA BF16 autocast. A temporary prehook on the final head captures its actual `x` and `e` arguments, then raises one private sentinel before the head executes. The hook is removed in `finally`; the caught sentinel's traceback is cleared so it cannot retain native intermediate tensors until cyclic garbage collection. No native forward, attention, rotary embedding, time embedding, head or unpatchify method is replaced.

The existing adapter runs on the captured FP32 hidden values. Its own code disables autocast. The original native head then runs with its original inner FP32 context, followed by the original unpatchify method. These operations retain gradients to the adapter. All 30 transformer blocks and all foundation parameters remain frozen. The head's weights being frozen does not remove its activation gradient path.

| Profile | Noisy latent | Tokens | Observed tokens | Grid passed to native unpatchify |
| --- | --- | ---: | ---: | --- |
| `baseline` | `[1,48,5,18,32]` | 720 | 144 | `[5,9,16]` |
| `spatial` | `[1,48,5,44,78]` | 4,290 | 858 | `[5,22,39]` |

Both profiles represent 17 RGB frames. Inputs are already on the core device: FP32 latents, int64 `[1,tokens]` times, one FP32 `[L,4096]` text context with `1 <= L <= 512`, FP32 `[1,16,6]` commands, and an independently encoded FP32 `[1,48,1,H,W]` observation. All observed times are zero; future times share one integer in `[0,999]`. The observation must exactly equal the input's initial latent. The bridge rejects a mismatch and does not repair or replace it.

Command units and order are unchanged: local right/up/forward meters, yaw-left/pitch-up radians and an interaction pulse. The existing GRU starts at zero on every call and encodes requested command prefixes within the clip. It is not persistent visual memory. The adapter masks its residual from observed tokens. The sampler, which is not implemented here, must still preserve the clean initial latent after each solver step. The bridge does not force predicted initial velocity to zero.

## Explicit API

Load and verify the original CUDA model through the existing `cuda_reference.native.load_model`, **outside `torch.inference_mode()`**. Keep its original FP32 storage and normal frozen parameters. A production bridge rejects CPU, MPS, meta, selective-BF16 core storage, instance method replacements, pre-existing forward hooks and anything other than the pinned literal model with FlashAttention 2 only. It performs no model load or device fallback. The caller must separately bind the loader's original 825 value hashes and complete source provenance.

```python
from experiments.wan22_native.action_adapter.model import PostBlockActionAdapter
from experiments.wan22_native.action_cuda.bridge import NativeCUDAActionBridge

# core is the already verified literal native FP32 CUDA model.
adapter = PostBlockActionAdapter().to("cuda:0")
bridge = NativeCUDAActionBridge(core, adapter, profile="spatial")

features = bridge.extract_features(noisy, times, contexts)
velocity = bridge.predict_from_features(features, commands, observation)
loss = future_flow_loss(velocity)  # supplied by a separately reviewed trainer
loss.backward()

# Equivalent direct call. Invalid conditions fail before frozen extraction.
velocity = bridge(noisy, times, contexts,
                  commands=commands, observation=observation)
```

`predict_from_features` defaults to `track_grad=True`, which explicitly enables its gradient path even inside an outer `no_grad` context. Use `track_grad=False` for an explicit inference call. Results stay on the core device; the tracked path does not detach or copy them to CPU. The feature bundle is an ephemeral constant owned by one bridge and is not a supported cache or checkpoint format.

Give the bridge exclusive use of the core. Concurrent wrappers, external calls to the same core, compilation and third-party hooks are unsupported. The bridge rejects existing hooks without removing them. Its own lock rejects recursive or concurrent bridge calls; it does not claim to control callers that bypass the bridge.

## CPU evidence and remaining gates

[CPU report](cpu-results/v1/report.json), [log](cpu-results/v1/pytest.txt) and exact source snapshots retain **30 passing checks** using small random CPU stand-ins. They cover both spatial grids, captured hidden/time identity, exact zero-adapter equality against the stand-in's complete forward, head and unpatchify gradient flow, second-update GRU/input gradients, frozen parameter values, observed-token masking, invalid-input rejection, explicit gradient mode, hook cleanup on success and failures, foreign-sentinel propagation and prompt release of native-frame temporaries without cyclic collection.

These fixtures use no foundation weights, CUDA or FlashAttention. They do not validate the numerical behavior of the complete native CUDA transformer. The measured command used Python 3.11.9 and Torch 2.5.1 from the existing Wan environment with the already installed pytest location appended. Reproduce the bounded suite from the repository root with:

```sh
python -m pytest experiments/wan22_native/action_cuda/test_cpu.py -q -p no:cacheprovider
```

The separate [independent review](cpu-results/independent-v1/report.json) passed five CPU cases using the literal upstream forward, head and unpatchify with zero attention blocks and small random weights. It checks both token grids, exact zero-adapter identity, activated paired gradients against a direct native-head path and cleanup. Its original report and ten checked source snapshots are copied byte for byte. The published warning log replaces one private local workspace prefix with `<LOCAL_WORKSPACE>`; [publication hashes](cpu-results/independent-v1/publication.json) retain the raw and published log hashes and the replacement record. The initial independent launch could not import pytest and executed no tests; the successful invocation used the already installed pytest path without installing packages. This remains a CPU mechanics check, not a full 30-block CUDA measurement.

Before a real numerical probe, retain a separate source-bound review and admission, exact data/text/noise identities, original weight verification, native-versus-zero-adapter comparison, finite gradients on both updates, a nonzero recurrent gradient after the zero-output first update, all foundation values unchanged and measured hardware limits. The existing low-resolution MPS cache is not a high-resolution CUDA training cache. Numerical success would not establish image quality, action control, generalization or a reason to run a longer pilot automatically.

The separate [numerical probe guide](PROBE.md) documents the implemented plan-first runner, its 20 passing CPU checks, required new CUDA cache and exact-plan admission. No real CUDA adapter run has been admitted or executed by this package's preparation work.

## Attribution

This bridge and its fixtures are original Apache-2.0 Worldline code under the repository [license](../../../LICENSE). They reuse the unchanged original [action adapter](../action_adapter/README.md). Wan2.2 architecture, model parameters and native source remain external Apache-2.0 components, attributed in [NOTICE](../NOTICE), the [native provenance](../cuda_reference/upstream-provenance.json), and the retained [Apache license](../LICENSE-APACHE-2.0.txt). Original trained adapter checkpoints are included in the result bundles; external foundation weights are not redistributed. Original Atrium RGB and metadata have their separate [CC0 license](../../atrium_data/DATA-LICENSE).


The subsequent [fresh sixteen-update A100 result](results/fixed16-a100-v1/README.md) completed and passed its independent retained-file audit. Its original adapter checkpoint and full recovery archive are public. This measures training mechanics; it contains no generated clip from that checkpoint.
