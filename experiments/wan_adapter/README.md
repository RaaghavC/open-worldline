# Frozen Wan action-adapter runtime experiment

This optional experiment tests whether one optimizer update of a small original adapter can run through the frozen, externally pretrained Wan2.1 1.3B video model on a 24 GB Mac. It uses synthetic video latents and synthetic text embeddings. It does not establish generated-video quality, prompt following, controllability on real scenes, persistent memory, causal streaming, or Genie 3 parity. It is separate from the Worldline app.

The later [real Atrium training and sampling study](../../docs/wan-atrium-pilot.md) completed ten updates with genuine VAE and text caches. Its generated clips failed the visual and control checks. [Training instructions](TRAINING.md) use the combined `requirements-real.txt` dependencies. A [same-latent decoder check](codec/results/generated-base-fp32/README.md) retained the artifacts in full precision, so decoder precision does not explain the failure. The declared adaptation differs from native Wan sampling and does not establish native Wan's quality.

The original `adapter.py` conditions three Wan transformer blocks on ordered actions and 32 tokens pooled from the first observed latent frame. The output projections start at zero. Actions are concatenated in their original order inside each group of four video-frame intervals. This implementation has no pose input, no future-observation input, and no persistent store. Action and observation adapters are established ideas; this implementation alone is not a research novelty claim.

The frozen foundation belongs to the Alibaba Wan authors. Selected Apache-2.0 code is extracted from [minWM revision 75322cc](https://github.com/shengshu-ai/minWM/tree/75322cc41e1d8386b32919a54a058d7841acbf90). [The source manifest](vendor/source-manifest.json) records original file hashes. [The extraction diff](vendor/extraction.patch) shows changed imports and removal of the unused causal block. [compat.py](compat.py) implements real-valued positional rotations, CPU timestep embeddings and dtype-preserving, padding-aware bidirectional SDPA. This experiment does not include Hunyuan code, pretrained minWM checkpoints, or minWM's Hunyuan-generated example data.

The official [Wan checkpoint release](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/37ec512624d61f7aa208f7ea8140a131f93afc9a) supplies Apache-2.0 weights. The native generator has 1,418,996,800 parameters and is stored in float32. Its original file is 5,676,070,424 bytes. The profiler verifies its SHA256 and converts tensors locally to the requested compute dtype without replacing the source file. The VAE is an optional additional 507,609,880 bytes. The 11.36 GB text encoder is not downloaded or loaded here.

## Reproduce the CPU checks

Run these commands from this directory. Use a Python 3.11 environment. The full dependency lock records the versions used for the checked run.

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python test_cpu.py --output results/cpu-tests.json
```

The checks use a random tiny Wan core and test numerical agreement with upstream complex128 CPU positional math, padding masks, full-core outputs and input gradients, exact zero-initialization, nonzero finite adapter gradients, agreement with activation checkpointing, preservation of action order after one update, and unchanged base-weight hashes. They do not use pretrained weights or a GPU.

## Explicitly download the external weights

```sh
.venv/bin/python fetch_weights.py --output external-weights
```

Add `--include-vae` only if preparing a later video-encoding experiment. The runtime test does not need the VAE. The downloader uses the fixed upstream revision and verifies file sizes and hashes. Checkpoints remain external and are not bundled with this repository.

## Run one bounded MPS update

Close other GPU workloads before profiling so the timing has a clear meaning.

```sh
.venv/bin/python runtime_probe.py \
  --weights external-weights \
  --output results/mps-runtime \
  --device mps \
  --dtype float16 \
  --synthetic-runtime-only \
  --steps 1 \
  --max-seconds 600 \
  --max-memory-gib 18
```

The requested shape is `[1,16,5,36,64]` video latents, corresponding to 17 decoded frames at 512 x 288. Wan patches these into 2,880 tokens. The synthetic context has 512 tokens of 4,096 features. The adapter remains float32; the frozen core uses the requested dtype. Per-block non-reentrant checkpointing preserves the adapter gradient through the frozen core. The entire core is not wrapped in `no_grad`.

The profiler records model-load time, forward time, backward time, optimizer and gradient-check time, sampled process RSS and Metal allocations, source and converted base hashes, and adapter gradients. A watchdog stops at the time or memory limits, or if available system memory drops below 2 GiB. RSS and Metal allocations overlap and must not be added. Sampled peaks can miss short-lived allocations. A stopped or failed attempt is evidence of that attempt, not a completed training step.

Any emitted `runtime-only-adapter.safetensors` was optimized against random targets. It is not a useful trained world-model checkpoint and should not be used for quality demonstrations. A useful adaptation needs original video clips, genuine cached VAE latents and text embeddings, held-out scene splits, and controlled action and observation ablations.

## Measured result on September 7, 2026

One complete synthetic update passed on an M4 Pro with 24 GB unified memory, macOS 26.3.1, PyTorch 2.5.1. The frozen core used float16 and the 1,349,376-parameter adapter used float32. The [raw report](results/mps-runtime.json) and [memory samples](results/mps-memory.jsonl) are included.

| Measurement | First completed update |
|---|---:|
| Load, conversion and initial weight verification | 7.499 s |
| Forward | 8.458 s |
| Backward | 10.536 s |
| Gradient validation and optimizer | 0.422 s |
| Complete update | 19.416 s |
| Whole process, including final hashing and saving | 31.097 s |
| Sampled peak Metal driver allocation | 5.53 GiB |
| Sampled peak active Metal tensors | 3.81 GiB |
| Sampled peak process RSS | 3.84 GiB |

Automatic MPS-to-CPU operator fallback was disabled. All 42 adapter-gradient tensors were finite, and their combined norm was nonzero. The initial zero output projections mean some internal adapter gradients can be zero on the first update. The converted core hashes matched before and after training. No warm repeated-update timing was measured.

The [first attempt](results/initial-import-failure.json) stopped before Metal tensor allocation: the program was named `profile.py`, which shadowed Python's standard profiling module during a PyTorch import. Renaming it to `runtime_probe.py` fixed the import. This failure is retained separately from the completed update.
