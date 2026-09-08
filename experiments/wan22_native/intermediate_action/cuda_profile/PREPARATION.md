# Intermediate placement CUDA numerical profile

The code and CPU preparation are complete. No real CUDA profile, foundation weight load, generated video or action-quality evaluation has been run by this preparation task. The 11 CPU tests passed in 3.606 seconds using a tiny literal native model and scoped CPU attention. They do not establish FlashAttention backward compatibility or actual GPU memory use.

This new profile compares injection after native block 28 with injection after block 29. Block 28 leaves one frozen attention block in the differentiable suffix. Block 29 is the matched final-block control. Both retain the original 947,712-parameter FP32 adapter, literal native FP32 foundation weights, BF16 CUDA autocast, FlashAttention 2, original head and unpatchify. The published full and cached bridges are imported unchanged.

## Fixed inputs and computation

`selection.json` binds eleven exact files from the public action-effect training release. The two start0 windows share the exact independent observation; their commands differ only at `[0,0,5]`. Targets retain their original bytes. Only `noise_0000` and `noise_0004` are materialized from the original sixteen-draw shard. They came from historical updates 1 and 5, with k506 and k265. Both profile updates repeat start0 with the corresponding saved draw. This is a new two-update placement profile, not a replay of the historical first two updates. No evaluation-only noise or newly generated noise is used.

Every placement starts from the same saved zero-output adapter and empty AdamW state. The saved CPU RNG is restored per placement; CUDA RNG is initialized with the same declared seed and retained in each checkpoint. Saved tensor hashes, rather than cross-platform Gaussian regeneration, establish input identity.

Before **any** optimizer is created, two literal native predictions at pure-noise time999 are retained, one per genuine text context. Each placement then produces full and cached predictions for both commands and both contexts. All sixteen comparisons must be bit-exact to the corresponding native output. The report also records the previous max-absolute/relative-L2 diagnostics, but those descriptive values do not relax this exact gate. Completed predictions remain available if a comparison or later call fails.

Each placement then runs two unchanged `effect.paired_update` calls with auxiliary enabled. Each call accumulates the two half-weight future FM losses and the original four-head lambda1 endpoint-contrast loss before one clip at L2=1 and one AdamW step. The main calls use the full bridge. Each auxiliary context gets one frozen-prefix extraction and two fresh adapter/suffix graphs. The first recurrent gradient must be zero and the second finite and positive; a rejected update does not advance the last-valid checkpoint. No model equation or optimizer implementation is copied or patched.

Counts are two native parity outputs, sixteen bridge parity outputs, eight main training outputs and sixteen auxiliary training outputs. Each placement retains checkpoints 0, 1 and 2, clipped gradients for both updates, raw predictions and original condition hashes. Gradient bundles contain the combined clipped main-plus-auxiliary gradients. They do not separately reconstruct the two backward contributions.

## CPU preparation and preflight

The existing dependency environment is Python 3.11 / Torch 2.5.1. From the workspace root:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=outputs/open-worldline \
work/wan-adapter-env/bin/python work/intermediate-action-cuda-profile-v1/run.py \
  --prepare \
  --input-root work/wan22-action-effect-training-public-download-v1/recovered \
  --cpu-report work/intermediate-action-cuda-profile-v1/cpu-v1/report.json \
  --output work/intermediate-action-cuda-profile-prepared-v1

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=outputs/open-worldline \
work/wan-adapter-env/bin/python work/intermediate-action-cuda-profile-v1/run.py \
  --preflight --prepared-directory work/intermediate-action-cuda-profile-prepared-v1
```

The shown prepared directory already exists and passed. Preparation refuses to overwrite it. Use a fresh name for another preparation. Running `run.py` without a mode prints a plan and initializes no CUDA. CPU tests can be run with pytest against `test_cpu.py`, with this directory and the repository on `PYTHONPATH`. The recorded invocation used the existing system pytest package appended after the pinned environment's packages; it installed nothing.

Frozen CPU report: `cpu-v1/report.json`, SHA256 `9b1682b3ef622d77f1f263c89b59d8bbb9a4f5b56416bda7a614e10002b51896`. Its exact 69 source snapshots are retained. An AST import check found no missing local/repository dependency. Prepared plan SHA256: `09cbbad19ff0f4ca3caadf8508e577b84ede72d8d208870456288ef163e18e7d`. The separate successful `prepared-read-v1.json` has SHA256 `ce7f08a99262dfe63fbf25f25313b509b2695222c3e1aaadc4ffd3ac15b2d6fa`. The packet contains 82 files totaling 68,887,581 bytes.

## Explicit future execution

The checked repository must be on `PYTHONPATH`; `engine.py`, `packet.py`, `run.py`, `test_cpu.py` and `selection.json` must remain together. On the proposed remote layout:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/workspace/intermediate-source/open-worldline \
python /workspace/intermediate-profile-src/run.py \
  --execute --prepared-directory /workspace/intermediate-profile-prepared \
  --weights /workspace/wan22-ti2v5b-weights \
  --expected-gpu NVIDIA\ A100-SXM4-80GB \
  --admission /workspace/intermediate-profile-admission.json
```

These are intended paths, not an assertion that a remote resource exists. The parent must issue a separate JSON decision accepted by `packet.admission`: this schema and numerical scope, `decision="admit"`, exact prepared plan/source/input-selection/CPU-report hashes, caller-declared GPU, unchanged limits, `quality_admitted=false`, and `image_generation=false`. An old training admission cannot admit this profile. The runner does not create a provider resource or an admission.

One parent and one worker share a 900-second deadline covering preflight, loading, hashing, parity, updates and final verification. Existing limits are 60 GiB CUDA reserved memory, 48 GiB combined host RSS, at least 8 GiB available host and GPU memory, one supported GPU with at least 70 GiB total memory, and the exact Torch 2.5.1 / CUDA 12.4 / FlashAttention 2.7.4.post1 environment. Timeout escalation and failure retention use the existing guards. Provider lease deletion remains a separate parent responsibility; a worker deadline does not cap cloud billing.

The prepared root receives `attempt.json`, `launch.json`, `executed-admission.json`, parent metrics/terminal/logs, and worker `result/`. Attempts are single-use. The worker verifies all 825 original FP32 parameter values before computation, after all parity calls and after each placement. Only one parameter verification copy is held at a time. Initial/final adapter state, optimizer/RNG recovery bundles, raw parity/training outputs, gradients, memory samples and source/input identity records remain available on success or partial failure. Full foundation weights are external and are not copied into results.

## Timing and interpretation limits

The fixed order is block28 then block29. It is not a randomized timing study. The worker releases each adapter and feature bundle, collects Python garbage and clears unused CUDA allocator cache between placements while retaining one shared foundation. Peak allocated and peak reserved bytes are recorded separately for parity and training. They include the foundation and all live graphs. Cached intermediate features are large, and auxiliary backward retains four suffix graphs until the original combined loss backward finishes. Actual fit must be measured.

Reported operation wall times include synchronization, required output retention and applicable CPU checks; they are not isolated GPU kernel timings. Training times also include prescribed prediction retention, with checkpoint and core-hash timings reported separately where defined. The native reference uses `inference_mode(False)` plus `no_grad()` so the literal first-call transfer of complex rotary frequencies creates a normal constant usable by subsequent backward. Its value is checked before/after; no alternative rotary equation is used.

The CPU fixture verifies exact full/cached zero outputs, original main-plus-auxiliary update wiring, frozen native parameters, later recurrent gradients, fresh initialization, failure retention and parent cleanup. It uses two tiny native blocks with CPU mathematical attention. Actual 30-block CUDA parity, FlashAttention backward, suffix activation memory and speed remain unmeasured. This profile does not independently compare nonzero full-versus-cached CUDA gradients; the mixed full-main/cached-auxiliary route is the intended integration being measured. Two updates cannot establish visible door/camera control, learning superiority, generalization or a usable video foundation.

Original experiment code remains Apache-2.0; the imported Wan source and external weights retain their existing upstream attribution/license records in the repository source graph. Atrium inputs retain their documented original capture provenance. No novelty claim is made.
