# Native CUDA reference

This package completed an [initial prediction pair](results/a100-pair-v1/README.md) and a separate [50-step clip](results/a100-clip50-v1/README.md) on one NVIDIA A100-SXM4-80GB on September 7, 2026. The pair completed in 243.19 seconds and the clip in 476.44 seconds, including model loading and verification. All 17 decoded frames are retained. Generated future frames still develop severe colored, warped surfaces. The numerical execution passed; visual quality failed.

The first mode computes one positive and one negative prediction using the original Wan2.2 TI2V-5B model equations, original FP32 parameters, native CUDA BF16 autocast and FlashAttention 2. The separate second mode performs 50 solver updates from the exact same retained initial noise, image latent and text. Neither mode trains parameters or adds an original Worldline model contribution. The [completed cloud record](../../../docs/cloud-gpu-diagnostic-results-2026-09-07.md) includes setup, result recovery and confirmed instance deletion.

The completed preparation includes [9 author CPU checks](cpu-results/author-v2/report.json), [12 independent CPU checks](cpu-results/independent-v1/report.json), and a [validated plan](plans/pair-v1/metrics.json). The plan's A100 80GB name is explicitly hypothetical in [plan-context.json](plans/pair-v1/plan-context.json); no GPU or account was inspected. Earlier author-v1 records are preserved. They did not include the CLI import check, and their earlier source does not qualify for the current runner.

The clip is a **17-frame, 512 × 288 diagnostic** with CPU UniPC. It is not the full official pipeline at its intended resolution and duration. It reuses an independently encoded initial image and the genuine cached UMT5 contexts. No future RGB, target latents or commands enter this runner. Frame 0 is a conditioned reconstruction; the other 16 frames are generated futures.

The command defaults to planning. `--execute` is required for each mode. Pair mode cannot continue into clip mode automatically. Clip mode requires a completed, source-matching CUDA pair with the same inputs, GPU name, runtime properties and fixed precision. A failed or stopped pair is rejected. All 100 clip predictions use the clean image prefix and zero time on the first 144 tokens; the prefix is restored after every CPU solver step.

## Source and numerical contract

The four files in `vendor/` other than `__init__.py` are byte-exact upstream files. `native.PINS` binds their hashes. The pinned [Wan2.2 source commit](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc) supplies the model, original attention dispatch and VAE. The [official UniPC file](https://github.com/Wan-Video/Wan2.2/blob/42bf4cfaa384bc21833865abc2f9e6c0e67233dc/wan/utils/fm_solvers_unipc.py) has SHA256 `0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9`. The sampler uses one shift of 5, 50 integer timesteps, and `negative + 5 * (positive - negative)`.

The transformer contains 4,999,787,712 parameters, occupying 19,999,150,848 bytes in FP32. All three official shards are checked before any parameter value is loaded. The loader copies one tensor at a time to CUDA without changing dtype. An explicit CPU verification copy checks the CUDA transfer and is then released. Both copies count toward the sampled resource usage. There is no selective BF16 parameter conversion, translated model AST, real-valued RoPE replacement, SDPA substitution or imported portable forward method in this package.

`expected-weights.json` contains only the original 825 names, shapes, shard names and tensor hashes from the completed CPU pair, with its source-artifact hash. It contains no weights. Clip admission requires the CUDA pair's loaded FP32 records to match that complete table.

Only the prior `official_cpu.inputs` and `official_cpu.streaming` file/hash readers are reused. Their exact sources are part of every preflight and run snapshot. `upstream-provenance.json` is an unchanged historical metadata record from the parent package. Its source paths describe that original record; `vendor/shared_config.py.txt` is not imported here. The actual source list and hashes are `evidence.sources()`. The cached prompt identity is checked directly, including its manifest and tensor hashes.

The pinned [official weight revision](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e) and Wan code are Apache-2.0. This repository does not redistribute the external weights. FlashAttention 2 is an external BSD-licensed dependency. PyTorch and the other installed libraries retain their own licenses. No remote inference API is used.

## Fixed resource limits

Use one visible Ampere, Ada or Hopper GPU with at least 70 GiB reported device memory and native BF16 support. This deliberately excludes the 24 GB Mac and smaller CUDA GPUs. The caller must provide the exact expected GPU name. This recipe fixes Torch 2.5.1 with CUDA 12.4 and FlashAttention 2.7.4.post1. FA3 and attention fallback are rejected.

Each explicit run has a 900-second combined deadline. The guards require at most 48 GiB combined host RSS, at most 60 GiB CUDA reserved memory, and at least 8 GiB available host and CUDA memory. These are sampled limits, not guarantees that no allocation briefly exceeds a sample. The parent can terminate and then kill a stalled child. The transformer and FP32 decoder run in separate child processes so their model weights do not coexist.

Clip admission estimates `1.2 * (measured core load + 50 * measured first pair) + 120 + 30` seconds. The 120-second decoder/load allowance and 30-second artifact allowance are declared estimates, not measured CUDA performance. The runner rejects estimates above 900 seconds and keeps the actual deadline unchanged. There is no intended-size 121-frame mode.

**A worker deadline does not stop cloud billing, stop a pod or delete its storage.** This package has no account, provisioning or teardown commands. A separately authorized operator must control and close any rented instance.

## Installation recipe

The commands below describe direct installation. The completed diagnostic instead used the pinned PyTorch container and matching FlashAttention binary recorded in [setup evidence](results/a100-pair-v1/README.md). Run it in a fresh Python 3.11 environment on Linux with the CUDA 12.4 toolkit and a compatible NVIDIA driver. The [official PyTorch archive](https://pytorch.org/get-started/previous-versions/) documents the CUDA 12.4 wheel index. The pinned [FlashAttention installation instructions](https://github.com/Dao-AILab/flash-attention/tree/v2.7.4.post1#installation-and-features) require the CUDA toolkit and recommend limiting parallel build jobs on hosts with less than 96 GB RAM. Compilation is outside this experiment's 900-second deadline.

From the repository root:

```sh
python3.11 -m venv .venv-cuda-reference
.venv-cuda-reference/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
.venv-cuda-reference/bin/python -m pip install -r experiments/wan22_native/cuda_reference/requirements.txt
.venv-cuda-reference/bin/python -m pip install packaging==25.0 ninja==1.11.1.3 setuptools==75.8.0 wheel==0.45.1
MAX_JOBS=4 .venv-cuda-reference/bin/python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Keep the installation log, `pip freeze`, `nvidia-smi` output and compiler version in a separate setup-evidence directory. The runtime records the reported GPU, Torch/CUDA/FlashAttention versions, cuDNN settings and relevant environment variables. Those records describe the actual machine; they do not establish CUDA numerical equivalence before execution.

If the operator explicitly chooses to download external weights, this command requests only the pinned core, index/config and native VAE. It does not download the text encoder:

```sh
.venv-cuda-reference/bin/huggingface-cli download Wan-AI/Wan2.2-TI2V-5B \
  diffusion_pytorch_model-00001-of-00003.safetensors \
  diffusion_pytorch_model-00002-of-00003.safetensors \
  diffusion_pytorch_model-00003-of-00003.safetensors \
  diffusion_pytorch_model.safetensors.index.json config.json Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir ../wan22-reference-weights
```

## CPU preflight and run commands

The CPU tests inspect schedule, input identity, source hashes, failure handling and gates. They execute no CUDA kernel or foundation model. They cannot validate GPU quality. Both output directories below must be new:

```sh
.venv-cuda-reference/bin/python -m experiments.wan22_native.cuda_reference.test_cpu \
  --output ../cuda-reference-author
.venv-cuda-reference/bin/python -m pip install pytest==7.4.4
.venv-cuda-reference/bin/python experiments/wan22_native/cuda_reference/cpu-results/independent-v1/run-independent-review.py \
  --repo . --output ../cuda-reference-independent
```

Supply both report files below. Replace the GPU-name placeholder with the exact name chosen for the experiment, rather than changing the hardware gate.

```sh
.venv-cuda-reference/bin/python -m experiments.wan22_native.cuda_reference.run \
  --mode pair --expected-gpu '<exact GPU name>' \
  --weights ../wan22-reference-weights \
  --pair-directory experiments/wan22_native/core-results/cpu-pair-v1 \
  --text-directory experiments/wan_adapter/text_cache/native-results \
  --cpu-report ../cuda-reference-author/report.json \
  --independent-report ../cuda-reference-independent/report.json \
  --output ../cuda-reference-pair-plan
```

After separately authorizing execution, use the same inputs with a fresh output directory and add `--execute`. Review the pair's complete outputs before requesting a clip. A separately requested clip uses `--mode clip --cuda-pair ../cuda-reference-pair-run`, with the same other settings and a fresh output directory. Omitting `--execute` still creates only a plan. Nothing here authorizes a paid run.

Pair artifacts include the exact copied input and text files, source/test snapshots, original FP32 tensor hashes, partial positive/negative predictions, final guided velocity, terminal records and memory samples. Clip artifacts additionally retain all 50 completed latent states, 17 decoded frames, raw FP32 RGB and final latents. Preview playback is 8 fps, separately labeled from measured generation throughput. A failure retains completed artifacts and a failed/interrupted or watchdog terminal record. No output is automatically judged photorealistic, correct, novel or comparable to Genie 3.
