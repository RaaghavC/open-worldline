# Native VAE cache source for six factorial arms

This directory preserves the exact cache implementation prepared in `work/atrium-factorial-native-cache-v1`. It passes the published native 1248×704 RGB arrays directly to the unchanged Wan2.2 VAE encoder. There is no resize, crop, old canonicalization, prefix replacement or transformer load.

The source, tests, original CPU report and its 71 source snapshots are unchanged. [PREPARATION.md](PREPARATION.md) retains the original method, commands and limitations. [source-integration.json](source-integration.json) records exact copies and the disclosed workspace-prefix removal in the public independent review. The original raw evidence is unchanged. Large captures and actual encoded results remain separate release artifacts; this folder contains source and small CPU evidence only.

The [copied-path CPU check](integration-v1/report.json) passed all 12 tests in a fresh process and verified the same 71 source hashes as the original CPU report. CI runs this file in its own process. The [actual A100 run](../../wan22_native/intermediate_action/results/a100-profile-v1/README.md#native-six-arm-cache) subsequently completed all eight encodes in a 68.365-second parent interval. All 12 prefix comparisons were bit-exact and all 196 VAE value records remained unchanged.

The fixed sequence is one independent shared first image, six 17-frame target clips and a repeat of the first image. Six cross-length prefix comparisons retain the original dual 1e-5 bounds. Five same-length comparisons and the repeated-image check require bit-exact equality. All 196 original FP32 VAE values are checked before and after encoding. No command or target latent is supplied as VAE conditioning.

From the repository root, use the existing pinned CPU dependencies and a separate Python process:

```sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/wan_adapter/requirements-real.txt \
  -r experiments/wan22_native/requirements-core.txt \
  -r experiments/wan22_native/requirements-sample.txt

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:experiments/atrium_factorial/native_cache \
python -m pytest -q experiments/atrium_factorial/native_cache/test_cpu.py
```

Keep this test file separate from the CUDA-profile tests. Both frozen scripts use bare `packet` and `run` imports; the separate process and shown `PYTHONPATH` avoid importing the other experiment's helpers.

Download and verify the public capture with the existing [factorial fetcher](../fetch.py), then prepare outside the repository:

```sh
python experiments/atrium_factorial/fetch.py --output /path/to/factorial-download

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/atrium_factorial/native_cache/run.py \
  --prepare --capture /path/to/factorial-download/capture \
  --cpu-report experiments/atrium_factorial/native_cache/cpu-v1/report.json \
  --output /path/to/fresh-prepared-cache

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/atrium_factorial/native_cache/run.py \
  --preflight --prepared-directory /path/to/fresh-prepared-cache
```

The current source graph must match the original CPU report. Source changes require new source-bound CPU evidence and a fresh plan. Running `run.py` without a mode prints a CPU-only plan. Actual encoding additionally requires `--execute`, a separate exact parent admission, the original external VAE file and the supported CUDA runtime. One 600-second guard covers the complete stage, with the unchanged 60 GiB CUDA-reserved and 48 GiB aggregate-host caps. The provider lease and serialized execution are separate parent responsibilities.

`run.read_completed(root, arm, conditioning_only=False)` exposes target, independent observation and commands only after complete parent/worker/monitor and source/input/value checks. It remeasures all twelve prefix checks from saved tensors. Conditioning-only reads inspect initial target slices for validation but never materialize future target latents.

These are CC0 development data from one seen room, with fixed-position yaw and a programmed remote door toggle. A completed cache is not evidence of learned door/camera control or generalization. The CPU tests use explicit arrays and a stand-in encoder, with no foundation weights or CUDA execution.
