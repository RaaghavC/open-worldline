# Intermediate action CUDA profile source

This directory preserves the exact numerical-profile implementation prepared in `work/intermediate-action-cuda-profile-v1`. It compares injection after block28 with injection after block29 using the same original saved start0 inputs, zero adapter and fresh optimizer. All native/full/cached zero-output comparisons precede two main-plus-auxiliary updates at each placement.

The source, tests, original CPU report and its 69 source snapshots are unchanged. The original preparation instructions and limitations are retained in [PREPARATION.md](PREPARATION.md). [source-integration.json](source-integration.json) maps each copied file to its original hash and discloses the path-only public derivatives of the review and warning log. Original raw evidence remains unchanged. This source package contains no model weights, prepared RGB/noise packet or actual model outputs. Actual run results are documented separately after recovery and audit.

The [copied-path CPU check](integration-v1/report.json) passed all 11 tests in a fresh process and verified the same 69 source hashes as the original CPU report. CI runs this file in its own process.

From the repository root, install the existing pinned CPU dependencies, then run this test file in its own Python process:

```sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/wan_adapter/requirements-real.txt \
  -r experiments/wan22_native/requirements-core.txt \
  -r experiments/wan22_native/requirements-sample.txt

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:experiments/wan22_native/intermediate_action/cuda_profile \
python -m pytest -q experiments/wan22_native/intermediate_action/cuda_profile/test_cpu.py
```

Do not combine this test file with the native-cache test file in one pytest process. The exact historical code imports bare `packet` and `run` modules. Separate processes and the shown `PYTHONPATH` preserve the intended imports without changing the frozen source.

The no-argument command prints a CPU-only plan:

```sh
PYTHONPATH=. python experiments/wan22_native/intermediate_action/cuda_profile/run.py
```

To create a new exact input packet, first recover the public [action-effect training artifacts](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-effect-training-a100-v1), then supply the directory containing `action-results/`:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/wan22_native/intermediate_action/cuda_profile/run.py \
  --prepare --input-root /path/to/recovered \
  --cpu-report experiments/wan22_native/intermediate_action/cuda_profile/cpu-v1/report.json \
  --output /path/to/fresh-prepared-profile

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
python experiments/wan22_native/intermediate_action/cuda_profile/run.py \
  --preflight --prepared-directory /path/to/fresh-prepared-profile
```

All eleven input files are pinned by [selection.json](selection.json). Saved noises and initial adapter values are not regenerated. The current source graph must still match the historical CPU report for preparation to pass. Changed source requires fresh CPU evidence and a fresh plan; no historical gate is silently bypassed.

Execution additionally requires `--execute`, the original external weights, the exact supported CUDA environment, a caller-declared GPU and a separate source/input-bound parent admission. The unchanged 900-second, 60 GiB CUDA-reserved and 48 GiB aggregate-host limits apply. No provider resource is created by this runner. The CPU tests establish orchestration on tiny native fixtures; they do not establish CUDA performance or visible action control.
