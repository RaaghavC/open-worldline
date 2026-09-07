# Compare the completed CUDA pair

Run this only after the complete pair directory has been recovered locally. The directory must contain its top-level metrics, exact copied inputs/text, measured sources, `core/terminal.json` and the complete `core/result/` output. A timeout, incomplete pair or changed retained file is rejected. There is no cloud or key access in the comparator.

From the task working directory, replace `RECOVERED_PAIR_DIRECTORY` with the local pair directory and choose a fresh report filename:

```sh
work/wan-adapter-env/bin/python work/runpod-launch-v1/compare-cuda.py \
  --repo outputs/open-worldline \
  --cuda-run RECOVERED_PAIR_DIRECTORY \
  --output work/runpod-launch-v1/cuda-pair-comparison-v1.json
```

The script reuses the exact published pure-NumPy comparator v2, SHA256 `30c0de0f3b1ecebcfe195ffd616d5763cbee75a43405f2e62d6430af6c382e70`. It does not import Torch or invoke a model. It checks source snapshots, original FP32 tensor identities, shared input/text identities, completed partial/final outputs and terminal status before computing statistics.

The saved baselines are:

| Baseline | Directory relative to repository root |
| --- | --- |
| Streamed official CPU | `experiments/wan22_native/official_cpu/results/pair-v1/` |
| Portable CPU | `experiments/wan22_native/core-results/cpu-pair-v1/` |
| Portable MPS | `experiments/wan22_native/core-results/pair-v1/` |

For each positive, negative and guided velocity, the report compares **CUDA minus each saved baseline**. Every relative RMSE uses the streamed official CPU RMS for the same region as its denominator, including CUDA versus portable CPU and CUDA versus MPS. The future region is the four latent frames `[:, 1:5, :, :]`, 110,592 values per velocity. All-frame and conditioned-prefix results are reported separately.

RMSE, absolute differences and cosine use the existing float64 NumPy definitions. Guidance residuals preserve v2's separate eager float32 subtract/multiply/add calculation and use that recomputed guidance as their own clearly labeled denominator. Saved velocities are never replaced by recomputed values.

There is no acceptance threshold or automatic equivalence, quality or failure-cause conclusion. These are predictions at the retained initial step, not comparisons of complete generated videos. No comparison has been run against CUDA outputs during preparation.
