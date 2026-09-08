# Block-28 training on six camera and door sequences

This is the exact training code used by the [completed 128-update A100 run](../results/factorial128-a100-v1/README.md). It trains the original 947,712-parameter action adapter before the final frozen Wan2.2 transformer block. The foundation weights remain unchanged. The run completed and its recovered-file audit passed; generated-video control has not yet been evaluated for this checkpoint.

Three motion pairs cycle through stationary, left and right camera commands. Each pair contains closed-door and interaction targets from the [native six-arm capture](../../../atrium_factorial/README.md). Both targets retain their original encoded prefixes and share an independently encoded starting image. Main flow matching uses target-corrupted noise; the additional paired objective uses shared pure-noise future conditioning. [PREPARATION.md](PREPARATION.md) retains the exact original method and its prospective limitations.

The same 128 saved noise/time draws, initial adapter, positive/negative text, paired objective and AdamW settings are preserved. The data and insertion point differ from the old final-block experiment. This run therefore cannot isolate a placement advantage.

[source-integration.json](source-integration.json) records byte-exact source and report copies. The copied-path check passed all 15 CPU tests and preserved the complete 88-file source map. The tests use small explicit fixtures; they are separate from the real CUDA result.

From the repository root, with the existing Wan experiment dependencies installed:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=.:experiments/wan22_native/intermediate_action/factorial_training \
python -m pytest -q experiments/wan22_native/intermediate_action/factorial_training/test_cpu.py
```

The production entry point is `training_run.py`. It defaults to a plan, with separate explicit preparation, input verification and guarded CUDA execution modes. Use the commands and artifact requirements in PREPARATION.md. An execution binds exact source, inputs, CPU evidence and actual prior CUDA/cache receipts; it retains a combined 900-second worker guard and original memory limits. No automatic resume, image generation or model-quality promotion is implemented here.
