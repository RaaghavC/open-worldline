# Completed official-equation CPU reference pair

The attributed Wan2.2 TI2V-5B reference completed one positive and one negative prediction on the retained 17-frame, 512 × 288 latent input. It kept the official model equations and original FP32 parameter values, using CPU BF16 autocast, independently implemented CPU BF16 SDPA, and one module of weights resident at a time. This is a numerical diagnostic of an external model. It performed no training, solver step, decoding or video generation.

The [parent record](metrics.json) reports 509.274 s total, including a 507.002 s [worker](result/metrics.json). Shard verification took 12.984 s; the positive and negative forwards, including streamed weight reads, took 245.937 s and 247.245 s. The sampled [parent/worker RSS peak](terminal.json) was 1,480,048,640 bytes (1.378 GiB), with at least 5.597 GiB available system memory. The guard remained 900 s, 18 GiB combined RSS and 2 GiB minimum available memory.

The [independent artifact audit](independent-review/report.json) matched all 825 original FP32 tensor hashes to the saved official-shard records, verified identical retained inputs and genuine text, and checked all 140 load/eviction events. Each pass recorded 35 groups and 825 released parameter owners. These are retained runtime checks, not direct inspection of memory after the process exited.

## First-step differences

The [comparison report](comparison/wan22-official-reference-comparison-v1.json) divides the RMSE by the **streamed official CPU reference RMS**. Percentages below use the future four latent frames, excluding the observed initial latent frame.

| Compared output | Positive | Negative | CFG-5 guided |
| --- | ---: | ---: | ---: |
| Earlier portable CPU | 0.360994% | 0.372924% | 2.270204% |
| Earlier portable MPS | 0.342459% | 0.422573% | 1.988637% |

The independent audit uses a different denominator: the **earlier portable CPU RMS**. Its corresponding CPU percentages are 0.361052%, 0.372855% and 2.274225%. Both reports preserve their original definitions and numbers. The raw difference RMSE agrees between them.

These measurements describe one initial prediction pair. They do not establish CUDA FlashAttention equivalence, agreement over 50 steps, a cause of the earlier prismatic images, visual quality, or an original-model advance. No numerical acceptance threshold was assigned to these differences. The previous failed videos remain failed results.

## Saved evidence and reproduction

[publication.json](publication.json) records every original run file and each published file. All 34 raw run files are present. Exactly four operational fields in `launch.json` are changed to repository/work-relative paths; all other raw files are byte-exact. The manifest records both file hashes and the exact field mapping. The old monotonic deadline is historical metadata, so the launch record is not a replay configuration.

The [comparison script](comparison/compare-wan22-official-reference.py), [frozen comparator preflight v2](comparison/preflight-v2/publication.json), [independent audit script](independent-review/audit-official-cpu-pair.py), reports, retained model sources, initial tensors and all three final velocity tensors are included. The frozen preflight README retains its original workspace commands and local previous-preflight reference. Use the public paths below for this package. Official model weights are not redistributed.

From the repository root, recompute only the saved-output comparison with NumPy and the pinned runtime dependencies:

```sh
python experiments/wan22_native/official_cpu/results/pair-v1/comparison/compare-wan22-official-reference.py \
  --repo . \
  --reference-run experiments/wan22_native/official_cpu/results/pair-v1 \
  --output work/NEW_OFFICIAL_REFERENCE_COMPARISON.json
```

The independent read-only audit accepts public paths too:

```sh
python experiments/wan22_native/official_cpu/results/pair-v1/independent-review/audit-official-cpu-pair.py \
  --run experiments/wan22_native/official_cpu/results/pair-v1 \
  --saved-cpu-pair experiments/wan22_native/core-results/cpu-pair-v1 \
  --source-package experiments/wan22_native/official_cpu \
  --text-cache experiments/wan_adapter/text_cache/native-results \
  --output work/NEW_OFFICIAL_REFERENCE_AUDIT.json
```

Neither command runs a model. A new audit of the public directory will list the relocated launch record and additional publication files, so its file inventory differs from the preserved audit of the original 34-file run. For a new pretrained forward, use the plan/execute instructions in the [package README](../../README.md), which establish a fresh deadline and recheck the sources and weights.
