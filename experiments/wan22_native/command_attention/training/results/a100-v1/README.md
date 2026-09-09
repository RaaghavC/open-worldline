# Fixed 512-update A100 training result

All 512 planned updates completed and the [saved-artifact audit passed](records/audit.json). Door comparisons improved in the fixed-noise latent evaluation. Camera comparisons remained close to zero response. **Generated-video evaluation is pending; this result does not demonstrate visible door or camera control.**

The run fits one already-seen room with six camera/door sequences. It replaces the previous residual adapter with 4,936,448 trainable parameters in command-conditioned rank-32 attention projections across blocks 24–29. It also adds direct camera contrast targets. Architecture and objective changed together, so these results cannot identify which change caused an improvement. There is no new-scene, generalization or novelty claim.

## Measurements

| Measurement | Recorded value |
| --- | ---: |
| Completed updates | 512 |
| Parent / worker elapsed | 1,126.617 / 1,117.084 seconds |
| Foundation loading | 178.987 seconds |
| Training plus initial/final evaluation | 808.569 seconds |
| Peak allocated / reserved CUDA memory | 55,111,928,320 / 56,503,566,336 bytes |
| Main / auxiliary training predictions | 1,024 / 512 |
| Auxiliary updates | 128 |
| Initial / final evaluation predictions | 48 / 48 |
| Foundation current-value records unchanged | 825 |

The [worker metrics](records/training/result/metrics.json) include all 512 scalar rows, the declared schedule and both evaluations. A [matching profile on the training lease](records/profile/result/metrics.json) and its [independent audit](records/profile-audit.json) passed before this run. The training parent used the unchanged 1,800-second and memory limits and reserved 600 seconds before the fixed external lease deadline. The separate [first profile](../../../profile/results/a100-v1/README.md) had correctly refused training on its shorter remaining lease.

## Seven fixed latent comparisons

Each range below spans the same four saved evaluation noises at the final checkpoint. The initial controller produced zero contrast, with normalized MSE exactly 1.0 for every pair. Lower normalized MSE is better. Cosine measures direction; prediction and target RMS show response magnitude. These values describe one ideal clean-endpoint estimate from shared pure future noise at sigma 1 and native time 999, not a sampled video.

| Pair, first to second | Normalized MSE | Cosine | Prediction RMS | Target RMS |
| --- | ---: | ---: | ---: | ---: |
| stationary_closed → stationary_interact | 0.817302 to 0.838479 | 0.627510 to 0.640391 | 0.086833 to 0.099265 | 0.607036 |
| left_closed → left_interact | 0.885702 to 0.899128 | 0.441983 to 0.448187 | 0.086383 to 0.098923 | 0.642549 |
| right_closed → right_interact | 0.885111 to 0.895426 | 0.422546 to 0.426808 | 0.087258 to 0.099582 | 0.585996 |
| stationary_closed → left_closed | 0.999684 to 1.000078 | 0.005811 to 0.017856 | 0.017054 to 0.017435 | 1.055066 |
| stationary_closed → right_closed | 1.000483 to 1.000757 | -0.013366 to -0.005954 | 0.016951 to 0.017412 | 1.007468 |
| stationary_interact → left_interact | 0.999930 to 1.000657 | -0.011406 to 0.010947 | 0.016942 to 0.018361 | 1.017872 |
| stationary_interact → right_interact | 0.999935 to 1.000574 | -0.007619 to 0.010673 | 0.016878 to 0.017578 | 0.969255 |

The three door pairs reduce normalized error by about 10.1% to 18.3% relative to the zero-contrast initialization. For camera pairs, normalized error stays between 0.999684 and 1.000757, cosine stays near zero, and prediction RMS is only about 0.017 to 0.018 against target RMS near 1. These camera responses remain weak. The [full-precision summary](numerical-summary.json) is derived directly from the unchanged producer records; the existing audit separately recomputed all 28 initial and all 28 final scores from retained arrays.

For an edge from a to b, the scored prediction is `-(G_b - G_a)`, where `G = N + 5(P - N)`, and the target is `z_b - z_a`. Only the four future latent groups enter the reductions. Main flow matching still conditions on target-corrupted noisy futures. Only the auxiliary contrast uses shared pure-noise future conditioning, with future targets entering its loss alone.

## Checkpoint and audit limits

The declared result is [checkpoint 512](records/training/result/checkpoint-0512/manifest.json), SHA256 `c3ece8c1b9c6f7eadedbd9db235ad63cab28b2c2ad2d097936c02f7efaf4393d`. Its 58 FP32 tensors contain 4,936,448 parameters. The five checkpoints at 0, 128, 256, 384 and 512 were checked; no intermediate checkpoint was selected for evaluation. The executed plan has SHA256 `e705ec9e1e8ef769317aef9f13b9e7a385106d7837236869964481c0a5490198`.

The exact [training audit](records/audit.json), SHA256 `ba8e27201b649e0f8e700c39c7c4cc2b78c166a74ebf71e7e4d01ba82e916c41`, checked all scalar/schedule rows, checkpoints, source/input/profile/lease bindings and retained arrays. It recomputed main and auxiliary losses from the 22 saved training predictions, and gradient norms from five combined post-clip gradient bundles at updates 1, 2, 5, 509 and 512. It also recomputed both fixed-noise evaluations from all 96 retained velocities.

The other 507 updates have scalar records, without raw prediction or gradient arrays. The audit does not replay native forwards, backward passes, omitted gradients, AdamW or sampling. Optimizer and CPU RNG pickle files were checked by file hash without loading their contents. Foundation weights are absent from the recovery, so the audit checks saved before/after current-value hash maps rather than hashing the foundation again. Resource sampling cannot capture every instantaneous allocation.

## Included records and verification

The [publication inventory](publication.json) lists 67 byte-exact files totaling 5,650,985 bytes: 46 small records, 15 existing auditor files and 6 executed experiment source files. All 66 recorded producer source hashes match the repository files listed there. No selected file needed redaction. Nonsecret `/workspace` paths, process IDs and the owned task's pod ID remain in the records. The unchanged auditor README retains its original local example command; the portable command below uses the copied source.

The [recovery receipt](records/recovery/recovery-verified.json) covers 303 files and 864,298,597 bytes, transported in 37 pieces totaling 581,737,911 bytes. Raw controller weights, optimizer states, velocities, gradient bundles and transport pieces remain outside Git. **Their raw release has not been created or published.** These compact records alone cannot reproduce the raw artifact audit, and no raw download URL is claimed here. The [original transfer binding](records/recovery/original-transfer-binding.json) identifies the separately retained source/input packet.

From the repository root, verify these small files and their source mappings without Torch or a model:

```sh
python experiments/wan22_native/command_attention/training/results/a100-v1/verify_records.py --repository .
```

When the complete original recovery and source/input transfer are available, the [copied NumPy auditor](audit/audit.py) can recheck them:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 python \
  experiments/wan22_native/command_attention/training/results/a100-v1/audit/audit.py \
  --recovered-root /path/to/training-recovery/recovered \
  --original-transfer /path/to/original-transfer-extraction \
  --recovery-verified /path/to/training-recovery/recovery-verified.json \
  --output /path/to/fresh-audit-output
```

The original transfer must contain its `inputs`, `repository`, `controller`, `training` and `profile` directories. The recovery includes both profile/training runs and their dispatch records. Use NumPy 1.26.4 for the recorded audit environment. The [three existing CPU fixtures](audit/cpu-v1/report.json) and [independent schema review](audit/cpu-v1/independent-schema-review.json) remain unchanged. Copying and verifying this compact result did not rerun training tests, the full artifact audit or any model.
