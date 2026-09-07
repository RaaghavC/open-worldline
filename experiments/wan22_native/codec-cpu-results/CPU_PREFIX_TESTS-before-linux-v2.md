# CPU prefix checks across platforms

The Linux tiny-codec fixture produced a small difference between a standalone one-frame encode and the first latent of a full 17-frame encode. It did **not** change the first latent when future RGB pixels changed at the same clip length. The separate portable CPU test preserves that distinction.

The strict original test and its CPU reports remain unchanged in `action_data/`. Production `exact_prefix`, the cache readers, the native codec and the completed real MPS caches remain unchanged. The real eight-window cache still passed all 11 exact checks with zero difference.

## Recorded Linux result

The retained Linux diagnostic used Torch `2.5.1+cpu` on x86-64. All four combinations of oneDNN enabled/disabled and one/two CPU threads produced the same result:

| Comparison | Differing elements | Maximum absolute difference |
| --- | --- | --- |
| Full-clip prefix versus standalone encode | 134 of 192 | 2.086162567138672e-7 |
| Full-clip prefix versus full-clip prefix after future RGB changes | 0 of 192 | 0 |
| Standalone encode repeated after other encodes | 0 of 192 | 0 |

The Mac diagnostic reproduced exact equality in all comparisons. Its fixture weights, input RGB and checked source hashes were identical to the Linux diagnostic. oneDNN is unavailable in that Mac Torch build. Disabling oneDNN on Linux did not remove the cross-length difference, so that toggle is not a supported fix.

These observations distinguish a cross-length numerical difference from the tested same-length future dependence. They do not identify the responsible CPU operation. The native encoder concatenates its temporal chunks before applying its final 1×1×1 projection, so the standalone and full-clip calls do contain different tensor shapes at that stage. The optional second diagnostic records values before and after this projection to localize the first observed discrepancy.

## Separate portable CPU test

`test_codec_cpu_portable.py` uses the identical tiny native codec, seed 762 and 17-frame RGB draw. It has four checks:

1. Changing future RGB at the same 17-frame length leaves the first latent **bit-exact**.
2. Repeating the standalone initial encode after other encodes remains **bit-exact**, with unchanged weights and clear caches.
3. Comparing full-clip and standalone initial latents uses only the declared absolute bound of eight FP32 machine epsilons, `9.5367431640625e-7`, with relative tolerance zero. This bound applies only to this fixed tiny CPU fixture's cross-length comparison.
4. The unchanged production `exact_prefix` function still rejects a deliberately introduced one-epsilon difference, even though that difference is smaller than the CPU fixture's cross-length bound.

There is no tolerance on the same-length causality or repeat checks. This fixture does not authorize approximate prefixes in a production cache or change the actual recorded tensors.

For Linux CI, exclude only the old bundled method `experiments.wan22_native.action_data.test_codec_path.CodecPathTests.test_initial_encode_is_independent_and_future_perturbation_causal`, and run this separate fixture. Keep the remaining original tests. The old source and failed CI record remain retained; the method is not rewritten into a different test.

```sh
python -m experiments.wan22_native.test_codec_cpu_portable \
  --output work/wan22-codec-cpu-portable-linux-v1.json
```

The Mac run passed all four checks. The portable Linux run is a separate CI measurement and must be reported from its actual result.

## Diagnostic commands and boundaries

The first diagnostic records all four backend/thread combinations without choosing a backend or altering a gate:

```sh
python -m experiments.wan22_native.codec_cpu_diagnostic \
  --output work/wan22-cpu-prefix-linux-diagnostic-v1.json
```

The optional second diagnostic copies the first encoder output, the final projection's input/output and the normalized latent. It verifies that attaching its read-only hooks leaves a full encode bit-exact, removes all hooks, and independently replays the same projection input at one and five latent frames. It also changes only later projection-input frames and checks whether the first output changes at the same length.

```sh
python -m experiments.wan22_native.codec_cpu_projection_diagnostic \
  --output work/wan22-cpu-projection-linux-diagnostic-v2.json
```

All four Mac projection traces preserved the unhooked result exactly and showed no same-length future dependence. They did not reproduce the Linux discrepancy. A Linux projection result is required before naming that operation as the cause. These are small CPU fixtures with no external weights, GPU operations, training or production-source edits.

Retained reports and their exact checked source snapshots are in `codec-cpu-results/`.
