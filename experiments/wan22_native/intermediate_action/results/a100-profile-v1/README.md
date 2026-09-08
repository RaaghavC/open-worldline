# Actual A100 placement profile and native six-arm cache

The intermediate action adapter completed the bounded CUDA profile. Both placements matched the unmodified Wan2.2 model exactly before training, and both completed two updates with gradients reaching the command recurrent unit. The new six-arm native-resolution VAE cache also completed. These results validate the execution paths and training inputs. They contain no generated video and establish no usable action control or graphical improvement.

## Placement comparison

The original 947,712-parameter adapter was inserted after block 28, leaving the final frozen transformer block to process its output. The control inserts the same adapter after block 29, directly before the native head. Each placement starts from the same saved zero adapter and fresh optimizer. Both receive identical previously published targets, commands, text, independent observation and noise draws. [Method](method.md) and [executed source](../../cuda_profile/README.md) describe the complete protocol.

| Measurement | After block 28 | After block 29 |
| --- | ---: | ---: |
| Exact initial native/full/cached comparisons | 8 of 8 | 8 of 8 |
| Completed paired updates | 2 | 2 |
| Time for both training updates | 4.089 s | 3.873 s |
| Peak CUDA allocation during training | 25.012 GiB | 20.093 GiB |
| Peak CUDA reservation during training | 25.432 GiB | 20.303 GiB |
| Recurrent gradient norm, second update | 0.00009064 | 0.00007969 |

The combined parent took 353.189 seconds, including 128.511 seconds loading weights. All 825 original foundation parameter records matched before the comparison, after initial parity and after each placement. The independent raw audit checked 46 finite prediction/gradient bundles, six checkpoints, four recomputed losses and all 16 exact initial comparisons. Saved final-block control predictions and all 18 gradient tensors from the first update also match the original historical experiment exactly.

The two updates use different saved noise/time draws, originally selected for updates 1 and 5. Their losses are not a fixed-input learning curve. Frozen suffix layers still require backward activations; the measured extra allocation is about 4.92 GiB in this specific profile.

## Saved command-response diagnostic

The already saved second auxiliary predictions measure response after one completed update, on the previously seen room and a saved noise draw. [The analysis](saved-analysis-v1.json) finds normalized target-difference error of 1.000194 after block 28 and 0.999994 after block 29; zero response has error 1. Corresponding cosine similarities are 0.003020 and 0.002627. Both responses remain poorly aligned with the desired change. The larger response at block 28 does not establish better control.

This analysis makes no new model calls. Any target-fitted multiplier in the retained report is a post hoc diagnostic, not a deployable model or a video score. Two training updates cannot decide whether longer training will succeed.

## Native six-arm cache

The [published factorial capture](../../../../atrium_factorial/README.md) supplies six 17-frame, 1248 × 704 clips from one known Atrium room: stationary, left and right camera movement, each with the door closed or remotely opened. The native VAE encoder processed the original pixels without resize, crop or prefix replacement.

Eight encodes completed: the shared first image, all six full clips and the first image again. All 12 prefix checks were bit-exact, including every full-clip prefix against the separately encoded image. All 196 original VAE parameter records and normalization values remained unchanged. The parent took 68.365 seconds; peak CUDA allocation was 9.201 GiB and reservation was 10.168 GiB. [Cache source and reader](../../../../atrium_factorial/native_cache/README.md) are included.

No model has been trained on these six clips in this release. These are programmed development targets from one seen scene, with a remote door toggle and fixed-position yaw. They do not represent contact physics or held-out scenes. Future training on this data is a new study; comparison with an older checkpoint trained on different images cannot isolate placement effects.

## Evidence and cloud cleanup

The [independent raw review](audit/raw-review.md) and [machine report](audit/raw-audit-report.json) cover all 511 originally recovered files. All scientific inputs, executed source, predictions, gradients, checkpoints and encoded tensors are retained. The audit checks retained weight-value records; it does not contain or reread the external foundation weights, replay model backward passes or independently reconstruct AdamW updates. Its first failed local wrapper and corrected wrapper are both retained.

The full artifacts are attached to the [A100 profile and cache release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-intermediate-factorial-a100-v1). The public archive discloses workspace-prefix changes in four review Markdown files. Scientific files are unchanged. The original audit script is an archival source bound to the original raw index and original work layout; the public transport verifier uses the separately published derivative index.

From the repository root, download and verify all 511 files without a provider account:

```sh
python3 experiments/wan22_native/intermediate_action/results/a100-profile-v1/fetch_artifacts.py \
  --metadata experiments/wan22_native/intermediate_action/results/a100-profile-v1/release-download.json \
  --output /absolute/path/to/fresh-profile-and-cache-download
```

The helper verifies each of the 39 download pieces, the combined archive and every extracted file. [Artifact details](ARTIFACTS.md) explain public derivation, licensing and compatibility with the original scientific readers.

The exact GPU instance was deleted, with explicit provider 404 and absence from the complete instance list. The temporary management key was revoked and returned HTTP 401. All three local private key files were removed. [Cleanup record](operations/pod-cleanup.json) and [key retirement](operations/key-retirement.json) are retained. No subsequent training started on this rental.

Wan2.2 is an attributed external open-weight foundation. The original adapter and experiment code retain the repository licenses; source and data notices remain in the archive. No foundation weights are redistributed here. Neither stage establishes Genie 3 parity, general world generation, visible interaction or a new scientific breakthrough.
