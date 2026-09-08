# Fresh 128-update action-adapter training on A100

**The original action adapter completed 128 paired optimizer updates, with all retained-artifact checks passing. This training result does not establish visible door or camera control.** The preceding [16-update rendered comparison](../visual-a100-v1/README.md) failed the requested actions. Checkpoint-128 fixed-input and rendered evaluations are outside this training record.

The run started from the exact original checkpoint-zero bytes and an empty AdamW optimizer. It did not continue from checkpoint 16. The first 16 saved noise/timestep draws match the earlier run, and its checkpoint-16 adapter is byte-identical to that earlier final adapter. The remaining 112 draws were prepared once from the retained CPU generator continuation and read as saved bytes. No cross-platform noise regeneration was used.

| Verified result | Value |
| --- | --- |
| Original trainable adapter parameters | 947,712, FP32 |
| Completed paired updates | 128 |
| Retained training velocity predictions | 256 |
| Retained clipped-gradient bundles / arrays | 128 / 2,304 |
| Retained checkpoint steps | 0, 16, 32, 48, 64, 80, 96, 112, 128 |
| Original foundation value identities unchanged | 825 tensors, 4,999,787,712 parameters |
| Native versus zero-adapter comparisons | Both bit-exact |
| Parent / worker time | 388.465 / 373.591 seconds |
| Core load time | 131.622 seconds |
| Sum of paired-update timings | 116.367 seconds |
| Sampled peak CUDA reserved memory | 21,583,888,384 bytes |

The original [training metrics](metrics/training.json), [parent metrics](metrics/parent.json), [worker metrics](metrics/worker.json) and [terminal](metrics/terminal.json) are unchanged. The [actual independent audit](audit/report.json) checked all 532 files in the training run, 1,873,031,208 bytes, without rerunning a model. It verified saved draws, exact source identities, all 256 future-only losses, all nine adapter/optimizer/RNG checkpoint bundles, 128 gradient bundles and sampled resource limits. Its [source](audit/audit.py) and [original run-file inventory](audit/inventory.json) are retained unchanged. Only the audit report's operational local-path prefix is replaced for publication, as disclosed below; every numerical and identity field remains exact.

The recurrent gradient is zero at the first update under zero output initialization, becomes 0.00006245405530 on the second, and is 0.0004438415926 on the last update. These values verify the retained gradient path. Different training rows use different windows, noises and timesteps, so their first and last losses are not a controlled learning curve.

## Fixed training contract

Training uses the four paired starts `[0,8,32,49]` repeated 32 times, closed then open with sequential batch size one. Each update averages half of each branch's mean future-latent velocity MSE and excludes the initial latent. The independently encoded observation remains the exact clean prefix. The original target encodings are unchanged. AdamW remains learning rate 1e-4, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01 and gradient clipping at L2 1.

The original command-prefix GRU and observation-attention adapter is inserted after native transformer block 29. The original Wan foundation weights, native head, upstream FlashAttention 2 and recorded CUDA BF16 execution remain frozen. No foundation parameter update, checkpoint selection or resume is claimed. These are eight windows from one fixed original room, not independent new scenes. Only the start-0 pair isolates wait versus interact with the same canonical observation; the later paired starts have the same commands but different observed images.

The [preceding fourteen-prediction diagnostic](../diagnostic-a100-v1/README.md) compared checkpoints 0 and 16 on the original k=506 corruption and same-input k=999 command panel. It found small seen-example objective gains and a nonzero first-step command difference. It did not support simple first-step CFG cancellation. The [explicit admission](plans/executed-admission.json) used those results to admit a fresh fixed-duration training experiment. The fourteen-prediction result does not evaluate checkpoint 128. No image generation was admitted or performed by this training runner.

## Final checkpoint and source

The compact payload contains only [checkpoint 128](checkpoint-0128/adapter.safetensors), SHA256 `ac1160b22531804c473746d2fb8ee9edd75710e0e43d6f1eabc86d268c864996`, and its original [manifest](checkpoint-0128/manifest.json). The manifest also identifies the optimizer/RNG file in the full recovery. The final checkpoint was fixed in advance. It is an experimental adapter, not a validated controller.

The [source map](code-provenance.json) binds six exact local files, ten exact reference files and 66 shared repository source files through existing public bundles. The dependency packet is commit `099eeae6268da852b0acb580258e4c168f900520`; the exact work-only additions are retained under [source/local](source/local) and [source/reference](source/reference). The [accepted eight-check CPU report](preflight/accepted-cpu.json), [independent source review](preflight/independent-source-review.json), and both historical CPU outcomes remain available. The earlier failure was a missing pytest import path during fixture startup, not a model run; it is preserved in the [redacted historical log](preflight/historical-cpu-output-v1.txt). The copied preparation README describes its historical unexecuted status; this page and the actual audit describe the completed run.

## Combined recovery and privacy

The combined public archive retains all **957 recovery members**, including the complete fourteen-prediction diagnostic, completed 128-update training, original cache/probe inputs and exact executed programs. Its public uncompressed total is **2,096,526,695 bytes**, compressed with gzip level 1 into **1,514,792,115 bytes** across **100 parts**: the first 90 are 16,000,000 bytes each and the final 10 are at most 8,000,000 bytes. The [combined release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-fixed128-a100-v1) is public. All 126 remote asset sizes and SHA256 digests passed verification, and a [fresh public HTTPS download](recovery/public-download-verification.json) recovered all 957 members. [Download status](release-status.json) binds those records. See [DOWNLOADER.md](DOWNLOADER.md) for the pinned metadata and verification procedure.

The [original index](recovery/original-index.json) remains unchanged and records 2,096,526,751 uncompressed bytes. In the public derivative, exactly two private workspace prefixes in one historical dependency traceback become `[WORKSPACE]/`. Every other byte in that log remains; all other **956 files are byte-identical**, including every source, protocol, metric and tensor. No failure record or archive member is discarded. The [derivation record](recovery/public-derivation.json) binds original/public hashes and the exact replacement rule. Recompression produces a different public stream identity; this is explicitly a public derivative, not the original compressed archive. The 536 MB interim archive is not separately uploaded.

The independent audit's `run` field receives the same one-prefix replacement. Its original report SHA256 is `e987d7b31ce754977ce8ee547aa9e2cf4a2b31afc136a4f57b893de6950caa7b`; the published hash and exact change are in [code-provenance.json](code-provenance.json). The original recovery and audit remain untouched locally for source-bound evaluation. Public derivatives preserve every numerical and identity field. [Privacy verification](recovery/privacy-verification.json) describes the inspected files and pattern-scan limits.

Run `python verify_payload.py` to verify this small payload, unchanged source references, final checkpoint and publication identities without models or network access. The full archive is needed to recompute the retained-tensor audit. A passed artifact check proves recovered bytes and recorded arithmetic, not rendered behavior or generalization.

Original Worldline code and adapter weights use [Apache-2.0](licenses/WORLDLINE-APACHE-2.0.txt). External Wan code, model and VAE retain their [Apache license](licenses/WAN-APACHE-2.0.txt) and [NOTICE](licenses/WAN-NOTICE.txt). Original procedural Atrium imagery and metadata use [CC0](licenses/ATRIUM-CC0.txt). No external foundation checkpoint, provider credential or account-lifecycle record is included. No new architecture novelty or frontier-model parity is claimed.

The earlier verified 45 MB part layout failed during upload. The same public gzip stream was repartitioned into 95 parts at most 16 MB. [Earlier public index](recovery/history/public-index-45mb.json), [earlier metadata](recovery/history/public-download-45mb.json) and [earlier local verification](recovery/history/local-verification-45mb.json) preserve that transport history. No member or compressed-stream byte changed during repartitioning.

Part 090 later exhausted its two bounded 16 MB upload attempts. Only the five unuploaded parts were repartitioned into ten parts at most 8 MB; the first 90 uploaded parts remain exact. The final layout contains 100 parts. [Prior 16 MB index](recovery/history/public-index-16mb.json), [metadata](recovery/history/public-download-16mb.json) and [local verification](recovery/history/local-verification-16mb.json) preserve that intermediate transport. The public gzip stream and every member remain unchanged.

The [transport history](recovery/transport-history.json) records the failed upload layouts and successful final layout. The exact [published release record](recovery/published-release.json) and [remote asset check](recovery/remote-asset-verification.json) retain publication evidence. The latter was taken while the release was a draft; all names, sizes and digests also match the published record. Upload errors did not change scientific files or disable certificate verification.
