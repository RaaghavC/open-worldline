# Action-effect adapter training: local publication staging

**The original 947,712-parameter adapter completed a fresh 128-update CUDA training experiment. The saved-artifact audit passed. This record does not establish visible door control, camera movement or generalization.** Separate numerical assessment is outside this training record and was pending when this staging directory was prepared. No release or public download is claimed.

| Verified training evidence | Result |
| --- | --- |
| Paired optimizer updates | 128 |
| Main training predictions | 256 |
| Auxiliary updates / predictions | 32 / 128 |
| Total training feature extracts / head predictions | 320 / 384 |
| Separate initial native / bridge parity predictions | 2 / 2, both comparisons bit-exact |
| Frozen foundation tensors | All 825 unchanged, 4,999,787,712 parameters |
| Saved checkpoint steps | 0, 16, 32, 48, 64, 80, 96, 112, 128 |
| Combined clipped-gradient bundles / arrays | 128 / 2,304 |
| Parent / worker duration | 412.050 / 396.287 seconds |
| Core load duration | 124.673 seconds |
| Sampled peak CUDA reserved memory | 21,692,940,288 bytes |

The [final adapter](checkpoint-0128/adapter.safetensors) is 3,792,296 bytes, SHA256 `4c9cd94ac04a45a4dad29d5e5efa05b6560af99c364d350657d5d5fa4f8ca3ce`. Its [original checkpoint manifest](checkpoint-0128/manifest.json) also identifies the optimizer/RNG file retained in the full original recovery. The final checkpoint was fixed in advance; no checkpoint selection is reported.

## What was trained

The original post-block-29 command-prefix GRU and observation-attention adapter remains FP32. Native Wan2.2 TI2V-5B preprocessing, all transformer blocks, head, unpatchify, foundation weights and recorded CUDA precision remain unchanged. Training begins from the original saved zero-output adapter and a fresh optimizer, rather than a previous trained checkpoint.

The main objective retains the original 128 saved noise/timestep draws and the paired starts `[0,8,32,49]` repeated 32 times. Closed and open branches use sequential batch size one and two half-loss backward calls. Only the four future latent frames enter the positive-context flow-matching MSE. The independently encoded observation is the exact clean input prefix; original target bytes remain unchanged.

At updates 1, 5, ..., 125, the additional start-0 objective uses one shared saved pure-noise future, the canonical observation, zero prefix token times and future token time 999. Two frozen feature extractions, one for each genuine text context, feed four adapter/head paths: closed/open commands with positive/negative text. For each command, the code forms `G = negative + 5 * (positive - negative)` through separate FP32 operations. It compares `-(G_open - G_closed)` against the untouched open-minus-closed target encoding using future-only MSE. This uses a declared ideal endpoint scale of 1; it does not equate that scale with the solver's literal first sigma.

On those 32 updates, the full objective is the main mean loss plus auxiliary loss with coefficient 1. The other 96 updates have only the main loss. All applicable backward calls precede one combined L2 clip at 1 and one original AdamW update: learning rate 1e-4, betas 0.9/0.999, epsilon 1e-8 and weight decay 0.01. The auxiliary branch explicitly trains with both positive and negative CFG contexts. No image generation is performed by the training runner.

These are eight overlapping windows from one fixed room. Only the start-0 pair uses the same canonical observed image with a wait/interact command difference. Later paired starts have equal commands and different observations. Training losses across different windows and noise draws are not a controlled learning curve. The separate assessment must test command response on fixed inputs; this staging record claims no such result.

## Evidence and reproducibility

The [parent](metrics/parent.json), [worker](metrics/worker.json), [training](metrics/training.json), [terminal](metrics/terminal.json), [plan](plans/plan.json), [admission](plans/executed-admission.json) and [final checkpoint](checkpoint-0128/manifest.json) retain their original bytes. [Before](metrics/core-before.json) and [after](metrics/core-after.json) foundation value records and the [weight-load report](metrics/weight-load.json) bind the 825 original tensors.

The [actual audit](audit/v2/report.json) verifies all 669 training-run files, totaling 2,303,196,556 bytes. It checks all main and auxiliary predictions, exact shared inputs and schedules, future-only losses, all nine checkpoint/optimizer/RNG bundles, saved combined clipped gradients and recorded resource samples. It does not rerun a foundation model or backward pass. Only combined post-clip gradients were retained, so separate main and auxiliary parameter gradients cannot be independently reconstructed. The AdamW arithmetic comparison is descriptive, not a CUDA replay. Sampled resource checks cover the recorded instants.

The [executed-source archive](reproducibility/executed-source.tar.gz) preserves all 88 exact dependency files and snapshots. [Source and input identities](reproducibility/source-and-inputs.json) bind their paths and hashes, the original saved initial adapter/RNG/noise files, cache/probe/text identities and external foundation identities. The unchanged source/input transfer manifest is retained separately. The complete 544,375,873-byte source/input transfer archive remains local; it is not duplicated here. The original dependency commit is `099eeae6268da852b0acb580258e4c168f900520`, with exact work-only source additions in the archive. Saved tensor bytes are the reproducibility reference; a seed alone does not assert identical Gaussian draws across architectures.

The [original full recovery index](recovery/original-index.json) covers 1,073 files totaling 2,524,879,425 bytes. Its compressed stream is 1,732,704,325 bytes with SHA256 `b640e364f8b121b212cdb481e8d947bbdc2ffbfc8dfcf53682cfc157a2435d02`; the index SHA256 is `61cd2d48160ee1db5b74a288328516850356cd1fc15b1bff4969c7f83d16c142`. [Local recovery reference](recovery/local-original-reference.json) gives the relative location and verification record. The full archive contains the earlier probe, cache and diagnostic needed by the auditor. Full raw publication remains undecided; this compact directory is not sufficient to rerun the retained-tensor audit.

## Retained invocation error and privacy

The [first audit invocation](audit/v1/report.json) failed before numerical checks because relative prior-input paths were resolved after a working-directory change. The unchanged auditor then passed with absolute arguments in fresh v2 output. The [correction record](audit/v2/invocation-correction.json) binds both original report hashes. The v1 failure is not discarded or presented as a training failure.

Only the `run` field in each of those two audit reports receives a prefix-only path replacement for staging. Every other byte is preserved. [Derivations](derivations.json) records original and staged hashes, the exact replacement rule, every copied-file origin and the original retained records. The original v2 report SHA256 remains `05b1d6ef0a32c4dfb228e7517ad27365240d75f105c007a18c779fc7e617f7c1`; source-bound assessment uses that original report, not the path-redacted derivative. No scientific source, metric, protocol or tensor byte changed.

The [privacy check](privacy-check.json) covers selected text and states its limits. No credential, account-lifecycle record or private key is selected. Original evidence remains untouched locally. This directory has not been committed, uploaded or published.

Original Worldline code and adapter weights use [Apache-2.0](licenses/WORLDLINE-APACHE-2.0.txt) with the [project NOTICE](licenses/WORLDLINE-NOTICE.txt). External Wan code, model and VAE retain [Apache-2.0](licenses/WAN-APACHE-2.0.txt) and their [NOTICE](licenses/WAN-NOTICE.txt). Original procedural Atrium imagery and metadata use [CC0](licenses/ATRIUM-CC0.txt). No external foundation checkpoint is included. No new architecture novelty or frontier-model parity is claimed.
