# Action-effect visual result: door and camera control failed

**The door stayed closed in both generated branches, and the requested camera turn did not occur.** Review covered all 34 generated frames and all four full-resolution endpoints. The earlier four-noise assessment improved its endpoint-contrast error by 0.513–0.532%, but this small numerical gain did not produce the requested visible action response.

The [parent visual review](visual-review/parent-all-frame-review.json) records the failure. The [independent artifact audit](audit/report.json) passed: the saved inputs, checkpoint, sources, all 100 latent states, first-step velocities, decoded frames and frozen model values were intact. Artifact integrity and successful control are separate questions.

## Exact generated previews

The original GIFs are retained byte-for-byte: [wait branch](clips/closed.gif) and [interact branch](clips/open.gif). Each shows a coherent beige room and brown door with small changes in framing. Both sequences contain 17 frames; they retain the original 120 ms per-frame playback, totaling 2.04 seconds. Playback timing is not model throughput.

See the [all-34-frame contact](comparison/all-34-frames.png) for every generated frame, displayed at 25% with nearest-neighbor resizing and no enhancement. The [checkpoint/target comparison](comparison/checkpoint-target-comparison.png) shows selected frames from effect128, previous128, previous16 and the recorded targets at 50%. Four full-resolution PNG copies retain their original 1248×704 bytes:

| Branch | Initial frame | Final frame |
| --- | --- | --- |
| Wait | [Original PNG](comparison/full-frame-previews/closed-initial.png) | [Original PNG](comparison/full-frame-previews/closed-final.png) |
| Interact | [Original PNG](comparison/full-frame-previews/open-initial.png) | [Original PNG](comparison/full-frame-previews/open-final.png) |

Only the first command differs: wait versus interact. The next 15 commands are left turns, requesting 112.5 degrees in total. The recorded target opens the door after interaction and changes view during those turns. Generated images retain a closed door and nearly stationary framing. There is no visible door-control or camera-control success in this result.

## Target correspondence

Mean future-frame RGB absolute error uses the `[0,1]` scale and excludes conditioned frame0. Lower is closer to the recorded target, but it is not a measure of successful control by itself.

| Result | Wait/closed target MAE | Interact/open target MAE |
| --- | ---: | ---: |
| Effect128 | 0.1835658690 | 0.1934661785 |
| Previous128 | 0.1835971495 | 0.1955583772 |
| Previous16 | 0.1835505914 | 0.1955229214 |
| Repeat canonical starting image | **0.1789626381** | **0.1903559489** |

**Both generated branches are worse than repeating the starting image.** Effect128 reduces target error slightly against previous128, particularly for the interact branch; its wait-branch error is slightly higher than previous16. Camera mismatch and lighting affect these numbers. The requirement that both branches be strictly closer to their own target than the opposite target remains **0/16** future frames for every checkpoint, both over the full frame and in the paired-target difference region. The latter includes lighting differences and is not a segmented door mask.

The [full comparison report](comparison/report.json) retains every frame identity, all per-frame scores, exact recomputation of both historical analyses, and source hashes. Its [interpretation](comparison/interpretation.md) distinguishes pixel differences from control success. The [summary](summary.json) records the target table, resource results and outcome. The separate [numerical assessment summary](assessment/summary.json) retains the small endpoint-contrast improvement and earlier main-loss regressions.

## Fixed evaluation and verified execution

The [final adapter](checkpoint/effect128-final.safetensors) has SHA256 `4c9cd94ac04a45a4dad29d5e5efa05b6560af99c364d350657d5d5fa4f8ca3ce`. It is the predeclared final checkpoint of the 128-update auxiliary experiment. No training or checkpoint selection occurred during this visual evaluation.

Both arms use the exact earlier saved first observation, Gaussian noise, genuine positive/negative text and command arrays. The unchanged native sampler uses 50 UniPC steps, shift 5, guidance 5 and 17 frames at 1248×704. Both CFG contexts receive the corresponding command sequence. The effect adapter's main training objective uses positive text; its auxiliary objective uses both CFG contexts. That training distinction is retained in the new-family metadata without changing the numerical sampler.

The [plan](plans/plan.json), [executed admission](plans/executed-admission.json), [actual-plan review](plans/actual-plan-review.json), [CPU report](preflight/visual-cpu-report.json), [independent source review](preflight/visual-independent-review.json), [parent metrics](metrics/parent.json), [core metrics](metrics/core-metrics.json) and [decoder metrics](metrics/decode-metrics.json) retain exact original bytes. The parent completed in 459.635 seconds. The original 1800-second limit and 60 GiB CUDA, 48 GiB host and 8 GiB free-memory requirements remained unchanged. Resource checks describe the recorded sample instants.

The audit checked all 303 current-run files without changing them. It verified all 825 original foundation tensor records before/after sampling and the original 196 VAE records. All 50 states per arm retained their exact observed prefix. Only first-step raw velocities were saved; later prediction counts rely on the reviewed source and recorded counters. All 34 raw RGB frames match their authoritative PNGs. No model or solver replay was used for the independent artifact or RGB audits.

## Source, input and raw-recovery records

The [executed-source archive](reproducibility/executed-source.tar.gz) retains 72 exact runtime/test dependencies in one archive. The [prepared-input archive](reproducibility/prepared-inputs.tar.gz) retains 29 other preparation files, including the original tensor inputs, final checkpoint and audited training metadata. [RGB analysis dependencies](reproducibility/rgb-analysis-dependencies.tar.gz) retain the unchanged target/score readers. [Source/input inventory](reproducibility/source-and-inputs.json) lists every archive member and checksum. The original repository dependency commit is `099eeae6268da852b0acb580258e4c168f900520`; work-only additions are identified by their exact source hashes. No external foundation weights are included.

The [original recovery index](recovery/original-index.json) covers 338 files, 786,411,030 raw bytes and a 708,595,902-byte transport stream. It includes the complete current-run folder, all 100 saved latent states, all raw frame shards and PNGs, and the other recovery records. Original index SHA256: `9897bddfcc99b2ce0db29e9a4b7dbd1e7f88f51f0c671d07fee2e2db82a6866e`; original stream SHA256: `d597d212947410526c657b8e32a88fcc27dc589cdbbbebfc41456aabdda3ff14`. The [local original pointer](recovery/local-original-reference.json) preserves its location and verification record. Large raw transport parts remain outside this compact package.

A separate public-archive derivative retains all 338 files: 337 byte-exact, with only the original local prefix in `training-audit.json`'s `run` field changed to `[WORKSPACE]/`. The [original/public member mapping](recovery/public-derivative/member-mapping.json), [public index](recovery/public-derivative/index.json) and [derivation record](recovery/public-derivative/public-derivation.json) identify both versions exactly. The prepared public stream is 708,835,028 bytes in 45 parts, SHA256 `c762b5aa63f2ef19d0e1d491c75008f957219009f48acebcddd87eab0c5bfd04`. Its index SHA256 is `66a3eb0ea3a470e27de1d9443ded653ee92131c805dbdad20c93c0c3a2f3b2fa`. Its separate owner verified a fresh local download. This staging task does not create a release or claim an available public URL.

The compact package has two disclosed path-only derivatives: that training-audit field inside the prepared-input archive, and the three local operational path fields in [audit/binding.json](audit/binding.json). [Derivations](derivations.json) records original and staged hashes, exact changed fields and every copy origin. Sources, metrics, protocol values, tensor bytes and image bytes are unchanged. Original plans and audit records retain their canonical original hash references. Path-redacted derivatives do not satisfy those original execution preflight checks; no original hash is reconstructed or falsely claimed. The original full evidence remains untouched. Public frame inspection and saved-score recomputation are distinct from execution admission.

[Payload manifest](payload-manifest.json) verifies every staged file other than itself. [Privacy scan](privacy-check.json) records its bounded checks; no credential or account files were selected. This directory is prepared locally only, with no repository or GitHub writes by its staging helper.

Original Worldline code and adapter weights retain [Apache-2.0](licenses/WORLDLINE-APACHE-2.0.txt) and the [project NOTICE](licenses/WORLDLINE-NOTICE.txt). Wan code and weights retain [Apache-2.0](licenses/WAN-APACHE-2.0.txt) and their [NOTICE](licenses/WAN-NOTICE.txt). Procedural Atrium data retain [CC0](licenses/ATRIUM-CC0.txt). No new architecture novelty, Genie parity, successful action control or generalization is demonstrated.
