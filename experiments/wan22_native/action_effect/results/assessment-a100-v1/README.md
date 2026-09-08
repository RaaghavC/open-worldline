# Action-effect assessment: a small numerical improvement

**The new adapter passed the predeclared comparison on all four held-out Gaussian noise inputs. Its predicted action contrast had 0.513–0.532% lower mean squared error than the previous 128-update adapter. This assessment does not test whether the door visibly opens or the camera turns.** This is local publication staging; no release or public download is claimed here.

| Held-out noise | Zero-adapter MSE | Previous128 MSE | Effect128 MSE | Improvement over previous128 | Effect128 alignment cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.1963713394 | 0.1964000009 | 0.1953865167 | 0.5160% | 0.080930 |
| 1 | 0.1963713394 | 0.1964004828 | 0.1953932767 | 0.5128% | 0.080254 |
| 2 | 0.1963713394 | 0.1964164425 | 0.1953999489 | 0.5175% | 0.079888 |
| 3 | 0.1963713394 | 0.1963830852 | 0.1953382364 | 0.5320% | 0.083432 |

Each noise had to yield lower error than **both** the zero adapter and previous128, and positive target alignment. All four passed without changing those inequalities. The predicted contrast RMS was 0.0185–0.0187, compared with the target contrast RMS of 0.4431. The response remains small. These are four noise inputs on one scene used in training, not four unseen scenes.

The [full-precision summary](summary.json) is derived from the [independent saved-output audit](audit/report.json). The audit passed in 2.455 seconds without running a model. All 66 saved predictions, input/context/command/checkpoint identities, 825 unchanged foundation tensors, completion records, and sampled resource limits passed. The parent runtime was 248.156 seconds; the recorded worker peak CUDA reservation was 21,581,791,232 bytes. The unchanged limits were 900 seconds, 60 GiB CUDA reservation, 48 GiB aggregate host memory, and at least 8 GiB available CUDA and host memory.

## Exact comparisons and retained regressions

All 12 repeated predictions from the previous 20-output run were **file- and tensor-byte identical**, with maximum absolute and RMS differences of zero. Those are the zero/previous128 checkpoints on both original time-506 training corruptions and both commands with both text contexts on the original time-999 visual input. The new assessment reused frozen feature bundles; the measured equality applies to those 12 calls.

The original time-506 main-objective check slightly regressed against previous128:

| Own original corruption | Previous128 MSE | Effect128 MSE | Relative error increase |
| --- | ---: | ---: | ---: |
| Closed | 0.1479280550 | 0.1479366282 | 0.00580% |
| Open | 0.1615609700 | 0.1616613961 | 0.06216% |

Those two loss inputs contain their own future targets and are not isolated command comparisons. On the separate original time-999 shared visual input, predicted action-contrast RMS rose from 0.004942 to 0.018254, and alignment cosine changed from approximately -0.000087 to 0.084282. That single velocity measurement is not a generated video or a successful action demonstration.

## What was measured

The final adapter was fixed in advance: [effect128-final.safetensors](checkpoints/effect128-final.safetensors), SHA256 `4c9cd94ac04a45a4dad29d5e5efa05b6560af99c364d350657d5d5fa4f8ca3ce`. It has 947,712 FP32 parameters. The previous128 and zero checkpoint bytes are also retained in the prepared-input archive. There was no optimization, image sampling, or checkpoint selection in this assessment.

Each held-out input uses the canonical independent first observation and one previously saved Gaussian future. Closed/open commands differ only in the first wait/interact channel. The same input and genuine positive/negative text are reused across all three checkpoints. For each command the code computes `G = negative + 5 * (positive - negative)` in FP32, then the predicted clean contrast `-(G_open - G_closed)`. The target is the untouched open-minus-closed encoded video difference. Only the four future latent frames enter the FP64 score. This defines an ideal pure-noise endpoint scale of 1; it does not replace the pinned solver's literal sigma with 1 or claim that this is an ordinary time-999 training corruption.

Four held-out noises × three checkpoints × two commands × two texts produce 48 saved heads. The two original time-506 corruptions produce six more heads; the original time-999 input produces 12 more. Twelve frozen feature extractions serve all 66 heads. Both text contexts enter the new auxiliary training objective; the unchanged main flow-matching objective uses positive text. Targets enter scoring, not the feature or prediction inputs.

## Files and reproducibility

The [plan](plans/plan.json), [executed admission](plans/executed-admission.json), [actual-plan review](plans/actual-plan-review.json), [exact saved-file binding](plans/audit-binding.json), [parent](metrics/parent.json), [result](metrics/result.json), and [terminal](metrics/terminal.json) retain their original bytes. Source-bound [CPU checks](preflight/assessment-cpu-report.json), [program review](preflight/assessment-independent-review.json), [auditor CPU checks](preflight/auditor-cpu-report.json), and [auditor review](preflight/auditor-independent-review.json) remain separate from actual measured evidence.

One [source archive](reproducibility/executed-source.tar.gz) retains all 83 exact executed dependencies, including the reviewed work-only method and the original dependency commit `099eeae6268da852b0acb580258e4c168f900520`. A separate [prepared-input archive](reproducibility/prepared-inputs.tar.gz) retains the other 149 bound members, including the four noise tensors, original input/text/command bytes, three checkpoints and prior evidence. [Source/input inventory](reproducibility/source-and-inputs.json) records every archive member's size and checksum. All numerical input and source bytes remain exact.

Two earlier training-audit records inside the input archive contain a path-only publication change: only the local prefix in their `run` field becomes `[WORKSPACE]/`. [Derivations](derivations.json) records the precise field, both file hashes and every copy origin. No other bytes in those reports change. The original plan and binding still refer to the original reports. This published input derivative cannot satisfy that original byte-identity preflight until those two original records are restored from the retained original recovery. No gate has been weakened or replaced. The actual 66-output audit and review are copied exactly.

The [complete original recovery index](recovery/original-index.json) covers 320 files and 294,746,992 raw bytes, including all 66 FP32 output tensors and the entire current-run folder. The two transport parts total 169,239,483 bytes, with stream SHA256 `48a07e2e849c0b8d606a3433d709975e2c08f9d7a9572914b85ba176886074eb`; the index SHA256 is `7cbe124f446623d4d3fa87cd8364478f0cb09cb040bb47f5d86bc4dbb710b0e8`. The [local reference](recovery/local-original-reference.json) preserves the relative location and exact verification record. Complete raw transport parts remain outside this compact staging package. No public URL is invented. The current-run auditor verified 312 files; the other eight recovery files are the separate assessment lifecycle records and packer.

The historical comparison uses the separately retained original20 run at the workspace-relative location recorded in the audit pins. The full earlier training recovery is also retained separately. The compact package does not duplicate those full recoveries or the external foundation weights. [Payload manifest](payload-manifest.json) covers every staged file other than itself. [Privacy scan](privacy-check.json) records the selected-text and archive-member checks and their limits; no credential files were read.

Original Worldline code and adapter weights retain [Apache-2.0](licenses/WORLDLINE-APACHE-2.0.txt) and the [project NOTICE](licenses/WORLDLINE-NOTICE.txt). External Wan code and weights retain [Apache-2.0](licenses/WAN-APACHE-2.0.txt) and their [NOTICE](licenses/WAN-NOTICE.txt). Procedural Atrium data retain [CC0](licenses/ATRIUM-CC0.txt). This experiment makes no new model architecture, Genie, broad quality, action-success, or generalization claim.
