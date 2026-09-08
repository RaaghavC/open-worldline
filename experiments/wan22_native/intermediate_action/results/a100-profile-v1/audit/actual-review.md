# Actual intermediate CUDA profile: metadata review

Status: retrieved metadata is internally consistent. **Full raw output verification is pending the complete recovery.** No model replay or tests were run for this review.

The current files are under `/Users/rc/Documents/Codex/2026-09-07/re/work/intermediate-action-profile-live-summary-v1`:

| File | SHA256 |
| --- | --- |
| parent.json | `432c4c2a7e36311e53c13da7b2aa34f2ea0c9cf71ddb4a535a32a9532509254e` |
| result.json | `d51c6fab4b948ea0d3a35d1dec88f12ebbfc94e5cbf426b409f36621c1f4aaf4` |
| terminal.json | `b001cdc0b4581be9021533a8409d495ffc35ebab960a6c548f1754a35af9131a` |
| memory.jsonl | `392629603159dcd439dee5d5decb6628562c35eac18868deb66aaf39c1d96d11` |

The parent hashes match the retrieved result/terminal bytes. All three report success/normal exit within the same900-second limits, with no terminal cleanup error. Source maps, exact plan `09cbbad19ff0f4ca3caadf8508e577b84ede72d8d208870456288ef163e18e7d`, selection and local admission `cd119728c9e64dd6c7db8ff47703493946c0249b4c5176acdea9fb4a538f504b` match the reviewed prepared packet.

All16 parity rows cover both blocks28/29, contexts, commands and full/cached paths exactly once. They report exact_equal=true, matching native/bridge tensor hashes and zero max/L2 errors. Four updates have the agreed main/auxiliary counts, lambda1 and unchanged input identities across placements. Reported total objectives equal half each main branch loss plus auxiliary loss. Both first updates report zero recurrent gradient; second updates report9.0638670e-5 at28 and7.9693527e-5 at29. Combined gradient norms remain below1 and before/after clip values are equal. These establish reported gradient flow, not useful learned control.

The worker reports all825 original foundation values unchanged, with identical declared hashes for before/after-parity/after-block28/after-block29 record files. Rotary before/after hashes match. These record contents and every raw comparison remain to be independently opened after recovery.

All1,367 retrieved worker samples contain finite nonnegative timing and integer memory fields, ordered below900seconds, and pass48GiB host/60GiB reserved CUDA/8GiB available-memory limits. Sampled host RSS peaks2.425125GiB and reserved CUDA25.431641GiB; minimum available GPU53.301697GiB. Parent terminal reports combined RSS peak3,202,613,248B; its full sample series has not arrived. Allocated/reserved values are separately timed reads and no atomic ordering relation was required.

Allocator training peaks are25.012216GiB allocated/25.431641GiB reserved for28, versus20.093074GiB allocated/20.302734GiB reserved for29. The0.25-second samples saw only22.469615GiB allocated and must not replace the allocator peak. Native suffix backward was feasible under the declared limits in the reported two-update run; these data do not establish a long-run peak or an all-workload bound.

Parent elapsed353.189368seconds, worker348.416377seconds. Loading128.511055seconds plus four full foundation hash passes200.972806seconds account for329.483861seconds. Training segments were4.089295seconds at28 and3.872722seconds at29. Ordered single-run timing is descriptive; no isolated speedup or longer-training estimate is claimed. The two updates use different saved noise/k draws, so their increasing main losses are not a matched-input learning curve.

## Post hoc analyzer read

Source `/Users/rc/Documents/Codex/2026-09-07/re/work/intermediate-action-profile-publication-v1/analyze_saved.py`, SHA `687c1201570bd5c4bc5ddd69c4ea2fabdf55d3ca0272f31cc44fd6d20e8e9c09`; imported analytical helper SHA `58ee2088f08054e80cf4ddc7d8f6503a499031512f5506a21b8ed6811686815a`. The report now records both dependencies, closing the only reproducibility omission I found.

The historical control selects the exact original first-update closed/open main predictions, four auxiliary predictions and after-clip gradient bundle for comparison with block29. It keeps equality and differences descriptive. These are correctly selected historical inputs/initial state; actual equality is not yet established by this source review.

The new response selects auxiliary-0002, which was emitted before optimizer update2 and therefore reflects exactly one completed update. It uses saved noise_0004 and the shared pure-noise endpoint, not the k265 main corruption. FP32 guidance order, open-minus-closed clean-contrast sign, future-only scalar fit and spatial patch measurements match the reviewed producer. The target-fitted scalar is labeled post hoc, and no video-quality/control claim is made. The analyzer is layered on the full audit: it does not itself replace terminal/admission/foundation validation. No remaining material math/interpretation issue found.

## Pending raw verification

After explicit recovery confirmation: verify original file inventory and parent/result/terminal identities; open all18 parity prediction files and recompute16 equalities; open both placements'24 main/auxiliary predictions, four gradient bundles and six checkpoints; confirm each saved input/checkpoint/draw identity; compare all four825-record foundation maps with original loaded records; verify raw parent/worker samples. Preserve this metadata-only note and add a separate raw-result record. No actual raw-output pass is asserted here.
