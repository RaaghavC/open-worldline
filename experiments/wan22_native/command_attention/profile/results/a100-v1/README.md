# First A100 command-attention resource profile

The native 1248 × 704, 17-frame profile completed 24 recorded exact zero-initialization comparisons and two optimizer updates. Its [saved-file audit passed](records/audit.json). This measures resource use and saved numerical consistency. It does not establish camera control, door control or generated-video quality. The planned 512-update run did not start on this pod.

| Measurement | Recorded value |
| --- | ---: |
| GPU | NVIDIA A100-SXM4-80GB |
| Trainable controller parameters | 4,936,448 |
| Worker elapsed time | 358.207319 seconds |
| Parent elapsed time | 361.986703 seconds |
| Original foundation load | 161.059168 seconds |
| Mixed FM/CFG update, including profile recording | 3.154871 seconds |
| FM-only update, including profile recording | 1.318617 seconds |
| Maximum reserved CUDA memory | 56,461,623,296 bytes |

The [exact producer metrics](records/profile/result/metrics.json) have SHA256 `f66110f4dcde1df703cb34b35d542a65653f4069a7b57682fb70402330b4a510`. The first update accumulated two sequential flow-matching backwards and a joint four-branch CFG auxiliary backward before one AdamW step. The second used flow matching only. Both complete 58-tensor gradient bundles were retained and checked as finite. The three saved maps of 825 foundation-value hashes, before the profile, after parity and after the updates, match the pinned original records.

The audit independently recomputed losses from four retained main predictions and four auxiliary predictions, and checked both gradient bundles and the initial/final controller tensors. Its exact report SHA256 is `0fbc81ffcde580b3917f44c43ab377cfc12796afdb9536f7d8539120c83f9964`.

The producer discarded the 24 full/cached controller velocities after comparison. The audit checked their recorded hashes against the two retained native-reference arrays; it could not compare the discarded arrays again. It did not replay native forwards, backward passes, optimizer moments or CUDA execution. Foundation weights are absent from the recovery, so the audit compared saved hash records rather than hashing the underlying foundation again. Resource sampling does not capture every instantaneous allocation.

## Why 512 updates did not start

The [recorded forecast](records/forecast.json) was 1,573.155046 seconds under the fixed formula `1.2 × (384×FM +128×mixed +16×prefix +96×cached_head +load +before_hash +after_hash) +120`. It fit the 1,800-second training cap, but only 1,336.297815 seconds remained after reserving 600 seconds on the original lease. The recovered index shows that neither the training result nor training dispatch directory existed. The two-step profile controller is not the declared final 512 checkpoint.

The [fixed cleanup plan](records/cleanup/plan.json) ended at 23:43:08.984135 UTC on September 8, 2026. The [completion request](records/cleanup/completion.json) was saved at 23:27:38.390491 UTC after recovery. The [cleanup receipt](records/cleanup/cleanup-result.json) confirms deletion through an acknowledged delete, an explicit GET404 and absence from a successful complete pod list. This receipt concerns the first profile pod only.

## Evidence included here

The [publication inventory](publication.json) maps 20 existing small records and nine existing auditor files to their original hashes. These are exact copies, without redaction. Owned, nonsecret `/workspace` paths, the task's pod ID and process IDs remain in the records; no private keys, credentials or local user paths are included. All 16 producer source/configuration hashes recorded by the profile match the repository files listed in that inventory. The auditor's pre-execution freeze remains unchanged; the later actual report is the result linked above.

The [recovery index](records/recovery/index.json) and [verification receipt](records/recovery/recovery-verified.json) cover 63 files totaling 114,158,423 bytes, transported as five parts totaling 72,262,339 bytes. The unchanged original source/input transfer is retained separately and identified by its [binding](records/recovery/original-transfer-binding.json). Raw tensors, controller files, memory logs and transport parts are preserved outside Git. Their public release is pending; this directory does not provide a raw download or enough data to repeat the full audit by itself.

The [exact auditor source](audit/audit.py) uses Python and NumPy, with no Torch or model loading. Its [source notes](audit/README.md) list the checks and required original transfer plus full recovery. The [two small CPU fixtures](audit/cpu-report.json) passed with Python 3.11.9 and NumPy 1.26.4. For a later full recheck, supply the recovered 63-file tree, its verification receipt and the original 901-file source/input tree to this copied auditor. No new audit or model execution was performed when copying these records into the repository.
