# First numerical probe failed before an optimizer update

The first admitted two-update numerical probe exited with an MPS pooling error. It completed **zero optimizer updates** and generated no images. This result does not establish full-model gradient feasibility, unchanged final foundation values, action control or better video quality. The parent admission explicitly retained a failed visual foundation and authorized only the numerical probe.

The first forward reached the original adapter's observation pooling operation. PyTorch's MPS adaptive average pool rejected the 18-by-32 input mapped to 4-by-8 because 18 is not divisible by 4. The error and complete stack trace remain in [worker.log](worker.log). The follow-up fix is a separately tested implementation of the same overlapping adaptive bins. This retained run is not repaired or relabeled by that change.

| Retained measurement | Value |
|---|---:|
| Worker elapsed time | 109.263940 seconds |
| Parent process elapsed time | 112.365922 seconds |
| Verify/load external frozen foundation | 87.387165 seconds |
| Hash all 825 loaded foundation parameters | 16.482218 seconds |
| Completed optimizer updates | 0 |
| Sampled peak process RSS | 0.552216 GiB |
| Sampled peak MPS active memory | 9.605913 GiB |
| Sampled peak MPS driver memory | 10.131317 GiB |
| Minimum sampled available system memory | 3.398727 GiB |

The failed forward has no completed stage timing. Elapsed time includes checks and initialization; the two listed stages do not cover the entire run. Memory samples occur about every half second and may miss shorter peaks. The limits remained 900 seconds, 18 GiB and 2 GiB available memory. This was an operation-support error rather than a recorded watchdog stop.

The [artifact audit](artifact-audit.json) verified 82 exact source snapshots across the parent and worker, all 825 before-training value records against the loader, saved noise and prompt identities, and the initial 947,712-parameter FP32 adapter plus empty optimizer state. The initial output projection is zero. [checkpoint-0000](result/checkpoint-0000/manifest.json) contains only fresh initialization, optimizer metadata and RNG state. There is no trained adapter checkpoint, no checkpoint 1, and no completed final foundation hash comparison. The metric `base_unchanged` remains null.

[result/metrics.json](result/metrics.json), [terminal.json](terminal.json), the parameter files, RNG state, draws, prompt context, memory samples and all measured source snapshots preserve their executed bytes. [publication.json](publication.json) records both the original run hashes and published hashes. Only operational paths in `launch.json` and the workspace prefix in `worker.log` were transformed. Every transformation is listed. The raw work directory remains unchanged.

[admission/executed.json](admission/executed.json) is the byte-exact parent decision. [admission/portable.json](admission/portable.json) changes only the three evidence image paths, with its own hash and explicit transformation record. Worker metrics retain the original admission identity. The admission permits neither a changed-source retry nor the fixed16 pilot. External model weights and the local raw capture are not bundled; their launch fields are explicit placeholders.

Our original initialization and training code use Apache-2.0. The genuine prompt context uses the attribution in the [text-cache record](../../../../wan_adapter/text_cache/native-results/README.md). The external frozen foundation remains attributed in the native [NOTICE](../../../NOTICE). No external foundation parameter file is included here.
