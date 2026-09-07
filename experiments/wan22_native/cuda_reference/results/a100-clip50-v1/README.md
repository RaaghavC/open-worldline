# Wan2.2 native CUDA clip: completed, visual quality failed

The external pretrained Wan2.2 TI2V-5B completed 50 CPU UniPC steps and 100 native CUDA predictions on one NVIDIA A100-SXM4-80GB. It produced all 17 prescribed frames at 512 × 288, including one conditioned reconstruction and 16 generated futures. The future frames contain severe prismatic room distortions.

The review covered all 17 original frames, using the contact sheet and the seven original frames omitted from that sheet. Frame 0 retains a sharp room and doorway. Frame 1 introduces colored, silvery foreground shapes at the lower left. From about frame 6, distortions spread across the floor and right wall. Frames 11–16 contain broad neon bands and warped structures. This is the same qualitative failure category seen in the earlier Mac clips. It does not establish pixel identity, numerical equivalence or the cause of the failure.

[View the exact contact sheet](recovered-clip-v1/results/clip-run-v1/decode/result/comparison.png), [all original PNG frames](recovered-clip-v1/results/clip-run-v1/decode/result/frames/) and the [retained GIF](recovered-clip-v1/results/clip-run-v1/decode/result/preview.gif).

| Measured interval | Seconds |
| --- | ---: |
| Core loading and its checks | 221.542057 |
| Fifty-step sampling loop | 198.348099 |
| Complete core worker | 422.342891 |
| Native FP32 VAE loading and its checks | 37.150694 |
| Native VAE decode | 1.127770 |
| Complete decoder worker | 45.012308 |
| Whole parent run | 476.444111 |

The worker totals include their load and operation intervals, so these rows must not be added together. Setup and weight download are outside the parent run. The admission estimate was 562.973035 seconds; the actual parent run completed within its fixed 900-second limit. Completion and image quality are separate outcomes.

The audit's sampled peak CUDA reserved memory was 20,430,454,784 bytes for the core and 8,153,726,976 bytes for decoding. Sampled combined host RSS peaked at 2,778,460,160 and 4,437,463,040 bytes for those separate stages. These values are sampled and overlap memory categories; they are not additive. The reported host available memory is a system figure, not a cgroup quota measurement.

The GIF requested 125 ms per frame (8 fps) but encodes 120 ms per frame, about 8.33 fps, because GIF delays use 10 ms units. These are playback timings, not model throughput. The table reports the actual measured execution intervals.

## What the retained audit verifies

The [independent saved-file audit](independent-audit/report.json) passed source, input, admitted-pair and completion checks. It matched all 825 original core tensor records and 196 original VAE tensor storages, checked the 48-channel normalization against the retained source, and verified all 50 saved step hashes and exact post-step image prefixes. The source makes two predictions per step with the same restored image prefix and appropriate token times. Separate tensors for all 100 call inputs were not retained.

The decoder produced finite `[1, 3, 17, 288, 512]` RGB. All 17 PNGs match the declared rounding of retained clamped FP32 pixels, and the contact-sheet crops match their originals. The decoder cache-clear record and both successful terminals are preserved. The [audit source](independent-audit/audit-cuda-clip.py) is included unchanged; its historical local relative paths must be supplied by a separate wrapper if rerun after relocation.

The run used the exact saved initial observation, noise and genuine text from the [preceding pair evidence](../a100-pair-v1/README.md), pinned original FP32 parameter storage, native CUDA BF16 autocast and FlashAttention 2, CPU UniPC50, shift 5 and CFG 5. It contains no training or original action adapter. This reduced spatial and temporal shape is not the model's configured 720p, 121-frame example.

## Complete evidence contents

All 146 recovered files are included unchanged under [recovered-clip-v1](recovered-clip-v1/): the actual run and preceding plan, exact inputs and text, CPU preflight reports, executed source snapshots, original load records, all 50 intermediate latents, final latent, clamped FP32 decoded RGB, all original images, previews, terminal/resource records and empty logs. No frame or step was selected for removal.

The transfer archive was 66,228,413 bytes with SHA256 `a1d6e1840a8ee0a94ab5b10dc6d860eeafccd36348a829b7b639addf80cff2d1`. Its remote and recovered hashes matched. The archive itself remains a separate local transfer artifact; its exact extracted files are in this payload. The [recovery record](recovery/clip-recovery.json), [copy provenance](provenance.json), [text-scan record](privacy-scan.json) and file manifest account for the contents. Safe `/workspace` and `/opt` paths remain unchanged in measured launch and report files, preserving their original hashes. No model checkpoint, environment dump, account balance, key or payment data is included.

Run `python3 verify-payload.py` from this directory to verify the indexed file identities without a model or cloud call. This is a prepared local publication payload; a GitHub upload is not asserted here. The pair link is intended for the sibling public `a100-pair-v1` result directory.

## Conditional next test and limits

The [next-diagnostic note](planning/next-diagnostic.md) proposes a separately prepared test of spatial size, keeping 17 frames and the existing conditioning settings. It remains unimplemented, unexecuted and unadmitted under the present runner. The [original pre-completion note](planning/next-diagnostic-original.md) and [exact publication edits](planning/publication-changes.json) preserve its history. Numerical planning estimates were not presented as measured runtime.

The task owner confirmed that the paid Pod was deleted using an explicit GET 404 and a successful complete-list absence check. Cleanup and credential-revocation records are maintained separately. No final billing amount is inferred or published here.

This is an external pretrained reference diagnostic with a recorded visual failure. It establishes no original Worldline model, long-term world memory, Genie 3 parity or novel research advance. The CUDA result narrows one question: Mac-specific execution was not necessary for this failure category at this reduced shape. Shared shape, conditioning and model/input behavior remain possible factors; this run does not identify the cause.
