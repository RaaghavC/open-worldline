# Checkpoint 128 evaluation: both action controls failed

**The door stays closed in both clips, and the requested 112.5-degree left turn is absent.** The room remains recognizable with natural colors, soft surfaces and small edge/texture changes. The [parent review](visual-review/visual-review.json) inspected all 34 frames in complete contact sheets and the initial/final raw images. The [saved-artifact audit](audit/visual/report.json) passed, but valid files and coherent appearance do not establish action control.

The original 947,712-parameter adapter is the fixed [checkpoint 128](../fixed128-a100-v1/checkpoint-0128/adapter.safetensors), SHA256 `ac1160b22531804c473746d2fb8ee9edd75710e0e43d6f1eabc86d268c864996`. Both branches use exactly the same initial observation, saved noise and text. Their commands differ only at the first transition, wait versus interact, followed by the same 15 left turns of 7.5 degrees. The native foundation, adapter and VAE remain unchanged during evaluation. Training used positive text only. The fixed sampler applies commands to both positive and negative CFG calls; negative-text action conditioning was not trained separately.

| Rendered pair | Retained evidence |
| --- | --- |
| Two clips | 17 frames each, 1248×704 |
| Complete parent time | 398.633 seconds for both clips, including loading, hashing and decoding |
| Sampling | Native 50-step UniPC, shift 5, CFG5; 100 model predictions per arm |
| Frozen foundation | All 825 original tensor values verified before/after |
| Observed prefix | Exact at all 50 saved steps in each arm |
| Requested interaction and camera movement | Both failed the full visual review |

[Wait preview](previews/closed.gif) · [Interact preview](previews/open.gif) · [All wait frames](visual-review/closed-all17.png) · [All interact frames](visual-review/open-all17.png). GIF playback speed is a display setting, not generation throughput. The full raw archive includes all 34 native PNGs and FP32 RGB frames, every saved latent state and initial guided velocities.

## Matched images and original targets

The [six-row comparison](rgb-comparison/checkpoint-comparison.png) shows both checkpoint 128 branches, both checkpoint 16 branches and both original targets at fixed frames 0,1,2,4,8,12,16. Display uses 50% nearest-neighbor resizing; no image enhancement is applied. The [numeric reader](rgb-comparison/report.json) scores every frame. Checkpoint 128 is slightly farther from each original target than checkpoint 16, and both are worse than simply repeating the starting image on mean future-frame RGB error:

| Mean absolute RGB error over future frames 1–16, range 0–1 | Wait | Interact |
| --- | ---: | ---: |
| Checkpoint 128 versus own target | 0.1835971495 | 0.1955583772 |
| Repeat starting image versus own target | 0.1789626381 | 0.1903559489 |
| Checkpoint 128 target error minus checkpoint 16 error | +0.0000465581 | +0.0000354559 |

Both checkpoints score 0/16 on the strict test requiring both branches to be closer to their own truth than the opposite truth, with ties failing. This applies to whole images and the paired-truth difference region. That region includes indirect lighting, not just door pixels. Camera alignment strongly affects these comparisons, and the original camera turns away from the door later in the sequence. These numbers describe correspondence, not general visual quality.

Targets retain exact original frame 0–16 mapping. The original 512×288 RGB is LANCZOS-resized to 1252×704 and centered-cropped by 2 pixels on each side to 1248×704. Only the derived closed-start frame 0 is canonicalized to the original open-start frame 0 before preprocessing, matching training and prior sampling. The raw captures remain unchanged, including their 18 one-level channel differences. Frame 0 is a conditioned reconstruction and is excluded from future means. [Reader interpretation](rgb-comparison/interpretation.md) states the scope and limits.

## Twenty saved predictions

A separate unchanged numerical run compared checkpoints 0,16,128 on the original k=506 corruption and one shared k=999 sampling input. The [v2 independent audit](audit/comparison20-v2/report.json) verifies all 20 predictions, all 825 original core identities, source/input/checkpoint hashes and recorded resource caps. Four checkpoint 0/native comparisons remain bit-exact. Separately, [all 14 repeated reference prediction files](audit/comparison20-v2/original14-reproduction.json) are byte-identical to the earlier diagnostic, with complete call identities matched. No training or image generation occurred inside this diagnostic.

| Fixed seen-example future flow MSE, FP64 reduction | Checkpoint 0 | Checkpoint 16 | Checkpoint 128 |
| --- | ---: | ---: | ---: |
| Closed example | 0.148484407142 | 0.147919876994 | 0.147928055027 |
| Open example | 0.162152676773 | 0.161552561582 | 0.161560970040 |

Checkpoint 128 is descriptively worse than 16 by 0.00552869% and 0.00520478% on these two fixed cases, while both remain slightly below initialization. This is not a held-out test. On the common initial input, the future guided command RMS increases from 0.001961411819 to 0.004942172282, a factor 2.51970149. A larger numerical response did not produce the requested rendered behavior. The first-step CFG decomposition does not support simple near-total cancellation; it does not diagnose the complete trajectory.

The first reader invocation [failed](audit/comparison20-v1/failed.json) because it imposed an extra `allocated <= reserved` relation on separately sampled CUDA counters. The frozen monitor reads those counters sequentially. The [narrow correction](audit/reader-v2/correction.json) removed only that extra relation; all actual resource limits and nonnegative-type checks remained. The [independent review](audit/reader-v2-independent/report.json) passed. Original v1 source/failure, v2 source and fixture report remain retained. No producer source, protocol, raw tensor, metric or limit changed.

## Sources, full recovery and limits

Exact local programs, CPU reports, admissions and audits are retained here or in the complete archive. The [source map](code-provenance.json) links 66 shared repository files through existing public bundles. The executed dependency commit is `099eeae6268da852b0acb580258e4c168f900520`; the release target commit is publication context. Copied historical source comments and pre-run reports retain their original preparation status.

The [download instructions](DOWNLOADER.md) recover all 608 members from one combined visual-plus-diagnostic archive. The [release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-post128-a100-v1) is public. All 111 remote asset sizes and SHA256 digests matched, and a [fresh public HTTPS download](recovery/public-download-verification.json) verified all 608 members. [Release status](release-status.json) binds the exact records. Three copies of the prior training audit receive only the approved workspace-prefix replacement in `report.run`; all other 605 members remain byte-identical. Original and public file/stream hashes and exact replacements are in the [derivation](recovery/public-derivation.json). All numerical and scientific identity fields are unchanged. The untouched original recovery stays local; the [original index](recovery/original-index.json) remains public. Plans retain original audit hash references, with derivative mappings explicitly disclosed.

The copied audit scripts retain their historical work-layout constants and original training-audit hash checks. Their passed reports come from the untouched recovery. The downloader verifies the declared public derivative; replaying those scripts requires restoring the expected work layout and explicitly handling the three operational-path derivatives.

Run `python verify_payload.py` for offline compact identity checks. Full raw evidence is needed for tensor-based audits. No foundation checkpoint, installed environment, credential or paid-compute access is included. No additional test/model run was performed for publication.

This is one seen room, one initial noise and a fixed short command sequence. The result does not establish successful action learning, generalization, a new architecture claim or frontier-model parity. Original Worldline code and adapter weights use [Apache-2.0](licenses/WORLDLINE-APACHE-2.0.txt); external Wan retains its [Apache license](licenses/WAN-APACHE-2.0.txt) and [NOTICE](licenses/WAN-NOTICE.txt); original Atrium imagery and metadata use [CC0](licenses/ATRIUM-CC0.txt).

The first 17 uploaded 16 MB parts are preserved. One bounded upload failed for part 017, so only the absent tail was divided into 65 pieces at most 8 MB. The final layout has 82 parts. The earlier [50-part index](recovery/history/public-index-16mb.json), [metadata](recovery/history/public-download-16mb.json) and [local verification](recovery/history/local-verification-16mb.json) retain transport history; the public gzip stream and every member are unchanged.
