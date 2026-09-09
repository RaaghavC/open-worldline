# Worldline project status, September 8, 2026

Worldline has a working local world editor, original small models trained from scratch, and a broad primary-source research review. It has not achieved Genie 3 quality or demonstrated three new scientific breakthroughs.

| Requested outcome | Current result |
| --- | --- |
| Current and less-mainstream research | More than 40 model and research entries, with source dates, release checks, licensing, and separate September 8 findings |
| Original open-source implementation | Original spatial flow model, ecological transition model, training data generator, editor, and action adapter code |
| Intuitive local use | Double-click `start.command`, choose a world preset, explore, paint changes, save, restore, and branch |
| Impressive general neural video | Not achieved; the main editor renders learned 64 × 64 fields with Three.js |
| Reliable video action control | Not achieved; the newest high-resolution room videos keep the door closed and omit the camera turn |
| Genie 3 parity | Not established by any matched evaluation |
| Three high-impact scientific advances | Not established; editing, persistence, and branching are useful software features with existing prior art |

## Research

Start with the [September 7 review](research-2026-09-07.md), then the [September 8 update](research-2026-09-08.md). The linked catalogs cover interactive video, persistent 3D generation, memory, action use, robotics, training methods, compute, and evaluation. The review is broad, rather than a claim to cover every paper or private system. Author-reported scores are distinguished from our measurements.

## Latest experiment

The original 947,712-parameter action adapter was trained for 128 updates around a frozen Wan2.2 TI2V-5B foundation. An added objective on 32 updates asks the difference between two commands to predict the observed difference in their outcomes. Main flow-matching training still uses target-corrupted noisy future inputs; the added command-difference conditioning uses shared pure noise and the independently encoded starting image.

The final checkpoint passed the predeclared four-noise numerical criterion. Its contrast error fell by 0.513–0.532% relative to the prior adapter. The four samples hold out noise in a single seen room, not new scenes or new tasks. All 825 foundation tensors remained unchanged.

The subsequent two 17-frame, 1248 × 704 videos failed the visible controls. The door remained closed in both videos, and the requested left turn did not occur. Mean future-frame target error was 0.183566 for wait and 0.193466 for interact, compared with 0.178963 and 0.190356 when simply repeating the initial image. A small numerical improvement therefore did not produce usable interaction.

The [complete results](../experiments/wan22_native/action_effect/results/README.md) retain the exact checkpoint, previews, source, predictions, audits, and download instructions. Historical records remain unchanged where possible; the publication notes disclose operational path redactions and changes to download-piece metadata. No further training under this hypothesis is admitted.

## Follow-up diagnosis and implementation

A [saved-prediction analysis](../experiments/action_effect_diagnostics/README.md) found that even an ideal, target-fitted multiplier explains less than 0.7% of the required initial latent change. The response is poorly aligned, rather than merely too small. This is a post hoc analysis of the same room, not a new quality result.

The [intermediate action bridge](../experiments/wan22_native/intermediate_action/README.md) lets the final native transformer block process the original adapter's commands. Its [actual A100 profile](../experiments/wan22_native/intermediate_action/results/a100-profile-v1/README.md) passed 16 exact native/full/cached comparisons and two updates at each of two placements. All 825 foundation records remained unchanged. The new placement needs about 4.92 GiB more peak training allocation in this profile. Its saved response after one update remains poorly aligned with the intended change; no new video was generated.

The [new camera/door capture](../experiments/atrium_factorial/README.md) contains six native 1248 × 704 sequences with independently varied camera and door commands. All 102 original training images passed file/label validation. The original VAE encoded all six clips without resizing or replacing their prefixes. All 12 prefix comparisons were exact and all 196 VAE records remained unchanged.

A fresh [128-update block-28 run](../experiments/wan22_native/intermediate_action/results/factorial128-a100-v1/README.md) trained on those six sequences. It completed in 469.203 seconds. The independent audit checked all 384 saved training predictions, 128 gradient bundles, nine checkpoints and 825 unchanged foundation records. Its complete training evidence is now [publicly downloadable](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-factorial-training-a100-v1).

The subsequent [six-command video evaluation](../experiments/wan22_native/intermediate_action/results/factorial-video-a100-v1/README.md) completed all 102 frames in 644.107 seconds. It failed the original control requirement. The inspected stationary-interact and left-interact clips keep the door closed in every frame; the left-interact clip also lacks the requested progressive turn. The independent artifact audit verified all 300 saved states and all 102 raw RGB/PNG correspondences. Small pixel-error improvements on some arms do not establish correct controls. The partial visual review, exact scalar results and cleanup records are retained. Because both the images and insertion point changed, comparison with the older run cannot isolate a placement advantage.

The next experiment uses a [4,936,448-parameter command-attention controller](../experiments/wan22_native/command_attention/README.md), replacing the single residual adapter with rank-32 query/key/value/output updates across blocks 24–29. Its prepared training objective includes both door and camera contrasts. The controller, fixed 512-update training runner and resource profiler have passed their small CPU checks. Its [first A100 profile](../experiments/wan22_native/command_attention/profile/results/a100-v1/README.md) completed 24 recorded exact zero-initialization comparisons and two optimizer updates in 358.207 worker seconds, with peak reserved CUDA memory of 56,461,623,296 bytes. The saved-file audit passed; it checked recorded hashes for the discarded controller outputs and did not replay backward or optimizer computations. The 1,573.155-second fitting forecast exceeded the original lease's available time after reserving 600 seconds for recovery, so the 512-update run did not start on that pod. The profile supplies no final trained checkpoint or video result and establishes no visible control improvement or novelty. Its raw recovery release is pending.

On a later lease, the [fixed 512-update run](../experiments/wan22_native/command_attention/training/results/a100-v1/README.md) completed in 1,117.084 worker seconds after a new matching profile passed. Its independent audit checked all 512 scalar/schedule rows, five checkpoints, 96 initial/final evaluation velocities, 22 retained training predictions, five gradient bundles and 825 unchanged foundation hash records. It did not replay omitted gradients, backward passes or AdamW. The three door comparisons improved to normalized latent MSE 0.8173–0.8991 across four fixed noises, from the zero-controller baseline of 1.0. The four camera comparisons remain weak at 0.99968–1.00076, with cosine near zero and response RMS about 0.017–0.018 against target RMS near 1. The final checkpoint is fixed at update 512; generated-video evaluation is pending. This remains a one-room fitting experiment. Architecture and objective changed together, so it establishes neither a causal explanation nor a new scientific advance. The small exact records and auditor are committed separately from raw binaries; the raw training release has not been created.

## Run the editor

Return to the [installation and controls](../README.md#run). The shipped small models run locally without an inference API. The experimental Wan adapter uses external open foundation weights and requires a separate CUDA environment; it is not the model powering the main editor.

The GPU instances used for the earlier adapter experiments, including the placement profile and six-arm encoding, were deleted. Their temporary provider keys were revoked and rejected by the provider, and local private key files were removed. No job from those completed adapter runs remains active. The first command-attention profile pod was also deleted after its 63-file recovery was verified; its exact deletion receipt is linked from that result. This statement does not cover a later command-attention training lease.
