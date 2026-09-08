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

## Run the editor

Return to the [installation and controls](../README.md#run). The shipped small models run locally without an inference API. The experimental Wan adapter uses external open foundation weights and requires a separate CUDA environment; it is not the model powering the main editor.

The two GPU instances used for the latest experiment were deleted, the temporary provider key was revoked and rejected by the provider, and the local private key files were removed. No training job from this experiment remains active.
