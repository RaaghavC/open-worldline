# Original Atrium observation/action adapter pilot

Protocol version: `worldline-wan-atrium-17-v1`, September 7, 2026. Freeze this version, the data manifest and the chosen settings before a real-data training run. Any later change to action units, input observations, text, splitting or inference conditioning requires a new version.

This experiment freezes the official Wan2.1-T2V-1.3B foundation and trains Worldline's original action/initial-observation adapter. The first objective is an actual update on original rendered observations and commands, with verified inputs and finite gradients. A 17-frame bidirectional clip model is not causal streaming. Initial-frame observation tokens are not persistent world memory. The completed synthetic MPS update measured 19.416 seconds and a sampled 5.53 GiB driver allocation; it establishes runtime feasibility, not image quality or real-data learning.

## Current data and claim boundary

The dense Atrium capture has two arms, 66 frames per arm at native 512×288, and 65 transitions per arm. Use its complete manifest and image hashes. Earlier sparse captures contain only selected frame indices and must never be treated as dense video or silently interpolated.

Atrium v1 is one architectural layout. The seed varies accent color and leaf placement, not independent geometry. The camera remains at one position and changes yaw. The door is a programmed remote toggle, with no reach, contact or collision model. Indirect light reveals some offscreen door state; do not assert identical hidden histories or a strict memory test. All windows and both arms of the current capture are **development data**. Window holdout within it is not scene generalization. A learning or visual inspection result must retain these limits.

## Six action channels, units and time alignment

Every command is a float32 vector in this exact order:

| Index | Name | Meaning and units |
|---|---|---|
| 0 | `local_right_m` | Requested rightward displacement in meters, in the camera-local pre-action frame |
| 1 | `local_up_m` | Requested upward displacement in meters, in the camera-local pre-action frame |
| 2 | `local_forward_m` | Requested forward displacement in meters, in the camera-local pre-action frame |
| 3 | `yaw_left_rad` | Requested leftward yaw increment in radians |
| 4 | `pitch_up_rad` | Requested upward pitch increment in radians |
| 5 | `interact_pulse` | One for a requested toggle at this transition, otherwise zero; never a held state |

There is **no action normalization in v1**. Translation and pitch are reserved and always zero in the current capture; no ability on those channels is demonstrated. Future data must record commanded translation directly. Never recover it from realized pose and then label the input RGB-plus-command prediction. A world-from-camera transform, door angle, arm name and object labels are evaluation metadata only.

Current action mapping is exact:

```text
wait     [0, 0, 0, 0,       0, 0]
left     [0, 0, 0, +pi/24, 0, 0]
right    [0, 0, 0, -pi/24, 0, 0]
interact [0, 0, 0, 0,       0, 1]
```

`action[t]` causes the transition from `RGB[t]` to `RGB[t+1]`. In the renderer manifest it is stored as `records[t+1].action_from_previous`. `records[0].action_from_previous` must be null. In the open arm the first command is interact; in the closed arm it is wait. Both continue with 24 left turns, 16 waits and 24 right turns. Do not use `records[t].action_from_previous` as the outgoing action for frame t.

For a window starting at frame `s`, the RGB sequence is `[s, ..., s+16]`, and the action sequence is `[s, ..., s+15]`. Valid dense starts are integers from 0 through 49. Preserve the 16 actions in chronological order. At the adapter boundary the action tensor has shape `[B,16,6]`. The five latent-time groups are:

```text
latent 0: a reserved all-zero initial action group, with no invented preceding command
latent 1: concatenate actions[s+0], actions[s+1], actions[s+2], actions[s+3]
latent 2: concatenate actions[s+4], actions[s+5], actions[s+6], actions[s+7]
latent 3: concatenate actions[s+8], actions[s+9], actions[s+10], actions[s+11]
latent 4: concatenate actions[s+12], actions[s+13], actions[s+14], actions[s+15]
```

Each later group is 24 values. Averaging or summing four controls would lose toggle/movement order and is prohibited by this protocol. The grouping describes causal VAE support intervals, not an assertion that each latent represents a single instantaneous RGB frame.

## RGB, target latents and observation conditioning

Read native RGB PNGs using their recorded AgX/display transform. Convert uint8 RGB to float32 `2*(RGB/255)-1`, producing `[B,3,17,288,512]`. Do not silently substitute scene-linear EXR, resize, crop, reverse channels or apply a second display transform. Record the conversion and source image hashes.

The pinned official Wan VAE encodes a complete 17-frame window into `[B,16,5,36,64]` using its documented mean/std normalization. Reset its temporal cache for every window and every independent clip. These full clean latents are the **training target**. They may be noised for a flow-matching loss; they must never become clean future conditioning.

Encode `RGB[s]` alone, with a fresh VAE temporal cache, to obtain the adapter observation `[B,16,36,64]`. Only this independently encoded initial frame enters its 32 pooled observation tokens. Keep target and conditioning in distinct named fields. Do not condition on `full_clip[:, :, 1:]`, future images, future pose, depth, door state, renderer access or a ground-truth continuation. Passing noisy targets through the denoiser during training is valid; leaking their clean values through the adapter is not.

The current 4×8 pooled observation tokens have no added position coordinates or position embeddings. Their cross-attention is invariant to a permutation of the token vectors. It can attend to their feature content, including any position cues already present in the VAE features, but it does not explicitly know which pooled cell was left, right, high or low. Pooling also discards fine spatial detail. This is a baseline limitation, not proof that all source images become indistinguishable. Preserve this baseline for the pilot; a later position-aware variant needs its own equal-budget comparison. Do not call these tokens a spatial map or a learned persistent memory.

For the real-data smoke update, record the exact flow convention, time distribution and loss reduction. A permitted convention is `z_t=(1-t)*z_clean+t*epsilon`, predicting `epsilon-z_clean` with MSE. Sampling then integrates from t=1 to t=0 using that same sign. Another convention requires explicit matching equations. Record whether the initial latent contributes to the loss; keep this choice fixed across ablations. Fixed starting observations may be supplied throughout denoising. Clamping observed latents is a separate declared sampling choice and must not clamp future targets.

## Causal VAE checks before using cached observations

Pass all checks for the exact VAE weights, dtype, device, implementation and normalization used to create the cache:

1. Encode the first RGB alone and as the first latent of a full 17-frame clip. Require finite outputs with matching shape and numerical agreement. The current ceiling test predeclares `atol=1e-5, rtol=1e-5`; if a different dtype needs a different threshold, document and justify it before evaluating results.
2. Keep the first RGB identical, replace future RGB frames with a substantially different finite sequence, reset caches and re-encode. The first latent must remain unchanged within the declared tolerance. Repeat with another real initial frame. This directly tests dependence on future pixels.
3. Encode clip A, clip B, then A again. A must reproduce within tolerance, proving no temporal cache survives between calls. A failed cache-isolation test stops observation-cache creation.
4. If any later prefix latent will be used as an observation in a future protocol, separately test every claimed prefix boundary. The first-latent check alone does not establish causality of arbitrary later latent slices or justify slicing a long encoded clip as a freshly observed window.

The simplest safe v1 reader encodes each 17-frame window from a fresh cache and caches the starting frame separately. Do not slice a long-video encoding and assume it equals independent window encoding. Decoding target latents measures the external codec's reconstruction ceiling. It does not measure a generated model rollout.

## Genuine cached text embeddings

Use the existing `text_cache/prompts.json` exactly. Both arms use the same positive `atrium` prompt and the same `unconditional` empty-string prompt. The latter must be an actual UMT5 encoding of the empty string, not an all-zero tensor. Door state, arm, action labels and future states must not appear in text.

Cache real pinned official UMT5 encoder output and tokenizer behavior. Store finite per-prompt sequence tensors `[L,4096]` and exact lengths or masks. Pass unpadded UMT5 outputs to the core. The selected Wan core then pads them with zeros to 512 positions before its learned text projection, with `context_lens=None`, matching its upstream behavior. Do not silently change this to masked attention while claiming identical foundation semantics; the projection makes those padding values nontrivial. Record exact UTF-8 prompt text and hash, encoder weight hash, tokenizer file hashes, implementation revision, dtype, normalization/preprocessing, special-token/truncation settings and cache tensor hashes. Record the actual prompt token counts. Missing cache entries, failed hashes or non-finite tensors must fail; no zero/mock fallback is allowed for real-data quality training. Synthetic context remains a separately labeled runtime probe.

## Required assertions and reports

Before the first real-data update:

- Manifest is complete; both arms contain all 66 consecutive PNG frames; each selected image matches its hash and native dimensions. Every 17-frame window has exactly 16 valid outgoing commands and no gaps. Paths must remain within the selected capture root.
- Preserve paired-arm identity and source-frame indices in cache metadata. Keep all windows from this scene family in development. Future train/validation/test separation must group complete layout families, assets, camera paths and both branches, with independently designed test geometry and no overlapping frames.
- Positive text is identical across arms; no prompt, filename, arm index, realized pose, depth or door label enters the adapter input. Train inputs contain only target noise/time, actual cached text, ordered commands and independently encoded starting observation.
- Initial zero-output adapter reproduces the frozen core under the same inputs. After backward, expected adapter gradients are present and finite; base parameters are frozen and unchanged. Keep hooks attached until checkpointed backward finishes. Never wrap the gradient path through the frozen core in `no_grad`.
- Validate the chosen VAE causal/cache tests, genuine text provenance, latent dimensions, normalization, finite values and deterministic cache hashes. Record foundation, adapter, data, source and dependency identities.
- Predeclare steps, seed, batch/window sampling, optimizer, loss, validation usage, time/memory caps and output directory. Fail on non-finite loss/gradients. Preserve an incomplete run's status and error instead of overwriting it with a passing report.

For the bounded single-layout development pilot, compare fixed-noise predictions before and after the update and retain the untrained adapter, shuffled-action and masked-observation controls. One update establishes a functioning real-data gradient path, not improvement. Multiple updates require actual measured before/after losses and fixed preselected generated videos; training loss alone is insufficient. No held-out-scene, door-memory, physics, streaming-speed, frontier-quality or novelty claim is supported by this capture.

For any later closed-loop rollout, initialize from observed RGB only once, then reuse generated history or declared learned state. Supply future user commands as they arrive. Never append renderer truth after initialization. A model that denoises a bidirectional clip with a known future action sequence must be described as action-planned clip generation until causal deployment has been independently implemented and tested.
