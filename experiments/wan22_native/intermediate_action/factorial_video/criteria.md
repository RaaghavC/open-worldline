# First generated-video test of the six-arm model

Prepare one fixed six-clip evaluation of the final, independently audited checkpoint 128. Generate all stationary/left/right × closed/interact combinations with the same observation, saved Gaussian noise, text, model and sampling settings. This directly tests whether visible controls occur. Training completion, lower flow loss or an endpoint contrast score cannot substitute for these videos. No model has run under this plan, and the future checkpoint/audit hashes are not yet known.

## Exact shared inputs and inference

Use the independently encoded observation from the completed new native-resolution cache at `work/intermediate-profile-recovered-stages-v1/recovered/action-results/factorial-native-cache-v1/result/observation.safetensors`. Its file SHA256 is `dc97ee856cadc78520e9f87a9b7d3ba0534c6270239affce146d31281269785a`; its FP32 `[1,48,1,44,78]` tensor SHA256 is `222f3792096029fba5d4cd8c0ea5a4b5919f0f92a6046cba5acd87ffa153cba3`. Use each of the six cache files' exact `[1,16,6]` commands. The completed cache's full targets and camera/door labels belong only in the later CPU scoring process.

Reuse **only** `initial_noise` from the prior fixed visual packet, `work/wan22-action-effect-visual-recovered-final-v1/recovered/action-results/effect-visual-v1/sampling-inputs.safetensors`. That file is `0b1daf39df7630971ae67ae1d0db43ede1bc0fa20a893a295d4eed6e878faade`; the FP32 `[48,5,44,78]` noise tensor is `d3a4f22635d595df7ddc0b9f3bc7b7d621c23ea540c8f36f759ee6a145adeace`. Do not reuse its older observation or preclamped initial latent. Construct a new initial latent by copying this noise and replacing exactly its first latent frame with the new independent observation. Save this deterministic derived tensor and its hash before execution. Generate no fresh noise and select no seed after looking at results.

Reuse the exact `contexts.safetensors` beside that prior packet: file SHA256 `40f59cf54f2819555ff37a95a116e41f3fe1e247af14cf824704f6191df327ee`. Its positive context is FP32 `[25,4096]`, tensor SHA `f152d177bf53650f4ac8658665fd6b7e86d4bad68106309c1ecd19e56a4bb093`; negative is FP32 `[126,4096]`, SHA `aa6911f67cb7a4e5cf934131594b6fe264dff9983f84ab188e9199c706cf60cc`.

Keep the native CPU UniPC sampler at 50 steps, shift 5, CFG 5, 17 frames and 1248×704. Both CFG branches receive the same current arm's commands. Initial-token times stay zero, and the exact independent observation is restored after every solver update. Native FP32 stored core/adapter parameters, CUDA BF16/FA2 forward and original FP32 VAE decode remain unchanged. The new prediction callback must explicitly use the published bridge at **block 28**; the previous final-block sampler wrapper cannot silently serve this checkpoint. Reuse `spatial_reference.sampling.sample` with that reviewed callback. Feature bundles cannot be reused across changing solver states/times.

Require the actual completed training plan, checkpoint-128 manifest/adapter, passed saved-file audit, all 825 unchanged foundation records and current source/CPU review before preparing an executable plan. Do not select checkpoints 16 through 112. No training or optimizer is permitted in evaluation. Retain the unchanged latent clamp even if the decoded first image has ordinary VAE reconstruction error; do not replace generated RGB with original pixels.

## Six outputs and a small control

Generate these arms in fixed order: `stationary_closed`, `stationary_interact`, `left_closed`, `left_interact`, `right_closed`, `right_interact`. They require 600 predictions and 300 solver updates in total. One trained adapter is loaded once; every arm starts from the same newly clamped latent, with a fresh solver instance. Keep all outputs, including failures.

The CPU repeat-initial-image baseline is mandatory and costs no model calls. Also report a baseline that repeats each generated clip's own decoded first frame, so VAE reconstruction error is visible separately.

If admitted in the prepared budget, generate **one** unadapted native clip with the same new observation/noise/text/settings, outside all adapter hooks. It has no command input, so six identical native reruns add no information. Score this one clip against all six targets. It adds 100 predictions/50 steps. Prefer planning the seventh clip up front because its marginal cost is small compared with loading the core. If it is omitted, explicitly leave improvement over unadapted Wan unresolved. Do not replace any required trained arm with this control.

## Visible success and failure

Review all 17 original PNGs per trained arm, not just a contact sheet or endpoints. Have two reviewers independently record camera state, door state, first frame of opening, and obvious scene collapse. Compare with all 17 matching captured targets. A numerical difference between generated arms is not sufficient.

| Command | Required visible result |
| --- | --- |
| stationary | Room viewpoint stays fixed throughout both door conditions. Object changes do not move the whole camera. |
| left | Both door conditions show coherent progressive left yaw. The final view moves toward the target's window/plant side, with the door moving toward the right edge. Tiny jitter, a still camera or an opposite turn fails. |
| right | Both door conditions show coherent progressive right yaw. The final view moves toward the target's bench side, with the door moving toward the left. Tiny jitter, a still camera or an opposite turn fails. |
| closed | The original door remains visibly closed through frames 1–16 in all three camera motions. |
| interact | The doorway becomes visibly open at frame 1 and remains open through frame 16 in all three motions, revealing the room behind it. Lighting changes, disappearing geometry or a painted opening without a coherent doorway do not count. |

The capture commands are ±1.5° per transition, ending at ±24°, and an instantaneous remote opening to 102°. Review direction, progression, endpoint view and opening timing against those actual images. Do not infer precise camera or door angles from unconstrained generated pixels. Record a correct but smaller/later motion as **partial**, not exact command execution. If reviewers cannot establish a control because the image is ambiguous or distorted, record **not demonstrated**. Overall six-arm visible success requires correct camera and door behavior in every arm plus a coherent recognizable room. Preserve per-control results if only some controls work.

A minimal quantitative companion uses the unchanged pixel rules and reports future-frame and final-frame RGB MAE/RMSE against the exact 102 captured PNGs, together with the two repeat baselines. Report each arm separately and the paired interact-minus-closed image difference separately. Target-derived difference regions are image-difference supports, not door segmentations. Lower pixel error corroborates reproduction; it cannot overrule a failed visible door or camera check.

The `stationary_closed` captured target repeats its first image exactly, so the original repeat baseline has zero target error. Do **not** demand improvement over zero. Report its reconstruction error and drift from its own decoded first frame. For the other five arms, report whether the generated future and final errors beat original repeat-start. If not, the model has not demonstrated better target reproduction than a still image, even if one control is recognizable.

Publish six per-arm verdict rows, both reviewers' disagreements, all-frame contacts, clips and original PNGs. State small numerical improvements separately. One fixed noise on one training scene is a development test, not unseen-scene or noise generalization. It cannot establish a block-28 advantage without matched block-29 training on the same new cache and schedule.

## Runtime and recovery

The latest old final-block two-arm visual run took 459.635 seconds overall: core load 168.496 seconds, before/after hashes 65.438/62.305 seconds, sampling 45.322/44.754 seconds, and decoder stage 61.167 seconds. These are measured old-run timings, not a block-28 sampling benchmark. Reusing one core load and one VAE load gives a rough six/seven-clip expectation of 12–17 minutes including file handling; actual block-28 sampling and recovery remain unmeasured.

Use one combined **1,800-second** parent budget with separate core and decoder lifetimes, the original memory caps, one before/after 825-value verification, native codec verification, and partial-output preservation. Admit only with the full cap plus recovery reserve remaining. For this larger output, plan **600 seconds** of recovery reserve rather than relying on the 300-second minimum. Do not automatically chain this after training on the same lease unless that complete allowance still fits.

Retain all 50 latent states per clip, exact initial noise/clamp/context/command/checkpoint identities, first-step positive/negative/guided velocities, every decoded FP32 frame shard, all 17 original PNGs, and parent/worker/resource/terminal records. Later prediction counts rely on the reviewed sampler, as before; do not claim every later velocity was retained. Expect roughly 2.2–2.8 GB raw for six/seven clips, based on earlier full retention, so prepare transport parts and complete-file verification before renting. One RGB frame is about 10.5 MB and one latent state about 3.3 MB; no file needs to exceed 100 MB.

No extra training-objective gate should delay these videos after training integrity passes. Saved losses and fixed-input predictions can diagnose a visual failure afterward. They do not satisfy this plan's visible-control criteria.
