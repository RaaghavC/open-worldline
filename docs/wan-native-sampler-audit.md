# Audit of the failed Atrium sampler against official Wan T2V

Checked September 7, 2026. Source/read-only audit. No training, inference, model edits or protocol changes were performed. Official source is pinned to `Wan-Video/Wan2.1` revision `9737cba9c1c3c4d04b33fcad41c111989865d315`; the official weight revision remains `37ec512624d61f7aa208f7ea8140a131f93afc9a`. The [Atrium study](wan-atrium-pilot.md) records the measured local evidence.

## Finding

The failed clips are evidence that the present adaptation failed. They are not a clean native Wan quality test. Both the untrained and trained arms use non-native image-prefix clamping, a different solver/step budget, different 1.3B sampling settings, a different negative context and altered numerical precision. The CPU checks establish the implemented equations and internal comparisons; they did not establish equivalence to the literal official mixed-precision Wan forward path.

The flow direction itself is consistent: the model predicts noise minus clean data, and generation integrates from high sigma toward zero. There is no evidence here of an accidental reversed Euler sign, future-target access during generation, mismatched base/trained noise or a CFG subtraction error. Those behaviors have independent CPU checks. Do not replace the measured failure with a claim that a particular remaining difference caused the lattice artifact. That cause is not yet isolated.

## Official contract versus the current experiment

| Component | Pinned official implementation | Current Atrium experiment | Implication |
|---|---|---|---|
| Model task | T2V: every latent starts as Gaussian noise; text is the condition | T2V foundation plus image clamp and new adapter | This is an adaptation to image/action conditioning, even with a zero-output adapter. |
| Solver | CLI defaults to UniPC; DPM++ is another supported option | First-order Euler | Euler is a valid flow solver, but it does not reproduce the recommended generation path. |
| Steps | T2V CLI and function default to 50 | 20 | Lower numerical accuracy may matter; increasing steps alone would not isolate all other differences. |
| 1.3B settings | README recommends guidance 6 and shift 8–12; example uses 8 | Guidance 5, shift 5 | Generic CLI defaults are 5/5, but the model-specific README advice differs. |
| Negative text | Empty `n_prompt` is replaced by the configured negative prompt | Genuine UMT5 embedding of the empty string | Our embedding is real, but it is not the native default negative condition. |
| Shape | 1.3B CLI supports 832×480 or 480×832; default 81 frames; VAE formula permits 4n+1 | 512×288, 17 frames | 17 is structurally valid; 512×288 is not a listed CLI size or a verified recommended quality setting. |
| Precision | BF16 autocast with explicit FP32 paths for time conditioning, modulation/residual operations and head | FP16 parameters and intermediate outputs through the extracted core | Matching keys, shapes and hashes does not establish matching numerical behavior. |
| Attention/text implementation | Native module choices and precision behavior | Selected minWM core, portable SDPA, altered LayerNorm path, exact text GELU | Further forward-parity validation is needed. |

The official [CLI validation and solver options](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/generate.py#L60) set the T2V defaults. One stale T2V function docstring mentions 40 steps; the actual signature and CLI both use 50. The [model-specific README guidance](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/README.md#L145) recommends guidance 6 and shift 8–12 for 1.3B. Its example enables neither a required hosted API nor mandatory prompt extension. A separate Diffusers example is centered on 14B with other shift guidance; it should not override the explicit 1.3B note without saying so.

## Conditioning and timestep details

The [native T2V generator](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/text2video.py#L146) builds a full noise tensor and passes text context plus token count to the model. It does not supply a clean first latent or a per-frame mask. In contrast, the current run presents a clean first latent under the same scalar diffusion timestep as its noisy future. The adapter was trained with that condition, so training and inference agree internally. Ten updates do not establish successful adaptation to this new condition.

Official [image-to-video inference](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/image2video.py#L192) uses a different conditioning path: encoded reference features, a mask and image features supplied to an I2V model. The released 1.3B T2V checkpoint is not silently converted into that pretrained I2V model by replacing its first latent with a clean image.

The [1.3B architecture configuration](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/configs/wan_t2v_1_3B.py) specifies VAE stride (4,8,8) and patch size (1,2,2). Thus 17 frames at 512×288 produce five latent frames of 36×64 and 2,880 patches, as used locally. The arithmetic is correct. The [supported size list](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/configs/__init__.py#L27) nevertheless lists only the two 480p orientations for this CLI model.

The official [UniPC schedule](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers_unipc.py#L73) starts from a training sigma maximum near 0.999, applies the requested shift, stores float32 sigmas and converts model timesteps to int64. Its final sigma is zero. The current hand-written schedule starts at exactly 1 and passes fractional `1000*sigma` values as floating tensors. The official [DPM++ sigma helper](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/utils/fm_solvers.py#L21) also starts at 1, so a timestep of 1000 is not by itself an established bug. For a native comparison, use one pinned official scheduler and its returned timesteps and state updates, rather than combining formulas from different solvers.

UniPC is a multistep predictor/corrector. Replacing its `step` with a single Euler increment while retaining the same sigma list would not be equivalent. Its `flow_prediction` convention is compatible with noise-minus-clean velocity, supporting the current sign audit.

The [shared configuration](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/configs/shared_config.py) sets BF16 compute, 1,000 training timesteps, 16 fps output metadata and a specific negative prompt. The current 10 fps GIF is a preview playback choice, not generation throughput. Cache a real UMT5 encoding of the exact configured negative prompt for a native test; do not substitute zeros or silently reuse the empty-string context. Retain the present positive Atrium prompt to avoid another unnecessary variable.

## Numerical differences that the prior checks did not cover

The official [model source](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/wan/modules/model.py) explicitly computes LayerNorm on float32 input, requires FP32 time embeddings/projections, preserves FP32 modulation/residual operations and computes the head through an FP32 path. Its text projection uses approximate-tanh GELU. The extracted minWM model instead uses exact GELU there; its LayerNorm wrapper lacks the explicit input upcast; its local all-FP16 conversion also lowers time and modulation precision.

The existing numerical tests compare positional substitutions and a tiny FP32 selected minWM model against that selected model's upstream functions. They do not compare a native pretrained FP32/BF16 Wan forward against the complete extracted FP16 path. The mismatches are source-observed. Their contribution to the visible artifact remains unmeasured.

The external text encoder already demonstrated values exceeding FP16 range before its final normalized embeddings. That is evidence for keeping its separate streamed FP32 path; it is not evidence that Wan's transformer has the same overflow. Likewise, finite transformer outputs do not prove adequate numerical fidelity.

Before a full native-style clip, perform a tiny no-adapter forward comparison against the literal pinned Wan model in FP32, using the same compatible SDPA and already checked real-valued RoPE substitutions. Compare at least timestep 999, 500 and 50; include final velocity and intermediate time/head outputs. Keep FP32 model parameters and the native activation choices for this reference. Then separately measure error introduced by the intended memory-saving precision path. A failed parity check is a reason to fix the port before another quality run. No such comparison was executed by this audit.

## Smallest proposed single-clip diagnostic

The same-latent FP32 decode has now completed. Root reports that the lattice and blur remain, with future-frame MAE 0.00105 and SSIM 0.99907 relative to the rounded FP16 decode. This makes decoder precision an unlikely main cause for these saved latents; it does not validate the denoiser precision path.

Generate **one pure T2V control clip**, with no adapter attached and no image/action input:

- Same official foundation files and same genuine positive Atrium text as the failed run.
- Genuine cached native configured negative prompt, recorded separately from the existing empty-string context.
- One seed, 20260908, with the complete initial Gaussian tensor retained and hashed. All five latent frames start from noise. Never read a capture-cache tensor, clamp an image, load an adapter checkpoint or access renderer truth.
- 17 frames at 512×288 for the first bounded diagnostic. This preserves the failed run's spatial/temporal size and measured memory regime; label it **a reduced-resolution native-style control**, not a complete reproduction of the official 480p/81-frame example.
- Pinned official `FlowUniPCMultistepScheduler`, 50 steps, shift 8, guidance 6, official timestep values and scheduler updates. Run negative and positive passes sequentially to bound memory. Preserve the latent solver state in FP32.
- Use the numerically validated native-compatible forward path. Prefer a full-FP32 reference if it fits the measured cap; otherwise use the explicit native-style FP32 islands and a separately validated lower-precision matrix path. Merely retaining the current all-FP16 core would leave the known precision question unresolved. Do not present an unvalidated precision compromise as exact native reproduction.
- Release the transformer before decoding. Use the FP16 decoder supported by the completed same-latent comparison, with native temporal cache behavior. Save all 17 output frames, initial noise, final latents, text identities, source hashes, scheduler config/timestep list, per-step latent ranges and timing/memory samples.

Keep the existing 18 GiB memory limit and 2 GiB minimum available-memory guard. Use a new diagnostic output directory and an explicit maximum of 900 seconds. Before launching all 50 steps, time one negative/positive pair at the actual shape/precision and estimate the remaining cost. The earlier 20-step batched-FP16 clip took about 210 seconds to denoise; multiplying by 2.5 gives 525 seconds only if cost per step were unchanged. Sequential passes and FP32 may be slower. If the measured estimate cannot fit the cap, retain the profile and do not silently reduce steps, change dtype or exceed the cap.

This single control intentionally restores the T2V conditioning contract and the model-specific recipe together. It can separate a working pretrained-generation path from the failed image/action adaptation at this shape. It cannot identify which restored setting produced a change; that would need a later one-factor ablation.

### Checks required before calling the control valid

1. A tiny FP32 native-versus-port test must preserve native activation, normalization, time-conditioning and residual rules at timesteps 999, 500 and 50. Predeclare maximum absolute velocity error 0.0001 and relative L2 error 0.00001, using the same attention implementation. If these fail, inspect the differing layer before sampling. This tiny test checks the equations; it does not establish numerical fidelity of all pretrained weights.
2. For the chosen lower-precision path, compare one actual-weight no-adapter forward against the FP32 reference at the diagnostic shape if the memory cap permits. Save maximum absolute error and relative L2 error, activation ranges and finite checks. Treat relative L2 error above 0.01 as a stop-and-review signal, not a known Wan quality boundary. If the reference cannot complete within the memory cap, record the unresolved precision limitation and call the clip a partial diagnostic.
3. The saved solver config and 50 timesteps must match direct construction of the pinned official UniPC scheduler with shift 8. Record 50 scheduler updates and 100 sequential denoiser calls. The CFG equation must be negative + 6 × (positive − negative). Assert finite model outputs and latent states at every step.
4. No capture-cache reads, adapter state, first-frame replacement, renderer call or ground-truth RGB access may occur inside generation. Every initial latent value comes from the retained Gaussian tensor; the output has exactly 17 decoded frames at 512×288.
5. Preserve all outputs before judging appearance. Review frames 0, 4, 8, 12 and 16 and the full clip. Record whether room boundaries and furniture remain recognizable, whether repeated lattice texture covers large surfaces, and whether adjacent frames retain the same scene. This single-seed visual check is diagnostic evidence, not a calibrated image-quality benchmark. No action-following requirement applies to pure T2V.

### Interpretation fixed before execution

| Outcome | Conclusion supported | Next bounded action |
|---|---|---|
| Pure T2V is coherent while current clamped clips fail | The base can generate coherent content at this tested shape; the failed adaptation/sampling bundle is implicated | Restore the image-conditioning experiment one factor at a time. No new large training run yet. |
| Pure T2V still shows lattice artifacts and native forward parity fails | There is a concrete port/numerical mismatch | Repair parity and rerun the same seed; do not attribute the failure to data scarcity. |
| Pure T2V fails despite forward parity and faithful solver/negative context | The reduced-size/short-clip setting, broader input/codec path or base behavior still needs diagnosis | Next consider one supported 832×480 clip only after its memory profile, or a tiny native T2I control. The 512×288 failure does not establish failure of the recommended native 480p recipe. |
| FP32 decode alone substantially fixes the saved latents | Decoder precision explains a material part of the visual failure | Preserve both decodes and repeat comparisons using the justified decoder; action correctness remains a separate measurement. |
| Time/memory guard stops the control | The chosen native-style path was not completed on this configuration | Record the limit. It supplies no positive or negative generation-quality result. |

Even a successful pure T2V control would be an external foundation-model diagnostic. It would not satisfy the user's action-conditioned world-model goal. Its purpose is to prevent spending another training budget on an unverified generation path.
