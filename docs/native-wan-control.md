# A coherent pretrained-video control on the Mac

September 7, 2026. One pure text-to-video control using the externally pretrained Wan2.1-T2V-1.3B model completed on the 24 GB M4 Pro. It generated all 17 frames at 512 × 288 from Gaussian noise and real text context. The clip depicts a recognizable sunlit doorway, plaster, paving and plants. Composition changes slowly, and the rendering is saturated and partly stylized. The repeated surface lattice in the earlier adapted clips is absent in this inspected sequence.

This is a working external-foundation diagnostic, not an original trained world model, action-control result or Genie 3 comparison. The result supports continuing development on the checked generation path. It does not establish which particular difference caused the previous failure.

![Actual generated frames from the fixed single control](../experiments/wan_adapter/native_control/results/clip50/comparison.png)

## What ran

The run uses the same pinned official 1,418,996,800-parameter Wan foundation as the failed adapter pilot. The network follows the literal official model source with full float32 parameters and activations. Explicit compatibility changes provide float32 SDPA attention, real-valued positional rotation and CPU timestep calculations. Small CPU tests compare the port against the literal source, including intermediate time, block and output values. This is not bitwise reproduction of native CUDA BF16/FlashAttention execution.

The positive prompt is unchanged: a sunlit interior with warm plaster walls, a wooden door, limestone flooring, brass details and green plants. The negative condition uses the actual [official configured text and its genuine UMT5 encoding](../experiments/wan_adapter/text_cache/native-results/README.md). The [source comparison](../experiments/wan_adapter/text_cache/official-source-audit/README.md) confirms matching selected and official text-network/tokenizer definitions; numerical precision remains separately declared.

Seed 20260908 supplies a saved, hashed full Gaussian latent tensor. No initial image, action adapter, camera pose, renderer call or future target enters generation. Every latent starts noisy. The unchanged official UniPC scheduler performs 50 updates with shift 8 and guidance 6, using 100 sequential negative/positive model calls. The small solver runs explicitly on CPU in float32 because this PyTorch MPS backend lacks its required linear solve. Latents and velocities transfer explicitly between the CPU solver and GPU denoiser; automatic operator fallback is disabled. A changing-velocity test and an actual-shape solver/transfer check verified this boundary against the all-CPU solver.

The denoiser is released before the official VAE decodes in float32, with the previously validated allocator cleanup after each original temporal chunk. The initial-image clamp from the failed experiment is absent. All 17 decoded frames are generated, including frame zero.

## Measured cost and failures

| Measurement | Result |
|---|---:|
| Network load | 7.82 s |
| 50-step denoising | 617.32 s |
| VAE load | 0.88 s |
| VAE decoding | 30.93 s |
| Measured validation/load/generation/decoding/artifact interval | 663.76 s |
| Sampled peak Metal driver allocation | 10.40 GiB |

The clip produced approximately 0.026 newly generated frames per second when counting denoising and decoding. Preview playback speed is independent of generation speed. This is an offline clip, with mostly static composition, rather than an interactive simulator. The chosen 512 × 288, 17-frame shape is smaller and shorter than the official 480p, 81-frame example.

A reviewed one-pair profile predicted 736.99 seconds including explicit solver/transfer and decoder allowances. The completed run stayed inside its 900-second, 18-GiB and minimum-2-GiB-available guards. Sampled memory can miss short peaks; driver allocation, active tensors and process RSS overlap.

Two pre-generation compatibility failures are retained. The first profile attempted an unsupported float64 MPS conversion while moving a timestep to CPU; separating the transfer and cast fixed it, with new source-matched parity checks. A solver preflight then found the unsupported MPS linear solve; the explicit CPU solver resolved that boundary without changing its equations. Neither failed attempt generated a video or supplied a quality result.

## Starting-image comparison

The previous [Atrium adaptation](wan-atrium-pilot.md) used a different precision path, negative context, solver, step count and first-image condition. The coherent result here shows that the pretrained model can generate a recognizable scene through the checked local path at this resolution. Restoring several choices at once does not isolate any one cause of the earlier lattice.

The subsequent single comparison added the encoded first frame from the original Atrium capture, holding it fixed during every denoiser call and solver update. It retained the pure control's saved noise, positive and negative text, scheduler, precision, guidance and step count. Only the observation tensor was loaded from the capture cache. No action adapter, actions or future target entered generation. The output contains one observed reconstruction and 16 generated future frames.

The comparison completed 50 solver steps in 613.64 seconds and decoded in 30.87 seconds, within a measured 660.30-second interval. The sampled peak Metal driver allocation was 9.49 GiB. The fixed prefix remained exact, and outputs were finite. Those checks establish execution, not image quality.

The observed reconstruction is sharp. Later generated frames change the door shape and room layout, introduce a bench and plant, and develop a repeating lattice-like surface texture. For this fixed seed and input, adding the starting-image clamp reintroduced a visible defect despite the otherwise successful settings. This does not establish a population failure rate or isolate the model's internal cause. It does show that this text-to-video checkpoint does not yet provide a usable starting-image world simulator through the tested clamp.

More updates of the previous adapter are not justified by its denoising loss alone. The next design needs a training and inference treatment of observed frames that the model can learn, with an explicit comparison to a native image-conditioned model. A pure text-to-video result contains no evidence of following a supplied camera command, reacting to an interaction, remembering a hidden object or supporting editable persistent worlds.

The [native-control directory](../experiments/wan_adapter/native_control/) contains the measured source, numerical checks, failed preflights, runtime profile, full output sequence and original latent/noise tensors. External foundation weights remain separate and attributed under their upstream Apache-2.0 license. No hosted generation API was used.
