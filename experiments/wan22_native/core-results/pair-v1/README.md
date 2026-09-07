# Actual native 5B core: one pair completed

The pinned Wan2.2 TI2V-5B transformer completed one positive-text and one negative-text forward pass on the M4 Pro, followed by one official CPU UniPC solver step. This is a core execution measurement. It does not contain a generated video or demonstrate action control.

| Stage | Measured interval |
|---|---:|
| Verify and stream the official transformer | 96.110 s |
| Positive-text forward | 5.765 s |
| Negative-text forward | 3.263 s |
| CPU UniPC step | 0.003 s |
| Complete worker | 105.398 s |

The two forwards totalled 9.028 seconds. Multiplying that single-pair interval by 50 gives 451.41 seconds, excluding loading, decoding and later-step variation. The first positive call includes its own initial execution overhead. This estimate does not admit a full clip by itself.

The reduced diagnostic uses a 17-frame, 512 × 288 latent layout: 48 channels, five latent times and 720 spatial/temporal tokens. The first 144 tokens represent the independently encoded starting image and receive time zero. The remaining tokens receive the native solver time. The positive and negative contexts, saved noise, velocities and one-step result are retained. All outputs were finite and the observed prefix was exactly preserved after the solver step.

The original transformer has 4,999,787,712 parameters in 825 tensors. The declared execution policy stores selected time/head/norm/modulation parameters in FP32 and other specified weights in BF16. The source checkpoint and each loaded conversion were verified. Sampled peaks were 922,566,656 process RSS bytes, 10,428,402,176 active MPS bytes and 10,864,607,232 driver bytes. These values exclude unobserved between-sample peaks and are not a general hardware requirement.

This run used its recorded default allocator thresholds with automatic CPU fallback disabled. Only the separately specified CPU solver and helper operations run on CPU. The decoder's later allocator experiment is a different process policy. External model weights are not included in this publication.

All 23 measured files, including checked source, complete runtime logs and original tensor artifacts, are preserved byte for byte. [publication.json](publication.json) records their hashes. [metrics.json](metrics.json) and [weight-load.json](weight-load.json) contain the complete measured details. Full native video sampling still requires a successful complete decoder cost measurement and the separate sampling checks.
