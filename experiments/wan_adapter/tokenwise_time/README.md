# Untrained tokenwise diffusion-time extension

This CPU prototype adapts the existing Wan2.1 T2V parameter layout to accept a separate diffusion timestep for each video token. It adds no parameters, loads no checkpoints, trains nothing, and generates no images. The change follows the established conditioning design in official Wan2.2 TI2V-5B. It does not reproduce the 5B model or establish an image-quality improvement.

The completed controlled clamp experiment showed clear initial-frame reconstruction followed by lattice-like textures and changing scene geometry. The tested Wan2.1 core received one global timestep for both its clean observed prefix and noisy future frames. Official Wan2.2 TI2V instead supplies timestep 0 for the clean observed tokens and the current scheduler timestep for future tokens. It also has weights trained for that use. Correctly assigning time is necessary to test this design; it does not supply the missing image-conditioning training.

## Source and changes

[The new module](portable.py) reuses the exact verified [Wan2.1 FP32 portable core](../native_control/portable.py). It binds three forward methods to the created model instance only: time embeddings are computed per token, block modulation broadcasts per token, and head modulation broadcasts per token. Parameter names, shapes and values remain unchanged. Existing native-control and clamp code and measured sources are not edited.

Official Wan2.2 source at commit `42bf4cfaa384bc21833865abc2f9e6c0e67233dc` is retained under [source-reference/](source-reference/), with its Apache-2.0 license, exact URLs and hashes. These files document the source design; they are not loaded as our 1.3B network and carry no 5B weights.

`token_times(grid_sizes, timestep, seq_len, observed_latent_frames=1)` constructs integer times in the exact F,H,W flatten order used by patch embedding. With temporal patch size 1, all Hpatch*Wpatch tokens of the initial latent frame receive 0. Future and padding tokens retain the supplied integer scheduler time. The helper does not change latent values. Model forward requires explicit `[B,sequence]` integer times and full FP32 parameters, inputs and contexts. It has no action, image-cache, future-target, adapter or sampler argument.

## CPU evidence

[The first parity report](results/cpu-parity.json) compares final velocities, time/text projections, block outputs, block time inputs, expanded block modulation and head modulation. It uses the actual 128-wide attention heads, two blocks, unequal video/text lengths, padding and batches with different timesteps, including 999, 500, 50 and 0. All 56 checked uniform-token-time comparisons had maximum absolute difference 0.0; the declared absolute/relative tolerance was 0.00002. Mixed-time outputs are finite. Parameter names and tensor bytes remain identical, and subsequently created original models keep their original methods.

These are small random-weight CPU checks. They do not show full-model GPU equality, trained image continuation, action response or persistent memory. No image generation should be inferred from a passed numerical test.

From `experiments/wan_adapter` in the existing isolated environment:

```sh
python -m pip install -r requirements-real.txt
python tokenwise_time/test_parity.py --output results/tokenwise-parity.json
```

Each evidence output must be new. No external weights are required for these checks. A later training experiment must use the same observation/noise/time contract at training and inference, with independently encoded observed data, no future-target conditioning and explicit development/held-out splits. No training or GPU generation is performed by this package.

The [foundation and licensing audit](../../../docs/wan-image-conditioning-design.md) compares the native I2V options and the actual download and memory requirements. [Raw source and parameter-count evidence](source-audit/) is retained separately from this CPU prototype.

[Six independent CPU checks](results/independent-tests.json) passed in 0.788 seconds. They cover all 50 official solver timesteps on the actual 5 × 18 × 32 patch grid, variable padding, Conv3d flatten order, scalar time-embedding oracles for every block/head, independent shift/scale/gate equations, and unchanged parameter objects and original model instances. No external weights or GPU operations were used.

```sh
python tokenwise_time/test_independent.py --output results/tokenwise-independent.json
```
