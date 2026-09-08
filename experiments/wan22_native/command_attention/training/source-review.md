# Bounded training-source review

This is a textual integration review by the video_review_close agent. It covers the four source files listed below. No tests were rerun, no images were inspected, and no model or provider action was performed for this review.

No open material finding remains in the reviewed versions. This conclusion concerns the source logic; it does not establish that native CUDA parity, the resource profile, 512 updates, or visible controls have passed.

The reviewed code uses each future target as an FM loss label and as part of the standard noisy FM training input. Auxiliary and evaluation future inputs use saved Gaussian noise with only the shared observation prefix restored. Commands and text are the other conditions. The endpoint contrast has the declared orientation, `-(G_b-G_a)` against `z_b-z_a`, with `G=N+5*(P-N)`. Two sequential main half-loss backwards precede the optional four-graph auxiliary backward, one gradient clip and one AdamW step. Foundation parameters remain outside the optimizer.

The input reader binds the manifest and file hashes, validates all 512 scheduled rows, and checks each selected saved noise and RNG state. Preparation preserves the first eight original draw shards, including `rng_initial` in the first shard. It continues the private generator from `rng_after_0127` with integer timestep then Gaussian-noise draw order. The runner reads `rng_initial` for checkpoint zero and the corresponding saved `rng_after` value for checkpoints 128, 256, 384 and 512.

The runner cycles three motion pairs and the seven declared auxiliary edges every fourth update. It retains all 512 scalar update records, five declared raw updates, five checkpoints and 48 initial plus 48 final endpoint predictions. Evaluation uses four fixed noises, six arms and both text contexts. Targets enter only the subsequent contrast scoring. The fixed final checkpoint is used; no score-based checkpoint selection appears in this path. Omitted training velocities and gradients remain disclosed as unavailable for complete replay.

The source/profile/input hashes, explicit admission, exact GPU/runtime comparison and UTC lease bind the planned run. The parent and worker use the existing 1800-second guard and preserve 600 seconds for recovery. Two-step timing extrapolation remains an estimate; actual runtime and memory guards still apply.

Two integration issues were sent to root and resolved before closure: the reused source collector needs non-Python resource files beyond the three native JSON configs, which root added to the transfer stage; and the preparer imported `math_steps` before setting its repository path, which root moved into `prepare` after path setup. The input packet was not regenerated for the import fix.

Reviewed source SHA-256 values:

- `math_steps.py`: `bd460ae93d34670da702cf7cd3d70f9d118f654286437ea523459e641d79db50`
- `packet.py`: `f838763a9777c61a08179a70c1fee8f8a3588063853abb243ee4d11b35bdcaa3`
- `prepare_inputs.py`: `e40f1792e034944cd334dbf6bf92bb5532fdd05c2180f3cc4f462f7d8ba7d724`
- `run_training.py`: `80cb395b202995219b2d2c561cc8ed2f453a27fd5700644ab2cdcdf028254c2a`

The final training freeze was received and its four reviewed source hashes match this report. Freeze SHA-256: `de512d91be83d2a127104c31f46c323a7dd3bff28e1d98cfa28d29c339351ab2`.
