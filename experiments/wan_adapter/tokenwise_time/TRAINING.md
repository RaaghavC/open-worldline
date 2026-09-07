# One-update tokenwise training profile

This separate path prepares one real-data adapter update on the existing original Atrium development capture. Five mechanics tests and five independent tests passed with small random-weight models on CPU. The independent checks also verified the actual original data-cache hashes and tensor separation. The separately scheduled MPS profile has now completed one adapter update, taking 15.658 seconds for the update and 54.659 seconds for the full measured interval. [The exact evidence](results/update-v1/README.md) is retained. No image generation or generalization evaluation was performed.

The pilot uses **100% image-conditioned examples and 0% ordinary text-to-video examples**. This is an explicit mechanics experiment. It makes no claim that this mixture is sufficient to preserve general text-to-video behavior or learn image continuation across scenes.

## Trainable parameters and conditioning

The exact official Wan2.1 T2V 1.3B parameter layout remains frozen in full FP32: 1,418,996,800 parameters. The separately tested tokenwise-time methods replace only the time broadcasting. Existing native-control code and the earlier adapter checkpoints remain unchanged.

A fresh instance of the original [ActionObservationAdapter](../adapter.py) trains all 1,349,376 of its parameters. It receives the same `[B, token, hidden]` block outputs as before, so its residual interface is compatible with the tokenwise time change. Three residuals run after blocks 9, 19 and 29. Their output projections start at zero. The runner checks that attaching this fresh adapter gives exactly the same prediction as the extended frozen core before the update. It does not load the earlier adapter trained under the failed scalar-time clamp contract.

At batch size 1, the tensors are:

| Tensor | Shape | Permitted use |
|---|---|---|
| Original encoded target clip | `[1,16,5,36,64]` | Future corruption and flow loss |
| Independently encoded first observation | `[1,16,1,36,64]` | Clean prefix and adapter's observed-image input |
| Original commands | `[1,16,6]` | Four ordered intervals for each future latent frame |
| Noisy model input | `[1,16,5,36,64]` | Clean first latent followed by noised future latents |
| Token times | `[1,2880]`, integer | First 576 spatial tokens at 0; remaining 2,304 at k |
| Transformer block output | `[1,2880,1536]` | Input to each attached residual |
| Genuine positive text context | `[25,4096]` | Native padding to 512 tokens before text projection |

The adapter receives `observation[:,:,0]`, retaining batch and channel dimensions. It pools this observation into 32 tokens. These are initial-image features, not persistent memory. All command values come from the original capture. A mechanics profile does not establish that the trained adapter follows those commands.

## Noise, labels and loss

A seeded CPU generator draws integer `k` uniformly from 50 through 950 inclusive. The training arithmetic uses FP32 `sigma = k / 1000`. Future latents are `(1-sigma) * target + sigma * noise`. The initial latent comes from the independent observation tensor and is never selected from the full target clip.

Observed tokens receive integer timestep 0. Future tokens receive the exact sampled integer k. This is the same per-token contract required by a later image-conditioned sampler. The profile records k, its exact fractional form, the actual FP32 sigma, noise, times and noised training input. Model time must not be rounded from a separately sampled continuous sigma.

The prediction target is `noise - target` on future positions. Mean squared error excludes the observed latent frame. There is no classifier-free guidance during training, no negative-prompt forward, no uniformly noised T2V mixture, no target-derived observation and no video decoding.

## Gradients and checkpointing

Each frozen block's forward is wrapped with non-reentrant activation checkpointing. Adapter forward hooks execute outside the checkpointed callable. Their outputs remain in autograd, so gradients from the future loss can pass through later recomputed frozen blocks into earlier adapter residuals. Hooks stay attached through backward, then are removed even on errors. Original block methods are restored after the update.

CPU tests compare checkpointed and uncheckpointed loss and every adapter gradient after nonzero adapter weights are activated. They also test two optimizer updates, nonzero gradients in the earliest residual, unchanged frozen parameters, exact zero-initialized behavior, future-only loss, independent observation use and partial hook-attachment cleanup. The runtime update requires a finite gradient tensor for every adapter parameter and a finite positive aggregate norm in each residual. Individual zero gradients are allowed at initialization.

The FP32 model's RoPE table is moved and cloned outside inference mode before the zero-initialization check. Otherwise a plain tensor attribute first created during inference could later be invalid for backward.

## Bounded command after CPU review and GPU scheduling

From `experiments/wan_adapter`, with the existing isolated dependencies, official base files and verified data/text caches:

```sh
python -m pip install -r requirements-real.txt
python tokenwise_time/test_training.py --output results/tokenwise-training-tests.json
python tokenwise_time/test_training_independent.py --output results/tokenwise-training-review.json
PYTORCH_ENABLE_MPS_FALLBACK=0 python tokenwise_time/profile_update.py \
  --weights external-weights \
  --capture-cache data_cache \
  --text-cache text_cache/native-results \
  --window open-0000 \
  --training-report results/tokenwise-training-tests.json \
  --independent-report results/tokenwise-training-review.json \
  --output results/tokenwise-update1
```

The last command reproduces the completed one-update MPS profile in a new directory. It performs exactly one AdamW update with learning rate 0.0001 and gradient clipping at 1. It requires current passed CPU reports, disables automatic MPS CPU fallback and uses the existing 900-second, 18-GiB and 2-GiB-available guards. Explicit CPU noise, time calculations and source hashing remain part of the declared path.

Every output directory must be new and outside the source directory. The runner preserves exact executed source, input hashes, sampled tensors, initial and updated adapter checkpoints, finite-gradient checks, per-block recomputation counts, timings and memory samples. It verifies the frozen base's actual tensor bytes before and after the update and checks the official source/config files again. External foundation weights are not bundled or downloaded by this command.

The measured full-FP32 update peaked at 8.687 GiB sampled Metal driver allocation. All 42 adapter gradient tensors were finite and every residual received a positive aggregate gradient norm. These measurements establish mechanics and resource use for this one update. They do not establish image quality, generalization, action control or a new foundation model. A reproduction remains subject to the same guards.
