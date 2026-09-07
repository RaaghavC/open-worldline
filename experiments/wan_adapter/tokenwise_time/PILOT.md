# Fixed 16-update development pilot

This separate path passed seven implementation tests and six independent CPU tests. It has not run on the GPU and has produced no trained 16-update checkpoint or video. The [CPU reports and source hashes](results/pilot-preflight-v2/provenance.json) preserve that review. The completed [one-update profile](results/update-v1/README.md), its source snapshots, and all previous native/clamp experiments remain unchanged.

The experiment asks whether the original small adapter can learn under the same image-conditioning rules used by its sampler. The independently encoded first observation is clean and its 576 spatial tokens receive time 0. The other 2,304 tokens receive the exact sampled time `k`, with future latent noise fraction `k/1000`. Training draws integer `k` uniformly from 50 through 950. Loss applies only to the four future latent frames. All examples are image-conditioned; there is no uniformly noised text-to-video mixture.

The frozen 1,418,996,800-parameter Wan2.1 T2V foundation uses the separately attributed, CPU-tested FP32 tokenwise extension. A fresh 1,349,376-parameter original action/observation adapter is the only trained module. It receives the 16 ordered six-channel commands and 32 pooled tokens from the first observed image. Those image tokens are not persistent memory. This training path does not reproduce Wan2.2 TI2V-5B or establish that the frozen T2V foundation becomes a general image-to-video model.

## Fixed protocol

Adapter initialization seed is 20260912. The training order is:

```text
closed-0000, open-0000, closed-0008, open-0008,
closed-0032, open-0032, closed-0049, open-0049,
closed-0000, open-0000, closed-0008, open-0008,
closed-0032, open-0032, closed-0049, open-0049
```

There are exactly 16 AdamW updates, batch size 1, learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, and gradient clipping at norm 1.0. No CFG is used in training. All core and adapter computation is FP32. Non-reentrant checkpointing replays the frozen block forward while the adapter hook stays outside replay, retaining gradients into earlier adapter injections. The existing tested one-update implementation is imported unchanged.

A separate CPU generator seeded 20260913 supplies the 16 training noise/time draws. Another seeded 20260914 supplies eight fixed diagnostic draws, one per window. All draws are saved before the core is loaded. Every diagnostic uses exactly the same noised latent, velocity target and token times before and after training. Predictions and future-only flow MSE are retained for all eight windows. These are development diagnostics on the same one-layout capture used for training. They do not measure generated-image correctness or generalization, and they cannot select a checkpoint. Only update 16 is eligible for the prescribed sampling comparison.

The runner checks all 42 adapter gradient tensors, positive aggregate gradient norms in each of the three residuals, finite updated parameters, hook removal and unchanged base tensor/file hashes. After each checked update, an atomic recovery bundle stores adapter weights, optimizer state, the completed schedule and CPU RNG state. The complete set of pre-sampled inputs is a separate hashed artifact. The recovery bundle preserves evidence after interruption; automatic resume is not implemented.

Each process retains the existing 900-second, 18 GiB sampled RSS/Metal driver, and 2 GiB minimum available-memory stops. Automatic MPS CPU fallback must be disabled. The proven single update took 15.66 seconds on this Mac; multiplying it by 16 excludes the new before/after diagnostics, data validation, loading and checkpoint writes. It is not a measured duration for this pilot.

## CPU review and execution

Run from the repository root with the isolated Wan dependencies in [requirements-real.txt](../requirements-real.txt). The tests load no foundation weights and use no GPU. Evidence paths must be new:

```sh
python experiments/wan_adapter/tokenwise_time/test_pilot.py \
  --output /path/to/new/pilot-cpu.json
python experiments/wan_adapter/tokenwise_time/test_pilot_independent.py \
  --output /path/to/new/pilot-independent.json
```

The training runner requires passed reports whose hashes match its current helpers, both entry points, the tests and imported shared sources. Keep the reviewed source frozen during the run. Launch the following only after the GPU reservation is granted:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/wan_adapter/tokenwise_time/train_pilot.py \
  --weights /path/to/verified/wan-adapter-weights \
  --capture-cache experiments/wan_adapter/data_cache \
  --text-cache experiments/wan_adapter/text_cache/results \
  --cpu-report experiments/wan_adapter/tokenwise_time/results/pilot-preflight-v2/cpu-tests.json \
  --independent-report experiments/wan_adapter/tokenwise_time/results/pilot-preflight-v2/independent-tests.json \
  --output /path/to/new/tokenwise-train16
```

No dependency or weight download is performed by the runner. `initial-adapter.safetensors`, `final-adapter.safetensors`, `fixed-inputs.safetensors`, `recovery-last.pt`, full diagnostics, sources, runtime samples and metrics are saved. A stopped or failed run is ineligible for sampling. Checkpoints are new original adapter weights; the attributed unchanged external foundation is not copied into results.

## Prescribed video comparison

The prescribed input is `open-0000`. Each clip contains one reconstructed initial observation and 16 newly generated future frames at 512 by 288 pixels. The sampling reader materializes only the separately encoded observation and actual command tensor from the public cache. It does not load the future target latent or future RGB images. Hashing the opaque cache file checks integrity without materializing its target tensor.

Run two separate processes, first with `--arm zero`, then with `--arm trained`. Both load the completed training record. The first uses its exact fresh initial adapter; the second uses update 16. Both use the retained Gaussian noise from the successful pure T2V control, seed 20260908, the same positive and native negative text tensors, FP32 core/adapter/decoder, and the unchanged official CPU UniPC solver with 50 steps, shift 8 and guidance 6. Both use the same actual commands and independently encoded initial observation. The saved noise file and tensor hashes define this input exactly. The loader does not require a different CPU platform to regenerate those bytes from the same seed. The [earlier Linux CI failure](https://github.com/RaaghavC/open-worldline/actions/runs/34138436083) and [updated checks](results/pilot-preflight-v2/provenance.json) are retained. PyTorch documents that identical results across platforms are not guaranteed, even with controlled seeds. [PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

Before both negative/positive calls, the initial latent is clamped and its tokens receive time 0. All future tokens receive the solver's exact integer time. The initial latent is clamped again after every solver update. Each arm records all 100 input prefix checks and all 50 post-update checks. Callers' original noise and observations remain unchanged.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/wan_adapter/tokenwise_time/sample_pilot.py \
  --weights /path/to/verified/wan-adapter-weights \
  --capture-cache experiments/wan_adapter/data_cache \
  --text-cache experiments/wan_adapter/text_cache/native-results \
  --training-run /path/to/completed/tokenwise-train16 \
  --cpu-report experiments/wan_adapter/tokenwise_time/results/pilot-preflight-v2/cpu-tests.json \
  --independent-report experiments/wan_adapter/tokenwise_time/results/pilot-preflight-v2/independent-tests.json \
  --arm zero --output /path/to/new/tokenwise-zero50
```

The second command changes only `--arm trained` and its new output directory. The sampler retains the first actual step and uses its measured time to estimate the remaining 49 steps plus a 60-second decoder/artifact allowance. An estimate over 900 seconds stops the arm and preserves that evidence. Each arm still has an independent 900-second watchdog. No step count or precision changes automatically to fit the cap.

All 17 images, generated latents, conditions, initial Gaussian noise, source files, checkpoint/text/data hashes, prefix checks, timings and failures are retained. Preview playback is 10 frames per second; this is unrelated to generation speed. Quality assessment must inspect both completed clips and the same original observations. One window and 16 development updates cannot establish broad action control, persistent world memory, photorealism or Genie 3 parity.
