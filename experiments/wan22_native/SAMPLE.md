# One native Wan2.2 clip

`sample_clip.py` prepares one 512 by 288 clip from the attributed Wan2.2 TI2V-5B port. The output has one reconstruction of the observed starting image and 16 generated future frames. This is an external pretrained baseline, with no trained Worldline adapter, action input, persistent memory or new-model claim. The 17-frame, 512 by 288 experiment is smaller than Wan2.2's usual advertised resolution and duration.

The new runner preserves the existing core and codec implementations. The transformer uses the measured explicit selective-BF16 storage/execution policy, with the native FP32 time, normalization, residual and head paths. The separate 48-channel codec uses FP32. This port's CPU equation checks do not establish numerical identity with the original CUDA kernels.

## Inputs and fixed sampling

The runner requires a successful actual one-pair core profile and a successful actual full 17-frame decode-cost profile. The pair must have both a passed worker report and a completed parent terminal with exit code 0. Any watchdog stop, incomplete stage, changed source, changed weight conversion or changed tensor hash prevents generation.

The pair's retained `inputs.safetensors` defines the exact Gaussian noise, independently encoded clean starting observation and first-step token times. The saved bytes are canonical across platforms. The seed is metadata; the runner does not regenerate noise and compare it across ARM and x86. Genuine cached positive text and the exact native negative text must match the pair's hashes. Copies of both text tensors are included in the new run's input file.

There are exactly 50 official CPU UniPC updates, shift 5 and guidance 5. The scheduler is created with shift 1 and receives shift 5 once when setting the time sequence. Each update calls the positive context, then the negative context, on the same latent and token times. Guidance is `negative + 5 * (positive - negative)`. The first 144 of 720 tokens receive time 0. All future tokens receive the scheduler's exact integer time. The clean prefix is restored before each of the 100 denoiser calls and after each of the 50 solver updates. These boundaries are recorded individually.

The starting native sigma is slightly below 1, approximately 0.99979985, and the final sigma is 0. A constant-velocity CPU oracle uses that actual interval. An earlier test assumed exactly 1; its failed report and source are retained separately. The integration code itself matched a separately written projected UniPC loop.

Only the initial observation, noise and real text enter inference. No action tensor, adapter checkpoint, future RGB or training target is accepted. The core and decoder run in separate child processes. The decoder starts after the transformer process has exited successfully and its final latent file has been verified.

## Measured admission and stops

Before loading model weights, a CPU-only, full-shape 50-step solver oracle supplies a measured solver cost. The admission estimate adds:

- Measured transformer loading time.
- 50 times the measured pair time.
- 50 times all other elapsed pair-profile time except loading. This deliberately repeats some one-time input/output work to include unassigned guidance, transfers and other overhead.
- The measured full 50-step CPU solver cost again, covering later multistep solver work beyond the first measured step.
- The complete measured full-decoder elapsed time, including its own weight load and saved diagnostic images.
- 45 seconds for child startup and additional artifact work, plus actual preflight elapsed time.

The estimate must fit within 900 seconds. It is a conservative projection from one measured pair, not a runtime guarantee. An infeasible result stops before generation; there is no automatic reduction in steps, resolution or precision.

One parent watchdog covers validation, both child processes and output writing under a combined 900-second deadline, 18 GiB combined sampled RSS ceiling and 2 GiB minimum available system memory. Each child also samples its Metal allocation under the existing 18 GiB ceiling. A stop retains the available inputs, source, memory record, completed-step latent and error records. A stopped run cannot be reported as complete.

The core and decoder each use their own measured environment. Low/high allocator watermarks, automatic CPU fallback, fast-math and preferred Metal-kernel settings are explicitly set or removed before each child starts. The child checks those values and recommended device memory before loading weights. A decoder profile with low watermark `0.6` does not silently change the core's measured configuration.

The decoder also preserves the measured per-chunk and optional per-convolution cleanup policies. If per-convolution cleanup was used, every layer's invocation count must match the completed cost profile. The resulting callback log is retained; no cleanup policy is silently added to the core.

## Reproduction

Use the existing isolated environment, or install `requirements-sample.txt` into a new environment. Weights must already be downloaded and match the pinned official identities in `provenance.json`; the runner downloads nothing. Run commands from the repository root. CPU checks load no foundation weights:

```sh
python -m experiments.wan22_native.test_sample_clip --output /path/to/new-sampler-cpu-report.json
```

After both actual profiles have completed, this command validates them, runs the small CPU solver benchmark and retains an admission report without launching model workers:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m experiments.wan22_native.sample_clip \
  --weights /path/to/wan22-ti2v5b-weights \
  --pair-profile /path/to/completed-native-core-pair \
  --decode-profile /path/to/completed-native-decode17 \
  --text-cache experiments/wan_adapter/text_cache/native-results \
  --cpu-report /path/to/new-sampler-cpu-report.json \
  --device mps --plan-only --output /path/to/new-native-clip-plan
```

For a separately scheduled generation, use the same inputs, omit `--plan-only` and choose another new output directory. There are no CLI controls for changing the declared step count, guidance, shift or precision. The first two stopped decoder records do not satisfy these prerequisites. The completed third decoder profile supported the [first full clip](sample-results/clip50-v1/README.md), which completed execution but failed visual inspection.

The output retains source snapshots, measured-profile records, input and weight identities, exact noise/observation/text tensors, every step's time and prefix checks, raw final latents, raw decoded FP32 pixels, all 17 unannotated PNG frames and an annotated contact sheet. GIF preview playback is 10 frames per second for inspection. It is separate from generation throughput. Count 16 new future frames when reporting generated-frame speed.
