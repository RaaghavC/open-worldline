# Original RGB to native CUDA training data

This stage prepares eight development windows from the original Atrium capture: closed/open door branches starting at frames 0, 8, 32 and 49. Each has 17 RGB frames and 16 recorded commands. It has no transformer, optimizer, generated-video sampler or provider API.

The spatial profile applies the existing 1248 × 704 resize/crop to the original 512 × 288 images. Enlarging an image adds no original scene detail. Before resizing, the established start-zero correction changes exactly 18 one-level channel values in the derived closed first frame, giving both branches the same original starting image. The source capture remains unchanged.

The native FP32 CUDA VAE independently encodes starting observations and complete targets. Raw target latents are retained. Cross-length first-latent differences have predeclared maximum-absolute and relative-L2 limits of 1e-5. Same-length future-only perturbation and repeated single-image checks require exact equality. A failed check stops completion and retains partial evidence. These limits are software/data admission rules, with no real CUDA cache result yet.

`cache_run.py` owns one separate `cache_worker.py` process. The existing codec supervisor imposes a 600-second total deadline, including planning, loading, encoding and final verification. The worker samples CUDA memory and host memory; the parent also checks their combined host memory. Limits remain 48 GiB combined host RSS, 60 GiB CUDA-reserved memory, and at least 8 GiB available host/GPU memory on the reviewed A100 80 GB environment. This does not create or delete a rental; external instance cleanup remains separate.

From the repository root, run the bounded CPU tests and retain their exact source identities:

```sh
python -m experiments.wan22_native.action_cuda.cache_review \
  --output /absolute/fresh-cache-cpu-report.json
```

Prepare a plan with the original capture and the reviewed report. The weights path is recorded but no weights are opened in plan mode:

```sh
python -m experiments.wan22_native.action_cuda.cache_run \
  --profile spatial --expected-gpu 'NVIDIA A100-SXM4-80GB' \
  --capture /absolute/original-atrium-capture \
  --weights /absolute/Wan2.2-TI2V-5B \
  --cpu-report /absolute/fresh-cache-cpu-report.json \
  --output /absolute/fresh-cache-plan
```

Execution additionally requires `--execute` and `--input-plan-sha256` with the exact `input_plan_sha256` returned by that plan. Use another fresh output directory. The worker must have the pinned native CUDA environment and original VAE checkpoint. Existing MPS latents cannot satisfy this cache contract.

Successful output contains `result/manifest.json`, `result/completion.json`, all eight window tensors, seven unique starting observations and every causal-check tensor. Original codec load records and monitor reports are under `worker/`. The supervisor retains `terminal.json`; the parent writes `metrics.json` with a completed-result hash. A partial cache or failed monitor cannot admit a training probe.

This is one original scene and fixed-position camera rotations. Only the shared start-zero branches isolate the outgoing door command. Later pairs have different observed door states and identical commands. Training losses on these development windows cannot establish learned action control, unseen-scene quality, persistent world memory or Genie 3 parity.
