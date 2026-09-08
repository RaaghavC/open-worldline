# Six-arm native VAE cache preparation

The source and CPU preparation are complete. No VAE weights, CUDA model or cloud resource was executed by this preparation task. Twelve CPU checks passed in 6.387 seconds. A separate read of the complete prepared RGB packet passed. Actual encode time, GPU memory use and causal prefix agreement remain unmeasured.

This cache uses the published native-resolution Atrium camera-by-door capture: stationary, left and right motion, each with closed and interact commands. Each arm contains 17 original 1248×704 images and 16 destination-aligned commands. All six first images are already byte-identical. The command yaw is 0 or ±π/120 per transition, reaching 0 or ±24 degrees at frame16. These are not the old enlarged images or the old 15×7.5-degree sequence.

The unchanged `experiments.atrium_factorial.reader` supplies only initial RGB, future RGB targets and commands. Its exact FP32 normalization is retained. `cache.video` only concatenates and reorders those arrays into `[1,3,17,704,1248]`. It performs no resizing, cropping, compositing, normalization, initial-image replacement or pose-derived command generation. No old latent cache is used. The VAE receives RGB only; commands are retained beside the latents.

## Encoding and checks

The worker imports the unchanged `cuda_reference.decode.load_codec` and calls `action_cuda.data.native_encode(model, scale, video, 'spatial')` directly. It never calls the old `data.encode_cache`, `processed_rgb` or canonicalization code. The original 48-channel VAE runs in FP32 with CUDA autocast disabled. No transformer, text encoder, action adapter, decoder or optimizer is loaded.

Eight calls run sequentially:

1. Encode the shared first image independently.
2. Encode the six complete clips in published arm order: stationary_closed, stationary_interact, left_closed, left_interact, right_closed, right_interact.
3. Encode the same independent first image again after the six video calls.

All returned tensors are saved before checking numerical values or prefix limits, so a completed failed output remains inspectable. Each target keeps its original encoded prefix. Six comparisons use the existing cross-length limits, max absolute error ≤1e-5 and relative L2 error ≤1e-5, while reporting exact equality separately. Five comparisons require bit-exact first-latent agreement between equal-length videos. The repeated one-image result must be bit-exact to the first encode. This is an A/video-calls/A cache-isolation check, not an additional independently encoded B image. All twelve checks retain unambiguous references to saved candidate/reference tensors and their hashes.

The original loader verifies the VAE file and its 196 FP32 CUDA copies. The worker independently hashes all 196 current parameter values before and after encoding, requiring the loader identities, frozen parameters and absent gradients. It also checks that the native normalization tensors are unchanged and all VAE feature caches are empty between calls and at exit. Verification holds only one parameter copy at a time. One arm's RGB is processed at a time.

## Preparation and future execution

From the workspace root, the completed preparation commands were:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=outputs/open-worldline \
work/wan-adapter-env/bin/python work/atrium-factorial-native-cache-v1/run.py \
  --prepare --capture work/atrium-factorial-public-download-v1/capture \
  --cpu-report work/atrium-factorial-native-cache-v1/cpu-v1/report.json \
  --output work/atrium-factorial-native-cache-prepared-v1

PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=outputs/open-worldline \
work/wan-adapter-env/bin/python work/atrium-factorial-native-cache-v1/run.py \
  --preflight --prepared-directory work/atrium-factorial-native-cache-prepared-v1
```

Preparation is single-use and will reject the already existing output. The packet contains 176 files totaling 108,241,087 bytes. It includes the exact 102 PNGs and capture manifest, a source snapshot, CPU report and plan. The public index, full capture validation receipt and data/source licenses are bound in the source graph. All 71 source entries have exact hashes; an AST import check found no missing project imports.

The plan SHA256 is `7986ffd63b5d3160f70b7084dc9b21f3367c7a504e852220996153e45d440ca6`. CPU report `cpu-v1/report.json` has SHA256 `bdfa0233757ef11f61e724a9b3b097f174110baa0abca54384db25ce6561fcc6`. Fresh `prepared-read-v1.json` has SHA256 `9911f2ba900f205fa99a18bd938333e4a26395a63008d1a8e90db0dc06ae40b5`.

The proposed remote layout keeps `packet.py`, `cache.py`, `run.py` and `test_cpu.py` together in `/workspace/factorial-cache-src`, with the checked repository on `PYTHONPATH`. The prepared root is `/workspace/action-results/factorial-native-cache-v1`. A separate parent-issued cache admission must bind this schema/scope, exact plan/source/capture/CPU-report hashes, caller-declared GPU and unchanged codec limits, with `core_model_loaded=false` and `training_admitted=false`.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/workspace/intermediate-source/open-worldline \
python /workspace/factorial-cache-src/run.py \
  --execute --prepared-directory /workspace/action-results/factorial-native-cache-v1 \
  --weights /workspace/wan22-ti2v5b-weights \
  --expected-gpu NVIDIA\ A100-SXM4-80GB \
  --admission /workspace/factorial-cache-admission.json
```

These are intended paths, not evidence that execution occurred. The VAE file must retain SHA256 `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`. There is no weight download or provider operation in this program.

One 600-second parent deadline covers input validation, loading, encoding, weight checks and final verification. The existing codec guards retain the 48 GiB aggregate host RSS cap, 60 GiB CUDA reserved cap and 8 GiB available host/GPU floors. The native runtime remains Torch2.5.1, CUDA12.4 and the existing supported GPU checks. The parent must serialize this process after the transformer has exited and ensure at least 600+300 seconds remain in the separate provider lease. Worker termination is not provider resource deletion or a billing cap.

## Result interface and limits

Parent metadata, attempt marker, exact admission, terminal and logs live in the prepared root. Worker load/196-value maps, timing and memory records live in `worker/`. The eight raw encoded files and `completion.json` live in `result/`. Failures preserve completed outputs and attempts; the runner refuses to reuse an attempted output.

`run.read_completed(root, arm, conditioning_only=False)` requires completed parent/worker/monitor terminals, all source/input/output identities, unchanged 196 values and no watchdog record. It returns separate model-facing tensors and provenance. Target shape is `[1,48,5,44,78]`, independent observation `[1,48,1,44,78]`, commands `[1,16,6]`. All are CPU FP32.

The reader remeasures all twelve prefix comparisons from the saved observation, repeated observation and first-latent target slices. It binds commands to the prepared published-reader receipts. A conditioning-only read returns observation and commands; it reads target-prefix slices for causal verification but never materializes future target latents. Descriptive FP64 norm fields may differ only within the pre-existing 1e-12 relative/1e-15 absolute CPU reduction allowance. Actual model prefix limits and exact tensor identities are unchanged.

The CPU tests use small explicit arrays and a stand-in encoder, not the pretrained VAE. They check axis/storage mapping, eight-call ordering, raw failure retention, recomputed prefix failures, command corruption, conditioning-only access, packet corruption, deadline rejection and admission scope. Original model/loader source is unchanged.

These are CC0 development captures from one seen room, with fixed-position yaw and a programmed remote door toggle. Encoding them does not demonstrate learned motion, door control, generalization, physical interaction or improved visual quality. No new training run or action-quality admission is created by this cache. The imported native source and model retain the existing Wan attribution/license; this preparation makes no novelty claim.
