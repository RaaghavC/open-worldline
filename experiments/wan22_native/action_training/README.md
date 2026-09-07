# Prepared paired action training

The [second numerical probe](results/probe-v2/README.md) completed two updates of the original 947,712-parameter adapter on the external Wan2.2 TI2V-5B foundation. It verified finite gradients, a nonzero second-update GRU gradient and unchanged foundation values. The [first probe](results/probe-failure-v1/README.md) failed at an unsupported MPS pooling shape before any update. Both results are retained. The native shift-5 and shift-3 clips still showed severe colored, warped surfaces. These numerical checks do not establish usable generated video or action control.

The plan command verifies the eight cached Atrium windows, original RGB and command hashes, genuine positive text context, and the separate 17-frame VAE reconstruction. It saves the exact proposed noise and timestep draws. It does not load foundation weights or start a model process.

## Fixed computation

The [original adapter](../action_adapter/README.md) has 947,712 FP32 parameters. All 825 named foundation parameters remain frozen. Each forward executes the frozen native transformer through block 29 without a gradient graph, then applies the adapter and differentiable frozen head. The adapter receives ordered commands and the independently encoded initial image. Geometry, camera poses, door-state labels and future RGB are not conditioning inputs. Future truth enters the standard training noise mixture and the velocity loss.

Each paired update executes closed then open as two sequential batch-one forwards. Each branch contributes half of its future-frame velocity MSE before one optimizer step. Frames 1 through 4 of the latent target enter the loss; the initial latent is excluded. There is no special door-region weighting or classifier-free guidance in training.

| Item | Prescribed value |
|---|---|
| Fresh adapter seed | 20260907 |
| Probe | 2 paired updates, starts 0 and 8 |
| Later pilot | 16 paired updates, starts 0, 8, 32, 49 repeated four times |
| Noise draw | Private CPU generator with the same seed; draw integer `k` in 50 through 950, then FP32 noise `[1,48,5,18,32]` |
| Pair identity | Both branches use the exact same saved noise tensor and `k` |
| Flow equation | `sigma=k/1000`; noisy future `(1-sigma)*target+sigma*noise`; velocity target `noise-target` |
| Initial observation | Exact independent one-image latent, with its 144 token times set to zero |
| Optimizer | AdamW, learning rate 0.0001, betas 0.9/0.999, epsilon 1e-8, weight decay 0.01 |
| Gradient clipping | Global L2 norm 1 |
| Child limits | 900 seconds, 18 GiB process/Metal limits, 2 GiB available-memory floor |

The first paired update must have finite nonzero output gradients and zero upstream GRU gradients because the output projection starts at zero. The second update must have finite, nonzero GRU gradients. A later 16-update run creates a fresh adapter and optimizer, verifies the same initial tensor bytes and draw prefix, and never loads probe weights.

Only start 0 is an existing pair with different outgoing commands and a shared derived observation. Its original PNGs differed in 18 channel values by one uint8 level. The cache uses the open first image for both branches and replaces the closed derived target's first image before VAE encoding. Raw captures remain unchanged. The other starts have equal command sequences and different observations. These overlapping windows come from one layout and cannot measure scene generalization or establish causal action control.

## Admission remains separate

Execution requires a current implementation CPU report, a separate independent report, all data and reconstruction checks, and a parent-authored JSON decision. The runner never creates an admitting decision.

The decision schema is `worldline-wan22-action-training-admission-v1`. It must record `decision="admit"`, `issued_by="parent-agent"`, the exact `mode`, current `source_sha256`, cache and reconstruction hashes, a written `visual_review`, and nonempty `foundation_evidence` file/hash records. The two scopes are deliberately separate:

- `mode="probe"` requires `scope="optimizer-feasibility-only"`. `foundation_visual_status` may be `failed` or `passed`, but the parent must explicitly permit these two numerical updates. A completed probe does not admit a quality study.
- `mode="fixed16"` requires `scope="fixed16-action-pilot"`, `foundation_visual_status="passed"`, and a completed matching probe with unchanged retained draws, context, checkpoint bundle and reports. A numerical probe decision with a failed visual status cannot authorize this run.

The parent issued separate decisions for the first failed probe and the fresh second probe with its 42-file source graph. Both decisions retained a failed visual assessment and permitted only two numerical updates. Fixed16 remains closed. These checks record the parent's assessment; they do not replace visual inspection or prove image quality.

The pinned [original RGB release](https://github.com/RaaghavC/open-worldline/releases/tag/atrium-rgb-v1) contains all 132 PNGs and the unchanged capture manifest. `python scripts/fetch_atrium_rgb.py --output /path/to/new/capture` downloads the 22,386,589-byte archive, verifies its fixed SHA-256 and exact file set, and refuses to overwrite an existing directory. Linux CI uses this same release for the original-data integration check.

## Commands

Use the pinned dependencies in [requirements-core.txt](../requirements-core.txt) and the existing [codec requirements](../../wan_adapter/codec/requirements.txt). The measured CPU environment used Python 3.11.9, PyTorch 2.5.1, NumPy 1.26.4, safetensors 0.5.3, Diffusers 0.34.0 and psutil 7.2.2. Commands below run from the repository root and require the original local capture directory.

On a clean checkout, prepare the capture at the exact location used by the CPU integration test:

```sh
python scripts/fetch_atrium_rgb.py --output ../../work/atrium-pilot/dense-pair
```

Skip this download when the original capture already exists there. The plan and training commands accept a separate capture path through `--capture`.

```sh
python -m experiments.wan22_native.action_training.test_cpu \
  --output /absolute/fresh-cpu-check/tests.json

python -m experiments.wan22_native.action_training.train \
  --mode probe \
  --capture /absolute/atrium-dense-pair \
  --cache experiments/wan22_native/action_data/results/cache-v1/result \
  --roundtrip-run experiments/wan22_native/action_data/results/roundtrip-v1 \
  --text-cache experiments/wan_adapter/text_cache/native-results \
  --output /absolute/fresh-probe-plan
```

An explicitly admitted execution uses the same command with a new output directory and these additional arguments:

```sh
  --execute \
  --weights /absolute/wan22-ti2v5b-weights \
  --cpu-report experiments/wan22_native/action_training/cpu-results/v5/tests.json \
  --independent-report experiments/wan22_native/action_training/cpu-results/independent-v3/report.json \
  --admission /absolute/parent-issued-probe-decision.json
```

This is a command template, not an admission or a scheduled run. A later fresh pilot uses `--mode fixed16`, its own fixed16 decision, and `--probe-run /absolute/completed-probe`.

## Retained evidence and failures

The parent directory holds `plan.json`, exact `draws.safetensors`, `positive.safetensors`, source snapshots and, only for execution, launch/terminal records and the worker log. The result directory holds per-update branch losses, gradient norms, synchronized timings, sampled memory, and all 825 loaded-value hashes before and after training. Core verification copies one parameter at a time to CPU and does not clone the entire foundation. The full model is still resident in device memory.

Each valid update publishes an immutable `checkpoint-NNNN/` bundle with adapter-only safetensors, optimizer/CPU RNG/private draw RNG recovery, tensor/file hashes and source/data identity. Recovery loads must use `weights_only=True`. There is no resume implementation. `last-valid.json` advances only after a fully checked update. A failed or partially applied optimizer step cannot replace that pointer. If the final 825-parameter verification does not finish, the report makes no unchanged-foundation claim.

All failures and partial files remain in their original output directory. Outputs refuse reuse. The parent and child enforce the limits; MPS fallback is explicitly disabled. The core's allocator settings are recorded separately from the VAE settings. No image generation, feature cache, multi-block adapter, checkpoint selection or quality evaluation is implemented here.

CPU reports are retained without rewriting their measured values:

- [v1](cpu-results/v1/tests.json): 13 tests attempted, 6 errors. The flow-input helper passed a Python integer to the token-time function, which requires an integer tensor. The one-line input-type correction is retained in later source snapshots. This was a fixture failure before any real training.
- [v2](cpu-results/v2/tests.json): 13 passed after the correction.
- [v3](cpu-results/v3/tests.json): 15 passed after retained-probe evidence checks were added.
- [v4](cpu-results/v4/tests.json): 16 passed in 1.012 seconds after the separate numerical-probe admission was added. All 41 declared source files are retained beside this report.
- [Independent v2](cpu-results/independent-v2/report.json): 7 checks passed in 0.042 seconds. The preceding [independent v1](cpu-results/independent-v1/report.json) retained a scalar fixture assertion failure: differently ordered FP32 reductions differed by 5.9604645e-8. Only that scalar assertion changed to an explicit absolute tolerance of 1e-7. Production code and the parameter/gradient checks did not change.
- [v5](cpu-results/v5/tests.json): 17 passed in 1.078 seconds after the original pooling fix. This version directly binds all 42 source files, including `action_adapter/pooling.py`, and tests rejection when the helper bytes change. The prior 41-file failed-probe snapshots remain unchanged. The separate [actual MPS operation check](../action_adapter/pooling-results/mps-v1/README.md) passed without a foundation model.
- [Independent v3](cpu-results/independent-v3/report.json): the unchanged seven independent checks passed again against the 42-file source graph. Its exact report and hash-matching snapshots are retained.

The first report retains its five local source files; the old report's foreign source hashes remain recorded, but their exact historical snapshots are not asserted to be present. Later reports have complete declared source snapshots. Publication manifests bind every copied file. CPU fixtures test equations, gradient accumulation, input validation and provenance. They do not measure full-model activation memory, real training latency, action fidelity or improved video quality.

## Attribution

The training code is Apache-2.0 under the repository license. It trains the original adapter against an external Wan2.2 foundation. The official model, VAE, tokenizer and text encoder attribution remains in the parent [NOTICE](../NOTICE), [source provenance](../provenance.json), and [text-cache attribution](../../wan_adapter/text_cache/native-results/README.md). The source RGB is original procedural Atrium data released under [CC0](../../atrium_data/DATA-LICENSE). This package includes no external foundation checkpoint and makes no claim that the pretrained foundation is original work.

The successful probe also has a separate [independent numerical audit](results/probe-v2/independent-review/numerical-audit.json) and [publication audit](results/probe-v2/independent-review/publication-audit.json). They verify saved draws, all 20 paired input hashes, all 825 before/after parameter records, three checkpoints and recovery state, and every declared publication path change. Gradient norms are independently reconstructed from Adam first moments within stated rounding bounds. These audits execute no foundation model and establish no image-quality result.
