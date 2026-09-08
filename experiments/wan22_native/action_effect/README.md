# Action-effect training, numerical assessment and visual evaluation

This directory contains the exact source used for the 128-update action-effect experiment. The existing 947,712-parameter action adapter receives an additional guided contrast loss on 32 start-0 updates. The 825 frozen Wan foundation tensors remain unchanged. The experiment completed training and its saved-artifact audits.

**The rendered action test failed.** Both generated doors stayed closed and the requested camera turn was absent. Small improvements on four held-out noise tensors did not establish visible command following. These noises use the same known room and starting image; they do not test new scenes. See the [combined results index](results/README.md) and the separate [training](results/training-a100-v1/README.md), [numerical assessment](results/assessment-a100-v1/README.md) and [visual results](results/visual-a100-v1/README.md).

The main objective uses the original two sequential half-loss backward passes. On updates 1, 5, ..., 125, the additional objective compares `-(G_open - G_closed)` with the true future open-minus-closed latent difference, where `G = N + 5 * (P - N)`. Its weight and endpoint scale are both 1. Main and auxiliary gradients accumulate before one original clip and AdamW step. The main loss uses positive text; the auxiliary uses both genuine text contexts. This is a fixed endpoint contrast objective, not the literal solver's initial sigma.

## Files and identities

| Directory | Contents |
| --- | --- |
| `training/` | Original main objective, additional loss, guarded runner, original saved input identities, tests and required reference modules. |
| `assessment/` | Four held-out noise comparisons, fixed k506 losses, original k999 comparisons, guarded runner and tests. |
| `visual/` | Matched wait/interact sampling and decoding, with common initial observation, noise and text; tests and input checks. |
| `audits/` | Independent saved-artifact readers for training, assessment and visual runs, including their frozen reference readers. |
| `comparison/` | All-frame RGB/target/repeat-first comparisons and the exact original RGB scorer. Pixel errors are descriptive and are not a quality measure. |
| `reviews/` | Historical independent source reviews. Historical CPU reports also remain beside the relevant source. |

[source-manifest.json](source-manifest.json) records the original relative location, file size and SHA256 of every copied source, test and historical record. All copied files are byte-exact. [repository-dependencies.json](repository-dependencies.json) lists the 66 repository dependencies plus three cache dependencies; those files remain in the existing repository. The model weights, optimizer states and large prediction arrays belong to the result archives, not this source directory.

`HISTORICAL-*` documents retain statements made before execution. The measured status is given above and in the results index. They are historical records, not current claims that the experiment is unexecuted.

## CPU checks

Use Python 3.11 from the repository root. The existing CPU environment uses Torch 2.5.1, NumPy 1.26.4, Diffusers 0.34.0, safetensors 0.5.3 and pytest 7.4.4. The pinned requirement files used by the repository's native CPU job supply the remaining imports:

```sh
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/wan_adapter/requirements-real.txt \
  -r experiments/wan22_native/requirements-core.txt \
  -r experiments/wan22_native/requirements-sample.txt

EFFECT_CODE="$PWD/experiments/wan22_native/action_effect"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest -q "$EFFECT_CODE/training/test_effect.py" "$EFFECT_CODE/training/test_runner.py"
python -m pytest -q "$EFFECT_CODE/assessment/test_reader.py" "$EFFECT_CODE/assessment/test_assessment.py" \
  -k 'not actual_packet and not actual_original_input_reader_and_small_checkpoint_loads'
python -m pytest -q "$EFFECT_CODE/visual/test_run.py"
```

Run those as separate processes. The unchanged historical files use sibling module names such as `inputs`, `runner` and `reader`; combining stages in one pytest process can collide. The two excluded assessment tests require the original saved input packets at their historical paths. The other listed tests use small CPU fixtures. `training/test_prototype.py` additionally runs the complete 128-update sequence on a tiny random CPU fixture, not on Wan weights.

The original bridge test module has a single-thread pytest fixture. Importing its small model factory into the assessment tests does not inherit that fixture. The first Linux CI run failed the exact cached-versus-direct prediction comparison. CI now sets the same one-thread limit explicitly, retaining the original `torch.equal` assertions and all frozen test files. This CI setting does not change the recorded A100 experiment.

The original reports record 14 training checks, 14 assessment checks and 11 visual checks. Relocation did not rerun those suites or create new CUDA evidence. [relocation-check.json](relocation-check.json) records three successful `--help` imports and three successful comparisons of the relocated source graphs with their existing CPU reports. CUDA remained uninitialized.

## Preparing actual inputs

The three main scripts below default to CPU preparation. Running them with `--help` reads their interface without preparing data. Every artifact path must be absolute; outputs must be fresh regular directories outside the repository. Keep the repository as the working directory so the frozen core dependencies resolve.

Set the following variables to verified artifact directories, not to an unverified archive extraction:

| Variable | Required artifact |
| --- | --- |
| `EFFECT_PROBE` | Completed original two-update native CUDA probe. |
| `EFFECT_CACHE` | Completed eight-window native CUDA cache parent directory. |
| `EFFECT_TEXT` | Genuine cached positive/negative text directory. |
| `EFFECT_OLD128` | Original fixed128 prepared/run directory containing the original saved draws and initialization. |
| `EFFECT_DIAGNOSTIC` | Original 14-prediction diagnostic prepared/run directory. |
| `EFFECT_OLD20` | Original comparison128 prepared directory, with its original14 subdirectory. |
| `EFFECT_HELDOUT` | The four saved evaluation-noise tensors and their original manifest. |
| `EFFECT_TRAINED` | Completed action-effect training run directory. |
| `EFFECT_AUDIT` | Its current source-bound passed independent training audit file. |
| `EFFECT_BASELINE` | Completed original spatial native clip directory. |
| `EFFECT_OUTPUT` | A new output directory outside the repository for this command. |

Do not regenerate saved Gaussian noise from its seed on another architecture. Saved file and tensor hashes are the canonical identities.

```sh
# Prepare the fixed training inputs. This does not train.
python "$EFFECT_CODE/training/runner.py" --profile spatial \
  --completed-probe "$EFFECT_PROBE" --cache-run "$EFFECT_CACHE" \
  --text-directory "$EFFECT_TEXT" --fixed128-prepared "$EFFECT_OLD128" \
  --diagnostic-prepared "$EFFECT_DIAGNOSTIC" \
  --cpu-report "$EFFECT_CODE/training/cpu-v3.json" --output "$EFFECT_OUTPUT"

# Use a different fresh EFFECT_OUTPUT for the numerical assessment.
python "$EFFECT_CODE/assessment/assessment.py" \
  --comparison-prepared "$EFFECT_OLD20" --heldout "$EFFECT_HELDOUT" \
  --training-run "$EFFECT_TRAINED" --training-audit "$EFFECT_AUDIT" \
  --cpu-report "$EFFECT_CODE/assessment/cpu-assessment-v1/report.json" \
  --output "$EFFECT_OUTPUT"

# Use a different fresh EFFECT_OUTPUT for the rendered comparison.
python "$EFFECT_CODE/visual/run.py" \
  --training-run "$EFFECT_TRAINED" --cache-run "$EFFECT_CACHE" \
  --baseline-run "$EFFECT_BASELINE" --training-audit "$EFFECT_AUDIT" \
  --cpu-report "$EFFECT_CODE/visual/cpu-v2.json" --output "$EFFECT_OUTPUT"
```

Actual execution separately requires `--execute`, a complete prepared packet, verified original weights and a new exact decision record. The CUDA environment is the existing [native CUDA reference](../cuda_reference/README.md): original FP32 foundation storage, native BF16 autocast and FlashAttention 2.7.4.post1 with Torch 2.5.1/cu124. A CPU check or a preparation command does not admit another training or rendering run. Worker deadlines do not bound provider billing; provider cleanup is outside these scripts.

## Public archives and historical audit paths

The [results index](results/README.md) supplies the public release/download status and manifests. Prior source/input material is also available from the [original CUDA probe](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-cuda-a100-v1), [original fixed128](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-fixed128-a100-v1) and [post128](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-post128-a100-v1) releases. Use their hash-verifying downloaders and disclosed file transformations.

Some public audit JSON files are disclosed derivatives: only their operational `run` path prefix was removed. Historical plans still bind the SHA256 of the original audit file. **A public path-redacted derivative is not a directly admitted historical execution packet.** Do not rewrite an old plan hash or relax a reader to force acceptance. A new run needs a fresh, internally consistent preparation and independent review. Original scientific source, prediction, target, schedule, metric and checkpoint bytes were not changed by these privacy transformations.

The three main preparation CLIs preserve imports after relocation, but the saved visual auditor, RGB comparison and two artifact-dependent assessment tests also derive historical workspace locations from `__file__`. [historical-layout.json](historical-layout.json) maps each exact source directory back to its original `work/...` name under an external workspace. That workspace needs a real repository at `outputs/open-worldline` and the separately verified input/result directories. Copy real files preserving the manifest hashes; symlinked artifact paths are rejected. `comparison/reference_rgb/analyze.py` belongs in its separate mapped original RGB directory. `assessment/heldout-generation/prepare.py` is the exact one-time historical noise generator, not a general `--help` CLI; do not run it to regenerate the published noises.

[executed-source.tar.gz](executed-source.tar.gz) preserves these source, test and review files at their original relative workspace paths. [executed-source-index.json](executed-source-index.json) binds the archive and every member. It contains no foundation weights or experiment result tensors. Restoring these sources alone does not supply the historical input packets or resolve their audit-file privacy transformations.

Saved auditors do not replay the foundation model. The training reader verifies retained main/auxiliary predictions, losses, combined clipped gradients, checkpoints, optimizer state and frozen core hashes. The retained combined gradients cannot reconstruct the separate main and auxiliary backward paths. Descriptive AdamW reconstruction is not a CUDA replay. The RGB scorer includes all 34 frames, both original targets and repeat-first baselines; camera alignment and lighting limit the meaning of pixel differences.

## Licenses

The original action adapter and experiment code use Apache 2.0. Wan model/code/VAE dependencies retain their upstream Apache 2.0 notices, and the procedural Atrium captures use CC0. Exact notices are in [licenses/](licenses/); the core vendored source remains in the parent repository. This experiment does not claim that the pretrained Wan foundation or the published research mechanisms are original work.
