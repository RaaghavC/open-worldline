# Fixed action-effect numerical assessment

`assessment.py` is the guarded executable. It performs no training or sampling. CPU preparation requires the completed new effect128 run and its passed independent audit. Execution requires a separate parent admission bound to that actual prepared plan. The three exact checkpoints are original cp0, original cp128, and the new final cp128. No intermediate checkpoint selection is permitted.

The current source-bound report is `cpu-assessment-v1/report.json`: 14 CPU checks passed in 5.96 seconds. They include the actual original-input reader, a real tiny bridge reusing frozen features while switching adapter states with exact direct-recomputation equality, fixed call counts, target-free callbacks, malformed audit/admission rejection, and deadline/failed-child cleanup. These are mechanics checks without external foundation weights or CUDA.

## Measurements and retained outputs

The executable produces 66 raw FP32 head outputs using 12 frozen feature extractions:

- Four held-out noises: two native context feature bundles per noise, reused across three checkpoints and both commands, totaling 48 heads and 8 extracts.
- Exact original seen k506 corruption: separate closed/open positive-context inputs, each reused across three checkpoints, totaling 6 heads and 2 extracts. Their noisy futures differ. Individual future flow losses remain separate from shared-input command contrasts.
- Exact original visual k999 input: two shared context bundles reused across three checkpoints and both commands, totaling 12 heads and 2 extracts.

Every completed raw velocity is saved before the next call. Reports retain all 12 held-out checkpoint/noise score groups, separate original-input scores, input/command/context/checkpoint hashes, source identities, both 825-tensor foundation records, sampled resource logs, terminal status and partial failure evidence. The adapter object and bridge owner remain the same as exact checkpoint state dictionaries are copied into that object. All assessment calls disable gradient tracking.

The four Gaussian tensors were generated once on CPU with dedicated seed 2026090801. The exact ten-file packet has manifest SHA256 `058e8c34570032f9a76a34ed040d89490fc25aede5bcaa9ffe40559c43111c19`. Both full tensors and future portions differ from every original saved training noise. These are held-out noises on the original scene, not unseen-scene generalization. Neither evaluation nor training regenerates them.

For shared-noise scores, observed-prefix tokens stay clean at time zero and future tokens use native model time 999. The declared ideal endpoint scale is s=1:

`G = negative + 5 * (positive - negative)`

`predicted_difference = -(G_open - G_closed)`

`target_difference = target_open - target_closed`

Guidance and subtraction use FP32 operation order. Errors, cosine and magnitude summaries use FP64 over future positions only. Normalized contrast MSE is `mean((predicted_difference - target_difference)^2) / mean(target_difference^2)`. A zero prediction gives normalized MSE 1 and undefined cosine, recorded as null. The report retains absolute magnitudes of all four raw paths, both guided paths, predicted contrast and target contrast. No threshold selects checkpoints or approves visual quality.

This ideal endpoint contrast is not a decoded reconstruction, a full generated endpoint, or the exact initial UniPC sigma. The sampler is unchanged. The commands differ only at the first interaction pulse, so these checks cannot isolate camera effects.

## Admission and limits

Preparation requires the exact frozen new training plan and all 88 source records, complete parent/worker/training reports, 825 original foundation value identities, final checkpoint file and tensor hashes, and the source-matching independent audit. It copies compact evidence into a self-contained prepared directory. The actual audit schema is `worldline-action-effect128-actual-independent-v1`, with passed status, 128 updates, 256 main predictions, 32 auxiliary updates, 64 auxiliary extracts, 128 auxiliary heads and the exact final checkpoint/source identity.

Execution retains the original 900-second parent/worker deadline, 48 GiB host RSS cap, 60 GiB CUDA reserved cap, 8 GiB host/device free-memory floors, and 70 GiB minimum GPU capacity. A worker deadline does not terminate a billed pod. External lifecycle control remains the parent's responsibility. Execution is single-use, and failed directories must be preserved.

The nested `heldout.checkpoint_admission_checked_here=false` and callback-only `meaning` describe `reader.evaluate_matrix` alone. The enclosing executable verifies completed training and explicit parent admission before invoking it.

## Exact CLI and remaining actual binding

Run from the verified repository root. With the normal local directory layout, CPU preparation after the actual new training audit passes is:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ../../work/wan-adapter-env/bin/python \
  ../../work/wan22-action-effect-evaluation-prep-v1/assessment.py \
  --comparison-prepared /absolute/path/to/original-comparison128-prepared \
  --heldout /absolute/path/to/wan22-action-effect-heldout-v1 \
  --training-run /absolute/path/to/completed-effect128-run \
  --training-audit /absolute/path/to/passed-effect128-audit/report.json \
  --cpu-report ../../work/wan22-action-effect-evaluation-prep-v1/cpu-assessment-v1/report.json \
  --output /absolute/path/to/fresh-assessment-prepared
```

The parent reviews the resulting actual `plan.json` and writes an admission matching `assessment.admission`: exact plan, source map, counts, limits, three checkpoint hashes, held-out manifest, new training-audit hash and CPU-report hash; `training_admitted=false` and `sampling_admitted=false`. After that separate decision, execution is explicit:

```sh
python /absolute/path/to/assessment-program/assessment.py --execute \
  --prepared /absolute/path/to/fresh-assessment-prepared \
  --weights /absolute/path/to/original-Wan2.2-TI2V-5B-weights \
  --decision /absolute/path/to/exact-plan-parent-admission.json
```

No actual new-training assessment plan is claimed before the new training run completes and is independently audited. CPU fixtures do not stand in for those artifacts.

## Transfer inputs and historical evidence

For the original comparison128 prepared folder, keep top-level `plan.json`, `cpu-report.json`, `checkpoint-0128.safetensors`, `training-audit.json`, and complete `original14/`, `source/`, `training-evidence/`, excluding only `__pycache__`. Add the full ten-file held-out packet, this program's 83 source files and current CPU report. The new actual training run and audit are consumed in place during preparation; execution uses selected compact copies. No external foundation weights belong in this transfer.

`cpu-v1` preserves the original seven NumPy-reader checks. `cpu-assessment-failed-v1` retains the first integrated run: 13 passed, one fixture stopped because it searched for `up.bias` while the existing adapter names that projection `output.bias`. Correcting the fixture name produced the current 14 passed checks. Frozen runtime math was unaffected. Saved-noise generation and its initial reader evidence remain unchanged.
