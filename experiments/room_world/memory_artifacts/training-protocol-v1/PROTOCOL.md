# Fixed development protocol v1: recurrent memory for Room RGB prediction

Status: **1,024 updates per arm, three seeds, batched path on MPS**, fixed from completed timing profiles before any quality evaluation. Training has not been launched by this implementation lane. No learned-memory result, protected-test result or frontier-model comparison is established by this protocol.

## Question and fixed models

Can a small recurrent state improve the prediction of a previously changed door after the camera turns away and returns, while preserving the frozen predictor's ordinary camera and interaction controls?

Use the published original Room predictor as a frozen base: 498,651 parameters, checkpoint SHA-256 `9376e684e3abd103313d13d04bd4a1afc1b202fccb7e7b6fe133e7232a486d69`. Add a GRU with 32 state values, receiving the pooled 96-channel bottleneck and existing 64-value action embedding. A zero-initialized output layer predicts scale and shift for that bottleneck. The memory module has 24,960 trainable parameters.

Compare carried state with state reset at every transition. Both modes have exactly the same parameters. Keep the original frozen predictor as a separate reference. Inference receives only four RGB observations, the current action and explicit session state. No door labels, pose, geometry or renderer state enter prediction.

## Training and checkpoint rule

Use initialization seeds **20260907, 20260908 and 20260909**. Within each seed, carry and reset load the exact same initialized memory file and follow the same shuffled scene schedule. Train on the 32 fixed scenes 5000 through 5031. Each update includes both branches of one scene and all 65 transitions.

Training uses observed RGB histories ending at the current action. The following true RGB frame enters only the loss. Use mean absolute RGB error in [0,1] units, equally averaged over both branches, all 65 steps, pixels and channels. Keep the complete recurrent gradient graph; do not detach state or add a special door-region loss. Reset state at the start of every paired episode.

AdamW settings are learning rate 0.0003, weight decay 0.0001, betas 0.9/0.999, epsilon 1e-8 and gradient-norm clipping at 1. The base remains frozen. Train **exactly 1,024 updates per arm** for all three seeds, using the explicit `batched` path. Every consecutive 32 updates is a shuffled pass over the same 32 training scenes. These 32 passes do not add independent scenes. Each arm therefore processes 2,048 branch sequences and 133,120 transitions; all six arms process 798,720 transitions in total. Initialize each seed freshly from the published base and its seeded zero-conditioning memory module. Do not warm-start from either profile. No checkpoint is selected by validation performance. All six arms must complete exactly 1,024 updates with the declared schedules, same initialization within seed, same optimizer and unchanged base. A failed or stopped arm leaves the matched study incomplete; do not compare a shorter surviving checkpoint as a completed study.

## Development evaluation

Use only scenes 300000 through 300007, with every result retained. Standard trajectories begin with wait versus successful interact, turn left for 24 actions, wait for 16, then turn right for 24. Initial RGB is identical; the four observed frames before return are also identical between branches.

Report these protocols separately:

1. **Observed prefix, generated return:** update memory from the real observed prefix, then recursively generate all 24 return frames. For the standard trajectory, warm actions 0 through 40, begin from observations 38 through 41, and generate observations 42 through 65. No later true frame enters generation.
2. **Uninterrupted generated history:** begin with four copies of the initial observation and recursively generate all 65 next frames. No later observed RGB enters generation.

Also evaluate the fixed paired wait-8 and wait-32 captures. Their first return actions are 33 and 57, respectively; both still have 24 return actions. Keep all three wait durations and both protocols separate.

After generation, define the return region from pixel/time locations where either paired true RGB differs. Use that same region for both predictions. A pair is correct only if **both** predicted branches are strictly closer to their own true branch than to the opposite branch on that region. Ties fail. Report region MAE, full-frame errors, eligible scene counts, pair-correct fractions and every prediction tensor.

## Ordinary control retention

Evaluate three separate 16-step controls on each of the same eight validation scenes: translation cycle; visible but out-of-reach failed interaction; and turn, open, then close. The independent audit binds the failed-interaction observations to their recorded files and verifies visible door pixels, distance beyond the interaction radius, sufficient facing and no toggle.

Initialize a fresh state per scene/control. Use observed histories at every step. Retain each transition's action and RGB MAE, per-action errors/counts and per-scene results. Average transitions within scene, then weight scenes equally within control type. The overall value weights the three types equally.

For each initialization seed, every control type and the equally weighted overall result must satisfy both `carry_MAE <= 1.05 * reset_MAE` and `carry_MAE <= 1.05 * frozen_MAE`. Report each exceeded threshold. A zero reference defines a zero threshold. Do not hide a failed seed/type inside a pooled average. Missing cases, references, matched budgets or visibility evidence make these gates unavailable.

## Point estimates and uncertainty

For each seed, protocol and wait duration, first average region MAE equally across eligible scenes. Define relative gain as `1 - mean_carry_MAE / mean_reset_MAE`; a zero reset mean makes the ratio undefined. Average the three seed gains equally. Compute pair correctness first across scenes within each seed, then equally across seeds.

The **provisional development point gates** are mean relative gain at least 30%, mean paired correctness at least 80%, and positive gain in every seed. Apply them separately to each protocol and wait duration. Combine them with the ordinary control-retention results only after complete measured evaluation. A result confined to observed prefixes must be described as such.

Use 10,000 whole-scene bootstrap resamples with fixed seed 20260907. Each replicate draws eight scene indices with replacement and applies the same indices to every model seed, arm, protocol, wait duration and control type. Recompute the full ratios and fractions in each replicate and report percentile 95% intervals. If a reference denominator is zero, retain the undefined result rather than adding an epsilon. The intervals describe variation across these scenes conditional on the three trained seeds. They do not support a p-value or a lower-confidence-bound success claim.

## Integrity and limits

Keep source, data, checkpoint, prediction and tensor hashes, completed schedules, timings and failures. Store the last validated memory/optimizer/random-state recovery bundle atomically. Retain final memory-only weights without copying the frozen base into them. Evaluation must verify the selected training path across study, run row, arm metrics and final checkpoint, and verify every retained training-source snapshot against its recorded hash. Record the original training source/study hashes and path in evaluation provenance; repeat the checkpoint/path check inside each worker. The frozen reference has no training path of its own and records the study path separately. Missing path fields in older reports mean stepwise. Reject changed model-forward source during evaluation.

The current local cap is 600 seconds and 18 GiB process RSS per worker, with available-memory checks and an MPS allocator limit. A cap is a stop condition, not permission to compare unequal completed budgets.

Scenes 400000 through 400031 remain reserved and unopened. Freeze any protected-test decision rule before accessing them. These development captures come from one programmed 64-pixel room renderer and fixed action scripts. Pixel improvement here would establish a limited experiment, not general physics, high-resolution visual quality, scientific novelty or Genie 3 parity.


## Timing basis and numerical limits

The separate batched profile completed 50 updates in both arms for seed 20260907, with identical initial checkpoint bytes, schedules, data and optimizer relative to the earlier stepwise profile. Measured training-arm time was 15.927366374991834 seconds for carry and 21.94787908301805 seconds for reset; median update times were about 0.2793 and 0.4147 seconds. Peak sampled MPS driver allocation was 2.09001159668 GiB. The independent audit verified unchanged base tensors and checkpoint/optimizer integrity. Linear estimates for 1,024 updates are 326.19 seconds for carry and 449.49 seconds for reset, about 38.8 minutes for all six training arms. These estimates exclude worker startup/base loading and separate evaluation, and are not completion guarantees. Keep the existing 600-second hard cap per worker; stop and report incomplete if it binds.

The previous MPS ordinary-L1 RGB-gradient equivalence check remains failed. A separate retained-tensor diagnostic found exactly three zero-to-one-unit float32 residual changes. Its common-upstream backward comparison passed the original tolerances, isolating that test's L1 sign changes. This does not establish equal optimization trajectories across stepwise and batched implementations; reset-mode MPS equivalence was not measured in that comparison. Both arms in this fixed study use the same batched implementation. No profile training loss is treated as a quality result.

## Frozen commands

Run from `outputs/open-worldline` with the existing `work/atrium-env` environment: Python 3.11.9, PyTorch 2.5.1, NumPy 2.4.2 and psutil 6.1.0. Use new output directories. Root owns GPU launch and keeps other GPU jobs idle during each run.

```sh
../../work/atrium-env/bin/python -m experiments.room_world.memory_train \
  --train ../../work/room-memory-data/train \
  --validation ../../work/room-memory-data/validation \
  --output ../../work/room-memory-training-batched-1024-v1 \
  --phase train --sequence-path batched --updates 1024 \
  --seeds 20260907 20260908 20260909 --device mps --max-seconds 600
```

Only after all six training arms complete with `matched_budgets: true`:

```sh
../../work/atrium-env/bin/python -m experiments.room_world.memory_evaluate \
  --study ../../work/room-memory-training-batched-1024-v1 \
  --data ../../work/room-memory-data/validation \
  --controls ../../work/room-memory-data/validation-controls \
  --control-audit experiments/room_world/memory_artifacts/control-validation/visibility-report.json \
  --output ../../work/room-memory-validation-batched-1024-v1 \
  --device mps --max-seconds 600
```

The plan-only record is `work/room-memory-fixed1024-plan-v1/study.json`; its schedule check verifies all three complete 1,024-entry schedules on CPU with zero optimizer updates. Profile checkpoints remain ineligible for evaluation. Do not change source while either actual run's source checks are active.

## Source and capture identity

| Training source | SHA-256 |
|---|---|
| memory_train.py | 78f233747956ada0f5a3f05deac9e504c01b943c8d85bef77b3f4a447b4201d8 |
| memory_evaluate.py | 2b082926775aeee38c03d96a092a594b0a17a8638b9488a9314a18f6166419fd |
| memory_model.py | a5544a4af4f2886e256e9715a5d68f02e6bdf310a418aed1c611cbec7248a67f |
| memory_data.py | 50c828d191866d2eaae0ab3b1b2e90675b30b8cdf41887adc9fb2540fe2edfe3 |
| model.py | 26e2c402312f6752455596f007986e0b8d70876b1bf402610811a9acd8745754 |
| memory_sequence.py | 173326ba7b2ec5332600ba1cf9814b6cc87619758524b8da1dfe51f84e142080 |

Training capture manifest: `f85c695dc192a1a726535e613e3df1517ed90029895cf0ea8be7552c8b35d8dc`. Validation capture manifest: `ad7e37e91de8fb26a3ad1ed43e1cc63d9239255b3f58915b5058955da96e5cf7`.

Separate control capture manifest: `b8b49ab3a6afef7541b33ee648b769f8a0b322daefba40c84875d1e2b5534b98`. Visibility audit: `3cafc4c158a0db9f72fbac046d1b366a914f45df68ce9152e0e9f952a7cb3981`.
