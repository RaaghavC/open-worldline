# Initial measured results

September 7, 2026. Apple M4 Pro with 24 GB shared memory, PyTorch 2.5.1. These measurements describe the shipped small synthetic-world prototype. External model results have not been measured here.

## Actual neural models

| Measurement | Result | Interpretation |
| --- | --- | --- |
| Spatial model size | 233,410 parameters | Original conditional flow network |
| Dynamics model size | 11,107 parameters | Original ecological transition network |
| Spatial training | 1,536 synthetic fields, 1,600 optimization steps, 74.44 s on MPS | Small training experiment, not foundation pretraining |
| Dynamics training | 1,024 synthetic states, 1,000 steps, 7.16 s on MPS | Learned approximation of one synthetic transition law |
| Untouched spatial test flow MSE | 0.10521 | 96 test examples separate from checkpoint-selection validation |
| Independent-pixel Gaussian flow baseline MSE | 0.52777 | Same field prediction task; this is not a frontier world-model baseline |
| Field generation | 208-402 ms CPU; 464 ms average MPS | 64 × 64, 24 Heun steps; excludes browser rendering and HTTP |
| Learned ecological transition | 1.629 ms median CPU, 3.931 ms p95 | One 64 × 64 three-channel transition |

Small networks can be limited by device-dispatch overhead. The app therefore defaults to CPU inference on the measured laptop. Training and larger experiments can use MPS or CUDA. The CPU values above come from the final benchmark with the browser app open; they vary with other work on the computer. Timing samples are limited and are not a server concurrency benchmark.

Source measurements: [training results](../checkpoints/training-metrics.json), [benchmark results](../tests/results/benchmark-metrics.json), [checkpoint SHA-256 values](../checkpoints/sha256.json).

## Control dependence

The same learned checkpoint starts from the same synthetic initial state. One arm receives the actual rain/heating inputs. A second arm receives zeroed inputs. Both are compared with the analytic teacher under the actual controls. This is an input ablation, not a separately trained alternative method.

| Predicted steps | Actual-input mean MSE | Zeroed-input mean MSE | Cases favoring actual inputs |
| --- | --- | --- | --- |
| 1 | 0.00000101 | 0.00059682 | 12 / 12 |
| 10 | 0.00003561 | 0.03439759 | 12 / 12 |
| 60 | 0.00040852 | 0.16676526 | 12 / 12 |

The model uses the control inputs in this synthetic domain. Error increases with the rollout horizon. This does not demonstrate real-world causal reasoning or accuracy under new dynamics laws.

## Persistence and editing

The benchmark applies 24 spatial edits and checks all saved revisions. It found zero changed values outside the selected brush. Restored and reloaded states match exactly. Branch edits leave their parent unchanged. JSON export/import preserves the state hash. The API tests also verify that an outdated revision is rejected instead of overwriting a newer edit.

Median snapshot save was 24.93 ms, restore 27.84 ms and branch creation 24.24 ms in this run. The final database was 8.02 MiB. This implementation stores full immutable snapshots; it does not claim compressed large-world memory or constant storage use.

The test suite contains 81 tests, including tests using actual learned weights and tests using explicit fixtures/model doubles to isolate software behavior. Browser download and file-picker import also preserve the state and hash after normalizing equivalent JSON number spellings. See the [evaluation protocol](evaluation-protocol.md) for the distinction.

## What these measurements leave open

They do not assess rendered-image beauty, user preference, large open-ended environments, natural-language understanding, real-world physics, multiplayer use, or Genie 3 parity. They do not establish three novel scientific contributions. Those questions require different data, methods and evaluations described in the [research plan](research-plan.md).
