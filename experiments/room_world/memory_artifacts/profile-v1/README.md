# First Room memory training profile

September 7, 2026. Both original memory arms completed exactly 50 full-sequence updates on the local M4 Pro. This run measures training feasibility and cost. Its checkpoints are ineligible for the planned quality evaluation.

| Arm | Completed updates | Training interval | Median update | Sampled peak Metal driver allocation |
|---|---:|---:|---:|---:|
| Carry recurrent state | 50 | 72.99 s | 1.384 s | 0.949 GiB |
| Reset recurrent state each step | 50 | 73.99 s | 1.429 s | 0.957 GiB |

Seed 20260907 initialized both arms from the exact same saved memory parameters. Both used the same paired scene schedule, original RGB/action captures and optimizer. Each update covered both branches and all 65 transitions, retaining the complete recurrent gradient graph. The 498,651 base parameters remained unchanged. Only the 24,960 original memory parameters were optimized. Every completed update passed finite-loss, finite-gradient and finite-parameter checks. The two training intervals include recovery writes but exclude worker startup and initial model loading.

The study reports matching completed budgets. Both supervised processes exited successfully, and at least 9.28 GiB of available system memory was recorded during each arm. This sampled measurement can miss short peaks. The full training, optimizer and timing records remain in the arm directories.

No validation predictions or reserved test scenes were used. Training loss from different scheduled scenes is not a controlled before/after quality comparison. This run does not establish learned door memory or retained camera and interaction controls. A later equal-budget three-seed study must start from fresh initialization and publish its separate quality results.

The run used Python 3.11.9, PyTorch 2.5.1, NumPy 2.4.2 and psutil 6.1.0, with MPS execution and automatic CPU fallback disabled. The [study](study.json) binds source snapshots, base and data hashes and the shared scene schedule. [Publication checksums](publication.json) cover the copied measured artifacts, including the exact source snapshots and all initialized, recovery and final profile checkpoints. The publication manifest was added after the completed run and does not replace its recorded source identity.

Code and original memory weights: Apache-2.0. Original captured training data is separately released under CC0-1.0. No external pretrained weights are included here.
