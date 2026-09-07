# Batched Room memory training profile

September 7, 2026. Both original memory arms completed 50 full-sequence updates on the local M4 Pro. This is a timing and resource profile. Its weights are ineligible for the planned quality evaluation.

| Arm | Completed updates | Training interval | Median update | Sampled peak Metal driver allocation |
|---|---:|---:|---:|---:|
| Carry recurrent state | 50 | 15.93 s | 0.279 s | 2.090 GiB |
| Reset recurrent state each step | 50 | 21.95 s | 0.415 s | 2.090 GiB |

The frozen image encoder and decoder now process the observed training sequence in one batch. The recurrent state still advances in time order and retains its complete gradient graph. This change applies to training on observed sequences; it does not batch future images during generated rollouts. The explicit `batched` setting is recorded in the study, worker results and checkpoints.

The earlier [stepwise profile](../profile-v1/README.md) took 72.99 and 73.99 seconds respectively for the same 50-update schedule and saved initialization. The measured intervals include recovery writes and exclude worker startup and initial model loading. These are individual local runs, without repeated timing trials. The faster path uses more Metal memory. No equality of the resulting optimizer trajectories is claimed.

[CPU comparisons](../sequence-cpu/) checked the batched equations and gradients. A subsequent [MPS comparison failed its original RGB-gradient tolerance](../sequence-mps-failure-v1/README.md). The separate [common-gradient diagnostic](../sequence-gradient-diagnostic-v1/README.md) traced that case to three tiny prediction differences at the absolute-error loss kink. Its shared-upstream-gradient comparison passed the original bounds. The original failed result remains a failure. This diagnosis covers one carried-state example and does not establish exact numerical equality across training.

Both profile arms use seed 20260907, identical original RGB/action captures, optimizer settings and paired scene schedule. Each update covers two branches and all 65 transitions. Only 24,960 original memory parameters are trained; the 498,651 base parameters remain unchanged. No validation predictions or reserved test scenes were used. Loss values from different scheduled scenes are not a before/after quality comparison.

The run used Python 3.11.9, PyTorch 2.5.1, NumPy 2.4.2 and psutil 6.1.0 with MPS and automatic CPU fallback disabled. The [study](study.json) records sources, data and schedule. [Publication checksums](publication.json) cover every copied measured file, including exact source snapshots and initial, recovery and final checkpoints. The publication manifest was added after the run.

Code and original weights: Apache-2.0. Original captured data: CC0-1.0. No external pretrained weights are included.
