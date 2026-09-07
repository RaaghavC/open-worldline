# Incomplete 1,024-update memory study

The first carried-state arm for seed 20260907 stopped at its predeclared 600-second wall-time limit. The parent measured 600.380 seconds. Both the last recorded metrics and authoritative recovery bundle contain 804 completed updates. All six saved optimizer step counters are 804. No final checkpoint was produced. The other five arms were not started, and no image-quality evaluation was performed.

The retained worker metrics still say `running` because the parent terminated the process at its hard limit. The parent `terminal.json` and `study.json` record the stop and failed matched study. This is not a completed 804-update experiment and its weights are ineligible for the quality comparison.

Training became slower than the short profile predicted: the first 512 completed updates took 337.500 seconds in the recorded update intervals, and the final 50 updates had a median interval of 0.849 seconds. Update intervals include recovery writes but exclude subsequent metrics writes and startup. The cause of the slowdown has not been isolated. Peak sampled process RSS was 0.506 GiB and minimum available system memory was 7.367 GiB. These sampled quantities are not total Metal memory.

The base was pinned and excluded from the optimizer throughout the executed training code. The forced stop prevented the end-of-arm base tensor comparison, so no completed before/after base audit is claimed. All initialized and last validated memory/optimizer/RNG artifacts, exact measured sources and parent resource samples are preserved with publication checksums. The later 512-update study starts fresh and does not reuse this recovery checkpoint.

Code and original memory weights: Apache-2.0. Original captured data: CC0-1.0. The original fixed protocol remains in `../training-protocol-v1/`.
