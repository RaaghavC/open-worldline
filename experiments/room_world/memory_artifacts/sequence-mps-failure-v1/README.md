# Retained failed MPS batching check

September 7, 2026. This first GPU comparison failed its declared RGB input-gradient limits. The failure and its thresholds are unchanged. No optimizer update or model-quality evaluation occurred.

The check used the original training scene 5023, both branches, all 65 transitions and 64 × 64 RGB. Both computational paths loaded the same nonzero memory parameters from the completed carry profile. It requested three comparisons for each state mode, but stopped after the first carry comparison failed. Reset and the remaining repetitions were not executed.

| Measurement | Result |
|---|---:|
| Both scalar losses | 0.010371977463364601 |
| Largest trainable-parameter gradient difference | 8.73115e-10 |
| Largest parameter-gradient relative L2 difference | 6.66338e-6 |
| Largest RGB input-gradient difference | 3.13000882e-7 |
| RGB input-gradient relative L2 difference | 0.00159644 |
| Allowed RGB absolute / relative differences | 1e-7 / 0.001 |

All six trainable-parameter gradient comparisons passed their declared limits. The RGB input-gradient comparison failed both limits. All retained values were finite. The [report](probe.json) and [original gradient tensors](gradients-carry.pt) preserve the measurements. [The supervised process record](terminal.json) reports a failed process, not a resource-limit stop.

The maximum RGB difference is approximately one absolute-error subgradient unit for this array shape. A pixel whose rounded prediction changes between exact equality and a tiny nonzero error can have such a derivative difference. This is a hypothesis at this stage: the failed probe did not retain its prediction/residual arrays, so its evidence alone does not prove that explanation. A separate diagnostic must retain those arrays and compare both paths with the same supplied output gradient. The original failed check must not be relabeled after that diagnostic.

The single recorded synchronized forward/backward calls took 2.757 s for the stepwise path and 1.174 s for the batched path. They include RGB input gradients, which ordinary training does not request. This stopped check is insufficient for selecting a full training budget or claiming a stable training speedup.

The exact executed script and module snapshots, original launch/terminal records, error log and [publication checksums](publication.json) are retained. Code and original gradient artifacts use Apache-2.0; source RGB is from the separately released original CC0 development capture. No reserved scenes or external model weights were used.
