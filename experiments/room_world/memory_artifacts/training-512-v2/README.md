# Complete fixed 512-update memory training

All six arms completed the [revised protocol](../training-protocol-v2/PROTOCOL.md): three initialization seeds, each with memory carried across steps and a matched model with memory reset at every step. Each arm started fresh and completed 512 updates over 16 full passes through 32 training scenes. No partial checkpoint from the stopped 1,024-update experiment was reused.

| Initialization seed | Carry training interval | Reset training interval | Updates per arm |
|---|---:|---:|---:|
| 20260907 | 282.84 s | 309.94 s | 512 |
| 20260908 | 294.31 s | 281.83 s | 512 |
| 20260909 | 302.61 s | 290.23 s | 512 |

These intervals are the individual training worker measurements on the M4 Pro. They include recovery writes and training bookkeeping. They are not video generation speeds or evidence of learned memory quality. Each worker stayed within the predeclared 600-second limit.

The directory preserves all 46 measured files byte for byte: the study, six final inference checkpoints, last inference and optimizer recovery checkpoints, complete update records, process terminal reports, empty worker logs and checked source snapshots. [publication.json](publication.json) records their sizes and hashes. The frozen original room predictor is referenced by its checkpoint hash and remains in the main room experiment.

All six final checkpoints are eligible for the separate, fixed evaluation. None was selected by a validation score. The validation evaluator compares all six against the frozen predictor and checks memory and ordinary controls separately. Training completion alone does not establish memory improvement, higher visual quality, scientific novelty or Genie 3 parity.

The original source and original trained memory weights use Apache-2.0. The original synthetic training captures are separately identified under CC0-1.0 in the [data directory](../data/README.md).
