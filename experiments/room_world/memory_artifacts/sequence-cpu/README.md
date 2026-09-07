# CPU check of batched observed-history computation

The separate [memory_sequence.py](../../memory_sequence.py) computes frozen image/action features and decoding across the 65 teacher-observed steps in a batch, while the GRU still advances sequentially. It is not connected to the measured stepwise trainer. The frozen predictor uses per-sample GroupNorm and has no BatchNorm or dropout.

This package preserves three CPU records, their source and five independent test cases. The predictor is the actual published original Room checkpoint, with 498,651 frozen parameters. The memory addition has 24,960 trainable parameters. Checks used deterministic synthetic RGB and nonzero conditioning weights. No optimizer update, renderer execution or GPU call produced these records.

| Record | Result |
|---|---|
| [Initial 16×16 check](failed-16-initial/comparison.json) | Failed because the checker incorrectly required reset's recurrent-weight gradient to be nonzero. Carry had passed before that assertion. |
| [Corrected 16×16 check](passed-16/comparison.json) | Carry and reset passed; both scalar losses matched exactly. |
| [Full 64×64 check](passed-64/comparison.json) | Carry and reset passed for two branches and all 65 transitions. |

Every prior state is zero in reset mode, so `gru.weight_hh` has an exactly zero gradient. The corrected checker explicitly requires that zero and requires the other gradients to be nonzero. The failed record remains unchanged. The [initial checker source](measured-source/check-room-memory-sequence-initial.py.txt) was reconstructed by reversing the two recorded assertion edits; its bytes match the source hash in the failed measurement. The [correction](assertion-correction.patch) and [corrected source](measured-source/check-room-memory-sequence.py.txt) are retained separately.

At 64×64, carry's loss differed by 7.45e-9 and reset's loss matched exactly. All six parameter-gradient tensors and the RGB input gradient were compared. The largest parameter-gradient absolute differences were 2.62e-10 for carry and 2.33e-10 for reset. RGB input-gradient maximum differences were 3.41e-11 and 3.14e-11; relative L2 differences were 1.67e-6 and 1.68e-6. Changing future observed frames left earlier predictions exactly unchanged. The complete numerical values, tensor hashes and declared tolerances are in the measured JSON files.

CPU batching was **slower in this single check**: carry forward/backward took 4.300 seconds versus 2.623 seconds stepwise; reset took 4.293 seconds versus 3.333 seconds. These are not repeated performance measurements. Recorded RSS values were taken after backward and are not peak memory. This CPU evidence establishes no MPS speed, MPS numerical-equivalence, optimizer-trajectory or model-quality result.

The [five independent test cases](measured-source/test_room_memory_sequence_independent.py.txt) also cover time-major action/history order, final-state equality, absence of future/session coupling, the full initial-state gradient in carry, no initial-state dependence in reset, and unchanged inputs/parameters. The reviewer reported five passes in 1.92 seconds; [independent-review.json](independent-review.json) identifies the exact test source and distinguishes that report from a retained raw test log.

To reproduce the full CPU check from the repository root with the project dependencies installed:

```sh
python experiments/room_world/memory_artifacts/sequence-cpu/measured-source/check-room-memory-sequence.py.txt \
  --repo . --output /path/to/a/new/cpu-comparison-directory --size 64
python -m pytest tests/test_room_memory_sequence_independent.py -q
```

The measured environment used PyTorch 2.5.1 and two CPU threads. The checker requires a new output directory and records current source hashes. Reproduction after changing those sources is a new measurement. The frozen predictor checkpoint remains in its existing repository location and is not copied into this package.

[PROVENANCE.json](PROVENANCE.json) binds all package files and identifies the measured source. Source and records are distributed under [Apache-2.0](LICENSE). This is a computation check for one original 64-pixel room experiment, with no claim of learned memory improvement or frontier world-model quality.
