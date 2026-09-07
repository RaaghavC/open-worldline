# What changes when the stored state is swapped?

The trained room-memory addition failed its [fixed validation tests](../validation-512-v2/README.md). This separate post-hoc diagnostic asks whether its saved state changes the returned doorway. It trains no parameters, fits no semantic probe and opens no reserved test scenes.

For each of three initialization seeds, both trained variants and eight validation scenes, the model observes actions 0 through 40. Its four recent images and subsequent actions are identical across the two door-history branches. At this boundary, three independent copies generate the 24 return frames using the normal state, the other branch's state, or an all-zero state. All 48 cases and 144 complete predictions are retained locally. Future true images are used only for scoring.

| Variant and intervention | Cases | Mean change in returned door-region RGB | Pairs with both outcomes correct |
| --- | ---: | ---: | ---: |
| Carried state, unchanged | 24 | Reference | 0 / 24 |
| Carried state, branches swapped | 24 | 0.00002816 | 0 / 24 |
| Carried state, set to zero | 24 | 0.02218057 | 0 / 24 |
| State reset each step, either intervention | 24 each | Exactly 0 | 0 / 24 |

RGB values use the 0 to 1 scale. The change is mean absolute difference from the normal prediction over the true paired-difference region. Both branches must be strictly closer to their own true outcome than to the opposite outcome to count as correct; ties fail. The two carried branch states differ by a mean L2 distance of 0.02460. All reset-control outputs are bit-exact across interventions.

The stored state can affect generated images when it is zeroed. The difference between the two trained branch states has little effect on the returned doorway. These measurements do not establish usable door-state storage or explain whether the failure originates in storage, decoding or training. Neither intervention solves the task.

The [summary](summary.json) contains all aggregate values. The [portable diagnostic report](diagnostic-portable.json) contains every case, score and raw tensor hash. Only six local checkpoint path strings have been replaced with repository-relative paths. The [publication record](publication.json) records the exact original and relocated metadata hashes; all other values are unchanged. The exact measured source is retained alongside the current implementation.

The [independent review](independent-review/README.md) verified all 48 cases, 384 saved tensors, 144 predictions and 3,104 numerical comparisons. Maximum numeric difference was 8.88 × 10⁻¹⁶. It also verified the source, checkpoint, data and raw file identities without repeating model inference. Its audit uses the original metadata and raw tensors, which remain local. Large GitHub asset transfers failed during this study, so this repository does not currently provide the 359 MB raw diagnostic package.

This diagnostic took 116.54 seconds on CPU. Reproduction uses the released six checkpoints and eight validation scenes with the existing fixed evaluation report:

```sh
python -m experiments.room_world.memory_state_diagnostic \
  --study experiments/room_world/memory_artifacts/training-512-v2 \
  --validation experiments/room_world/memory_artifacts/data/validation \
  --evaluation experiments/room_world/memory_artifacts/validation-512-v2 \
  --output /path/to/new-state-diagnostic --execute --max-seconds 600
```

The output directory must be new. Running this creates fresh results and timing records, with their own hashes; it does not recreate historical file hashes from metadata alone.
