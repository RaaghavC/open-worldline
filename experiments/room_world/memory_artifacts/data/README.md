# Room memory development data

These are original 64 × 64 rendered RGB trajectories for a recurrent-memory training experiment. They are teacher observations, not neural predictions. The original NumPy room renderer supplies every pixel. The training and validation scene seeds are separate from those used for the published direct predictor.

| Split | New scenes | Observations | Transitions | Compressed NPZ bytes | Capture time |
|---|---:|---:|---:|---:|---:|
| Training | 32 | 4,224 | 4,160 | 2,520,849 | 29.70 s |
| Validation | 8 | 1,056 | 1,040 | 661,475 | 7.58 s |

Each scene contains two 65-step branches with an identical initial closed-door image. One branch waits, and the other successfully interacts. Both then turn left 24 times, wait 16 times and turn right 24 times. Before the first return action, the last four observed frames are exactly equal between branches; the final door views differ. These properties were checked for all 40 development scenes. The renderer uses a simple direct-light model without Atrium's indirect-light cue.

Training scene IDs are 5000–5031, and validation IDs are 300000–300007. **Reserved test scenes 400000–400031 have not been generated.** Architecture and evaluation choices still require a frozen protocol before that set is opened. The scenes vary within one programmed room family, so the split does not establish general-world coverage.

Each NPZ contains only `observations` as uint8 `[2,66,3,64,64]` and `actions` as int64 `[2,65]`. An action at index `t` produces observation `t+1`. Branch order is wait, interact. A training history includes four observations ending at `t`, repeating the initial image for missing early history. Later RGB images are targets, not additional model inputs. Load NPZ files with `allow_pickle=False`.

The separate manifest includes scene IDs and teacher door states for validation. They are not input channels for the memory model. [memory_data.py](../../memory_data.py) creates the data, validates intervention/alignment/aliasing and prepares observed histories. It also supports separately evaluated unsuccessful interactions and varied waiting durations; these are distinct from the standard 65-step captures. Its development command deliberately offers only training and validation splits. The separate controls described below use the same eight validation scenes.

```sh
python -m experiments.room_world.memory_data --split train --output /path/to/new-training-data
python -m experiments.room_world.memory_data --split validation --output /path/to/new-validation-data
```

The output directories must be new. [provenance.json](provenance.json) and the two capture manifests contain file, raw-array and source hashes. Python 3.11.9 and NumPy 2.4.2 created the measured captures on the M4 Pro. These times measure CPU data creation and compression, not learned inference or training.

The original RGB/action data and metadata in this folder are dedicated under [CC0-1.0](DATA-LICENSE). Source scripts use Apache-2.0. No external images, assets or weights are included. No learned-memory improvement is claimed by this data release.

## Separate validation controls

[validation-controls/manifest.json](validation-controls/manifest.json) adds 24 short trajectories and 16 paired excursions on the same eight validation scenes. These are development controls, not additional independent scene examples. The original [control generator](../../memory_controls.py) verifies actual translations, an unsuccessful out-of-reach interaction, successful opening/closing, and the paired return conditions before saving.

| Control type | Cases | Transitions per branch | Branches per case |
|---|---:|---:|---:|
| Four forward, eight backward, four forward | 8 | 16 | 1 |
| Back away, attempt interaction, wait, approach | 8 | 16 | 1 |
| Turn away/back, open, wait, close, wait | 8 | 16 | 1 |
| Paired excursion with eight waits | 8 | 57 | 2 |
| Paired excursion with 32 waits | 8 | 81 | 2 |

The short controls cover all six command IDs and contain 384 transitions. The longer paired controls contain 2,208 transitions. All 2,648 RGB observations were captured in 20.52 seconds. Their NPZ files retain the same observations/actions-only interface, with a variable time dimension and a batch dimension of one or two. Teacher outcome annotations remain in the separate manifest. No reserved test scene was opened.

```sh
python -m experiments.room_world.memory_controls --output /path/to/new-validation-controls
```

These cases have been generated but have not yet established retained control accuracy or learned memory. Planned scoring keeps each control type separate and weights whole scenes equally.
