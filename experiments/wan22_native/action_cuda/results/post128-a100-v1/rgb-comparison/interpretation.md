# Matched checkpoint128 versus checkpoint16 RGB comparison

The frozen reader completed on its first invocation after the structural audit passed. It scored all 17 frames of both checkpoint128 arms, verified their raw RGB-to-PNG identities, and compared them with all corresponding checkpoint16 frames and the same original Atrium targets. Initial observation, saved noise, commands and text match exactly across checkpoints. Raw files remain unchanged.

| Future-frame mean absolute RGB error, range 0–1 | Closed | Open |
| --- | ---: | ---: |
| Checkpoint128 versus own target | 0.1835971495 | 0.1955583772 |
| Repeat starting image versus own target | 0.1789626381 | 0.1903559489 |
| Checkpoint128 target error minus checkpoint16 target error | +0.0000465581 | +0.0000354559 |
| Checkpoint128 versus checkpoint16 image difference | 0.0010593017 | 0.0011793926 |

Both checkpoints score 0/16 on the strict test requiring both generated branches to be closer to their own truth than the opposite truth, for both whole-frame and paired-truth-difference regions. Ties fail. The paired-truth region includes lighting changes; it is not a semantic door mask. These are correspondence measurements, not a numerical visual-quality standard.

The six-row contact was inspected at the displayed resolution. Its seven fixed frames per arm show a recognizable room with the door closed and framing almost unchanged for both checkpoints and both commands. The corresponding original open target opens the door at frame 1; later original frames show the requested turn away from the doorway. This contact inspection supports the parent's observed endpoint failure. It is not an inspection of every individual frame; root owns the separate all 34-frame visual review. All frames are nevertheless included in the numerical report.

This paired render does not demonstrate improved command following after 128 updates. Differences from checkpoint16 are small, and target correspondence is descriptively slightly worse in both branches. That conclusion applies to this seen room and one shared noise draw. It does not determine whether a different architecture, objective, dataset or budget would work, and it does not measure generalization.

Frame 0 is a conditioned reconstruction and is excluded from the 16 future-frame averages. Its target MAE is 0.001794696958 for both arms. Ground truth uses the exact prior LANCZOS resize to 1252×704 and centered crop to 1248×704, with only the same derived closed-start canonical-frame correction; all original capture bytes are unchanged. Commands remain initial wait/interact, followed by 15 left turns of 7.5 degrees. The paired-truth difference later becomes small as the requested camera turns away, so whole-frame error is sensitive to camera alignment and indirect lighting rather than door state alone.

Report SHA256: `f0fff1f533543372b6d901c12ae63e6081bc84930a6de703edb660ad26b9667f`.

Reader SHA256: `970ce554e04eba2774e7f78b7d499acefd8e036935504aa0b96c6a01634e2a50`. Exact reader, fixture, pre-run CPU report and independent read review are retained under `source/`. No model, solver, decoder or training replay occurred.
