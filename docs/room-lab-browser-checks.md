# Room Lab browser checks

September 7, 2026. Local in-app browser at http://127.0.0.1:8788, using the published original predictor checkpoint on CPU.

- The initial page showed disabled action controls until the first room was ready.
- Start room loaded scene seed 31415 and displayed a labeled initial rendered view at frame zero.
- Open/close and turn-left controls produced two labeled neural frames. The displayed per-call model/history timing was about 7 milliseconds on this particular short run, excluding browser transfer and display. This is not a throughput benchmark.
- Saving a branch at frame two, moving forward, and returning restored frame two. The saved branch remained available for another attempt.
- Browser error and warning logs were empty for these interactions.
- Visual inspection showed immediate blur and loss of door detail. The demo exposes that failure; the interface does not describe its output as frontier-quality graphics.

The separate automated session tests verify generated-frame feedback, absence of future renderer calls, branch random-state equality, independent subsequent branches, stale-request rejection, bounded storage and missing-checkpoint recovery. Model-quality claims rely on the [recorded room experiment](room-rgb-experiment.md), not these interface checks.

These browser checks cover a short desktop interaction. They are not a complete accessibility audit or an hour-long session test.
