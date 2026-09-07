# Fixed 512-update memory study: validation results

These values are copied from the completed evaluator. They describe a 64 × 64 pixel room model, three initialization seeds and eight validation scenes. They do not establish general world-model quality or novelty.

The paired test asks whether both generated door states are closer to their own true state than to the opposite branch. Ties fail. A positive error reduction means memory carry has lower doorway error than resetting memory at every step.

| Protocol | Hidden wait | Mean error reduction | 95% interval | Both branches correct | 95% interval | Memory gate |
|---|---:|---:|---|---:|---|---|
| Observed prefix, generated return | 8 | -0.07% | -0.54% to 0.34% | 0.00% | 0.00% to 0.00% | failed |
| Observed prefix, generated return | 16 | -0.09% | -0.56% to 0.32% | 0.00% | 0.00% to 0.00% | failed |
| Observed prefix, generated return | 32 | -0.10% | -0.57% to 0.31% | 0.00% | 0.00% to 0.00% | failed |
| Generated history throughout | 8 | 0.69% | -4.39% to 4.58% | 8.33% | 0.00% to 25.00% | failed |
| Generated history throughout | 16 | 0.61% | -5.24% to 4.82% | 8.33% | 0.00% to 25.00% | failed |
| Generated history throughout | 32 | 0.15% | -5.87% to 3.92% | 4.17% | 0.00% to 12.50% | failed |

Intervals are the evaluator's paired scene-bootstrap percentiles, conditional on the three trained seeds. Each memory gate requires at least 30% mean error reduction, at least 80% pair correctness and positive reduction in every seed. Control retention is a separate required check below. The plotted thresholds alone do not establish overall success.

Control retention across every required type and seed: **passed**.

| Seed | Control | Carry MAE | Reset MAE | Frozen MAE | Fixed control gate |
|---|---|---:|---:|---:|---|
| 20260907 | out_of_reach_interaction | 0.014587 | 0.014612 | 0.014556 | passed |
| 20260907 | overall_equal_type | 0.014919 | 0.014926 | 0.015124 | passed |
| 20260907 | translation_cycle | 0.015659 | 0.015670 | 0.015854 | passed |
| 20260907 | turn_open_close | 0.014511 | 0.014494 | 0.014962 | passed |
| 20260908 | out_of_reach_interaction | 0.014420 | 0.014442 | 0.014556 | passed |
| 20260908 | overall_equal_type | 0.014867 | 0.014858 | 0.015124 | passed |
| 20260908 | translation_cycle | 0.015581 | 0.015601 | 0.015854 | passed |
| 20260908 | turn_open_close | 0.014600 | 0.014531 | 0.014962 | passed |
| 20260909 | out_of_reach_interaction | 0.014483 | 0.014516 | 0.014556 | passed |
| 20260909 | overall_equal_type | 0.014889 | 0.014880 | 0.015124 | passed |
| 20260909 | translation_cycle | 0.015654 | 0.015617 | 0.015854 | passed |
| 20260909 | turn_open_close | 0.014529 | 0.014507 | 0.014962 | passed |

MAE uses normalized RGB values. Each carry control error must be at most 1.05 times both its reset and frozen references. Full per-seed results, intervals, predictions and control evidence remain in the evaluator output.

Evaluator file SHA256: `33b08c887b98af949eba677ab7b7070f85eb4d8b3e2a147a8c710461e4c8a02e`.

Presentation source: `summarize-room-memory.py`. This presentation reads saved statistics and does not change predictions, thresholds, intervals or training.
