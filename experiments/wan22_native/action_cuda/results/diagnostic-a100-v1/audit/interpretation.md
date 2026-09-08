# Actual fourteen-prediction audit

The frozen CPU reader passed on its first invocation. It verified all 14 prediction identities, all four saved command-difference tensors, the exact prepared-v3 inputs and sources, and the retained load/before/after records for all 825 original foundation tensors. All four checkpoint-zero comparisons with native predictions were bit-exact. This checks retained bytes and arithmetic without rerunning a model.

| Fixed k=506 future flow MSE | Checkpoint 0 | Checkpoint 16 | Relative decrease |
| --- | ---: | ---: | ---: |
| Closed target | 0.148484407142 | 0.147919876994 | 0.3801949032% |
| Open target | 0.162152676773 | 0.161552561582 | 0.3700926822% |

These values were independently reduced in FP64 from the saved FP32 predictions and targets. The recorded worker reductions agree within the reader's predeclared arithmetic tolerance. The examples were seen in training and share a fixed saved noise draw; each loss uses its own target-corrupted input. This is evidence of a small objective improvement on those examples, not a held-out result or an isolated command-response test.

The k=999 panel changes only the command while sharing the initial noisy latent, independent observation and text. Future-frame command-difference RMS at checkpoint 16 is 0.0003046884883 for the positive text branch, 0.0003113022430 for the negative branch and 0.001961411819 after FP32 CFG5. The guided difference has relative L2 0.001582771553 against the closed guided velocity. The positive and negative command-difference cosine is 0.006401127732. Guided difference L2 is 1.287486659 times five times the positive-branch difference L2, and 0.7084358145 times the sum of the two weighted component norms.

These first-step values do not support simple cancellation of the command difference by CFG as the explanation for the failed rendered door response. They also do not establish why the observed response failed across the full 50-step trajectory. The command difference is nonzero but small relative to the complete velocity. Compared with native predictions, the trained adapter changes the closed positive and negative velocities by RMS 0.01154690045 and 0.01155963593, respectively. Much of that total change therefore occurs without distinguishing the two commands in this one panel; this observation does not identify its cause.

Guided velocities were reconstructed with separate FP32 subtract, multiply and add operations. The runtime did not save guided tensors, so no saved-guided identity is claimed. The retained positive and negative tensors and four deltas match exactly; the recorded contrast metrics match independently recomputed values. The maximum FP32 difference between the two guided-delta decompositions is 7.450580597e-9.

No images were generated or assessed in this audit. It supplies neither a visual-quality conclusion nor authorization for a larger training run. Any further run requires the parent's separate decision and fixed protocol.

Report SHA256: `b18fbdce5a37fc3c7d93a512b3b47136dda9f0f788b111363c3b0db70f91277e`.

Executed reader SHA256: `a76be21884be026304970f649dbc1d400cf412c474ab8ec3c1472108e32f47d8`. Exact reader, fixture and pre-run CPU report are retained in `source/`. The actual model source and raw recovered outputs were not edited.
