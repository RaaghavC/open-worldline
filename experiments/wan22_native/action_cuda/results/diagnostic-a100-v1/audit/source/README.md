# Fourteen-prediction diagnostic reader

This work-only NumPy reader checks retained predictions without importing Torch, loading a model, using a GPU, or contacting the cloud. It is bound to the exact prepared-v3 plan `c17ad42baf1dcd9898c7fcecb1aa3458113576ad92dd977c9ee7376e7f50c1e6` and the frozen diagnostic source. No actual recovered diagnostic has been audited yet.

From the workspace root, after recovery is verified:

```sh
work/wan-adapter-env/bin/python work/wan22-action-cuda-diagnostic-analysis-prep-v1/audit.py \
  --run /absolute/path/to/recovered/diagnostic-run \
  --output /absolute/path/to/new-audit-directory
```

The run directory must contain `plan.json`, `metrics.json`, `terminal.json`, the prepared inputs and `result/`. Output must be a new directory. A failed check leaves `failed.json`; success writes `report.json`. Inputs remain unchanged.

The reader verifies all 14 call identities and prediction files, four saved command deltas, the original 825 before/after weight-hash records and load records, and both checkpoint-zero command branches against the native reference in both text contexts. Bounded raw safetensors parsing rejects invalid metadata, tensor shapes, offsets and nonfinite values.

It independently reconstructs the original k=506 corruption and scores four future-only flow losses. It reports both FP64 MSE and FP32 arithmetic results. The predeclared tolerance of relative 2e-6 and absolute 1e-8 checks the recorded FP32 reduction against NumPy; this is an arithmetic check, not a learning or quality threshold.

For the same-input k=999 panel, it computes each command's guided velocity using separate FP32 operations `negative + 5 * (positive - negative)`. The runtime retains positive and negative velocities, but does not retain guided velocities. The reader therefore reports reconstructed guided hashes and checks the recorded effect metrics and decomposition residual; it does not claim comparison with a saved guided tensor. It reports positive/negative command-effect cosine and guided-effect norms to describe possible cancellation. These values alone do not identify the cause of failed action control.

The four objective losses use seen training examples, one fixed noise draw and branch-specific target-corrupted inputs. The k=999 command comparison shares the initial noise, observation and text and does not condition on future targets. Neither panel measures rendered quality or generalization. Foundation checks verify retained records against the pinned original catalog; they do not reload all original weights.

Five small CPU checks passed in `cpu-check.json`: exact local prepared-v3 identities and corruption; exclusion of the initial latent from the loss; an analytic example with nonzero conditional effects that cancel under CFG5; independent binary-reader construction; and malformed-file rejection. `history/draft-v1/` preserves the earlier reader and report before adding the explicit current-catalog-to-plan hash check. No diagnostic execution or model test occurred in these checks.

License: Apache-2.0, as stated in the reader and fixture headers.
