# Saved fixed512 training audit

This CPU reader checks the recovered command-attention training records against the original, unchanged transfer. It imports NumPy and standard-library modules only. It does not load weights, execute a model, load a pickle, or regenerate noise.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 work/wan-adapter-env/bin/python \
  work/command-attention-training-audit-v1/audit.py \
  --recovered-root work/ACTUAL-RECOVERY/recovered \
  --original-transfer work/command-attention-transfer-v1/local-extraction-v1 \
  --recovery-verified work/ACTUAL-RECOVERY/recovery-verified.json \
  --output work/command-attention-training-actual-audit-v1
```

The recovery must contain the profile and training runs as sibling directories under `action-results/`, their dispatch records, and the retained dispatcher source. Output must be fresh. The report is retained even when a producer failed or an audit check fails. Exit code zero requires passed parent and worker records plus all audit checks; a partial run is never called a successful512-update result.

Checks cover:

- All512 saved schedule/scalar rows, exact original512 draws and their saved RNG hashes. The42 input artifacts plus the input manifest are verified against the original transfer. No Gaussian draw is regenerated.
- Five controller checkpoints at0,128,256,384,512: all parameter names, FP32 shapes, finite values, tensor hashes, binary hashes, exact initial values, identities and final pointer. The optimizer/CPU RNG pickle files are checked by file hash but never loaded; their internal state semantics are outside this audit. Separately retained CUDA RNG byte tensors are read.
- Initial and final48 raw predictions per evaluation, from four fixed noises, six arms and two contexts. All28 scores per phase are recomputed with the declared FP32 guided contrast followed by FP64 future-only reductions. These are measurements, without an invented quality threshold.
- The22 retained training prediction arrays and five combined post-clip gradient bundles at updates1,2,5,509,512. Main and auxiliary MSE and aggregate gradient norms are recomputed. The other507 updates have scalar records, without raw prediction/gradient replay.
- Exact training/source/input/admission/actual-profile receipts, same-Pod UTC lease,600-second reserve, sampled original time/memory caps, parent/worker/dispatch terminals and complete output hashes. Sequential allocated/reserved samples are not assigned an additional inequality gate.
- All825 before/after current-value hash records and original loader records. Raw foundation tensors are absent from recovery, so this does not perform a new foundation hash computation.

No full backward, optimizer, omitted-gradient, sampling or visual-quality replay is claimed. Model inputs and observed prefixes are bound through the original input bytes and producer source. Intermediate noisy inputs and attention activity were not retained, so their runtime behavior is not independently reconstructed.

`profile_reference.py` and `arrays.py` are byte-exact copies of the earlier profile audit's helpers; only read/check functions are used. `reference/` preserves the producer and checkpoint writer read during this review. Three small CPU tests check analytic losses/norms/checkpoints, corrupted records/hashes, and partial results. They do not simulate an actual GPU result.
