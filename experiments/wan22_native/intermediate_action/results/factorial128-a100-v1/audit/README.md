# Independent factorial training artifact audit

This CPU reader checks the completed block28 six-arm training experiment. It loads saved predictions, gradients and small adapter recovery states, never foundation weights or CUDA. The original recovery stays unchanged.

From the workspace root:

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  work/wan-adapter-env/bin/python work/factorial-intermediate-training-audit-v1/audit.py \
  --repo outputs/open-worldline \
  --recovery work/factorial-intermediate-training-recovered-v1 \
  --run-root work/factorial-intermediate-training-recovered-v1/recovered/action-results/factorial-intermediate128-v1 \
  --output work/factorial-intermediate-training-actual-audit-v1
```

The output must be fresh. The report is retained on failure. A remote use before full recovery can omit `--recovery`, but requires a completed parent and terminal; it is not a substitute for later full recovery verification. No remote audit ran during the expired video admission window.

The schema is `worldline-factorial-intermediate128-actual-audit-v1`. A passed report binds the exact parent, worker result, terminal, reviewed plan, final128 checkpoint manifest, final adapter, and all88 source hashes. It reports128 updates,256 main predictions,128 auxiliary predictions,64 auxiliary feature extracts and825 unchanged foundation tensor identities. This receipt matches the separately reviewed six-arm video preparation gate.

Independent arithmetic reuses the unchanged NumPy reader and loss helpers from the previously passed action-effect audit. Main losses use the four future latents. Auxiliary losses recompute separate FP32 CFG5 operations, then the negative open-minus-closed guided velocity and raw target difference. The previous scalar consistency limits remain relative2e-6 and absolute1e-8. These are agreement checks between recorded FP32 reduction results and CPU FP64 scoring, not quality thresholds.

Every saved combined gradient is checked against its recorded clip and recurrent/output norms. Checkpoint recovery uses `torch.load(weights_only=True, map_location='cpu')` with only the installed `TorchVersion` metadata type allowed. It verifies finite optimizer states, steps, parameter ownership and shapes, plus exact original CPU/draw RNG bytes. It does not replay backward, reconstruct separate main/auxiliary gradients, or claim an exact AdamW replay.

CPU history is retained: the first attempt passed six checks and failed one temporary-path fixture because macOS `/var` is a symlink. Resolving that fixture path preserved the strict production path check. CPU-v1 passed seven checks. Independent review then required strict nonnegative finite resource records and the minimum GPU capacity; CPU-v2 passed eight checks with those additions. No scientific data or producer code changed.

The experiment changes both the dataset and adapter placement relative to the older room training. A passed numerical audit establishes saved-file consistency and original training mechanics. It establishes neither a placement-only advantage nor visible door/camera control.
