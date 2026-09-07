# Independent CUDA protocol CPU review

All 12 independent CPU checks passed against the [retained source hashes](report.json). The measured pytest call took 0.79 s; the enclosing invocation took 0.973 s. The original output retains two non-failing warnings about the already imported anyio pytest plugin. No CUDA kernel, foundation model, actual weight values, cloud account or paid resource was used.

The checks cover exact original source bytes and retained input identities, all 100 prefix/timestep calls, a separate literal 50-step CPU solver calculation, state isolation and partial failures, required hardware records and sampled memory limits, explicit CPU verification copies through 825 scalar loader stand-ins, and complete pair-to-clip source/precision/artifact admission. Changed hashes, missing inputs/outputs, wrong timing, stopped workers and incomplete pairs are rejected.

These checks do not establish CUDA installation compatibility, GPU numerical equivalence, performance or image quality. The optional clip uses CPU UniPC at 17 frames and 512 × 288. Its 120 s decoder allowance is unmeasured. A 900 s worker deadline neither stops cloud billing nor deletes a pod or its storage.

The [review runner](run-independent-review.py), exact checked sources, report and original test output are included. To repeat the CPU checks, use the declared runtime dependencies plus the test dependency `pytest==7.4.4`. Run from the repository root with a fresh output directory:

```sh
python experiments/wan22_native/cuda_reference/cpu-results/independent-v1/run-independent-review.py \
  --repo . --output ../NEW_CUDA_PROTOCOL_INDEPENDENT_REVIEW
```

This command runs CPU fixtures only. [publication.json](publication.json) records the exact copied files and the additional explanatory README.
