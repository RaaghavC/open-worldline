# Independent native sampler review

Five CPU tests passed against these exact sources. All 50 native UniPC steps and 100 denoiser calls matched an independently written time/state/context-dependent velocity oracle bit-exactly. Checks cover input preservation, observed/future token times, expired deadlines, failed stdin handoff, interruption cleanup, resource accounting and rejection of mixed input keys. Both actual stopped decoder profiles were rejected.

This is a source and CPU mechanics review. It does not admit generation until successful actual core/decoder evidence passes the runner's measured 900-second gate. No GPU or pretrained model was executed. No scientific novelty or visual-quality claim follows from these tests.

From the repository root, using the pinned Wan runtime:

```sh
python -m experiments.wan22_native.test_sample_independent --output /path/to/new-tests.json
```

`report.json` records the independent read review; `tests.json` is the unchanged executed test record. `source/` and `reused-source/` retain the exact reviewed code. Full clip results, when measured, are separate artifacts.
