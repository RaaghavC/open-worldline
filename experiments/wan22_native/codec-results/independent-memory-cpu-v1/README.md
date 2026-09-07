# Independent decoder cleanup review

Four CPU tests passed against these exact sources. All 17 tiny native output frames remained bit-exact with per-convolution and existing per-chunk cleanup. The fixture registered 46 native causal convolutions and observed 129 calls. Checks cover preservation of unrelated hooks, event-sink failure, synchronization failure and current source-bound CPU gates. Codec arithmetic and the initial-image reader retain their previous hashes.

This review supports the separately measured synthetic decoder v3. It does not measure full-model memory savings or generated quality. Callback duration includes synchronization waits and allocator cleanup; it is not isolated allocator overhead. No GPU or pretrained weight values were loaded.

From the repository root, using the pinned Wan runtime:

```sh
python -m experiments.wan22_native.test_codec_memory_independent --output /path/to/new-tests.json
```

`report.json` records the independent read review; `tests.json` is the unchanged executed test record. `source/` retains the exact reviewed code.
