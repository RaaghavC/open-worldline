# Tiny decoder CPU evidence

Nine random-weight CPU checks passed in 0.474 s. The full original TAEHV architecture was evaluated at a small 1 × 2 latent spatial size. These checks establish streaming equations and input/failure boundaries; they do not measure reconstructed image quality or full-resolution latency. No official weight values were read and no GPU was used.

`tests.json` is copied byte-for-byte from the local run. `checked-source/` and `reused-source/` retain each measured file, with hashes verified against that report. The script preserves before/after source identity.

Run from the repository root in the pinned Wan environment:

```sh
python -m experiments.wan22_native.tiny_decoder.test_cpu --output /path/to/fresh-cpu-report/tests.json
```

Coverage: exact upstream sequential-versus-streamed 17-frame outputs; ABot 9/12-frame chunks; A/B/A cache reset and alternate chunk partitions; unchanged normalized inputs; FP32 outputs under ambient CPU BF16 autocast; cleanup after interruption; malformed tensor/checkpoint rejection before loading; source-bound gate rejection; and supervised child cleanup on failed handoff.
