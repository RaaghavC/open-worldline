# Core runtime metadata update

The core profiler now records the exact inherited values of `PYTORCH_MPS_LOW_WATERMARK_RATIO`, `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, `PYTORCH_ENABLE_MPS_FALLBACK`, `PYTORCH_MPS_FAST_MATH` and `PYTORCH_MPS_PREFER_METAL`, plus MPS recommended-memory bytes. Unset variables remain null. The parent additionally records the five values passed before child startup. The existing forced fallback value of `0` remains unchanged.

This addition records execution settings; it does not change model, loader, attention, precision, sampler or allocator-policy math. In particular, a lower watermark chosen for the separate decoder is not silently applied to the core.

`cpu-v3/` retains the current source and the affected CPU verification: 16 implementation cases and four independent profiler cases, all 20 passing in 2.72 seconds of pytest-reported time. CPU and mocked-MPS metadata tests verify exact raw values, unset/null behavior, device-memory reporting and absence of environment mutation. No actual MPS work or large checkpoint load occurred.

Earlier reports, source snapshots and `publication.json` remain unchanged. Use `cpu-v3/report.json` for the current profile gate. The original seven-test independent core gate remains valid because core and loader files are unchanged. This update is covered by the separate `publication-v2.json` file index.

The refreshed independent profiler review is retained in `independent-profile-v2/`: four checks passed in 0.061 seconds, with exact checked source and imported helper copies. The reviewer found no computation or guard change. Its report SHA256 is `bb6708f34054fba6ad6a3efab56c6672e4af72ac31e4fbaa7a2e333290ef4f0f`.
