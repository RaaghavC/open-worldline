# CPU and download evidence

These records do not contain a full-model GPU execution or image-quality measurement. CPU tests use tiny weights, and the profiler review uses a stand-in denoiser with the real official CPU UniPC solver. The full 5B tensor shapes and storage count were checked using a meta model and actual shard headers.

- `cpu-v1/` retains the first 14-test execution and its exact source. Its observation reader was subsequently corrected to reject additional tensor keys before reading any values. This historical report cannot authorize the current profiler.
- `cpu-v2/` retains the 14-test rerun against the corrected, current profiler and its exact source. The run passed in 4.73 seconds of pytest-reported time; the report also records total measured wall time. Two expected official CUDA-autocast warnings and two pytest plugin-rewrite warnings occurred on CPU.
- `independent-core/` retains the seven independent core/loader checks and their report. The report references an earlier version of the implementation test file; `publication.json` identifies any historical source not available byte-for-byte. The tested core, loader, vendor files and independent test remain unchanged.
- `independent-profile/` retains four additional CPU checks for the corrected input reader, exact positive/negative call order, CPU solver math and failure records.
- `weight-download/` retains only the actual completed download manifest and metadata identities. The three transformer shards and VAE checkpoint remain outside the repository.

All original execution reports are copied without modifying their measured values. Each bundle includes the source files available at its recorded hashes. `publication.json` in this directory hashes every published file except itself. Attribution and upstream license copies are in the parent directory. The previous startup failures remain recorded in the original reports; they did not execute a model.
