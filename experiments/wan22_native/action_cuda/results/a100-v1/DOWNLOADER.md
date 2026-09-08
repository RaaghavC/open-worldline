# Recover the exact evidence

The [public release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-cuda-a100-v1) has verified asset sizes and hashes. A [fresh public HTTPS download](recovery/public-download-verification.json) also verified all 283 recovered files. `release-download.json` pins its GitHub URLs and exact bytes. Run from this directory:

```sh
python3 fetch_artifacts.py --metadata release-download.json --output /absolute/fresh-action-recovery
```

No provider account, API key, model weights or model execution is required. The downloader verifies the index, ordered parts and complete compressed stream, then every extracted file. It rejects traversal, symbolic links, duplicate archive members, unexpected files, corrupt data and an existing output directory. TLS verification stays enabled.

To verify already downloaded assets without network access, put `index.json` and `evidence.tar.gz.part-000` together and add:

```sh
  --parts-directory /absolute/existing-action-assets
```

The recovered cache is `recovered/action-results/cache-spatial-run-v1`; the probe is `recovered/action-results/probe-spatial-v1`. Original parent/worker records, sources, targets, observations, parity tensors, all three adapter/optimizer/RNG bundles and logs remain unchanged. The archive excludes installed environments, wheel binaries, external foundation weights and credentials.

This reuses the tested spatial recovery implementation with action-specific schema names and an 83,000,000-byte per-part ceiling. [downloader-source.json](downloader-source.json) identifies its original and revised bytes. Five bounded offline cases are in `test_fetch_artifacts.py`. Artifact integrity checks do not independently establish scientific conclusions.
