# Recover the complete paired visual run

The [public release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-visual-a100-v1) is live. All 23 remote assets were verified, and a [fresh public download](recovery/public-download-verification.json) recovered all 443 original files. The adjacent metadata pins nine ordered part identities and their public URLs.

Run from this directory:

```sh
python3 fetch_artifacts.py --metadata release-download.json --output /absolute/fresh-visual-recovery
```

The standard-library downloader verifies the index, each part, the combined compressed stream and all 443 extracted files. It retains the unchanged action-recovery schemas and 90,000,000-byte part limit from the fixed16 release. It rejects path traversal, symlinks, duplicate or unexpected members, corrupt payloads and an existing output directory. TLS verification remains enabled. No provider account, key or model execution is needed.

Already downloaded `index.json` and all nine `evidence.tar.gz.part-000` through `part-008` files can be checked without network access by adding `--parts-directory /absolute/existing-visual-assets`.

The complete run is under `recovered/action-results/visual-spatial-v2`. Both `decode/result/<arm>/frames` folders retain all 17 native PNGs, and adjacent `rgb` folders retain the original FP32 shards. The archive also contains every solver state, initial conditions, final adapter, source/CPU snapshots, admission and immutable logs. Earlier v1 preparation is retained separately. External model weights, installed environments and credentials are excluded.

The five offline cases are byte-identical to those in the fixed16 downloader tests. [downloader-source.json](downloader-source.json) identifies the reused algorithm and description-only change. Recovery verifies bytes; it does not establish successful control or visual quality.

The original eight-part index is retained as the separate release asset `original-index.json` and [historical index](history/original-recovery-index.json). The derived public `index.json` changes only its parts list, splitting the original last part into two equal pieces. The complete compressed stream and every recovered file remain unchanged. The downloader code did not change for the nine-part transport.
