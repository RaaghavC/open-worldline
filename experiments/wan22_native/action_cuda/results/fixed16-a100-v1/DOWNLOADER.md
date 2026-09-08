# Recover the complete fixed16 evidence

The [public release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-fixed16-a100-v1) contains all six pinned parts. Its published asset sizes and SHA256 digests were verified, and a [fresh public HTTPS download](recovery/public-download-verification.json) recovered all 485 files. `release-download.json` supplies their exact identities and public URLs. Run from this directory:

```sh
python3 fetch_artifacts.py --metadata release-download.json --output /absolute/fresh-fixed16-recovery
```

The downloader verifies the index, ordered parts, complete compressed stream and every extracted file. It retains the existing action-recovery schemas and algorithm, with a 90,000,000-byte per-part ceiling. It rejects path traversal, links, duplicate/unexpected members, corrupt files and an existing destination. TLS verification stays enabled. No provider credentials, model weights or model execution are needed.

For a network-free check of downloaded assets, place `index.json` and all six `evidence.tar.gz.part-000` through `part-005` files together and add:

```sh
  --parts-directory /absolute/existing-fixed16-assets
```

The fixed16 run is `recovered/action-results/fixed16-spatial-v1`; its final adapter is `training/checkpoint-0016/adapter.safetensors`. All 17 checkpoints, optimizer/RNG state, predictions, gradients, earlier cache/probe records and exact sources remain in the archive. External pretrained weights, installed environments and credentials are excluded. Original source/input and preparation archives are separate release assets whose hashes are retained in `code-provenance.json`.

The five offline tests are the unchanged meaningful cases from the two-update downloader. [downloader-source.json](downloader-source.json) identifies the reused source and the sole limit/description changes. Artifact verification proves recovered bytes, not rendered action quality.
