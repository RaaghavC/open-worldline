# Recover the combined diagnostic and 128-update evidence

The pinned [metadata](release-download.json) identifies the combined release's exact index, 100 parts and compressed stream. The [release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-fixed128-a100-v1) is public, and a [fresh HTTPS recovery](recovery/public-download-verification.json) verified all 957 files. [Release status](release-status.json) binds the final publication records. From this directory:

```sh
python3 fetch_artifacts.py --metadata release-download.json --output /absolute/fresh-recovery
```

The unchanged existing downloader verifies HTTPS asset sizes and SHA256, index identities, ordered part concatenation, the full gzip stream and every extracted file. It rejects existing destinations, traversal, links, duplicate/unexpected members, invalid sizes and corrupt bytes. Its historical module description mentions fixed16; the supplied pinned metadata selects this exact combined release. TLS verification remains enabled. No provider account or model is needed.

For an offline check, add `--parts-directory /absolute/existing-assets` where `index.json` and all `evidence.tar.gz.part-000` through `part-099` are available. The code retains its 90,000,000-byte per-part ceiling; this release's actual pins cap every part at 16,000,000 bytes. The five existing offline fixtures are copied unchanged. The full local derivative was separately recovered and verified before upload.

The diagnostic is `recovered/action-results/diagnostic-spatial-v3`. Training is `recovered/action-results/fixed128-spatial-v1`, with final weights under `training/checkpoint-0128/adapter.safetensors`. The full archive contains all nine retained checkpoints, optimizer/RNG states, 256 velocity predictions, 128 gradient bundles, canonical saved draws, prior cache/probe inputs and source copies. Foundation weight files and installed environments are excluded.

The public archive is a disclosed privacy derivative: only two local workspace prefixes in the historical dependency traceback changed. [Original/public index records](recovery/public-derivation.json) prove that all other 956 files match their original hashes. The original index remains available separately. Do not compare the new public compressed-stream hash to the original stream hash as if they were expected to match.

Parts 000–089 retain their verified 16 MB bytes. Only the formerly unuploaded tail, parts 090–099, uses 8 MB or smaller pieces. The full gzip stream is unchanged from both earlier public transport layouts.
