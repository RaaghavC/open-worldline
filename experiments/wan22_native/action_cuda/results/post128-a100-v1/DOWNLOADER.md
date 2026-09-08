# Recover final 128 visual and twenty-prediction evidence

Use the pinned metadata with the existing downloader:

```sh
python3 fetch_artifacts.py --metadata release-download.json --output /absolute/fresh-recovery
```

The verifier checks HTTPS asset hashes and sizes, ordered part concatenation, the gzip stream and every extracted member. It rejects traversal, symlinks, duplicate/unexpected files and existing output directories. Its historical module description mentions fixed 16; the supplied metadata selects this exact release. Certificate checks remain enabled. No provider account or model is needed.

The actual public layout has 82 parts: the first 17 are 16,000,000 bytes each, and the remaining 65 are at most 8,000,000 bytes. The unchanged helper retains its existing 90,000,000-byte ceiling; exact metadata provides the narrower actual bounds. Add `--parts-directory /absolute/assets` for offline recovery. Full local recovery is verified before upload. The [release](https://github.com/RaaghavC/open-worldline/releases/tag/wan22-action-post128-a100-v1) is public and a [fresh HTTPS download](recovery/public-download-verification.json) verified all 608 files. [Publication status](release-status.json) binds the exact remote and download checks.

Runs appear at `recovered/action-results/final128-visual-v1` and `recovered/action-results/comparison128-spatial-v1`. The archive includes both full 17-frame clips,50 latent states per arm,20 predictions, sources, plans, canonical input files and checkpoints required by the measured evaluation. External foundation weights are excluded.

The public gzip stream is newly compressed and has a different identity from the original recovery. Only three prior training-audit copies replace the operational `run` workspace prefix; all 605 other members and every numerical/scientific field remain exact. Executed plans name the original audit hashes. Consult [public-derivation.json](recovery/public-derivation.json) when checking those three derivative copies against original hash references. All 608 original member names are preserved.
