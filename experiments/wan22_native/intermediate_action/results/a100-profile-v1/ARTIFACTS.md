# Intermediate profile and factorial VAE cache artifacts

Release tag: `wan22-intermediate-factorial-a100-v1`. This package is prepared locally. The download URLs become usable only after the maintainer publishes the release.

The archive preserves all 511 recovered files. Four Markdown review copies replace the local workspace prefix with `[WORKSPACE]/`, twice per file. The other 507 files are byte-exact, including all scientific source, input tensors, raw predictions, gradients, checkpoints, PNGs, encoded cache values and JSON evidence. `public-derivation.json` lists each original/public hash and the exact replacement rule. `original-index.json` preserves the original recovery inventory.

## Download and verify

Use Python 3.11 or later. The standalone verifier uses only the standard library and does not require Torch, a GPU or a provider account.

After obtaining the release's `fetch_artifacts.py` and pinned `release-download.json`:

```sh
python3 -B fetch_artifacts.py --metadata release-download.json --output downloaded
```

To verify release assets already in a local directory:

```sh
python3 -B fetch_artifacts.py --metadata release-download.json --parts-directory assets --output verified-local
```

Each transport part is at most 8,000,000 bytes. The verifier checks part lengths and hashes, the complete stream, safe regular archive members, every extracted file hash and exact file coverage. It writes `verification.json` only on success. Use a fresh output directory; retain an interrupted directory and choose a new one for another attempt. A local verification is recorded in `local-verification.json`; it is not a public HTTP download measurement.

## Scientific readers and recorded review hashes

The extracted data is below `downloaded/recovered`. The checked repository is `intermediate-source/open-worldline`; profile helpers are in `intermediate-profile-src`; cache helpers are in `factorial-cache-src`. Results are in `action-results/intermediate-profile-spatial-v1` and `action-results/factorial-native-cache-v1`.

The unchanged scientific Python preflight readers consume the original plans, sources, CPU evidence and input bytes. With the recorded Python/Torch dependencies installed, they can be called separately:

```sh
artifact_root="$PWD/downloaded/recovered"
export PYTHONPATH="$artifact_root/intermediate-source/open-worldline"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python3 -B "$artifact_root/intermediate-profile-src/run.py" --preflight --prepared-directory "$artifact_root/action-results/intermediate-profile-spatial-v1"
python3 -B "$artifact_root/factorial-cache-src/run.py" --preflight --prepared-directory "$artifact_root/action-results/factorial-native-cache-v1"
```

These calls check saved inputs without running a model. They do not authorize another execution.

Both unchanged readers passed against the fresh public extraction. `scientific-reader-compatibility.json` records their exact source and plan identities.

The archived shell prechecks also pin the original Markdown review hashes. Those specific checks will reject the four public review copies. Their original hash references and recorded admissions remain unchanged. Use the public index and derivative mapping to verify published bytes. A new execution needs a fresh preparation and admission with its own reviewed metadata; do not alter historical evidence to imply that a redacted file has its original hash.

`original-raw-audit-report.json` is the byte-exact passed audit of the original recovery. The public derivative changes its four review documents only. The archive contains a numerical placement profile and a six-arm VAE cache from one development scene; it does not establish trained visible control or scene generalization.
