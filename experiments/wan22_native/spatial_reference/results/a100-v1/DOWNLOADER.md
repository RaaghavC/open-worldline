# Public artifact verification

`fetch_artifacts.py` has verified a complete fresh public HTTPS download of all 536 original files. [Pinned metadata](release-download.json) names the actual uploaded assets; [verification](recovery/public-download-verified.json) records the result. It does not import the Pod SSH helper, use a provider account, read credentials or run a model. Python 3.11 or newer is sufficient; only the standard library is used.

The pinned local JSON metadata has exactly these fields:

- `schema`: `worldline-spatial-release-download-v1`.
- `index`: `{name, bytes, sha256, url}` for the exact final `index.json`. `name` is `index.json`.
- `parts`: ordered `{name, bytes, sha256, url}` records for every final `evidence.tar.gz.part-NNN`, beginning at `000`.
- `stream_sha256`: the complete compressed stream SHA256 from that final recovery index.

Every pinned URL names the same exact public release tag and its matching asset filename. The tracked publication record binds the metadata SHA256.

The final recovery index remains byte exact. The downloader fetches it by pinned size/hash, then requires its ordered part records and stream hash to match the local metadata. Each part is at most 90,000,000 bytes. It verifies each downloaded byte count and SHA256, the concatenated stream hash and every original archive file's length and SHA256. It streams extraction across the part files without assembling a large archive. It rejects unsafe paths, symbolic/hard links, duplicate/extra/missing members, malformed sizes/hashes, and an existing output directory. An error leaves an incomplete output directory without `verification.json`; use a new output directory after resolving the cause.

Download and verify the complete public evidence:

```sh
python3 fetch_artifacts.py --metadata release-download.json --output /absolute/new-directory
```

Verify assets already downloaded without network access:

```sh
python3 fetch_artifacts.py --metadata release-download.json \
  --parts-directory /absolute/existing-assets --output /absolute/new-directory
```

The local asset directory must contain the exact `index.json` and every named part. The metadata file still uses the actual public URLs so the same identity can be checked in either mode. Output contains the metadata bytes, index, unchanged parts, `recovered/` with every original file, and `verification.json` after complete verification. The measured remote `/workspace` and `/opt` path strings remain untouched inside their records.

Five bounded offline fixtures cover exact multi-part extraction including an empty log; corruption/truncation; traversal, links and duplicates; mismatching order, URLs, index and malformed sizes; and a simulated HTTPS asset path with network access forbidden. These exercise the existing recovery logic. These fixture tests are separate from the completed real HTTPS verification linked above.

The Python installation must have a trusted HTTPS certificate bundle. This Mac's isolated Python initially lacked one; setting `SSL_CERT_FILE` to its installed Certifi CA bundle allowed the complete verified download. Certificate verification remained enabled.
