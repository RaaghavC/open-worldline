# Current release transport

Use `release-download.json` with the unchanged `fetch_artifacts.py`. The current archive has 150 parts: the first two are the already verified 8 MB uploads, and the remaining 148 are at most 2 MB each. Intermittent TLS upload failures prompted this transport change.

The archive stream, all 511 indexed files, the four disclosed Markdown replacements, and every scientific source, input and output are unchanged. `original8-*` files and `README.md` retain the earlier 39-part preparation as history. Their part counts and index hash describe that earlier transport. `scientific-reader-compatibility.json` also records the earlier transport's extraction; its two exact prepared plans and all scientific reader inputs remain unchanged.

The current `index.json`, `release-download.json`, `public-derivation.json`, `local-verification.json` and `transport-change.json` describe this 150-part transport. The original raw index remains distinct. Historical shell prechecks still bind the original unredacted Markdown reviews and are not rewritten.

Publication and a fresh public HTTPS download remain separate final steps. No new model computation was performed.
