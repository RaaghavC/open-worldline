# SPDX-License-Identifier: Apache-2.0
"""Fetch the pinned official Wan UMT5 encoder and tokenizer into a local folder."""
import argparse
import json
from pathlib import Path
import shutil

from experiments.baselines.fetch_assets import fetch

MANIFEST = Path(__file__).with_name("weights-manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    expected_bytes = sum(item["size"] for item in manifest["artifacts"])
    if expected_bytes > 11_500_000_000:
        parser.error("Pinned artifact set exceeds the 11.5 GB download budget")
    if not args.verify_only:
        parent = args.output.resolve()
        while not parent.exists():
            parent = parent.parent
        if shutil.disk_usage(parent).free < expected_bytes + 2_000_000_000:
            parser.error("Insufficient free disk for the pinned artifact set plus 2 GB reserve")
    fetch(manifest["repository"], manifest["revision"], manifest["artifacts"], args.output,
          verify_only=args.verify_only, manifest_payload=manifest)


if __name__ == "__main__":
    main()
