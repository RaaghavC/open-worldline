"""Fetch the authors' DIAMOND checkpoint and seven small starting-frame bundles."""
import argparse
import json
from pathlib import Path

from fetch_assets import fetch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("diamond-assets-manifest.json").read_text())
    fetch(manifest["repository"], manifest["revision"], manifest["artifacts"], args.output)

