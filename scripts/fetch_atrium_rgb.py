"""Fetch the pinned original CC0 RGB capture for reproducible CPU data checks."""
import argparse
import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://github.com/RaaghavC/open-worldline/releases/download/atrium-rgb-v1/atrium-rgb-v1.zip'
BYTES = 22_386_589
SHA256 = '3e65d7e606fc9bd2929b6d832090bace6499d205117dc6f39eda46330c4383fe'
MANIFEST_SHA256 = '942eaf38badb1c2de5489fac59b7727e5c2a3d5699ec44aa415b7862c8440ef0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Verify an existing archive instead of downloading')
    parser.add_argument('--output', type=Path, required=True, help='Fresh capture directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be a new directory; existing captures are never overwritten')
    if args.archive:
        payload = args.archive.read_bytes()
    else:
        request = urllib.request.Request(URL, headers={'User-Agent': 'Worldline-reproducible-data'})
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = response.read(BYTES + 1)
    if len(payload) != BYTES or hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError('Archive size or pinned SHA-256 differs')
    names = {'manifest.json', 'README.md', 'DATA-LICENSE'}
    names.update(f'{branch}/{frame:04d}.png' for branch in ('closed', 'open') for frame in range(66))
    expected = {'atrium-rgb-v1/' + name for name in names}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if len(archive.infolist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError('Archive must contain exactly the documented 135 files')
        manifest = archive.read('atrium-rgb-v1/manifest.json')
        if hashlib.sha256(manifest).hexdigest() != MANIFEST_SHA256:
            raise ValueError('Original capture manifest differs')
        args.output.mkdir(parents=True)
        for name in sorted(names):
            destination = args.output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(archive.read('atrium-rgb-v1/' + name))
    print(f'Verified and extracted 132 original PNGs and their manifest to {args.output}')


if __name__ == '__main__':
    main()
