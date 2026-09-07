"""Explicit download of pinned official Apache-2.0 Wan files. No text encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import ssl
import urllib.request
import certifi

REPO = 'Wan-AI/Wan2.1-T2V-1.3B'
REVISION = '37ec512624d61f7aa208f7ea8140a131f93afc9a'
FILES = {
    'diffusion_pytorch_model.safetensors': (5676070424, '96b6b242ca1c2f24e9d02cd6596066fab6d310e2d7538f33ae267cb18d957e8f'),
    'Wan2.1_VAE.pth': (507609880, '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'),
    'config.json': (249, 'ab37994c43740513f94b3ba6233a784035a67b43c8cde83c8f31aa90468c67ce'),
    'LICENSE.txt': (11357, 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'),
}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--include-vae', action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (size, expected) in FILES.items():
        if name == 'Wan2.1_VAE.pth' and not a.include_vae:
            continue
        dest = a.output / name
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        if not dest.exists():
            temp = dest.with_suffix(dest.suffix + '.part')
            print(f'Downloading {name}: {size} bytes', flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Worldline-Wan-Runtime-Probe/1.0'})
            with urllib.request.urlopen(req, context=ssl.create_default_context(cafile=certifi.where()), timeout=90) as src, temp.open('wb') as out:
                count = 0
                for block in iter(lambda: src.read(8 << 20), b''):
                    count += len(block)
                    if count > size:
                        raise RuntimeError('File exceeded pinned size')
                    out.write(block)
            if temp.stat().st_size != size or (expected and sha(temp) != expected):
                raise RuntimeError(f'Integrity failure: {name}')
            temp.replace(dest)
        actual = sha(dest)
        if dest.stat().st_size != size or (expected and actual != expected):
            raise RuntimeError(f'Integrity failure: {name}')
        records.append({'file': name, 'bytes': size, 'sha256': actual, 'url': url})
        print(f'Verified {name}: {actual}', flush=True)
    (a.output / 'download-manifest.json').write_text(json.dumps({'repository': REPO, 'revision': REVISION, 'source_dtype': 'float32 transformer', 'files': records}, indent=2) + '\n')

if __name__ == '__main__':
    main()
