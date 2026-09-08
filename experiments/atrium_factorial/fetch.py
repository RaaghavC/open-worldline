# SPDX-License-Identifier: Apache-2.0
"""Fetch and verify the original public camera/door dataset, without credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import ssl
import tempfile
import time
import urllib.request
import zipfile

import certifi

BASE = 'https://github.com/RaaghavC/open-worldline/releases/download/atrium-camera-door-factorial-v1/'
INDEX = Path(__file__).resolve().parent/'results/dataset-index.json'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_extracted(archive, extracted):
    if extracted.is_symlink() or not extracted.is_dir():
        raise ValueError('Regular capture directory required')
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            path = extracted/member.filename
            if path.is_symlink() or not path.is_file() or path.stat().st_size != member.file_size:
                raise ValueError('Extracted member differs: '+member.filename)
            with z.open(member) as source:
                expected = hashlib.file_digest(source, 'sha256').hexdigest()
            if sha(path) != expected:
                raise ValueError('Extracted member bytes differ: '+member.filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True,exist_ok=True)
    if root.is_symlink():
        raise ValueError('Output must not be a symbolic link')
    if sha(INDEX) != 'ca29b3f1ac8f68d289b1d9fb54d3a4b15cf8d5d80dd41b917a4aa79dfcb8821a':
        raise ValueError('Bundled public part index differs')
    index = json.loads(INDEX.read_text())
    if index['archive_sha256'] != 'a41caa2e287bb7cb644a37d9912cde7aed84f49891e7ed692d9332b771358019':
        raise ValueError('Bundled public archive identity differs')
    context = ssl.create_default_context(cafile=certifi.where())
    for row in index['parts']:
        name = row['name']
        if Path(name).name != name or not name.startswith('atrium-camera-door-factorial-v1.zip.part-'):
            raise ValueError('Unexpected part filename')
        target = root/name
        if target.exists():
            if target.is_symlink() or target.stat().st_size != row['bytes'] or sha(target) != row['sha256']:
                raise ValueError('Existing download differs: '+name)
            continue
        partial = root/(name+'.partial')
        if partial.exists():
            raise ValueError('Prior partial file retained; inspect or remove before retry: '+str(partial))
        for attempt in range(3):
            try:
                with urllib.request.urlopen(BASE+name,context=context,timeout=30) as response, partial.open('xb') as out:
                    count=0
                    while chunk:=response.read(1024*1024):
                        count+=len(chunk)
                        if count>row['bytes']:raise ValueError('Oversized part')
                        out.write(chunk)
                if partial.stat().st_size != row['bytes'] or sha(partial) != row['sha256']:
                    raise ValueError('Downloaded part identity differs')
                partial.rename(target)
                break
            except Exception:
                if partial.exists():partial.unlink()
                if attempt==2:raise
                time.sleep(1)
        print(json.dumps({'verified':name,'bytes':row['bytes']}),flush=True)
    archive = root/index['archive']
    if archive.exists():
        if archive.is_symlink() or sha(archive)!=index['archive_sha256']:
            raise ValueError('Existing archive differs')
    else:
        # The final name is only visible after complete verification. A
        # interrupted assembly leaves the verified input parts reusable.
        with tempfile.TemporaryDirectory(prefix='assemble-',dir=root) as staging:
            candidate = Path(staging)/index['archive']
            with candidate.open('xb') as out:
                for row in index['parts']:
                    with (root/row['name']).open('rb') as source:
                        while chunk:=source.read(1024*1024):out.write(chunk)
            if candidate.stat().st_size!=index['archive_bytes'] or sha(candidate)!=index['archive_sha256']:
                raise ValueError('Assembled archive identity differs')
            candidate.rename(archive)
    if archive.stat().st_size!=index['archive_bytes'] or sha(archive)!=index['archive_sha256']:
        raise ValueError('Complete archive identity differs')
    extracted=root/'capture'
    if extracted.exists():
        verify_extracted(archive, extracted)
    else:
        with tempfile.TemporaryDirectory(prefix='extract-',dir=root) as staging:
            candidate = Path(staging)/'capture'
            with zipfile.ZipFile(archive) as z:
                names=z.namelist()
                if len(set(names))!=len(names) or sum(i.file_size for i in z.infolist())>400_000_000:
                    raise ValueError('Archive inventory or expanded size differs')
                for name in names:
                    p=Path(name)
                    if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive member')
                z.extractall(candidate)
            verify_extracted(archive, candidate)
            candidate.rename(extracted)
    if sha(extracted/'manifest.json')!=index['capture_manifest_sha256']:
        raise ValueError('Extracted manifest identity differs')
    report={'status':'download_verified','archive_sha256':index['archive_sha256'],'parts':len(index['parts']),
        'archive_bytes':index['archive_bytes'],'capture_manifest_sha256':index['capture_manifest_sha256'],
        'model_execution':False,'credentials_required':False}
    (root/'download-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
