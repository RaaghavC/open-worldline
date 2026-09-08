# SPDX-License-Identifier: Apache-2.0
"""Verify the small action-CUDA publication without network or model access."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'publication.json').read_text())
    assert manifest['schema'] == 'worldline-action-cuda-small-publication-v1'
    actual = {}
    for path in root.rglob('*'):
        if '__pycache__' in path.parts:
            continue
        assert not path.is_symlink(), path
        if path.is_file() and path.name != 'publication.json':
            with path.open('rb') as source:
                checksum = hashlib.file_digest(source, 'sha256').hexdigest()
            relative = str(path.relative_to(root))
            actual[relative] = {'bytes': path.stat().st_size, 'sha256': checksum}
            assert actual[relative]['bytes'] < 100_000_000, relative
    assert actual == manifest['files'], 'Unexpected, missing or changed payload files'
    assert manifest['file_count'] == len(actual)
    assert manifest['file_bytes'] == sum(row['bytes'] for row in actual.values())
    provenance = json.loads((root / 'code-provenance.json').read_text())
    for name, record in provenance['copied_files'].items():
        assert actual[name] == {key: record[key] for key in ['bytes', 'sha256']}
        assert record['sha256'] == record['source_sha256'] and record['transformation'] == 'none'
    index = json.loads((root / 'recovery/index.json').read_text())
    assert index['schema'] == 'worldline-action-recovery-v1'
    assert len(index['files']) == 283 and len(index['parts']) == 1
    metadata = json.loads((root / 'release-download.json').read_text())
    assert actual['recovery/index.json']['sha256'] == metadata['index']['sha256']
    assert index['stream_sha256'] == metadata['stream_sha256']
    for name in ['cache', 'probe']:
        audit = json.loads((root / f'audits/{name}/report.json').read_text())
        assert audit['status'] == 'passed'
        assert actual[f'audits/{name}/audit.py']['sha256'] == audit['audit_source_sha256']
    print(json.dumps({'status': 'passed', 'files': len(actual),
                      'bytes': sum(row['bytes'] for row in actual.values()),
                      'raw_parts_in_this_payload': False, 'model_execution': False,
                      'network_access': False}))


if __name__ == '__main__':
    main()
