"""Verify this small publication payload without network or model access."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'publication.json').read_text())
    assert manifest['schema'] == 'worldline-spatial-small-publication-v1'
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
    release = json.loads((root / 'release-status.json').read_text())
    assert manifest['raw_release_urls_pending'] is (not release['raw_artifacts_public'])
    provenance = json.loads((root / 'code-provenance.json').read_text())
    for name, record in provenance['original_file_copies'].items():
        assert actual[name] == {key: record[key] for key in ['bytes', 'sha256']}
        assert record['transformation'] == 'none'
    index = json.loads((root / 'recovery/index.json').read_text())
    assert actual['recovery/index.json']['sha256'] == provenance['raw_artifacts']['index_sha256']
    assert index['stream_sha256'] == provenance['raw_artifacts']['stream_sha256']
    assert len(index['files']) == 536 and len(index['parts']) == 8
    print(json.dumps({'status': 'passed', 'files': len(actual),
                      'bytes': sum(row['bytes'] for row in actual.values()),
                      'raw_parts_in_this_payload': False, 'model_execution': False,
                      'network_access': False}))


if __name__ == '__main__':
    main()
