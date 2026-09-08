# SPDX-License-Identifier: Apache-2.0
"""Verify fixed16 payload bytes and shared sources without network/model access."""
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'publication.json').read_text())
    assert manifest['schema'] == 'worldline-fixed16-cuda-small-publication-v1'
    actual = {}
    for path in root.rglob('*'):
        if '__pycache__' in path.parts:
            continue
        assert not path.is_symlink(), path
        if path.is_file() and path.name != 'publication.json':
            actual[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    assert actual == manifest['files'], 'Missing, changed or unexpected payload file'
    assert len(actual) == manifest['file_count']
    assert sum(v['bytes'] for v in actual.values()) == manifest['file_bytes']
    provenance = json.loads((root / 'code-provenance.json').read_text())
    for name, record in provenance['copied_files'].items():
        assert actual[name] == {key: record[key] for key in ['bytes', 'sha256']}
        if record['transformation'] == 'none':
            assert record['sha256'] == record['source_sha256']
        else:
            assert name == 'execution/remaining-time-public.json'
            assert record['transformation']['removed_fields'] == ['pod_id']
            assert 'pod_id' not in json.loads((root / name).read_text())
    for record in provenance['repository_source_references'].values():
        relative = record['path']
        assert relative.startswith(('source/repository/', '../a100-v1/source/'))
        path = root / relative
        assert path.is_file() and not path.is_symlink() and digest(path) == record['sha256']
    audit = json.loads((root / 'audit/report.json').read_text())
    assert audit['status'] == 'passed' and digest(root / 'audit/audit.py') == audit['audit_source_sha256']
    summary = json.loads((root / 'summary.json').read_text())
    assert summary['completed_updates'] == 16 and summary['image_generation'] is False
    for name, expected in summary['evidence_sha256'].items():
        assert actual[name]['sha256'] == expected
    index = json.loads((root / 'recovery/index.json').read_text())
    metadata = json.loads((root / 'release-download.json').read_text())
    assert len(index['files']) == 485 and len(index['parts']) == len(metadata['parts']) == 6
    assert digest(root / 'recovery/index.json') == metadata['index']['sha256']
    assert index['stream_sha256'] == metadata['stream_sha256']
    print(json.dumps({'status': 'passed', 'files': len(actual), 'bytes': manifest['file_bytes'],
                      'shared_repository_sources_verified': len(provenance['repository_source_references']),
                      'model_execution': False, 'network_access': False}))


if __name__ == '__main__':
    main()
