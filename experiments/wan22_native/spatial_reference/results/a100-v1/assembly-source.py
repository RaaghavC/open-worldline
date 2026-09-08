"""Copy the small verified spatial publication selection; never invoke a model or network."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

RUNTIME = 'ffee84129b9c29a7d3c29449e10b6a18a660aebf'
AUDITORS = '1142cbbd409285cd69fbc7c637a4153ad5eb91bc'
INDEX_SHA = '834053b3fd4b0428740bdc78853225dde07ee269bdb6e43bc2b88916f0f6690d'
STREAM_SHA = 'e900c18faab7de9747c2369ce5579e3ffd6352731d5ba4b3f9fa6139b5635136'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.workspace.resolve(); output = args.output.resolve()
    assert not args.output.exists() and not args.output.is_symlink()
    assert output.is_relative_to(base / 'work/wan22-spatial-publication-v1')
    repo = base / 'outputs/open-worldline'
    raw = base / 'work/wan22-spatial-recovered-final-v1'
    assert sha(raw / 'index.json') == INDEX_SHA
    index = json.loads((raw / 'index.json').read_text())
    verified = json.loads((raw / 'recovery-verified.json').read_text())
    assert verified['status'] == 'verified' and len(index['files']) == verified['files'] == 536
    assert index['file_bytes'] == verified['file_bytes'] == 803_541_766
    assert index['transport_bytes'] == verified['transport_bytes'] == 636_908_966
    assert index['stream_sha256'] == verified['stream_sha256'] == STREAM_SHA
    assert len(index['parts']) == 8 and all(0 < p['bytes'] <= 90_000_000 for p in index['parts'])
    assert verified['index_sha256'] == INDEX_SHA
    output.mkdir(parents=True)
    copies = {}

    def copy(source, relative, expected=None):
        source = Path(source); target = output / relative
        assert source.is_file() and not source.is_symlink() and source.stat().st_size < 100_000_000
        checksum = sha(source)
        if expected is not None:
            assert checksum == expected['sha256'] and source.stat().st_size == expected['bytes']
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(source, target)
        assert sha(target) == checksum == sha(source)
        copies[relative] = {'original_workspace_relative': str(source.relative_to(base)),
                            'bytes': target.stat().st_size, 'sha256': checksum, 'transformation': 'none'}

    def raw_copy(name, target):
        copy(raw / 'recovered' / name, target, index['files'][name])

    stages = {'codec': ('codec-both', ['codec']),
              'baseline-pair': ('pair-baseline', ['core']),
              'spatial-pair': ('pair-spatial', ['core']),
              'clip': ('clip-spatial', ['core', 'decode'])}
    runtime_sources = None
    for label, (prefix, workers) in stages.items():
        run = f'spatial-results/{prefix}-run-v1'
        parent = json.loads((raw / 'recovered' / run / 'metrics.json').read_text())
        assert parent['status'] == 'passed'
        if runtime_sources is None:
            runtime_sources = parent['source_sha256']
        assert runtime_sources == parent['source_sha256']
        raw_copy(run + '/metrics.json', f'metrics/{label}/parent.json')
        raw_copy(f'spatial-results/{prefix}-plan-v1/metrics.json', f'plans/{label}.json')
        for worker in workers:
            child = json.loads((raw / 'recovered' / run / worker / 'result/metrics.json').read_text())
            assert child['status'] == 'passed'
            for source_tail, dest_tail in [('result/metrics.json', 'metrics.json'),
                                           ('terminal.json', 'terminal.json'),
                                           ('result/monitor-terminal.json', 'monitor-terminal.json')]:
                raw_copy(f'{run}/{worker}/{source_tail}', f'metrics/{label}/{worker}/{dest_tail}')

    audits = {}
    for label in stages:
        directory = base / f'work/wan22-spatial-actual-{label}-audit-v1'
        audit = json.loads((directory / 'report.json').read_text())
        assert audit['status'] == 'passed' and audit['sources_unchanged'] and audit['inputs_unchanged']
        assert not audit['new_model_execution'] and not audit['provider_access'] and not audit['quality_assessed']
        copy(directory / 'report.json', f'audits/{label}/report.json')
        copy(directory / 'input-inventory.json', f'audits/{label}/input-inventory.json')
        assert sha(directory / 'input-inventory.json') == audit['input_inventory_sha256']
        audits[label] = audit
    audit_sources = audits['codec']['source_sha256']
    assert len(runtime_sources) == 29 and len(audit_sources) == 34
    assert all(a['source_sha256'] == audit_sources for a in audits.values())
    assert all(audit_sources.get(name) == checksum for name, checksum in runtime_sources.items())
    for name, checksum in audit_sources.items():
        for label in stages:
            assert sha(base / f'work/wan22-spatial-actual-{label}-audit-v1/source' / name) == checksum
        copy(base / 'work/wan22-spatial-actual-codec-audit-v1/source' / name, 'source/' + name)
    copy(base / 'work/wan22-spatial-actual-baseline-pair-audit-v1/earlier-a100-guidance-comparison.json',
         'audits/baseline-pair/earlier-a100-guidance-comparison.json')

    for name in ['index.json', 'recovery-verified.json', 'earlier-recovery-preservation.json']:
        copy(raw / name, 'recovery/' + name)
    raw_copy('spatial-results/codec-both-run-v1/cpu-report.json', 'preflight/runtime-cpu-v2.json')
    copy(repo / 'experiments/wan22_native/spatial_audit/reviews/cpu-v1.json', 'preflight/auditor-cpu-v1.json')
    copy(base / 'work/wan22-spatial-transfer-v2/transfer.json', 'preflight/source-transfer-v2.json')

    for profile in ['baseline', 'spatial']:
        raw_copy(f'spatial-results/codec-both-run-v1/codec/result/{profile}/reconstructed-initial.png',
                 f'previews/codec-{profile}-reconstructed-initial.png')
    for name in ['comparison.png', 'preview.gif']:
        raw_copy('spatial-results/clip-spatial-run-v1/decode/result/' + name, 'previews/clip-' + name)
    for source, target in [
        ('LICENSE', 'licenses/PROJECT-APACHE-2.0.txt'),
        ('experiments/wan22_native/LICENSE-APACHE-2.0.txt', 'licenses/WAN-APACHE-2.0.txt'),
        ('experiments/wan22_native/NOTICE', 'licenses/WAN-NOTICE.txt'),
        ('experiments/atrium_data/DATA-LICENSE', 'licenses/ORIGINAL-SCENE-CC0.txt')]:
        copy(repo / source, target)
    for name in ['fetch_artifacts.py', 'test_fetch_artifacts.py', 'DOWNLOADER.md', 'downloader-cpu-review-v1.json']:
        copy(base / 'work/wan22-spatial-publication-v1' / name, name)
    copy(Path(__file__).resolve(), 'assembly-source.py')

    def write(name, value):
        path = output / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + '\n')

    write('code-provenance.json', {
        'schema': 'worldline-spatial-small-payload-provenance-v1',
        'runtime_commit': RUNTIME, 'auditor_commit': AUDITORS,
        'runtime_source_sha256': runtime_sources, 'audit_source_sha256': audit_sources,
        'source_bundle': 'source/', 'original_file_copies': copies,
        'raw_artifacts': {'files': 536, 'bytes': index['file_bytes'], 'parts': index['parts'],
                          'transport_bytes': index['transport_bytes'], 'stream_sha256': STREAM_SHA,
                          'index_sha256': INDEX_SHA, 'copied_into_git_payload': False}})
    write('release-status.json', {
        'schema': 'worldline-spatial-release-status-v1', 'status': 'pending_release_creation_and_upload',
        'release_url': None, 'release_download_metadata': None, 'raw_artifacts_public': False,
        'local_recovery_verified': True, 'planned_part_count': 8,
        'index_sha256': INDEX_SHA, 'stream_sha256': STREAM_SHA,
        'note': 'No URLs are invented. Root will add actual pinned public asset metadata after upload verification.'})
    write('visual-review-status.json', {
        'status': 'pending_root_all_frame_visual_conclusion', 'numerical_audits_passed': True,
        'all_17_original_frames_retained': True,
        'note': 'Finite output and completion are measured. A visual verdict is separate and pending here.'})
    print(json.dumps({'status': 'assembled_pending_prose_and_release', 'byte_exact_copies': len(copies),
                      'copy_bytes': sum(row['bytes'] for row in copies.values()), 'output': str(output)}))


if __name__ == '__main__':
    main()
