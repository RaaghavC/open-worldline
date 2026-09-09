"""Verify the committed text payload and its derived seven-edge summary, without models."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(root, name):
    rel = PurePosixPath(name)
    if rel.is_absolute() or any(x in ('', '.', '..') for x in name.split('/')):
        raise ValueError('Invalid inventory path')
    path = root.joinpath(*rel.parts)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError('Missing or nonregular inventory member: '+name)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, help='Also check the 66 recorded producer sources in this repository')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    publication = json.loads((root/'publication.json').read_text())
    expected = set(publication['files']) | set(publication['supplemental_files'])
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Unexpected or missing files in compact result directory')
    for name, row in publication['files'].items():
        path = checked(root, name)
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Payload hash or size mismatch: '+name)
    summary = json.loads((root/'numerical-summary.json').read_text())
    worker_path = root/'records/training/result/metrics.json'
    worker = json.loads(worker_path.read_text())
    if (summary['source_metrics_sha256'] != sha(worker_path) or
        summary['source_audit_sha256'] != sha(root/'records/audit.json') or
        summary['completed_updates'] != worker['completed_updates'] or
        summary['final_checkpoint_sha256'] != json.loads((root/'records/audit.json').read_text())['identity']['final_checkpoint_sha256']):
        raise ValueError('Summary identity mismatch')
    scores = worker['final_evaluation']['scores']
    if len(scores) != 28 or len(summary['edges']) != 7:
        raise ValueError('Seven edges across four noises required')
    if any(r['normalized_mse'] != summary['initial_normalized_mse_all'] or
           r['prediction_rms'] != summary['initial_prediction_rms_all']
           for r in worker['initial_evaluation']['scores']):
        raise ValueError('Initial summary mismatch')
    for edge in summary['edges']:
        rows = [r for r in scores if r['edge'] == edge['edge']]
        if len(rows) != 4 or [r['noise'] for r in rows] != edge['noise_ids']:
            raise ValueError('Evaluation noise mismatch')
        for key, limits in edge['ranges'].items():
            if limits != [min(r[key] for r in rows), max(r[key] for r in rows)]:
                raise ValueError('Derived range mismatch: '+key)
    if args.repository:
        for row in publication['recorded_producer_sources_in_repository'].values():
            if sha(checked(args.repository.resolve(), row['repository_path'])) != row['sha256']:
                raise ValueError('Recorded repository source mismatch: '+row['repository_path'])
    print(json.dumps({'status':'passed', 'copied_files':len(publication['files']),
                      'summary_edges':7, 'repository_sources_checked':66 if args.repository else 0,
                      'meaning':'Committed text and scalar ranges verified. No raw numerical audit, model execution or quality assessment.'}, indent=2))


if __name__ == '__main__':
    main()
