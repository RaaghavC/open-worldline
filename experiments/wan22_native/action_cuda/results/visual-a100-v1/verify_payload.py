# SPDX-License-Identifier: Apache-2.0
"""Check the compact visual payload and shared measured sources without models."""
import hashlib,json
from pathlib import Path

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    root=Path(__file__).resolve().parent;publication=json.loads((root/'publication.json').read_text())
    files={}
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts:continue
        assert not path.is_symlink(),path
        if path.is_file() and path.name!='publication.json':
            files[str(path.relative_to(root))]={'bytes':path.stat().st_size,'sha256':sha(path)}
    assert files==publication['files'] and len(files)==publication['file_count']
    assert sum(x['bytes'] for x in files.values())==publication['file_bytes']
    provenance=json.loads((root/'code-provenance.json').read_text())
    for name,row in provenance['copied_files'].items():
        assert files[name]=={k:row[k] for k in ['bytes','sha256']}
        assert row['transformation']=='none' and row['sha256']==row['source_sha256']
    for name,row in provenance['repository_source_references'].items():
        assert (root/row['path']).is_file() and sha(root/row['path'])==row['sha256'],name
    for name,digest in provenance['executed_local_source_sha256'].items():assert sha(root/'source/v2'/name)==digest
    assert sha(root/provenance['checkpoint_source'])==provenance['sampling_checkpoint_sha256']
    summary=json.loads((root/'summary.json').read_text())
    assert summary['numerical_execution']=='passed' and summary['rendered_command_following']=='failed_on_this_layout_and_noise'
    for name,digest in summary['evidence_sha256'].items():assert files[name]['sha256']==digest
    audit=json.loads((root/'audit/report.json').read_text());assert audit['status']=='passed'
    review=json.loads((root/'reviews/parent-visual-review.json').read_text())
    assert all(row==list(range(17)) for row in review['frames_reviewed'].values())
    report=json.loads((root/'analysis/report.json').read_text())
    assert report['contact_sha256']==sha(root/'previews/comparison.png') and len(report['frames'])==17
    for row in report['frames']:
        for arm in ['closed','open']:assert review['frame_sha256'][arm][str(row['frame'])]==row['identities'][arm]['png_sha256']
    index=json.loads((root/'recovery/index.json').read_text());metadata=json.loads((root/'release-download.json').read_text())
    assert len(index['files'])==443 and len(index['parts'])==len(metadata['parts'])==9
    assert sha(root/'recovery/index.json')==metadata['index']['sha256']
    assert index['stream_sha256']==metadata['stream_sha256']
    original=json.loads((root/'history/original-recovery-index.json').read_text())
    assert len(original['parts'])==8 and {k:v for k,v in original.items() if k!='parts'}=={k:v for k,v in index.items() if k!='parts'}
    derivation=json.loads((root/'recovery/transport-derivation.json').read_text())
    assert derivation['replacement_concatenation_sha256']==original['parts'][-1]['sha256']
    assert derivation['replacement_parts']==index['parts'][-2:]
    assert original['parts'][:-1]==index['parts'][:-2]
    print(json.dumps({'status':'passed','files':len(files),'bytes':publication['file_bytes'],
      'repository_sources_verified':len(provenance['repository_source_references']),
      'model_execution':False,'network_access':False}))

if __name__=='__main__':main()
