# SPDX-License-Identifier: Apache-2.0
"""Verify compact diagnostic records and existing shared sources, without models."""
import hashlib,json
from pathlib import Path

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    root=Path(__file__).resolve().parent
    publication=json.loads((root/'publication.json').read_text());files={}
    for p in sorted(root.rglob('*')):
        if '__pycache__' in p.parts:continue
        assert not p.is_symlink(),p
        if p.is_file() and p.name!='publication.json':
            files[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    assert files==publication['files'] and len(files)==publication['file_count']
    assert sum(v['bytes'] for v in files.values())==publication['file_bytes']
    provenance=json.loads((root/'code-provenance.json').read_text())
    for name,row in provenance['copied_files'].items():
        assert files[name]=={k:row[k] for k in ('bytes','sha256')}
        assert row['transformation']=='none' and row['source_sha256']==row['sha256']
    for row in provenance['repository_source_references'].values():
        assert sha(root/row['path'])==row['sha256']
    for name,digest in provenance['executed_local_sources'].items():assert sha(root/'source'/name)==digest
    checkpoints=json.loads((root/'checkpoint-identities.json').read_text())
    cp0=checkpoints['checkpoint_0'];idx=json.loads((root/cp0['public_recovery_index']).read_text())
    assert idx['files'][cp0['raw_path']]=={k:cp0[k] for k in ('bytes','sha256')}
    cp16=checkpoints['checkpoint_16'];assert sha(root/cp16['path'])==cp16['sha256']
    assert sha(root/cp16['manifest_path'])==cp16['manifest_sha256']
    audit=json.loads((root/'audit/report.json').read_text());summary=json.loads((root/'summary.json').read_text())
    assert audit['status']=='passed' and audit['predictions_verified']==summary['predictions']==14
    assert audit['original_foundation_records_verified']==825
    assert all(row['exact_equal'] for row in audit['zero_checkpoint_native_parity'].values())
    assert summary['optimizer_updates']==summary['generated_images']==0
    for name,digest in summary['evidence_sha256'].items():assert files[name]['sha256']==digest
    raw=json.loads((root/'recovery/diagnostic-file-index.json').read_text())
    assert len(raw['files'])==raw['file_count']==136
    assert sum(row['bytes'] for row in raw['files'].values())==raw['file_bytes']==107319439
    status=json.loads((root/'release-status.json').read_text())
    reference=json.loads((root/'combined-release.json').read_text())
    assert status['full_raw_status']=='public_and_download_verified' and status['raw_release_url']==reference['release_url']
    assert status['downloader_available_for_this_diagnostic'] is True
    for key in ('metadata','index','verification','downloader'):assert sha(root/reference[key+'_path'])==reference[key+'_sha256']
    public=json.loads((root/reference['index_path']).read_text());verified=json.loads((root/reference['verification_path']).read_text())
    assert verified['status']=='verified' and verified['transport']=='public-https' and verified['index_sha256']==reference['index_sha256']
    assert all(public['files'][raw['source_directory']+'/'+name]==row for name,row in raw['files'].items())
    print(json.dumps({'status':'passed','files':len(files),'bytes':publication['file_bytes'],
        'shared_sources_verified':len(provenance['repository_source_references']),
        'model_execution':False,'network_access':False,'full_raw_publication':'public_and_download_verified'}))

if __name__=='__main__':main()
