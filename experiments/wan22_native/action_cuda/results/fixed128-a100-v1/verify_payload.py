# SPDX-License-Identifier: Apache-2.0
"""Check compact fixed128 publication identities without models or network access."""
import hashlib,json
from pathlib import Path

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    root=Path(__file__).resolve().parent;publication=json.loads((root/'publication.json').read_text());files={}
    for p in sorted(root.rglob('*')):
        if '__pycache__' in p.parts:continue
        assert not p.is_symlink(),p
        if p.is_file() and p.name!='publication.json':files[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    assert files==publication['files'] and len(files)==publication['file_count']
    assert sum(v['bytes'] for v in files.values())==publication['file_bytes']
    provenance=json.loads((root/'code-provenance.json').read_text())
    for name,row in provenance['copied_files'].items():
        assert files[name]=={k:row[k] for k in ('bytes','sha256')}
        assert row['transformation']=='none' and row['source_sha256']==row['sha256']
    for name,row in provenance['privacy_transformations'].items():
        assert files[name]=={'bytes':row['published_bytes'],'sha256':row['published_sha256']}
        assert row['replacement']=='[WORKSPACE]/' and row['replacement_count'] in (1,2)
    for name,row in provenance['repository_source_references'].items():assert sha(root/row['path'])==row['sha256'],name
    for group,key in [('local','executed_local_source_sha256'),('reference','executed_reference_source_sha256')]:
        for name,digest in provenance[key].items():assert sha(root/'source'/group/name)==digest
    summary=json.loads((root/'summary.json').read_text());audit=json.loads((root/'audit/report.json').read_text())
    assert summary['numerical_training']=='passed' and summary['completed_updates']==audit['completed_updates']==128
    assert audit['status']=='passed' and audit['recovered_bytes_unchanged'] and audit['foundation_values_unchanged']
    manifest=json.loads((root/'checkpoint-0128/manifest.json').read_text())
    assert manifest['completed_updates']==128 and sha(root/'checkpoint-0128/adapter.safetensors')==manifest['files']['adapter.safetensors']==summary['final_checkpoint_sha256']
    original=json.loads((root/'recovery/original-index.json').read_text());public=json.loads((root/'recovery/index.json').read_text())
    derivation=json.loads((root/'recovery/public-derivation.json').read_text());metadata=json.loads((root/'release-download.json').read_text())
    assert set(original['files'])==set(public['files']) and len(public['files'])==957
    changed=[n for n in public['files'] if public['files'][n]!=original['files'][n]]
    assert changed==[r['path'] for r in derivation['transformed_files']] and len(changed)==1
    assert derivation['unchanged_file_count']==956 and derivation['excluded_file_count']==0
    for row in derivation['transformed_files']:
        assert original['files'][row['path']]==row['original'] and public['files'][row['path']]==row['published']
    assert sha(root/'recovery/index.json')==metadata['index']['sha256']==derivation['public_index_sha256']
    assert public['stream_sha256']==metadata['stream_sha256']==derivation['public_stream_sha256']
    assert public['stream_sha256']!=original['stream_sha256']
    assert len(public['parts'])==100 and max(r['bytes'] for r in public['parts'])<=16000000
    local=json.loads((root/'recovery/local-verification.json').read_text())
    assert local['status']=='verified' and local['files']==957 and local['index_sha256']==metadata['index']['sha256']
    release=json.loads((root/'recovery/published-release.json').read_text());remote=json.loads((root/'recovery/remote-asset-verification.json').read_text())
    downloaded=json.loads((root/'recovery/public-download-verification.json').read_text());status=json.loads((root/'release-status.json').read_text())
    assert not release['draft'] and release['id']==remote['release_id']==status['release_id']==384516693
    assert remote['status']=='passed' and remote['count']==len(release['assets'])==126
    assert {a['name']:(a['size'],a['digest']) for a in release['assets']}=={a['name']:(a['size'],a['digest']) for a in remote['assets']}
    assert downloaded['status']=='verified' and downloaded['transport']=='public-https' and downloaded['files']==957
    assert downloaded['metadata_sha256']==sha(root/'release-download.json') and downloaded['index_sha256']==metadata['index']['sha256']
    assert downloaded['stream_sha256']==public['stream_sha256'] and downloaded['file_bytes']==sum(v['bytes'] for v in public['files'].values())
    assert status['status']=='public_and_download_verified' and status['raw_publication_verified']
    print(json.dumps({'status':'passed','files':len(files),'bytes':publication['file_bytes'],
        'shared_sources_verified':66,'exact_local_reference_sources_verified':16,
        'public_raw_original_files_unchanged':956,'disclosed_raw_log_changes':1,'model_execution':False,'network_access':False}))

if __name__=='__main__':main()
