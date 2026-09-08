# SPDX-License-Identifier: Apache-2.0
"""Verify compact post128 evidence without models, GPUs or network access."""
import hashlib,json
from pathlib import Path

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    root=Path(__file__).resolve().parent
    files={}
    for p in sorted(root.rglob('*')):
        if '__pycache__' in p.parts:continue
        assert not p.is_symlink(),p
        if p.is_file() and p.name!='publication.json':files[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    publication=read(root/'publication.json')
    assert files==publication['files'] and len(files)==publication['file_count']
    assert sum(r['bytes'] for r in files.values())==publication['file_bytes']
    provenance=read(root/'code-provenance.json')
    for name,row in provenance['copied_files'].items():
        assert files[name]=={k:row[k] for k in ('bytes','sha256')}
        assert row['transformation']=='none' and row['source_sha256']==row['sha256']
    for name,row in provenance['repository_source_references'].items():assert sha(root/row['path'])==row['sha256'],name
    for name,row in provenance['audit_analysis_source_references'].items():assert sha(root/row['path'])==row['sha256'],name
    for name,digest in provenance['visual_local_sources'].items():assert sha(root/'source/visual'/name)==digest
    visual=read(root/'audit/visual/report.json');comparison=read(root/'audit/comparison20-v2/report.json');review=read(root/'visual-review/visual-review.json')
    assert visual['status']==comparison['status']=='passed'
    assert comparison['predictions_verified']==20 and comparison['original_foundation_records_verified']==825
    assert comparison['zero_native_bit_exact_comparisons']==4
    repeated=read(root/'audit/comparison20-v2/original14-reproduction.json')
    assert repeated['status']=='passed' and repeated['repeated_original14_prediction_files_byte_exact']==14
    assert repeated['comparison20_audit_sha256']==sha(root/'audit/comparison20-v2/report.json')
    assert review['frame_count']==34 and not review['door_interaction_success'] and not review['camera_turn_success']
    assert review['structural_audit_sha256']==sha(root/'audit/visual/report.json')
    assert review['rgb_comparison_sha256']==sha(root/'rgb-comparison/report.json')
    assert sha(root/'visual-review/source-frame-hashes.json')==review['source_frame_hashes_sha256']
    for arm,digest in review['sheets'].items():assert sha(root/'visual-review'/f'{arm}-all17.png')==digest
    correction=read(root/'audit/reader-v2/correction.json')
    assert sha(root/'audit/reader-v1/audit.py')==correction['old_reader_sha256']
    assert sha(root/'audit/reader-v2/audit.py')==correction['new_reader_sha256']
    assert sha(root/'audit/comparison20-v1/failed.json')==correction['prior_failed_audit_sha256']
    assert sha(root/'audit/comparison20-v2/report.json')==correction['passed_report_sha256']
    assert read(root/'audit/reader-v2-independent/report.json')['status']=='passed'
    assert sha(root/'../fixed128-a100-v1/checkpoint-0128/adapter.safetensors')==comparison['checkpoint128_sha256']
    original=read(root/'recovery/original-index.json');public=read(root/'recovery/index.json');derivation=read(root/'recovery/public-derivation.json');metadata=read(root/'release-download.json')
    assert set(original['files'])==set(public['files']) and len(public['files'])==608
    changes={n for n in public['files'] if original['files'][n]!=public['files'][n]}
    assert changes=={r['path'] for r in derivation['transformed_files']} and len(changes)==3
    assert derivation['unchanged_file_count']==605 and derivation['excluded_file_count']==0
    for row in derivation['transformed_files']:
        assert original['files'][row['path']]==row['original'] and public['files'][row['path']]==row['published']
        assert row['replacement']=='[WORKSPACE]/' and row['replacement_count']==1
    assert sha(root/'recovery/index.json')==metadata['index']['sha256']==derivation['public_index_sha256']
    assert public['stream_sha256']==metadata['stream_sha256']==derivation['public_stream_sha256']
    assert len(public['parts'])==82 and max(r['bytes'] for r in public['parts'])<=16000000
    local=read(root/'recovery/local-verification.json')
    assert local['status']=='verified' and local['files']==608 and local['index_sha256']==metadata['index']['sha256']
    status=read(root/'release-status.json')
    if status['status']=='public_and_download_verified':
        release=read(root/'recovery/published-release.json');remote=read(root/'recovery/remote-asset-verification.json');download=read(root/'recovery/public-download-verification.json')
        assert not release['draft'] and release['id']==status['release_id']==remote['release_id']
        assert remote['status']=='passed' and len(release['assets'])==remote['count']
        assert {a['name']:(a['size'],a['digest']) for a in release['assets']}=={a['name']:(a['size'],a['digest']) for a in remote['assets']}
        assert download['status']=='verified' and download['transport']=='public-https' and download['files']==608
        assert download['index_sha256']==metadata['index']['sha256'] and download['metadata_sha256']==sha(root/'release-download.json')
        assert download['stream_sha256']==public['stream_sha256'] and download['file_bytes']==public['file_bytes']
    else:assert status['status'] in ('prepared_local_verified','draft_upload_in_progress') and not status['raw_public_download_verified']
    print(json.dumps({'status':'passed','files':len(files),'file_bytes':publication['file_bytes'],'shared_sources':len(provenance['repository_source_references']),'exact_raw_members':605,'disclosed_audit_path_derivatives':3,'public_status':status['status'],'model_execution':False,'network_access':False}))
if __name__=='__main__':main()
