"""Verify copied clip evidence and the declared planning-only edits."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()

def main():
    manifest=json.loads((HERE/'manifest.json').read_text());expected=manifest['files']
    actual={str(p.relative_to(HERE))for p in HERE.rglob('*')if p.is_file()and str(p.relative_to(HERE))!='manifest.json'}
    assert actual==set(expected),'Missing or extra payload files'
    for name,row in expected.items():
        p=HERE/name;assert p.stat().st_size==row['bytes']and sha(p)==row['sha256'],name
    provenance=json.loads((HERE/'provenance.json').read_text())
    for name,row in provenance['copied_files'].items():
        assert expected[name]['sha256']==row['sha256']and expected[name]['bytes']==row['bytes'],name
        assert row['transformed']is False
    assert provenance['path_transformations']==[]
    raw=HERE/'recovered-clip-v1';recovery=json.loads((HERE/'recovery/clip-recovery.json').read_text())
    assert {str(p.relative_to(raw))for p in raw.rglob('*')if p.is_file()}==set(recovery['files'])
    for name,row in recovery['files'].items():
        p=raw/name;assert sha(p)==row['sha256']and p.stat().st_size==row['bytes'],name
    planning=HERE/'planning';changes=json.loads((planning/'publication-changes.json').read_text())
    original=planning/changes['original_file'];public=planning/changes['public_file']
    assert sha(original)==changes['original_sha256']and sha(public)==changes['public_sha256']
    text=original.read_text()
    for row in changes['changes']:
        assert row['old']in text;text=text.replace(row['old'],row['new'])
    assert text==public.read_text(),'Undeclared planning edits'
    print(json.dumps({'status':'passed','indexed_files':len(expected),'indexed_bytes':sum(v['bytes']for v in expected.values()),
                      'byte_exact_copies':len(provenance['copied_files']),'recovered_files':len(recovery['files']),
                      'planning_changes_exact':True,'cloud_calls':0,'model_calls':0}))

if __name__=='__main__':main()
