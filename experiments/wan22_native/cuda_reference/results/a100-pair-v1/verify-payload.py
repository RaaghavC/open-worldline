"""Read-only payload identity check. Standard library only; no model or cloud."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    expected=manifest['files']
    actual={str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and p.name!='manifest.json'}
    assert actual==set(expected),'Missing or extra payload files'
    for name,row in expected.items():
        p=HERE/name;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],name
    provenance=json.loads((HERE/'provenance.json').read_text())
    for name,row in provenance['copied_files'].items():
        assert expected[name]['sha256']==row['sha256'] and expected[name]['bytes']==row['bytes'],name
        assert row['transformed'] is False
    assert provenance['path_transformations']==[]
    raw=HERE/'recovered-pair-v1'
    recovery=json.loads((HERE/'recovery/pair-recovery.json').read_text())
    actual_raw={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file()}
    assert actual_raw==set(recovery['files'])
    for name,row in recovery['files'].items():
        p=raw/name;assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes'],name
    print(json.dumps({'status':'passed','indexed_files':len(expected),'indexed_bytes':sum(v['bytes'] for v in expected.values()),
                      'byte_exact_copies':len(provenance['copied_files']),'recovered_files':len(actual_raw),
                      'cloud_calls':0,'model_calls':0}))

if __name__=='__main__':main()
