"""Verify the compact result files, nested archives and release transport bindings."""
import argparse, hashlib, json, re, tarfile
from pathlib import Path, PurePosixPath

HEX = re.compile(r'[0-9a-f]{64}')
KINDS = ('training', 'assessment', 'visual')

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def relative(name):
    need(type(name) is str and name and not name.startswith('/'), 'Relative name required')
    p=PurePosixPath(name)
    need(str(p)==name and '..' not in p.parts and '\\' not in name and '\x00' not in name, 'Unsafe member path')
    return p

def js(path):
    path=Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<16_000_000, 'Bounded regular JSON required')
    return json.loads(path.read_text())

def check_record(path, row):
    need(type(row) is dict and set(row)=={'bytes','sha256'}, 'Exact file-record fields required')
    need(type(row['bytes']) is int and 0<=row['bytes']<100_000_000 and type(row['sha256']) is str and HEX.fullmatch(row['sha256']), 'Invalid file size/hash')
    path=Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], 'File bytes differ: '+path.name)

def file_map(root, rows):
    need(type(rows) is dict and rows, 'Nonempty file inventory required')
    for name,row in rows.items():
        relative(name); path=root/name
        need(not any(p.is_symlink() for p in (path,*path.parents)), 'Symlink forbidden')
        check_record(path,row)

def check_transport(metadata,index):
    need(metadata.get('schema')=='worldline-action-release-download-v1', 'Download schema differs')
    parts=metadata['parts'];need(type(parts) is list and 0<len(parts)<=1000, 'Part count differs')
    for i,p in enumerate(parts):
        need(set(p)=={'name','bytes','sha256','url'} and p['name']==f'evidence.tar.gz.part-{i:03d}', 'Part order/name differs')
        need(type(p['bytes']) is int and 0<p['bytes']<=16_000_000 and HEX.fullmatch(p['sha256']), 'Part bounds/hash differ')
    expected=[{k:r[k] for k in ('name','bytes','sha256')} for r in parts]
    need(index.get('parts')==expected, 'Index and download part layouts differ')
    need(index.get('stream_sha256')==metadata['stream_sha256'] and HEX.fullmatch(metadata['stream_sha256']), 'Stream hash differs')
    need(index.get('transport_bytes')==sum(r['bytes'] for r in expected), 'Transport size differs')
    need(index.get('file_bytes')==sum(r['bytes'] for r in index['files'].values()), 'Recovered member bytes differ')

def check_archive(root,name,row):
    relative(name);path=root/name;check_record(path,{k:row[k] for k in ('bytes','sha256')})
    expected=row['files'];seen=set()
    with tarfile.open(path,'r:gz') as archive:
        for member in archive:
            relative(member.name)
            need(member.isfile() and member.name not in seen and member.name in expected, 'Archive member is linked, repeated or unindexed')
            record=expected[member.name];need(member.size==record['bytes'] and 0<=member.size<100_000_000, 'Archive member size differs')
            with archive.extractfile(member) as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
            need(checksum==record['sha256'], 'Archive member checksum differs')
            seen.add(member.name)
    need(seen==set(expected), 'Archive members missing')
    return len(seen)

def verify(root):
    root=Path(root).absolute();manifest=js(root/'integration-manifest.json')
    need(manifest['schema']=='worldline-action-effect-results-integration-v1', 'Integration schema differs')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    need(actual==set(manifest['files'])|{'integration-manifest.json'}, 'Unexpected or missing compact file')
    file_map(root,manifest['files']);reports={}
    for kind in KINDS:
        folder=root/(kind+'-a100-v1');record=manifest['results'][kind]
        need(sha(folder/'payload-manifest.json')==record['original_payload_manifest_sha256'], 'Original payload manifest changed')
        payload=js(folder/'payload-manifest.json');file_map(folder,payload['files'])
        need(payload['file_count']==len(payload['files']) and payload['file_bytes']==sum(r['bytes'] for r in payload['files'].values()), 'Original compact counts differ')
        sources=js(folder/'reproducibility/source-and-inputs.json')
        archives=sources.get('archives')
        if archives is None:
            row=sources['executed_source_archive'];archives={'reproducibility/'+row['file']:row}
        counts={name:check_archive(folder,name,row) for name,row in archives.items()}
        metadata=js(folder/'release-download.json');index=js(folder/'raw-recovery-public/index.json')
        check_record(folder/'raw-recovery-public/index.json',{k:metadata['index'][k] for k in ('bytes','sha256')})
        check_transport(metadata,index)
        tag='wan22-action-effect-'+kind+'-a100-v1'
        base='https://github.com/RaaghavC/open-worldline/releases/download/'+tag+'/'
        need(metadata['index']['url']==base+'index.json' and all(r['url']==base+r['name'] for r in metadata['parts']), 'Release URL differs')
        change=js(folder/'raw-recovery-public/transport-change.json')
        need(change['final_metadata_sha256']==sha(folder/'release-download.json') and change['final_index_sha256']==sha(folder/'raw-recovery-public/index.json'), 'Transport change does not bind current metadata/index')
        reports[kind]={'original_compact_files':len(payload['files'])+1,'archive_members_verified':counts,
            'raw_release_parts':len(metadata['parts']),'raw_files':len(index['files']),
            'payload_manifest_sha256':sha(folder/'payload-manifest.json'),'download_metadata_sha256':sha(folder/'release-download.json')}
    return {'schema':'worldline-action-effect-results-verification-v1','status':'passed',
        'integration_manifest_sha256':sha(root/'integration-manifest.json'),
        'files_verified':len(manifest['files'])+1,'results':reports,
        'original_indexed_payload_files_changed':False,'network_or_model_calls':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();print(json.dumps(verify(args.root),indent=2))
