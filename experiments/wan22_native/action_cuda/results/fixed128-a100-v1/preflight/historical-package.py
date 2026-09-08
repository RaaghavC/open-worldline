"""Local bounded assembly; --extract verifies all files before fresh extraction."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent
W=HERE.parent
MANIFEST='fixed128-transfer-manifest.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()

def inspect(archive,manifest_sha,output=None):
    if output is not None:
        output=Path(output).absolute()
        if output.exists() or any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Fresh regular extraction directory required')
    with tarfile.open(archive,'r:gz') as t:
        members=t.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive name')
        for m in members:
            p=PurePosixPath(m.name)
            if not m.isfile() or m.size>=100_000_000 or p.is_absolute() or '..' in p.parts or str(p)!=m.name:raise ValueError('Unsafe archive entry')
        data=t.extractfile(MANIFEST).read()
        if hashlib.sha256(data).hexdigest()!=manifest_sha:raise ValueError('Manifest hash differs')
        manifest=json.loads(data)
        if set(names)!=set(manifest['files'])|{MANIFEST}:raise ValueError('Exact archive coverage differs')
        for m in members:
            if m.name==MANIFEST:continue
            record=manifest['files'][m.name]
            h=hashlib.sha256();n=0
            with t.extractfile(m) as f:
                for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk);n+=len(chunk)
            if record!={'type':'file','bytes':n,'sha256':h.hexdigest()}:raise ValueError('Saved file differs: '+m.name)
        if manifest['file_count']!=len(manifest['files']) or manifest['file_bytes']!=sum(r['bytes'] for r in manifest['files'].values()):raise ValueError('Manifest counts differ')
        if output is not None:
            output.mkdir(parents=True)
            for m in members:
                target=output/m.name;target.parent.mkdir(parents=True,exist_ok=True)
                with t.extractfile(m) as source,target.open('xb') as dest:shutil.copyfileobj(source,dest,2**20)
            for name,row in manifest['files'].items():
                if sha(output/name)!=row['sha256']:raise ValueError('Extracted file differs')
        return manifest

def build():
    prepared=W/'wan22-action-cuda-fixed128-prepared-v1'
    if json.loads((prepared/'metrics.json').read_text())['status']!='prepared':raise ValueError('Passed actual preparation required')
    if json.loads((HERE/'prepared-read-v1.json').read_text())['status']!='passed':raise ValueError('Passed actual reader required')
    paths={}
    def tree(root,prefix):
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                if p.is_symlink():raise ValueError('No links')
                paths[prefix+'/'+str(p.relative_to(root))]=p
    for name in ('prototype.py','runner.py','extension.py','test_prototype.py','test_runner.py','reference-identities.json','cpu-v1.json','cpu-output-v1.txt','cpu-v2.json','cpu-output-v2.txt','prepared-read-v1.json','README.md','storage-estimate.json','package.py'):
        paths['fixed128-program/'+name]=HERE/name
    for folder in ('reference_fixed16','reference_diagnostic'):tree(HERE/folder,'fixed128-program/'+folder)
    tree(prepared,'action-results/fixed128-spatial-v1')
    previous=W/'wan22-action-cuda-recovered-final-v1/recovered/action-results'
    for folder in ('cache-spatial-run-v1','probe-spatial-v1'):tree(previous/folder,'action-results/'+folder)
    repo=W.parent/'outputs/open-worldline';text=repo/'experiments/wan_adapter/text_cache/native-results'
    for name in ('manifest.json','embeddings.safetensors','LICENSE-APACHE-2.0.txt','NOTICE'):paths['fixed128-inputs/text/'+name]=text/name
    for name in ('LICENSE','NOTICE'):paths['fixed128-program/licenses/'+name]=repo/name
    review=W/'wan22-action-cuda-fixed128-independent-review-v1'
    tree(review,'fixed128-program/independent-review')
    rows={n:{'type':'file','bytes':p.stat().st_size,'sha256':sha(p)} for n,p in sorted(paths.items())}
    manifest={'schema':'worldline-fixed128-transfer-v1','file_count':len(rows),'file_bytes':sum(r['bytes'] for r in rows.values()),'files':rows}
    mp=HERE/'transfer-manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    archive=W/'wan22-action-cuda-fixed128-transfer-v1.tar.gz'
    with tarfile.open(archive,'x:gz',compresslevel=1) as t:
        for name,p in sorted(paths.items()):
            if p.stat().st_size>=100_000_000:raise ValueError('Oversize member')
            t.add(p,arcname=name,recursive=False)
        t.add(mp,arcname=MANIFEST,recursive=False)
    verified=inspect(archive,sha(mp))
    result={'status':'built_and_verified','archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'manifest_sha256':sha(mp),'file_count':len(rows),'file_bytes':manifest['file_bytes'],'maximum_file_bytes':max(r['bytes'] for r in rows.values()),'plan_sha256':sha(prepared/'plan.json'),'model_execution':False}
    (HERE/'transfer.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extract',type=Path);p.add_argument('--archive-sha256');p.add_argument('--manifest-sha256');p.add_argument('--output',type=Path);a=p.parse_args()
    if a.extract:
        if not a.archive_sha256 or not a.manifest_sha256 or not a.output or sha(a.extract)!=a.archive_sha256:raise ValueError('Explicit exact archive/manifest hashes and fresh output required')
        m=inspect(a.extract,a.manifest_sha256,a.output);print(json.dumps({'status':'extracted_and_verified','file_count':m['file_count']}))
    else:build()
