#!/usr/bin/env python3
"""Check a saved probe publication and its declared JSON path relocation only."""
import argparse,copy,hashlib,json
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--published',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 pub=read(a.published/'publication.json');audit=read(a.audit);assert audit['status']=='passed'
 raw={str(f.relative_to(a.raw)):f for f in a.raw.rglob('*')if f.is_file()};assert set(raw)==set(pub['raw_run_files']) and len(raw)==105
 public={str(f.relative_to(a.published)):f for f in a.published.rglob('*')if f.is_file() and f.name!='publication.json'}
 assert set(public)==set(pub['files_sha256']) and len(public)==110
 for name,digest in pub['files_sha256'].items():assert sha(public[name])==digest,name
 for name,row in pub['raw_run_files'].items():
  assert raw[name].stat().st_size==row['bytes'] and sha(raw[name])==row['sha256'],name
  if name!='launch.json':assert sha(a.published/name)==row['sha256'],name
 launch=read(a.raw/'launch.json');admission=read(a.published/'admission/executed.json')
 assert sha(a.published/'admission/executed.json')==pub['raw_admission_sha256']==sha(Path(launch['admission']))
 assert sha(a.published/'admission/portable.json')==pub['portable_admission_sha256']
 transformed={'launch.json':copy.deepcopy(launch),'admission/portable.json':copy.deepcopy(admission)}
 paths=set()
 for row in pub['path_transformations']:
  assert row['file'] in transformed
  identity=(row['file'],row['json_pointer']);assert identity not in paths;paths.add(identity)
  parts=row['json_pointer'].lstrip('/').split('/');obj=transformed[row['file']]
  for part in parts[:-1]:obj=obj[int(part)] if isinstance(obj,list)else obj[part]
  key=int(parts[-1])if isinstance(obj,list)else parts[-1]
  assert obj[key]==row['original'];obj[key]=row['published']
 assert len([x for x in paths if x[0]=='launch.json'])==10 and len(paths)==13
 for name,value in transformed.items():assert value==read(a.published/name),name
 for row in transformed['admission/portable.json']['foundation_evidence']:
  path=(a.published/'admission'/row['file']).resolve();assert path.is_file()and sha(path)==row['sha256']
 for field in ('cpu_report','independent_report','cache','roundtrip_run','text_cache','admission','run','output'):
  assert (a.published/transformed['launch.json'][field]).exists(),field
 report={'schema':'worldline-action-probe-publication-independent-v2','status':'passed','audit_source_sha256':sha(__file__),
  'initial_audit_count_correction':'Preserved attempt1 mistakenly counted the manifest among the hashed files; 110 hashed files plus one manifest is 111 total.',
  'numerical_audit_sha256':sha(a.audit),'publication_sha256':sha(a.published/'publication.json'),
  'raw_files':105,'raw_bytes':sum(v['bytes']for v in pub['raw_run_files'].values()),'raw_unchanged':True,
  'public_hashed_files_excluding_manifest':110,'public_files_including_manifest':111,'public_bytes_excluding_manifest':sum(v.stat().st_size for v in public.values()),
  'raw_files_byte_exact_in_public':104,'launch_only_operational_paths_changed':10,'separate_admission_only_evidence_paths_changed':3,
  'all_other_json_content_equal':True,'all_public_file_hashes_match':True,'relative_evidence_and_available_launch_paths_resolve':True,
  'raw_admission_copy_exact':True,'model_inference':False,'gpu_used':False,'measured_output_changed':False}
 a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
