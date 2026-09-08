# SPDX-License-Identifier: Apache-2.0
"""Paired RGB correspondence for final128, using the unchanged prior reader."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
PRIOR_SOURCE=ROOT/'work/wan22-action-cuda-visual-analysis-prep-v1/analyze.py'
PRIOR_SHA='520c2af654749faf5f4d480cc52c0ac21d8f7aede7257e254e2cf4e9eb00d9bc'
PRIOR_RUN=ROOT/'work/wan22-action-cuda-visual-recovered-final-v1/recovered/action-results/visual-spatial-v2'
PRIOR_REPORT=ROOT/'work/wan22-action-cuda-visual-comparison-v1/report.json'
PRIOR_REPORT_SHA='6693ba5434dd07dd9016ca660613ce19e5ce833512ce4a8d0c8dc839550faae3'
PREPARED=ROOT/'work/wan22-action-cuda-final128-visual-prepared-v1'
PLAN_SHA='6a00ce5db2b3858dd23a616beadbf774fb8b157efbc26edd8a969bde501bed3b'
CHECKPOINT='ac1160b22531804c473746d2fb8ee9edd75710e0e43d6f1eabc86d268c864996'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
need(sha(PRIOR_SOURCE)==PRIOR_SHA,'Exact original reader required')
spec=importlib.util.spec_from_file_location('original_visual_rgb_comparison',PRIOR_SOURCE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def sources():return {'wrapper':sha(Path(__file__)),'reused':base.sources()}
def prepared_check():
 need(sha(PREPARED/'plan.json')==PLAN_SHA,'Exact final128 prepared plan required')
 need(sha(PRIOR_REPORT)==PRIOR_REPORT_SHA,'Exact completed cp16 analysis required')
 p=json.loads((PREPARED/'plan.json').read_text());old=json.loads(PRIOR_REPORT.read_text())
 need(p['artifacts']['adapter.safetensors']==CHECKPOINT and p['final_checkpoint_updates']==128,'Fixed final checkpoint required')
 need(p['artifacts']['sampling-inputs.safetensors']==sha(PRIOR_RUN/'sampling-inputs.safetensors') and p['artifacts']['contexts.safetensors']==sha(PRIOR_RUN/'contexts.safetensors') and p['artifacts']['commands.safetensors']==sha(PRIOR_RUN/'commands.safetensors'),'Exact prior sampling inputs/text/commands required')
 need(old['source_sha256']==base.sources(),'Historical comparison source changed')
 return p,old

def comparison(new,old,truth):
 need(set(new)==set(old)==set(truth)==set(base.ARMS),'Both branches required')
 cp128=base.paired_frame(new,truth);cp16=base.paired_frame(old,truth);arms={}
 for arm in base.ARMS:
  target=truth[arm].astype(np.float64)/255
  current=base.score(new[arm],target);previous=base.score(old[arm],target)
  arms[arm]={'checkpoint128_vs_checkpoint16':base.score(new[arm],old[arm]),'checkpoint128_vs_target':current,'checkpoint16_vs_target':previous,'target_mae_delta_128_minus16':current['mae_0_1']-previous['mae_0_1']}
 return {'arms':arms,'checkpoint128_pair':cp128,'checkpoint16_pair':cp16,'meaning':'Descriptive RGB difference and target correspondence. No numerical quality or control threshold.'}

def sheet(frames,path):
 keys=('checkpoint128 closed','checkpoint128 open','checkpoint16 closed','checkpoint16 open','truth closed','truth open');tw,th,label=624,352,42
 image=Image.new('RGB',(len(base.SELECTED)*tw,len(keys)*(th+label)),(242,242,238));draw=ImageDraw.Draw(image)
 for r,key in enumerate(keys):
  for c,i in enumerate(base.SELECTED):
   x,y=c*tw,r*(th+label);action='initial observation' if i==0 else ('wait' if key.endswith('closed') else 'interact') if i==1 else 'left'
   draw.text((x+8,y+5),f"{key.replace('checkpoint128','checkpoint 128').replace('checkpoint16','checkpoint 16')} | frame {i} | incoming {action}",fill=(16,16,16));draw.text((x+8,y+21),'50% nearest-neighbor display; original aspect; no enhancement',fill=(50,50,50))
   image.paste(Image.fromarray(frames[key][i]).resize((tw,th),Image.Resampling.NEAREST),(x,y+label))
 image.save(path)

def analyze(run,capture,audit_report,output):
 run=Path(run).resolve();output=Path(output);before=sources();plan,oldreport=prepared_check()
 need(not output.exists(),'Fresh comparison directory required');output.mkdir(parents=True)
 try:
  need(sha(run/'plan.json')==PLAN_SHA,'Actual final128 plan differs')
  integrity=json.loads(Path(audit_report).read_text())
  need(integrity['status']=='passed' and integrity['schema']=='worldline-action-cuda-final128-visual-actual-independent-v1' and integrity['recovered_bytes_unchanged'] is True,'Completed structural audit required')
  identity=integrity['identity'];need(identity['plan_sha256']==PLAN_SHA and identity['parent_sha256']==sha(run/'metrics.json') and identity['checkpoint128_sha256']==CHECKPOINT,'Actual structural audit binding differs')
  current_report=base.analyze(run,capture,output/'checkpoint128-versus-truth')
  oldrun,oldresult,oldparent,olddecode=base._completed(PRIOR_RUN);newrun,newresult,newparent,newdecode=base._completed(run)
  truth,provenance=base.ground_truth(capture)
  oldconditions=base._bind_conditions(oldrun,oldparent,provenance);newconditions=base._bind_conditions(newrun,newparent,provenance)
  need(oldconditions['arms']==newconditions['arms'],'Both checkpoints must use identical measured commands/initial inputs/text')
  need(current_report['data']==oldreport['data']==provenance,'Exact original target mapping and preprocessing differ')
  oldindices=base._generated_indices(oldresult,olddecode);newindices=base._generated_indices(newresult,newdecode)
  selected={key:{} for key in ('checkpoint128 closed','checkpoint128 open','checkpoint16 closed','checkpoint16 open','truth closed','truth open')};rows=[]
  for i in range(17):
   new={};old={};identities={}
   for arm in base.ARMS:
    new[arm],npix,nid=base._read_frame(newresult,newdecode,newindices[arm],arm,i)
    old[arm],opix,oid=base._read_frame(oldresult,olddecode,oldindices[arm],arm,i)
    need(oid==oldreport['frames'][i]['identities'][arm],'Exact cp16 frame identity differs')
    need(nid==current_report['frames'][i]['identities'][arm],'Exact cp128 frame identity differs')
    identities[arm]={'checkpoint128':nid,'checkpoint16':oid}
    if i in base.SELECTED:
     selected['checkpoint128 '+arm][i]=npix;selected['checkpoint16 '+arm][i]=opix;selected['truth '+arm][i]=truth[arm][i]
   rows.append({'frame':i,'identities':identities,**comparison(new,old,{a:truth[a][i] for a in base.ARMS})})
  sheet(selected,output/'checkpoint-comparison.png')
  result={'schema':'worldline-final128-paired-rgb-comparison-v1','status':'completed','model_execution':False,'cloud_operations':False,'source_sha256':before,'structural_audit_sha256':sha(audit_report),'checkpoint128_plan_sha256':PLAN_SHA,'checkpoint128_sha256':CHECKPOINT,'checkpoint16_original_analysis_sha256':PRIOR_REPORT_SHA,'checkpoint128_analysis_sha256':sha(output/'checkpoint128-versus-truth/report.json'),'conditions_identical_across_checkpoints':True,'all_frames_scored':17,'future_frames_in_means':16,'frames':rows,'future_means':{arm:{'target_mae_delta_128_minus16':float(np.mean([v['arms'][arm]['target_mae_delta_128_minus16']for v in rows[1:]])),'checkpoint128_vs_checkpoint16_mae':float(np.mean([v['arms'][arm]['checkpoint128_vs_checkpoint16']['mae_0_1']for v in rows[1:]]))}for arm in base.ARMS},'strict_pair_counts':{cp:{region:sum(r[cp+'_pair']['regions'][region]['strict_both_closer_to_own_truth']for r in rows[1:])for region in ('full_frame','paired_truth_difference')}for cp in ('checkpoint128','checkpoint16')},'contact_indices':list(base.SELECTED),'contact_sha256':sha(output/'checkpoint-comparison.png'),'raw_files_modified':False,'limitations':current_report['limitations']+['Pixel changes between checkpoints and target-error differences do not by themselves establish quality or door/camera response.','The final128 structural audit is a separate prerequisite. This wrapper does not replay sampling, decoding or training.']}
  need(sources()==before,'Comparison sources changed during inspection')
  (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result
 except BaseException as error:
  (output/'failed.json').write_text(json.dumps({'status':'failed','error_type':type(error).__name__,'error':str(error),'source_sha256':before},indent=2)+'\n');raise

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('run','capture','audit-report','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();r=analyze(a.run,a.capture,a.audit_report,a.output)
 print(json.dumps({'status':r['status'],'frames_per_arm':17,'report_sha256':sha(a.output/'report.json')}))
if __name__=='__main__':main()
