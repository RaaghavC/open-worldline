# SPDX-License-Identifier: Apache-2.0
"""All-frame effect128 RGB correspondence using unchanged prior metric functions."""
import argparse, hashlib, importlib.util, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
BASE_PATH=ROOT/'work/wan22-action-cuda-visual-analysis-prep-v1/analyze.py'
BASE_SHA='520c2af654749faf5f4d480cc52c0ac21d8f7aede7257e254e2cf4e9eb00d9bc'
PREPARED=ROOT/'work/wan22-action-effect-visual-prepared-v1'
PLAN_SHA='a6e239bdad20b699f5909d47a026f39312e4015135194a874587619d3d621ef3'
CHECKPOINT_SHA='4c9cd94ac04a45a4dad29d5e5efa05b6560af99c364d350657d5d5fa4f8ca3ce'
TRAINING_AUDIT_SHA='05b1d6ef0a32c4dfb228e7517ad27365240d75f105c007a18c779fc7e617f7c1'
PRIOR={
 'previous16':{'run':ROOT/'work/wan22-action-cuda-visual-recovered-final-v1/recovered/action-results/visual-spatial-v2','report':ROOT/'work/wan22-action-cuda-visual-comparison-v1/report.json','report_sha256':'6693ba5434dd07dd9016ca660613ce19e5ce833512ce4a8d0c8dc839550faae3'},
 'previous128':{'run':ROOT/'work/wan22-action-cuda-post128-recovered-final-v1/recovered/action-results/final128-visual-v1','report':ROOT/'work/wan22-action-cuda-final128-visual-comparison-v1/checkpoint128-versus-truth/report.json','report_sha256':'052d89a30bf90510963f89b73c8cc4fa96ad551e18098ed535587d87d9fa68f7'}}
MODELS=('effect128','previous128','previous16')
DISPLAY={'effect128':'effect 128','previous128':'previous 128','previous16':'previous 16'}

def sha(path):
 with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def need(ok,message):
 if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
need(sha(BASE_PATH)==BASE_SHA,'Unchanged original RGB reader required')
spec=importlib.util.spec_from_file_location('effect_rgb_original_reader',BASE_PATH)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def sources():return {'wrapper':sha(__file__),'reused':base.sources()}

def prepared_check():
 need(sha(PREPARED/'plan.json')==PLAN_SHA,'Exact prepared effect visual plan required')
 plan=read(PREPARED/'plan.json')
 need(plan['artifacts']['adapter.safetensors']==CHECKPOINT_SHA and plan['training_audit_sha256']==TRAINING_AUDIT_SHA,'Exact effect128 checkpoint and audit required')
 need(plan['negative_context_adapter_training'] is True,'Effect training used both auxiliary text contexts')
 prior_reports={}
 for name,item in PRIOR.items():
  need(sha(item['report'])==item['report_sha256'],'Pinned historical report '+name)
  record=read(item['report']);need(record['source_sha256']==base.sources(),'Historical metric source differs')
  for filename in ('sampling-inputs.safetensors','contexts.safetensors','commands.safetensors'):
   need(plan['artifacts'][filename]==sha(item['run']/filename),'Exact shared historical input '+name+'/'+filename)
  prior_reports[name]=record
 return plan,prior_reports

def compare_frame(generated,truth,initial):
 """Call the old score and paired_frame functions; no alternate equations."""
 need(set(generated)==set(MODELS),'All three model results required')
 rows={}
 for model in MODELS:
  need(set(generated[model])==set(base.ARMS),'Both branches required')
  rows[model]={'pair':base.paired_frame(generated[model],truth),'arms':{}}
  for arm in base.ARMS:
   target=truth[arm].astype(np.float64)/255
   rows[model]['arms'][arm]={'generated_vs_target':base.score(generated[model][arm],target),'repeat_start_vs_target':base.score(initial,target),'generated_vs_start':base.score(generated[model][arm],initial)}
 comparisons={}
 for previous in ('previous128','previous16'):
  comparisons[previous]={arm:{'effect_vs_previous':base.score(generated['effect128'][arm],generated[previous][arm]),'target_mae_delta_effect_minus_previous':rows['effect128']['arms'][arm]['generated_vs_target']['mae_0_1']-rows[previous]['arms'][arm]['generated_vs_target']['mae_0_1']}for arm in base.ARMS}
 return {'models':rows,'comparisons':comparisons}

def incoming(arm,index):return 'initial observation' if index==0 else ('wait' if arm=='closed' else 'interact') if index==1 else 'left'

def comparison_sheet(images,path):
 keys=[(model,arm)for model in MODELS for arm in base.ARMS]+[('truth',arm)for arm in base.ARMS]
 tw,th,label=624,352,40
 canvas=Image.new('RGB',(len(base.SELECTED)*tw,len(keys)*(th+label)),(242,242,238));draw=ImageDraw.Draw(canvas)
 for row,(model,arm)in enumerate(keys):
  for col,index in enumerate(base.SELECTED):
   x,y=col*tw,row*(th+label)
   draw.text((x+8,y+4),f'{DISPLAY.get(model,model)} {arm} | frame {index} | {incoming(arm,index)}',fill=(16,16,16))
   draw.text((x+8,y+21),'50% nearest-neighbor; native aspect; no enhancement',fill=(50,50,50))
   canvas.paste(Image.fromarray(images[(model,arm)][index]).resize((tw,th),Image.Resampling.NEAREST),(x,y+label))
 canvas.save(path)

def all_frame_sheet(images,path):
 """All 34 generated PNGs in order; quarter-size display with empty spare cells."""
 tw,th,label=312,176,40;columns=5;rows_per_arm=4
 canvas=Image.new('RGB',(columns*tw,2*rows_per_arm*(th+label)),(242,242,238));draw=ImageDraw.Draw(canvas);layout=[]
 for a,arm in enumerate(base.ARMS):
  need(set(images[arm])==set(range(17)),'All seventeen images required')
  for index in range(17):
   col=index%columns;row=a*rows_per_arm+index//columns;x=col*tw;y=row*(th+label)
   draw.text((x+6,y+4),f'effect 128 {arm} | frame {index} | {incoming(arm,index)}',fill=(16,16,16))
   draw.text((x+6,y+21),'25% nearest-neighbor; no enhancement',fill=(50,50,50))
   canvas.paste(Image.fromarray(images[arm][index]).resize((tw,th),Image.Resampling.NEAREST),(x,y+label))
   layout.append({'arm':arm,'frame':index,'column':col,'row':row,'original_png_pixel_sha256':base.arr_sha(images[arm][index])})
 canvas.save(path);return layout

def audit_gate(run,recovery,audit_path,plan):
 verified=read(recovery/'recovery-verified.json');audit=read(audit_path)
 need(verified['status']=='verified' and sha(recovery/'index.json')==verified['index_sha256'],'Original complete verified recovery required')
 need(run.is_relative_to((recovery/'recovered').resolve()),'Run must lie inside verified recovery')
 need(audit['schema']=='worldline-action-effect128-visual-actual-independent-v1' and audit['status']=='passed' and audit['recovered_bytes_unchanged'] is True,'Passed effect visual structural audit required')
 need(audit['recovery_verified_sha256']==sha(recovery/'recovery-verified.json'),'Structural audit recovery differs')
 expected={'plan_sha256':PLAN_SHA,'parent_sha256':sha(run/'metrics.json'),'core_sha256':sha(run/'core/result/metrics.json'),'decode_sha256':sha(run/'decode/result/metrics.json'),'admission_sha256':sha(run/'executed-admission.json'),'training_audit_sha256':TRAINING_AUDIT_SHA,'checkpoint128_sha256':CHECKPOINT_SHA,'source_sha256':plan['source_sha256']}
 need(audit['identity']==expected,'Exact completed structural identity required')
 need(sha(run/'plan.json')==PLAN_SHA,'Actual visual plan differs')
 return verified,audit

def analyze(run,capture,recovery,audit_report,output):
 run=Path(run).resolve();recovery=Path(recovery).resolve();output=Path(output).absolute();audit_report=Path(audit_report).resolve()
 need(not output.exists() and not output.resolve().is_relative_to(recovery),'Fresh output outside recovered evidence required')
 before=sources();plan,prior_reports=prepared_check();audit_gate(run,recovery,audit_report,plan)
 output.mkdir(parents=True)
 try:
  truth,provenance=base.ground_truth(capture);initial=truth['open'][0].astype(np.float64)/255
  handles={};conditions={};indices={}
  for model,path in [('effect128',run),*((n,x['run'])for n,x in PRIOR.items())]:
   actual,result,parent,decode=base._completed(path);handles[model]=(actual,result,parent,decode)
   conditions[model]=base._bind_conditions(actual,parent,provenance);indices[model]=base._generated_indices(result,decode)
   if model!='effect128':need(prior_reports[model]['data']==provenance,'Original target processing changed')
  need(all(conditions[m]['arms']==conditions['effect128']['arms'] for m in MODELS),'Commands/noise/observation/text must match all prior runs')
  rows=[];selected={(m,a):{}for m in (*MODELS,'truth')for a in base.ARMS};all_images={a:{}for a in base.ARMS};previews={}
  for index in range(17):
   generated={};identities={}
   for model in MODELS:
    generated[model]={};identities[model]={};actual,result,parent,decode=handles[model]
    for arm in base.ARMS:
     value,pixels,identity=base._read_frame(result,decode,indices[model][arm],arm,index)
     generated[model][arm]=value;identities[model][arm]=identity
     if model!='effect128':need(identity==prior_reports[model]['frames'][index]['identities'][arm],'Historical frame bytes changed')
     if index in base.SELECTED:selected[(model,arm)][index]=pixels
     if model=='effect128':
      all_images[arm][index]=pixels
      if index in (0,16):
       destination=output/'full-frame-previews'/f'{arm}-{"initial" if index==0 else "final"}.png';destination.parent.mkdir(exist_ok=True)
       shutil.copyfile(result/arm/'frames'/f'{index:04d}.png',destination)
       need(sha(destination)==identity['png_sha256'],'Full-frame preview must be an exact original PNG copy')
       previews[str(destination.relative_to(output))]={'arm':arm,'frame':index,'sha256':sha(destination),'original_size':[1248,704],'byte_exact_copy':True}
   current={a:truth[a][index]for a in base.ARMS}
   if index in base.SELECTED:
    for arm in base.ARMS:selected[('truth',arm)][index]=current[arm]
   scored=compare_frame(generated,current,initial)
   # Require precisely the already published baseline equations and results.
   for old in PRIOR:
    expected=prior_reports[old]['frames'][index]
    need(scored['models'][old]['pair']==expected['pair'],'Historical paired score changed')
    for arm in base.ARMS:
     for key,value in scored['models'][old]['arms'][arm].items():need(value==expected['arms'][arm][key],'Historical per-frame score changed')
   rows.append({'frame':index,'identities':identities,'frame_mapping':{a:provenance['arms'][a]['frames'][index]for a in base.ARMS},**scored})
  comparison_sheet(selected,output/'checkpoint-target-comparison.png')
  layout=all_frame_sheet(all_images,output/'all-34-frames.png')
  summary={model:{arm:{key:float(np.mean([row['models'][model]['arms'][arm][key]['mae_0_1']for row in rows[1:]]))for key in ('generated_vs_target','repeat_start_vs_target','generated_vs_start')}for arm in base.ARMS}for model in MODELS}
  report={'schema':'worldline-action-effect128-matched-rgb-comparison-v1','status':'completed','model_execution':False,'cloud_operations':False,'source_sha256':before,'structural_audit_sha256':sha(audit_report),'recovery_verified_sha256':sha(recovery/'recovery-verified.json'),'plan_sha256':PLAN_SHA,'effect128_checkpoint_sha256':CHECKPOINT_SHA,'prior_analysis_sha256':{n:v['report_sha256']for n,v in PRIOR.items()},'old_per_frame_metrics_recomputed_exact':True,'conditions_identical_across_all_three_runs':True,'conditions':conditions,'data':provenance,'all_frames_scored_per_arm':17,'future_frames_in_means':16,'frames':rows,'future_means':summary,'future_comparisons':{old:{arm:{'target_mae_delta_effect_minus_previous':float(np.mean([r['comparisons'][old][arm]['target_mae_delta_effect_minus_previous']for r in rows[1:]])),'effect_vs_previous_mae':float(np.mean([r['comparisons'][old][arm]['effect_vs_previous']['mae_0_1']for r in rows[1:]]))}for arm in base.ARMS}for old in PRIOR},'strict_pair_counts':{m:{region:sum(r['models'][m]['pair']['regions'][region]['strict_both_closer_to_own_truth']for r in rows[1:])for region in ('full_frame','paired_truth_difference')}for m in MODELS},'summary_contact_indices':list(base.SELECTED),'all_generated_contact_indices':list(range(17)),'all_generated_contact_layout':layout,'full_frame_previews':previews,'contacts':{n:sha(output/n)for n in ('checkpoint-target-comparison.png','all-34-frames.png')},'raw_files_modified':False,'quality_success_assessed':False,'limitations':['One seen fixed room and one common saved noise, not a test of new scenes or independent action generalization.','Metrics measure RGB correspondence, not visual quality or successful door/camera response. Camera mismatch and lighting changes affect target error.','Paired truth masks include indirect illumination and all RGB differences of at least two byte levels; they are not door segmentation.','Fifteen requested 7.5-degree left turns reach 112.5 degrees at frame16. Door visibility is not inferred from these scores.','Frame0 is a conditioned reconstruction and is excluded from future means. Both-branches-closer counts require strict inequalities; ties fail.','All three samplers use commands on both CFG contexts. Old16/old128 training used positive text only; effect128 uses positive-only main FM plus the auxiliary loss on both text contexts.','The separate structural audit verifies saved sampling/decoder evidence. This comparison does not replay a model or infer a successful control outcome.']}
  need(sources()==before,'Comparison source changed')
  (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  (output/'source').mkdir();shutil.copyfile(__file__,output/'source/analyze.py')
  return report
 except BaseException as error:
  (output/'failed.json').write_text(json.dumps({'status':'failed','error_type':type(error).__name__,'error':str(error),'source_sha256':before},indent=2)+'\n');raise

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('run','capture','recovery','audit-report','output'):parser.add_argument('--'+name,type=Path,required=True)
 args=parser.parse_args();result=analyze(args.run,args.capture,args.recovery,args.audit_report,args.output)
 print(json.dumps({'status':result['status'],'frames_per_arm':17,'report_sha256':sha(args.output/'report.json')}))
if __name__=='__main__':main()
