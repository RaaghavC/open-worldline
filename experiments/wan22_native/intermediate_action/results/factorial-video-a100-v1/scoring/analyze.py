"""CPU-only six-arm RGB scoring after verified complete recovery. No model imports."""
import argparse,hashlib,json,math,platform,shutil,time
from pathlib import Path
import numpy as np
from PIL import Image
import capture_reference as capture
import rgb_reference as raw
import metrics
import presentation
HERE=Path(__file__).resolve().parent
ARMS=presentation.ARMS
HEIGHT,WIDTH=704,1248
PLAN_SHA='07dc80a0daf69134e1985dc4649ad08f207fa6c2f85430cab23def82095d0a12'
CHECKPOINT_SHA='9147ef7a53a01c4399e7073cab97a4ccdc7c8d1195305333560871eeff004dba'
CAPTURE_MANIFEST_SHA='1872dea69d16b124fb7a0a9eefaff06ea2d850ff49f36bcf87fec8c10c82dada'
CRITERIA_SHA='e3c109730aab414ff4191bc3896a7f8f15bf58269e38506d6593c9e7774d82e6'
SOURCE_FILES=('analyze.py','metrics.py','presentation.py','test_cpu.py','rgb_reference.py','capture_reference.py','criteria.md','reference-provenance.json')
need,sha,read=raw._require,raw._sha,raw._json

def sources():
    values={n:sha(HERE/n) for n in SOURCE_FILES}
    need(values['criteria.md']==CRITERIA_SHA,'Original frozen criteria changed')
    for n,row in read(HERE/'reference-provenance.json').items():need(values[n]==row['sha256'],'Original independent reference source changed')
    return values

def cpu_review(path):
    r=read(Path(path));need(r.get('status')=='passed' and r.get('source_sha256')==sources() and r.get('sources_unchanged') is True,'Current source-bound CPU checks required')
    need(type(r.get('tests')) is int and r['tests']>=8 and all(type(r.get(k)) is int and r[k]==0 for k in ('failures','errors','skipped','exit_code')),'Complete CPU checks required')
    return sha(path)

def fresh_output(out,inputs):
    out=Path(out).absolute();need(not out.exists() and not any(p.is_symlink() for p in (out,*out.parents)),'Fresh output with regular ancestors required')
    need(all(not out.resolve().is_relative_to(Path(p).resolve()) for p in inputs),'Output cannot be within original inputs')
    return out

class Recovered:
    def __init__(self,folder,run):
        self.folder=Path(folder).resolve();self.root=Path(run).resolve();self.used={}
        need(self.root.is_relative_to(self.folder/'recovered'),'Run must belong to verified original recovery')
        self.prefix=self.root.relative_to(self.folder/'recovered').as_posix()
        self.verified=read(self.folder/'recovery-verified.json');self.index=read(self.folder/'index.json')
        need(self.verified.get('status')=='verified' and self.verified.get('index_sha256')==sha(self.folder/'index.json'),'Complete original recovery index required')
    def take(self,name,expected=None,maximum=raw.JSON_LIMIT):
        p=raw._file(self.root,name,maximum);row=dict(bytes=p.stat().st_size,sha256=sha(p))
        need(self.index['files'][self.prefix+'/'+name]==row,'Consumed file differs from verified recovery: '+name)
        if expected is not None:need(row['sha256']==expected,'Consumed file differs from worker/plan: '+name)
        self.used[name]=row;return p
    def json(self,name,expected=None):return read(self.take(name,expected))
    def finish(self):
        need(sha(self.folder/'index.json')==self.verified['index_sha256'],'Recovery index changed during scoring')
        for name,row in self.used.items():
            p=raw._file(self.root,name);need(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Consumed original changed during scoring')

def completed(reader,artifact_audit=None):
    plan=reader.json('plan.json',PLAN_SHA);parent=reader.json('metrics.json');terminal=reader.json('terminal.json')
    need(plan.get('schema')=='worldline-factorial-trained-video-v1' and plan['training_identity']['final_checkpoint_sha256']==CHECKPOINT_SHA,'Exact final128 six-arm plan required')
    need(parent.get('status')=='passed' and parent.get('model_frozen') is True and parent.get('model_execution') is True
        and parent.get('predictions')==600 and parent.get('solver_updates')==300 and parent.get('frames_per_arm')==17,'Complete six-video parent required')
    need(parent.get('plan_sha256')==PLAN_SHA and parent.get('source_sha256')==plan['source_sha256'],'Parent source/plan binding')
    need(parent.get('terminal_sha256')==reader.used['terminal.json']['sha256'] and terminal.get('status')=='complete' and type(terminal.get('exit_code')) is int and terminal['exit_code']==0,'Complete zero-exit video terminal required')
    need(not list(reader.root.rglob('watchdog-stop.json')),'No stopped video worker')
    stages={}
    for stage in ('core','decode'):
        child=reader.json(stage+'/result/metrics.json',parent['child_reports'][stage]);end=reader.json(stage+'/terminal.json',parent['child_terminals'][stage])
        need(child.get('status')=='passed' and child.get('plan_sha256')==PLAN_SHA and child.get('source_sha256')==plan['source_sha256']
            and child.get('admission')==parent['admission'] and end.get('status')=='complete' and type(end.get('exit_code')) is int and end['exit_code']==0 and end.get('cleanup_error') is None,'Complete source-bound child worker')
        need(set(child['arms'])==set(ARMS),'Complete six arm records required');stages[stage]=child
    need(stages['core'].get('model_frozen') is True and stages['core'].get('predictions')==600 and stages['core'].get('solver_updates')==300,'All six frozen-core samples required')
    need(stages['decode'].get('all196_unchanged') is True and all(v['images']['frames']==17 for v in stages['decode']['arms'].values()),'Complete original VAE decoder records')
    identity=dict(plan_sha256=PLAN_SHA,parent_sha256=reader.used['metrics.json']['sha256'],core_sha256=parent['child_reports']['core'],decode_sha256=parent['child_reports']['decode'],checkpoint_sha256=CHECKPOINT_SHA,capture_manifest_sha256=CAPTURE_MANIFEST_SHA,recovery_index_sha256=reader.verified['index_sha256'])
    if artifact_audit is not None:
        a=read(Path(artifact_audit));need(a.get('status')=='passed','Only a separately passed artifact audit can be attached')
        for key in ('plan_sha256','parent_sha256','core_sha256','decode_sha256','checkpoint_sha256'):
            need(a.get('identity',{}).get(key)==identity[key],'Attached artifact audit belongs to another run: '+key)
        identity['attached_artifact_audit_sha256']=sha(artifact_audit)
    return stages['decode'],identity

def frame_index(index,height=HEIGHT,width=WIDTH):
    need(index.get('schema')=='wan22-rgb-frames-v1' and index.get('purpose')=='generated_clip'
         and index.get('shape')==[1,3,17,height,width] and index.get('dtype')=='float32' and index.get('range')==[-1,1],'Exact raw RGB index required')
    rows=index.get('frames');need(isinstance(rows,list) and len(rows)==17,'All17 raw RGB rows required')
    for i,row in enumerate(rows):need(type(row.get('index')) is int and row['index']==i and row.get('file')==f'{i:04d}.safetensors','Exact zero-padded PNG/raw frame order required')
    return rows

def png_value(path,height=HEIGHT,width=WIDTH,allow_alpha=False):
    with Image.open(path) as image:
        need(image.format=='PNG' and image.size==(width,height) and getattr(image,'n_frames',1)==1,'One native PNG frame required')
        need(image.mode in (('RGB','RGBA') if allow_alpha else ('RGB',)),'8-bit RGB or opaque target RGBA required');value=np.array(image)
    need(value.dtype==np.uint8,'Original8-bit channels required')
    if value.shape[-1]==4:
        need(np.all(value[:,:,3]==255),'Opaque target alpha required');value=value[:,:,:3].copy()
    return value

def generated_frame(reader,decoder,index,arm,i,height=HEIGHT,width=WIDTH):
    row=index[i];relative=arm+'/rgb/'+row['file'];path=reader.take('decode/result/'+relative,decoder['output_sha256'][relative],raw.FRAME_LIMIT)
    need(row.get('bytes')==path.stat().st_size and row.get('sha256')==sha(path),'Raw frame index file binding')
    value,digest=raw._TensorFile(path,{'rgb':(1,3,1,height,width)}).read('rgb')
    need(row.get('tensor_sha256')==digest and value.min()>=-1 and value.max()<=1,'Raw frame tensor identity/range')
    relative=arm+'/frames/'+f'{i:04d}.png';png=reader.take('decode/result/'+relative,decoder['output_sha256'][relative],raw.IMAGE_LIMIT)
    pixels=png_value(png,height,width);need(np.array_equal(pixels,raw._pixels(value)),'Generated PNG must exactly match original FP32 rounding')
    return (value[0,:,0].transpose(1,2,0).astype(np.float64)+1)/2,png

def target_frame(root,row,used):
    path=raw._file(root,row['png'],capture.MAX_IMAGE_BYTES);digest=sha(path)
    need(digest==row['png_sha256'],'Exact original target PNG hash required')
    pixels=png_value(path,allow_alpha=True);used[row['png']]=dict(bytes=path.stat().st_size,sha256=digest)
    return pixels.astype(np.float64)/255,path

def copy_exact(source,target):
    target.parent.mkdir(parents=True,exist_ok=True);need(not target.exists(),'Fresh exact display copy required');shutil.copyfile(source,target)
    need(sha(source)==sha(target),'Original PNG copy changed')

def analyze(recovery,run,capture_root,output,cpu_report,*,artifact_audit=None,mp4=True):
    out=fresh_output(output,(recovery,run,capture_root));cpu=cpu_review(cpu_report);source_before=sources();reader=Recovered(recovery,run);decoder,identity=completed(reader,artifact_audit)
    capture_root=Path(capture_root).resolve();rows={}
    for arm in ARMS:
        _,rows[arm],_=capture.checked_manifest(capture_root,arm,CAPTURE_MANIFEST_SHA)
    generated_indices={}
    for arm in ARMS:
        name=arm+'/rgb/index.json';idx=reader.json('decode/result/'+name,decoder['output_sha256'][name]);generated_indices[arm]=frame_index(idx)
        need({p.name for p in (reader.root/'decode/result'/arm/'frames').iterdir()}=={f'{i:04d}.png' for i in range(17)},'All17 and only original ordered PNGs required')
    # All admission/metadata/capture checks above are read-only. Results start only with actual complete files.
    out.mkdir(parents=True);started=time.monotonic();used_targets={};report=dict(schema='worldline-factorial-six-video-rgb-scoring-v1',status='running',identity=identity,model_execution=False,quality_verdict='manual reviews pending')
    try:
        frame_rows={a:[] for a in ARMS};pairs={};summary={}
        for motion in ('stationary','left','right'):
            pair={a:motion+'_'+a for a in ('closed','interact')};first_g={};first_t={};previous_g={};previous_t={};pairs[motion]=[]
            for i in range(17):
                gs={};ts={}
                for door,arm in pair.items():
                    g,gp=generated_frame(reader,decoder,generated_indices[arm],arm,i);t,tp=target_frame(capture_root,rows[arm][i],used_targets);gs[door]=g;ts[door]=t
                    if i==0:first_g[door]=g;first_t[door]=t;previous_g[door]=g;previous_t[door]=t
                    frame_rows[arm].append(dict(index=i,**metrics.frame_scores(g,t,first_t[door],first_g[door],previous_g[door],previous_t[door])))
                    if arm=='stationary_closed':need(np.array_equal(t,first_t[door]),'Pinned stationary-closed target must repeat initial pixels exactly')
                    copy_exact(gp,out/'frames/generated'/arm/f'{i:04d}.png');copy_exact(tp,out/'frames/target'/arm/f'{i:04d}.png')
                pairs[motion].append(dict(index=i,**metrics.paired_scores(gs,ts,first_g)));previous_g=gs;previous_t=ts
            for arm in pair.values():
                records=frame_rows[arm];keys=set(records[0])-{'index'};future={k:metrics.average([v[k] for v in records[1:]]) for k in sorted(keys)}
                summary[arm]=dict(future=future,final=records[-1],future_vs_original_repeat=metrics.improvement(future['generated_vs_target'],future['original_repeat_vs_target']),final_vs_original_repeat=metrics.improvement(records[-1]['generated_vs_target'],records[-1]['original_repeat_vs_target']))
        pair_summary={motion:dict(future={k:metrics.average([v[k] for v in records[1:]]) for k in records[0] if k!='index'},final=records[-1]) for motion,records in pairs.items()}
        shutil.copyfile(HERE/'criteria.md',out/'criteria.md');copy_exact(capture_root/'manifest.json',out/'target-manifest.json');contacts=presentation.contacts(out);video=presentation.movies(out,mp4);presentation.viewer(out,identity,summary)
        if artifact_audit:shutil.copyfile(artifact_audit,out/'attached-artifact-audit.json')
        for name,digest in source_before.items():
            copy_exact(HERE/name,out/'source'/name);need(sha(out/'source'/name)==digest,'Scoring source copy changed')
        copy_exact(Path(cpu_report),out/'source/cpu-report.json')
        reader.finish()
        need(sha(capture_root/'manifest.json')==CAPTURE_MANIFEST_SHA,'Target manifest changed')
        for name,row in used_targets.items():need(sha(raw._file(capture_root,name))==row['sha256'],'Original target changed during scoring')
        need(source_before==sources(),'Scoring source changed')
        (out/'consumed-inputs.json').write_text(json.dumps(dict(generated=reader.used,targets=used_targets,capture_manifest_sha256=CAPTURE_MANIFEST_SHA),indent=2)+'\n')
        report.update(status='completed',source_sha256=source_before,cpu_report_sha256=cpu,frames_per_arm=17,total_generated_frames=102,total_target_frames=102,future_frames_per_arm=16,
            pixel_rules=dict(generated='Raw FP32[-1,1] converted to FP64, then(value+1)/2; no display rounding in scores',target='Original opaque PNG RGB uint8 converted directly to FP64/255; no resize/crop',png_validation='Exact equality to np.rint((FP32rgb+FP32(1))*FP32(127.5)).clip(0,255).astype(uint8)',reductions='FP64 mean absolute/squared error over every RGB channel; pooled future RMSE=sqrt(mean future-frame MSE), not mean of RMSE',paired='Signed interact-minus-closed RGB difference; same-motion pairs only; no door mask or angle estimate'),
            frames=frame_rows,arms=summary,paired_frames=pairs,paired_summary=pair_summary,contacts=contacts,mp4=video,
            consumed_inputs_sha256=sha(out/'consumed-inputs.json'),raw_files_unchanged=True,control_outcomes_automatically_assessed=False,
            limitations=['One fixed saved noise and one seen scene; no generalization claim.','Training data and placement both changed; no placement-only advantage is established.','Pixel errors and RGB drift do not identify camera pose, door state, or successful controls.','Stationary-closed original repeat baseline is zero; no beat-zero threshold exists.','No native unadapted clip is included, so improvement over unadapted Wan remains unresolved.','This scorer verifies consumed RGB/PNG and completion identities; it does not replay the solver, foundation or decoder.','Contact thumbnails and optional lossy MP4s are display derivatives. Both reviewers must inspect all original PNGs.'])
    except BaseException as e:report.update(status='failed',error_type=type(e).__name__,error=str(e));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started;(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    inventory={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'output-manifest.json').write_text(json.dumps(dict(files=inventory,file_count=len(inventory),file_bytes=sum(v['bytes'] for v in inventory.values())),indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('recovery','run','capture','output','cpu-report'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--artifact-audit',type=Path);p.add_argument('--no-mp4',action='store_true');a=p.parse_args()
    r=analyze(a.recovery,a.run,a.capture,a.output,a.cpu_report,artifact_audit=a.artifact_audit,mp4=not a.no_mp4);print(json.dumps(dict(status=r['status'],report_sha256=sha(a.output/'report.json'),model_execution=False)))
