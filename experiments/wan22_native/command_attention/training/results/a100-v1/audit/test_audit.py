"""Small independent arithmetic fixtures, not simulated CUDA execution."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
import audit
from arrays import inventory, sha, tensor_sha


def save(p,values):
    p.parent.mkdir(parents=True,exist_ok=True); header={}; chunks=[]; offset=0
    for n,x in sorted(values.items()):
        x=np.ascontiguousarray(x); raw=x.tobytes()
        header[n]=dict(shape=list(x.shape),dtype='U8' if x.dtype==np.uint8 else 'F32',data_offsets=[offset,offset+len(raw)])
        offset+=len(raw); chunks.append(raw)
    h=json.dumps(header,separators=(',',':')).encode(); h+=b' '*(-len(h)%8)
    p.write_bytes(struct.pack('<Q',len(h))+h+b''.join(chunks))


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(value)+'\n')


def fixture(root):
    shape=(1,1,5,2,2); initial={'a':np.ones((2,2),np.float32),'b':np.zeros((2,2),np.float32)}
    windows={}
    for i,a in enumerate(audit.ARMS):
        target=np.full(shape,i,np.float32); target[:,:,:1]=0
        windows[a]=dict(target=target,observation=target[:,:,:1].copy(),commands=np.zeros((1,16,6),np.float32))
    schedule=[]; rows=[]; noise=np.full(shape,3,np.float32); draws={}
    norm=float(np.sqrt(8*.125**2))
    for i in range(512):
        names=list(audit.ARMS[2*(i%3):2*(i%3)+2]); edge=list(audit.EDGES[(i//4)%7]) if i%4==0 else None
        s=dict(update=i+1,k=506,sigma=.506,noise_key=f'noise_{i:04d}',noise_sha256=tensor_sha(noise),rng_after_sha256='0'*64,
               branches=names,auxiliary_edge=edge)
        schedule.append(s); enabled=edge is not None
        rows.append(dict(update=i+1,schedule=s,main_predictions=2,optimizer_updates=1,
            all_controller_gradients_present_finite=True,foundation_gradients_absent=True,
            main=[dict(arm=a,future_flow_mse=.0625) for a in names],
            auxiliary=dict(enabled=enabled,edge=edge,weight=1.,feature_extracts=2*enabled,predictions=4*enabled,loss=0.),
            total_objective=.0625,gradient_l2_before_clip=norm,gradient_l2_after_clip=norm,main_seconds=.1,seconds=.2))
        if i+1 in audit.RAW:
            draws[s['noise_key']]=noise
            d=root/f'raw/update-{i+1:04d}'
            for a in names:save(d/f'main-{a}.safetensors',{'velocity':noise-windows[a]['target']+np.float32(.25)})
            if enabled:
                for j,a in enumerate(edge):
                    for c in ('positive','negative'):save(d/f'aux-{j}-{c}.safetensors',{'velocity':-windows[a]['target']})
            save(d/'gradients.safetensors',{n:np.full_like(v,.125) for n,v in initial.items()})
    worker=dict(completed_updates=512,updates=rows,main_predictions=1024,auxiliary_updates=128,auxiliary_predictions=512,auxiliary_feature_extracts=256)
    data=dict(plan={'schedule':schedule},initial=initial,windows=windows,draws=draws,evaluation={f'noise_{k:02d}':noise for k in range(4)})
    for phase,gain in (('initial',0),('final',.25)):
        scores=[]
        for n in data['evaluation']:
            for a in audit.ARMS:
                for c in ('positive','negative'):
                    save(root/f'evaluation/{phase}/{n}/{a}-{c}.safetensors',{'velocity':-np.float32(gain)*windows[a]['target']})
            for a,b in audit.EDGES:
                # Constant future difference D, exact ideal endpoint gain g.
                d=float(windows[b]['target'][0,0,1,0,0]-windows[a]['target'][0,0,1,0,0]); energy=d*d
                scores.append(dict(noise=n,edge=[a,b],future_mse=(gain-1)**2*energy,target_mean_square=energy,
                    prediction_mean_square=gain**2*energy,normalized_mse=(gain-1)**2,cosine=1. if gain else None,
                    target_rms=abs(d),prediction_rms=abs(gain*d),target_informative=True))
        worker[phase+'_evaluation']=dict(label=phase,predictions=48,feature_extracts=8,ideal_sigma=1.,native_time=999,
             model_quality_assessed=False,checkpoint_selection=False,scores=scores,seconds=.1)
    plan={'source_sha256':{'synthetic':'0'*64}}; plan_sha='1'*64; admission_sha='2'*64
    identity=dict(schema=audit.SCHEMA,scope=audit.SCOPE,plan_sha256=plan_sha,source_sha256=plan['source_sha256'],inputs_sha256=audit.INPUT_SHA,
        controller_parameters=4936448,admission_sha256=admission_sha,
        note='Reused checkpoint helper calls the new controller file adapter.safetensors; original residual adapter is absent.')
    for i in audit.CHECKPOINTS:
        d=root/f'checkpoint-{i:04d}'; values={n:v+np.float32(i/1024) for n,v in initial.items()}
        save(d/'adapter.safetensors',values); (d/'optimizer-and-rng.pt').write_bytes(b'opaque fixture only, never loaded')
        m=dict(schema='worldline-wan22-action-checkpoint-v1',completed_updates=i,identity=identity,
               external_core_weights_included=False,resume_supported=False,
               tensors={n:dict(shape=list(v.shape),dtype='float32',sha256=tensor_sha(v)) for n,v in values.items()},
               files={n:sha(d/n) for n in ('adapter.safetensors','optimizer-and-rng.pt')})
        write(d/'manifest.json',m); save(root/f'cuda-rng-{i:04d}.safetensors',{'rng':np.arange(16,dtype=np.uint8)})
    pointer=dict(directory='checkpoint-0512',manifest_sha256=sha(root/'checkpoint-0512/manifest.json'),completed_updates=512)
    write(root/'last-valid.json',pointer); worker['last_checkpoint']=pointer
    worker['output_sha256']={n:r['sha256'] for n,r in inventory(root).items()}
    return data,worker,plan,plan_sha,admission_sha


class Tests(unittest.TestCase):
    def test_tiny512_analytic_rows_predictions_gradients_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data,r,plan,ps,ads=fixture(root)
            self.assertEqual(audit.scalar_rows(r,data['plan']['schedule'],True)['scalar_rows'],512)
            v=audit.numeric(root,r,data,True)
            self.assertEqual(v['raw_prediction_files'],118)
            self.assertEqual(v['raw_gradient_bundles'],5)
            self.assertEqual(v['evaluations']['initial']['scores'][0]['normalized_mse'],1.)
            self.assertEqual(v['evaluations']['final']['scores'][0]['normalized_mse'],.5625)
            self.assertEqual(v['retained_updates'][0]['main_future_mse'],[.0625,.0625])
            self.assertEqual(v['retained_updates'][0]['auxiliary_future_mse'],0.)
            cp=audit.checkpoints(root,r,plan,ps,ads,data,True)
            self.assertEqual(len(cp['checkpoints']),5); self.assertFalse(cp['optimizer_payloads_unpickled'])
            audit.profile.check_output_hashes(root,r)

    def test_corrupted_schedule_score_gradient_checkpoint_and_binary_hash(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data,r,plan,ps,ads=fixture(root)
            bad=copy.deepcopy(r); bad['updates'][127]['schedule']['k']+=1
            with self.assertRaisesRegex(ValueError,'schedule'):audit.scalar_rows(bad,data['plan']['schedule'],True)
            bad=copy.deepcopy(r); bad['final_evaluation']['scores'][0]['future_mse']+=.5
            with self.assertRaisesRegex(ValueError,'scalar'):audit.numeric(root,bad,data,True)
            bad=copy.deepcopy(r); bad['updates'][511]['gradient_l2_after_clip']*=2
            with self.assertRaisesRegex(ValueError,'gradient'):audit.numeric(root,bad,data,True)
            m=read_fixture(root/'checkpoint-0256/manifest.json'); m['identity']['plan_sha256']='3'*64; write(root/'checkpoint-0256/manifest.json',m)
            with self.assertRaisesRegex(ValueError,'identity'):audit.checkpoints(root,r,plan,ps,ads,data,True)
            with self.assertRaisesRegex(ValueError,'inventory'):audit.profile.check_output_hashes(root,r)

    def test_partial_rows_do_not_certify512_or_discard_completed_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); data,r,*_=fixture(root)
            r.update(completed_updates=2,updates=r['updates'][:2],main_predictions=4,auxiliary_updates=1,auxiliary_predictions=4,auxiliary_feature_extracts=2)
            del r['final_evaluation']
            self.assertEqual(audit.scalar_rows(r,data['plan']['schedule'],False)['completed_updates'],2)
            v=audit.numeric(root,r,data,False)
            self.assertEqual(len(v['retained_updates']),2); self.assertNotIn('final',v['evaluations'])
            with self.assertRaises(ValueError):audit.scalar_rows(r,data['plan']['schedule'],True)


def read_fixture(p):return json.loads(p.read_text())
if __name__=='__main__':unittest.main()
