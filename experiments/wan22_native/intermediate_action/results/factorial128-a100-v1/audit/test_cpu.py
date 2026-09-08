"""Bounded failure fixtures for the independent reader and new schedule."""
import json,struct,tempfile,unittest
from pathlib import Path
import numpy as np
import audit
import checks as c

def write(path,values):
    header={};raw=bytearray()
    for n,v in values.items():
        v=np.ascontiguousarray(v);start=len(raw);raw.extend(v.tobytes());header[n]=dict(dtype={np.dtype('float32'):'F32',np.dtype('int64'):'I64',np.dtype('uint8'):'U8'}[v.dtype],shape=list(v.shape),data_offsets=[start,len(raw)])
    text=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(text))+text+raw)

class Checks(unittest.TestCase):
    def test_schedule_exact_and_wrong_motion_aux_draw_rejected(self):
        p=audit.read(audit.HERE/'expected-plan.json');original=[dict(r) for r in p['schedule']];audit.schedule(p['schedule'],original)
        for key,value in [('motion','right'),('auxiliary',False),('noise_sha256','0'*64),('k',950)]:
            rows=[dict(r) for r in p['schedule']];rows[0][key]=value
            with self.assertRaises(ValueError):audit.schedule(rows,original)
    def test_safe_reader_corruption_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.safetensors';a=np.arange(20,dtype=np.float32).reshape(1,1,5,2,2);write(p,{'velocity':a});self.assertTrue(np.array_equal(c.arrays(p,['velocity'])['velocity'],a))
            p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaises(ValueError):c.arrays(p)
            a.flat[0]=np.inf;write(p,{'velocity':a})
            with self.assertRaises(ValueError):c.arrays(p)
    def test_future_loss_excludes_prefix_and_rejects_wrong_score(self):
        a=np.ones((1,1,5,2,2),np.float32);b=np.zeros_like(a);a[:,:,0]=99
        self.assertEqual(c.loss_record(a,b,1.)['future_mse_fp64'],1.)
        with self.assertRaises(ValueError):c.loss_record(a,b,2.)
    def test_exact_flow_and_unchanged_target(self):
        target=np.full(c.SHAPE,2.,np.float32);obs=target[:,:,:1].copy();noise=np.full(c.SHAPE,3.,np.float32);window=dict(target=target,observation=obs,commands=np.zeros((1,16,6),np.float32));before=c.tensor_sha(target)
        x,t,v,identity=audit.flow(window,noise,506);self.assertTrue(np.array_equal(x[:,:,:1],obs));self.assertTrue(np.all(t[:,:858]==0));self.assertTrue(np.all(t[:,858:]==506));self.assertTrue(np.all(v==1));self.assertEqual(before,c.tensor_sha(target));self.assertEqual(identity['flow_target'],c.tensor_sha(v))
        window['observation']=obs+1
        with self.assertRaises(ValueError):audit.flow(window,noise,506)
    def test_cfg_order_and_auxiliary_four_paths(self):
        shape=(1,1,5,2,2);noise=np.ones(shape,np.float32);obs=noise[:,:,:1].copy();cmd=np.zeros((1,16,6),np.float32);opened=cmd.copy();opened[0,0,5]=1;pair=[dict(target=noise.copy(),observation=obs,commands=cmd),dict(target=noise.copy()+1,observation=obs,commands=opened)];ctx=np.ones((2,3),np.float32)
        _,_,target,identity=c.auxiliary_conditions(pair,noise,ctx,ctx);pred={a+'-'+t:np.zeros(shape,np.float32) for a in ('closed','open') for t in ('positive','negative')};pred['open-positive'][:]=-.2
        endpoint=-(c.guided(pred['open-positive'],pred['open-negative'])-c.guided(pred['closed-positive'],pred['closed-negative']));score=float(np.mean((endpoint[:,:,1:].astype('float64')-target[:,:,1:])**2));record=dict(enabled=True,**{'lambda':1.},endpoint_scale=1.,feature_extracts=2,head_predictions=4,prefix_excluded=True,target_conditioning=False,input_identity=identity,future_clean_difference_mse=score,weighted_loss=score)
        c.auxiliary_record(record,pred,pair,noise,ctx,ctx)
        record['head_predictions']=3
        with self.assertRaises(ValueError):c.auxiliary_record(record,pred,pair,noise,ctx,ctx)
    def test_combined_gradient_norm_and_poison(self):
        g={'command_gru.x':np.array([.3],np.float32),'output.x':np.array([.4],np.float32)};row=dict(gradient_l2_before_clip=.5,gradient_l2_after_clip=.5,command_gru_gradient_l2=.3,output_gradient_l2=.4);c.clipped_gradient_record(g,row)
        row['gradient_l2_after_clip']=.6
        with self.assertRaises(ValueError):c.clipped_gradient_record(g,row)
    def test_malformed_resource_counters(self):
        row=dict(seconds=1.,host_available_bytes=9*2**30,host_rss_bytes=2**30,cuda_reserved_bytes=30*2**30,cuda_allocated_bytes=29*2**30,cuda_available_bytes=40*2**30);audit.resource_sample(row)
        for key in row:
            for bad in (-1,True,float('inf')):
                changed=dict(row);changed[key]=bad
                with self.assertRaises(ValueError):audit.resource_sample(changed)
    def test_safe_relative_paths(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d).resolve();(p/'a').write_text('x');self.assertEqual(audit.file(p,'a'),p/'a')
            for name in ('../a','/tmp/a'):
                with self.assertRaises(ValueError):audit.file(p,name)
            (p/'link').symlink_to(p/'a')
            with self.assertRaises(ValueError):audit.file(p,'link')
if __name__=='__main__':unittest.main()
