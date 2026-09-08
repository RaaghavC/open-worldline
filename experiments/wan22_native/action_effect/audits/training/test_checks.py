"""Small deterministic CPU failures for the independent saved-artifact reader."""
import copy,json,math,struct,tempfile,unittest,subprocess,sys
from pathlib import Path
import numpy as np
import checks

S=(1,2,5,2,4)
def fixture():
    noise=np.arange(math.prod(S),dtype=np.float32).reshape(S)/128
    observation=np.full((1,2,1,2,4),.25,np.float32)
    commands=np.zeros((1,16,6),np.float32);commands[:,1:,3]=np.float32(math.pi/24)
    windows=[{'target':np.full(S,.125,np.float32),'observation':observation.copy(),'commands':commands.copy()},
             {'target':np.full(S,.25,np.float32),'observation':observation.copy(),'commands':commands.copy()}]
    windows[1]['commands'][0,0,5]=1
    positive=np.full((2,3),.25,np.float32);negative=np.full((3,3),-.125,np.float32)
    predictions={k:np.full(S,v,np.float32)for k,v in [('closed-positive',.125),('closed-negative',.25),('open-positive',.25),('open-negative',.125)]}
    _,_,target,identity=checks.auxiliary_conditions(windows,noise,positive,negative)
    # Independent literal scalar oracle: Gc=-.375; Go=.75; predicted difference=-1.125.
    loss=(-1.125-.125)**2
    record={'enabled':True,'lambda':1.,'endpoint_scale':1.,'feature_extracts':2,'head_predictions':4,'prefix_excluded':True,'target_conditioning':False,'input_identity':identity,'future_clean_difference_mse':loss,'weighted_loss':loss}
    return windows,noise,positive,negative,predictions,record

def write_tensor(path,value,metadata=None):
    dtype={np.dtype('float32'):'F32',np.dtype('int64'):'I64',np.dtype('uint8'):'U8'}[value.dtype]
    h={'value':{'dtype':dtype,'shape':list(value.shape),'data_offsets':[0,value.nbytes]}}
    if metadata is not None:h['__metadata__']=metadata
    raw=json.dumps(h,separators=(',',':')).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw+value.tobytes())

def count_fixture():
    schedule=[];steps=[]
    for i in range(1,129):
        start=(0,8,32,49)[(i-1)%4];schedule.append({'start':start})
        aux={'enabled':False,'lambda':1.,'feature_extracts':0,'head_predictions':0,'weighted_loss':0.}
        if start==0:aux={'enabled':True,'lambda':1.,'endpoint_scale':1.,'feature_extracts':2,'head_predictions':4,'weighted_loss':.25}
        steps.append({'update':i,'start':start,'optimizer_updates':1,'live_sequential_forwards':2,'auxiliary':aux,'paired_mean_future_flow_mse':.5,'total_objective':.5+aux['weighted_loss']})
    result={'updates':steps,'main_feature_forwards':256,'main_head_predictions':256,'total_training_feature_forwards':320,'total_training_head_predictions':384,'auxiliary_feature_extracts':64,'auxiliary_head_predictions':128,'parity_native_forwards':2,'parity_bridge_forwards':2,'negative_context_adapter_training':True}
    return result,schedule

class Checks(unittest.TestCase):
    def test_four_path_cfg_endpoint_oracle(self):
        w,n,p,m,pred,r=fixture();out=checks.auxiliary_record(r,pred,w,n,p,m)
        self.assertEqual(out['future_mse_fp64'],1.5625)
        expected=np.full(S,-1.125,np.float32)
        self.assertEqual(out['predicted_endpoint_difference_sha256'],checks.tensor_sha(expected))

    def test_future_only_loss_and_target_not_in_common_input(self):
        w,n,p,m,pred,r=fixture();before=copy.deepcopy(w);a=checks.auxiliary_conditions(w,n,p,m)
        w[1]['target'][:,:,1:]+=2;w[0]['target'][:,:,:1]+=7;b=checks.auxiliary_conditions(w,n,p,m)
        self.assertTrue(np.array_equal(a[0],b[0]));self.assertTrue(np.array_equal(a[1],b[1]));self.assertFalse(np.array_equal(a[2],b[2]))
        self.assertTrue(np.array_equal(a[0][:,:,:1],w[0]['observation']))
        pred['closed-positive'][:,:,:1]=10000;pred['open-negative'][:,:,:1]=-5000
        self.assertEqual(checks.auxiliary_record(r,pred,before,n,p,m)['future_mse_fp64'],1.5625)
        self.assertTrue(np.array_equal(n,np.arange(math.prod(S),dtype=np.float32).reshape(S)/128))

    def test_wrong_path_sign_missing_head_or_identity_rejected(self):
        w,n,p,m,pred,r=fixture()
        wrong=copy.deepcopy(pred);wrong['open-positive'],wrong['closed-positive']=wrong['closed-positive'],wrong['open-positive']
        with self.assertRaises(ValueError):checks.auxiliary_record(r,wrong,w,n,p,m)
        wrong=copy.deepcopy(pred);del wrong['open-negative']
        with self.assertRaises(ValueError):checks.auxiliary_record(r,wrong,w,n,p,m)
        rr=copy.deepcopy(r);rr['input_identity']['times_sha256']='0'*64
        with self.assertRaises(ValueError):checks.auxiliary_record(rr,pred,w,n,p,m)
        w[1]['commands'][0,2,0]=1
        with self.assertRaises(ValueError):checks.auxiliary_conditions(w,n,p,m)

    def test_auxiliary_scalar_four_path_derivatives(self):
        # Exact dyadic finite differences check all four signed CFG coefficients.
        w,n,p,m,pred,r=fixture();coeff={'closed-positive':5,'closed-negative':-4,'open-positive':-5,'open-negative':4};eps=2**-10
        def objective(values):
            d=-(checks.guided(values['open-positive'],values['open-negative'])-checks.guided(values['closed-positive'],values['closed-negative']))
            return float(np.mean((d[:,:,1:].astype(np.float64)-.125)**2))
        for key,c in coeff.items():
            lo=copy.deepcopy(pred);hi=copy.deepcopy(pred);lo[key][:,:,1:]-=eps;hi[key][:,:,1:]+=eps
            self.assertAlmostEqual((objective(hi)-objective(lo))/(2*eps),2*(-1.25)*c,places=9)

    def test_all128_schedule_and_exact_call_counts(self):
        report,schedule=count_fixture();self.assertEqual(checks.schedule_counts(report,schedule),list(range(1,129,4)))
        for mutate in [lambda r:r['updates'].pop(),lambda r:r['updates'][2].update(optimizer_updates=2),lambda r:r.update(auxiliary_head_predictions=124),lambda r:r.update(negative_context_adapter_training=False),lambda r:r['updates'][4]['auxiliary'].update(enabled=False),lambda r:r['updates'][1]['auxiliary'].update(weighted_loss=.1)]:
            wrong=copy.deepcopy(report);mutate(wrong)
            with self.assertRaises(ValueError):checks.schedule_counts(wrong,schedule)
        wrong=copy.deepcopy(schedule);wrong[-1]['start']=0
        with self.assertRaises(ValueError):checks.schedule_counts(report,wrong)

    def test_loss_record_corruption_rejected(self):
        a=np.full(S,.25,np.float32);b=np.zeros(S,np.float32)
        self.assertEqual(checks.loss_record(a,b,.0625)['future_mse_fp64'],.0625)
        for value in [float('nan'),float('inf'),-.1,True,.07]:
            with self.assertRaises(ValueError):checks.loss_record(a,b,value)

    def test_gradient_sum_one_clip_norms(self):
        # Combined main+aux vector [1,2] is clipped once, not each component separately.
        before=math.sqrt(5);scale=1/(before+1e-6)
        values={'command_gru.weight':np.array([scale],np.float32),'output.weight':np.array([2*scale],np.float32)}
        row={'gradient_l2_before_clip':before,'gradient_l2_after_clip':checks.norm(np.array([scale,2*scale])),'command_gru_gradient_l2':1.,'output_gradient_l2':2.}
        self.assertLess(checks.clipped_gradient_record(values,row)['saved_post_clip_l2_fp64'],1)
        bad=copy.deepcopy(values);bad['output.weight']*=.5
        with self.assertRaises(ValueError):checks.clipped_gradient_record(bad,row)
        bad=copy.deepcopy(values);bad['output.weight'][0]=np.nan
        with self.assertRaises(ValueError):checks.clipped_gradient_record(bad,row)

    def test_safe_reader_rejects_metadata_nan_missing_and_trailing(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t)/'tiny.safetensors';v=np.array([1.,2.],np.float32);write_tensor(p,v,{'role':'fixture'})
            self.assertTrue(np.array_equal(checks.arrays(p,['value'])['value'],v))
            with self.assertRaises(ValueError):checks.arrays(p,['other'])
            write_tensor(p,v,{'role':1})
            with self.assertRaises(ValueError):checks.arrays(p)
            write_tensor(p,np.array([np.nan],np.float32))
            with self.assertRaises(ValueError):checks.arrays(p)
            write_tensor(p,v);p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaises(ValueError):checks.arrays(p)
            write_tensor(p,v);link=Path(t)/'link';link.symlink_to(p)
            with self.assertRaises(ValueError):checks.arrays(link)

    def test_failed_preflight_retained_and_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t)
            args=[]
            for name in ('repo','runner-source','recovery','run-root','probe-root','cache-root','diagnostic-root'):
                path=root/name;path.mkdir();args.extend(['--'+name,str(path)])
            output=root/'failure-output';command=[sys.executable,str(Path(__file__).with_name('audit.py')),*args,'--output',str(output)]
            first=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(first.returncode,0)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['schema'],'worldline-action-effect128-actual-independent-v1')
            self.assertEqual(report['status'],'failed');self.assertFalse(report['cuda_initialized']);self.assertFalse(report['model_execution'])
            original=(output/'report.json').read_bytes()
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(second.returncode,0);self.assertEqual((output/'report.json').read_bytes(),original)

    def test_header_duplicate_offset_and_bound_rejected_before_payload(self):
        with tempfile.TemporaryDirectory()as t:
            p=Path(t)/'x';h=b'{"x":{},"x":{}}';p.write_bytes(struct.pack('<Q',len(h))+h)
            with self.assertRaises(ValueError):checks.arrays(p)
            p.write_bytes(struct.pack('<Q',65537)+b'{}')
            with self.assertRaises(ValueError):checks.arrays(p)
            h=json.dumps({'x':{'dtype':'F32','shape':[1],'data_offsets':[4,8]}}).encode();p.write_bytes(struct.pack('<Q',len(h))+h+b'12345678')
            with self.assertRaises(ValueError):checks.arrays(p)

if __name__=='__main__':unittest.main(verbosity=2)
