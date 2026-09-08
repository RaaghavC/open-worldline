"""Small CPU fixtures; no native weights, model execution, or cloud actions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import torch
import inputs as inp
import run


class VisualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory(prefix='visual-cpu-');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()

    def conditions(self):
        obs=torch.ones(1,2,1,4,4);commands={}
        for arm in ('closed','open'):
            c=torch.zeros(1,16,6);c[:,1:,3]=torch.pi/24;c[:,0,5]=int(arm=='open');commands[arm]=c
        noise=torch.randn(2,5,4,4);initial=noise.clone();initial[:,:1]=obs[0]
        return {'initial_noise':noise,'initial_latent':initial,'observation':obs,'token_times':torch.zeros(1,20,dtype=torch.int64)}, {'atrium':torch.ones(25,4096),'native_negative':torch.zeros(126,4096)},commands

    def test_only_declared_first_interaction_differs(self):
        values,_,commands=self.conditions()
        records={a:{'observation':values['observation'].clone(),'commands':c.clone()} for a,c in commands.items()}
        inp.validate_commands(records)
        self.assertEqual((commands['open']!=commands['closed']).nonzero().tolist(),[[0,0,5]])
        records['open']['commands'][0,1,3]=0
        with self.assertRaises(ValueError):inp.validate_commands(records)
        records['open']['commands']=commands['open'];records['open']['observation'][0,0,0,0,0]+=1
        with self.assertRaises(ValueError):inp.validate_commands(records)

    def fake_sampler(self,calls,fail_second=False):
        def sample(core,checkpoint,checksum,values,contexts,commands,*,profile,event,check):
            calls.append({'values':copy.deepcopy(values),'contexts':copy.deepcopy(contexts),'commands':commands.clone()})
            if fail_second and len(calls)==2:raise RuntimeError('second-arm fixture failure')
            state=values['initial_latent'].clone()
            for i,t in enumerate(run.sampler.native_sampling.scheduler().timesteps):
                state[:,1:]+=float(commands[0,0,5])*.001
                event(i,t,state.clone(),{k:torch.zeros_like(state) for k in ('positive_velocity','negative_velocity','guided_velocity')});check()
            values['initial_noise'].zero_();contexts['atrium'].zero_();commands.zero_()
            return state,{'predictions':100,'solver_updates':50}
        return sample

    def test_matched_private_inputs_and_all50_states(self):
        values,contexts,commands=self.conditions();original=copy.deepcopy((values,contexts,commands));calls=[]
        out=self.root/'core';out.mkdir()
        with patch.object(run.sampler,'sample_checkpoint',side_effect=self.fake_sampler(calls)):
            records=run.sample_arms(object(),out,self.root/'adapter','a'*64,values,contexts,commands,lambda:None)
        self.assertEqual(set(records),{'closed','open'})
        for name in values:self.assertTrue(torch.equal(calls[0]['values'][name],calls[1]['values'][name]))
        for name in contexts:self.assertTrue(torch.equal(calls[0]['contexts'][name],calls[1]['contexts'][name]))
        self.assertEqual((calls[0]['commands']!=calls[1]['commands']).nonzero().tolist(),[[0,0,5]])
        for arm in ('closed','open'):
            self.assertEqual(len(list((out/arm).glob('step-*.safetensors'))),50)
            rows=[json.loads(s) for s in (out/arm/'steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['timestep'] for r in rows],[int(t) for t in run.sampler.native_sampling.scheduler().timesteps])
            self.assertTrue(all(r['prefix_exact'] for r in rows))
        for actual,before in zip((values,contexts,commands),original):self.assertTrue(all(torch.equal(v,before[k]) for k,v in actual.items()))

    def test_second_arm_failure_retains_first_trajectory(self):
        values,contexts,commands=self.conditions();out=self.root/'core';out.mkdir()
        with patch.object(run.sampler,'sample_checkpoint',side_effect=self.fake_sampler([],True)):
            with self.assertRaisesRegex(RuntimeError,'second-arm'):
                run.sample_arms(object(),out,self.root/'adapter','a'*64,values,contexts,commands,lambda:None)
        self.assertTrue((out/'closed/latents.safetensors').is_file());self.assertFalse((out/'open/latents.safetensors').exists())

    def decision(self,out):
        inp.atomic(out/'plan.json',{'fixture':True})
        plan={'source_sha256':{'fixture':True},'input_identity':{'checkpoint':'128'},'artifacts':{'adapter':'sha'},'settings':inp.SETTINGS,'cpu_report_sha256':'a'*64,'training_audit_sha256':'b'*64}
        value={'schema':'worldline-action-cuda-visual-admission-v1','decision':'admit','issued_by':'parent-agent',
            'scope':inp.SCOPE,'plan_sha256':inp.sha(out/'plan.json'),'source_sha256':plan['source_sha256'],'input_identity':plan['input_identity'],
            'artifacts':plan['artifacts'],'limits':run.limits('clip'),'settings':plan['settings'],'profile':'spatial','arms':['closed','open'],
            'training_admitted':False,'final128_audit_sha256':plan['training_audit_sha256'],'cpu_report_sha256':'a'*64,'reason':'CPU fixture only'}
        path=self.root/'decision.json';inp.atomic(path,value);return plan,path,value

    def test_exact_plan_checkpoint_and_audit_admission(self):
        out=self.root/'prepared';out.mkdir();plan,path,value=self.decision(out)
        self.assertEqual(run.admit(path,out,plan)['sha256'],inp.sha(path))
        for field,replacement in [('final128_audit_sha256','0'*64),('input_identity',{'checkpoint':'2'}),('training_admitted',True)]:
            changed=copy.deepcopy(value);changed[field]=replacement;inp.atomic(path,changed)
            with self.assertRaises(ValueError):run.admit(path,out,plan)

    def test_core_failure_prevents_decode_and_refuses_reuse(self):
        out=self.root/'prepared';out.mkdir();plan,decision,_=self.decision(out)
        with patch.object(inp,'read_prepared',return_value=(plan,)),patch.object(run,'launch',side_effect=RuntimeError('core stopped')) as child:
            with self.assertRaisesRegex(RuntimeError,'core stopped'):run.execute(out,self.root/'no-weights',decision)
            self.assertEqual(child.call_count,1);self.assertEqual(child.call_args.args[0]['stage'],'core')
            with self.assertRaisesRegex(ValueError,'single-use'):run.execute(out,self.root/'no-weights',decision)
        self.assertEqual(inp.read(out/'metrics.json')['status'],'failed');self.assertFalse((out/'decode').exists())

    def test_interrupted_handoff_stops_child_and_retains_terminal(self):
        out=self.root/'prepared';out.mkdir();proc=Mock();proc.returncode=-15
        config={'prepared':str(out),'stage':'core','deadline':run.time.monotonic()+100.,'plan_sha256':'f'*64}
        with patch.object(run.subprocess,'Popen',return_value=proc),patch.object(run,'supervise',side_effect=KeyboardInterrupt()),patch.object(run,'stop_child') as stop:
            with self.assertRaises(KeyboardInterrupt):run.launch(config)
            stop.assert_called_once_with(proc)
        self.assertEqual(inp.read(out/'core/terminal.json')['status'],'failed')

    def test_expired_deadline_prevents_popen(self):
        out=self.root/'prepared';out.mkdir()
        with patch.object(run.subprocess,'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError,'before child'):
                run.launch({'prepared':str(out),'stage':'core','deadline':run.time.monotonic()-1})
            popen.assert_not_called()
        self.assertEqual(inp.read(out/'core/terminal.json')['status'],'failed')

    def test_hardware_allows_only_equal_or_greater_reported_capacity(self):
        trained={'name':'NVIDIA A100-SXM4-80GB','total_memory_bytes':85093777408,
                 'capability':[8,0],'bf16_supported':True,'dependencies':{'torch':'2.5.1+cu124'},
                 'environment':{'CUDA_VISIBLE_DEVICES':None}}
        self.assertEqual(run.require_hardware(trained,trained)['additional_reported_bytes'],0)
        larger=copy.deepcopy(trained);larger['total_memory_bytes']+=2097152
        self.assertEqual(run.require_hardware(larger,trained)['additional_reported_bytes'],2097152)
        for value in (trained['total_memory_bytes']-1,True,float(trained['total_memory_bytes'])):
            changed=copy.deepcopy(trained);changed['total_memory_bytes']=value
            with self.assertRaises(ValueError):run.require_hardware(changed,trained)
        for key,value in [('name','different'),('capability',[9,0]),('bf16_supported',1),
                          ('dependencies',{'torch':'2.6.0'}),('environment',{'CUDA_VISIBLE_DEVICES':'0'})]:
            changed=copy.deepcopy(larger);changed[key]=value
            with self.assertRaises(ValueError):run.require_hardware(changed,trained)
        changed=copy.deepcopy(larger);changed['new_field']=None
        with self.assertRaises(ValueError):run.require_hardware(changed,trained)
        self.assertEqual(run.limits('clip')['seconds'],1800.)
        self.assertEqual(run.limits('clip')['cuda_reserved_bytes'],60*2**30)
        self.assertEqual(run.limits('clip')['minimum_cuda_available_bytes'],8*2**30)
        self.assertEqual(run.limits('clip')['minimum_gpu_total_bytes'],70*2**30)

    def test_final128_audit_binds_checkpoint_and_complete_source(self):
        trained={k:chr(97+i)*64 for i,k in enumerate(('parent_sha256','worker_sha256','training_sha256','terminal_sha256','plan_sha256','checkpoint_manifest_sha256','checkpoint_sha256'))}
        identity={k:trained[k] for k in ('parent_sha256','worker_sha256','training_sha256','terminal_sha256','plan_sha256')}
        identity.update(final_checkpoint_manifest_sha256=trained['checkpoint_manifest_sha256'],final_checkpoint_sha256=trained['checkpoint_sha256'],source_sha256=inp.TRAINING_SOURCES)
        report={'schema':'worldline-action-cuda-fixed128-actual-independent-v1','status':'passed','completed_updates':128,'foundation_values_unchanged':True,'final_checkpoint_only':True,'identity':identity}
        path=self.root/'audit.json';inp.atomic(path,report);inp.validate_training_audit(path,trained)
        for key,value in [('completed_updates',127),('foundation_values_unchanged',False),('status','running')]:
            bad=copy.deepcopy(report);bad[key]=value;inp.atomic(path,bad)
            with self.assertRaises(ValueError):inp.validate_training_audit(path,trained)
        for key in ('final_checkpoint_sha256','source_sha256','terminal_sha256'):
            bad=copy.deepcopy(report);bad['identity'][key]='changed';inp.atomic(path,bad)
            with self.assertRaises(ValueError):inp.validate_training_audit(path,trained)

    def test_training_eligibility_rejects_prior16_and_incomplete128_before_loading(self):
        base=self.root/'training';base.mkdir()
        rows={'plan.json':{},'metrics.json':{'schema':'worldline-action-cuda-fixed16-run-v1','status':'passed','completed_updates':16,'model_execution':True},'worker/metrics.json':{},'training/metrics.json':{},'terminal.json':{},'training/last-valid.json':{}}
        with patch.object(inp,'file',side_effect=lambda b,n:Path(b)/n),patch.object(inp,'read',side_effect=lambda path:rows[str(Path(path).relative_to(base))]),patch.object(inp.sampler,'load_adapter_checkpoint') as load:
            with self.assertRaisesRegex(ValueError,'Completed fresh fixed128'):inp.training(base)
            rows['metrics.json'].update(schema='worldline-action-cuda-fixed128-run-v1',completed_updates=127)
            with self.assertRaisesRegex(ValueError,'Completed fresh fixed128'):inp.training(base)
            load.assert_not_called()


if __name__=='__main__':unittest.main()
