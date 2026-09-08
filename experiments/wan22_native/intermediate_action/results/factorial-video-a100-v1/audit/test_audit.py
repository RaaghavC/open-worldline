"""Small corrupt-evidence fixtures. No models or CUDA imports."""
import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import audit as a

def tensor(path,values):
    header={};payload=b''
    for n,v in values.items():
        raw=np.ascontiguousarray(v).tobytes();header[n]=dict(dtype='F32' if v.dtype==np.float32 else 'I64',shape=list(v.shape),data_offsets=[len(payload),len(payload)+len(raw)]);payload+=raw
    text=json.dumps(header).encode();path.write_bytes(struct.pack('<Q',len(text))+text+payload)

class Checks(unittest.TestCase):
    def test_actual_prepared_identity_and_commands(self):
        root=a.HERE.parent/'factorial-trained-video-prepared-v1';plan=a.references()
        a.prepared(root,plan);_,identity=a.conditions(root,plan)
        self.assertEqual(len(identity['commands']),6)
        self.assertEqual(sum(len(v) for v in plan['source_sha256'].values()),75)

    def test_schedule_against_saved_original_native_times(self):
        expected=[r['timestep'] for r in a.h.jsonlines(a.HERE/'reference/native-steps-reference.jsonl')]
        times=a.schedule();self.assertEqual(times,expected);self.assertEqual(len(times),50);self.assertEqual(times[0],999)
        self.assertTrue(all(x>y for x,y in zip(times,times[1:])));self.assertEqual(times[-1],92)

    def test_commands_all_six_endpoints(self):
        for arm in a.ARMS:
            v=a.commands_expected(arm);motion,door=arm.split('_')
            self.assertEqual(v.shape,(1,16,6));self.assertEqual(v[0,0,5],int(door=='interact'))
            self.assertEqual(v[0,1:,5].sum(),0)
            self.assertAlmostEqual(float(v[0,:,3].sum()),{'left':math.pi*24/180,'right':-math.pi*24/180,'stationary':0}[motion],places=6)

    def test_safe_tensor_rejects_nonfinite_and_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';v=np.array([1.,2.],np.float32);tensor(p,{'x':v})
            a.h.array_file(p,{'x':((2,),'F32')})
            p.write_bytes(p.read_bytes()+b'x')
            with self.assertRaises(AssertionError):a.h.array_file(p,{'x':((2,),'F32')})
            v[0]=np.inf;tensor(p,{'x':v})
            with self.assertRaises(AssertionError):a.h.array_file(p,{'x':((2,),'F32')})

    def test_saved_state_label_hash_and_prefix_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);shape=(2,5,2,2);v=np.ones(shape,np.float32);obs=v[:,:1].copy();tensor(p/'step-01.safetensors',{'latent':v})
            r=dict(step=1,timestep=999,prefix_exact=True,latent_sha256=a.tensor_sha(v))
            a.state(p,1,r,obs,shape)
            for key,value in [('step',2),('timestep',998),('prefix_exact',False),('latent_sha256','0'*64)]:
                bad=dict(r,**{key:value})
                with self.assertRaises(AssertionError):a.state(p,1,bad,obs,shape)
            v[0,0,0,0]=2;tensor(p/'step-01.safetensors',{'latent':v});r['latent_sha256']=a.tensor_sha(v)
            with self.assertRaises(AssertionError):a.state(p,1,r,obs,shape)

    def test_initial_cfg_corruption(self):
        p=np.array([.3,-2.],np.float32);n=np.array([-.4,2.],np.float32)
        v=dict(positive_velocity=p,negative_velocity=n,guided_velocity=n+np.float32(5)*(p-n));a.guided(v)
        v['guided_velocity'][0]+=np.float32(.1)
        with self.assertRaises(AssertionError):a.guided(v)

    def test_sample_actual_caps_without_sequential_relation(self):
        good=dict(host_rss_bytes=100,cuda_reserved_bytes=100,cuda_allocated_bytes=101,host_available_bytes=8*2**30,cuda_available_bytes=8*2**30)
        a.worker_sample(good)
        for key,value in [('host_rss_bytes',48*2**30+1),('cuda_reserved_bytes',60*2**30+1),('host_available_bytes',8*2**30-1),('cuda_available_bytes',8*2**30-1),('cuda_allocated_bytes',True)]:
            with self.assertRaises(AssertionError):a.worker_sample(dict(good,**{key:value}))

    def test_hardware_capacity_and_other_fields(self):
        trained=dict(total_memory_bytes=80*2**30,name='GPU',runtime={'x':1});actual=dict(trained,total_memory_bytes=80*2**30+2*2**20)
        recorded=dict(policy='All fields exact except reported total GPU bytes may be greater',training_total_memory_bytes=trained['total_memory_bytes'],actual_total_memory_bytes=actual['total_memory_bytes'],additional_reported_bytes=2*2**20)
        a.hardware(actual,trained,recorded)
        for bad in (dict(actual,total_memory_bytes=70*2**30),dict(actual,total_memory_bytes=float(actual['total_memory_bytes'])),dict(actual,runtime={'x':True})):
            with self.assertRaises(AssertionError):a.hardware(bad,trained,recorded)

    def test_missing_or_changed_map_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'value').write_text('one');rows={'value':a.sha(p/'value')};a.checked_map(p,rows)
            (p/'value').write_text('two')
            with self.assertRaises(AssertionError):a.checked_map(p,rows)
            (p/'link').symlink_to(p/'value')
            with self.assertRaises(ValueError):a.checked_map(p,{'link':a.sha(p/'value')})
            with self.assertRaises(AssertionError):a.checked_map(p,{})

    def test_arm_counter_command_and_checkpoint_corruption(self):
        arm=a.ARMS[0];identity=dict(values={'noise':'x'},contexts={'text':'y'},commands={arm:'z'});adapter={'checkpoint_sha256':'a'}
        row=dict(arm=arm,commands_label=arm,profile='spatial',block_index=28,predictions=100,clean_prefix_calls=100,solver_updates=50,
            settings=dict(steps=50,shift=5.,guidance=5.),input_tensor_sha256=identity['values'],text_tensor_sha256=identity['contexts'],
            commands_sha256='z',adapter=adapter,commands_on_cfg_branches=['positive','native_negative'],negative_context_adapter_training=True,
            negative_context_training_scope='Every fourth factorial paired update uses both CFG contexts; main FM uses positive text only',
            training=False,quality_assessed=False,foundation_values_verified_by_this_kernel=False)
        a.arm_contract(row,arm,identity,adapter)
        for key,value in [('predictions',99),('commands_sha256','bad'),('adapter',{'checkpoint_sha256':'other'}),('commands_on_cfg_branches',['positive'])]:
            with self.assertRaises(AssertionError):a.arm_contract(dict(row,**{key:value}),arm,identity,adapter)

    def test_recovery_exact_run_inventory(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);root=p/'recovered/action-results/test';root.mkdir(parents=True);(root/'data').write_text('raw')
            inventory=a.h.inventory(root);idx=p/'index.json';idx.write_text(json.dumps({'files':{'action-results/test/'+k:v for k,v in inventory.items()}}))
            rec=p/'recovery-verified.json';rec.write_text(json.dumps(dict(status='verified',index_sha256=a.sha(idx))))
            a.recovery_binding(root,rec,inventory)
            (root/'data').write_text('changed')
            with self.assertRaises(AssertionError):a.recovery_binding(root,rec,a.h.inventory(root))
            idx.write_text('{}')
            with self.assertRaises(AssertionError):a.recovery_binding(root,rec,inventory)

    def test_complete_parent_children_and_bad_terminal(self):
        plan=a.references()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def save(name,row):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(row));return a.sha(p)
            decision=dict(schema=plan['schema'],scope=plan['protocol']['scope'],decision='admit',issued_by='parent-agent',
                plan_sha256=a.PLAN_SHA,source_sha256=plan['source_sha256'],training_identity=plan['training_identity'],
                training_audit_sha256=plan['training_audit_sha256'],cpu_report_sha256=plan['cpu_report_sha256'],
                protocol=plan['protocol'],minimum_lease_remaining_seconds=2400,native_control=False,training_admitted=False,
                reason='fixture',lease_deadline_utc='2026-09-08T21:00:00+00:00')
            admitted=dict(sha256=save('executed-admission.json',decision),scope=plan['protocol']['scope'],lease_deadline_utc=decision['lease_deadline_utc'])
            trained=plan['training_details']['hardware'];cap=trained['total_memory_bytes']
            comparison=dict(policy='All fields exact except reported total GPU bytes may be greater',training_total_memory_bytes=cap,actual_total_memory_bytes=cap,additional_reported_bytes=0)
            save('evidence/training/result/metrics.json',{'runtime_flags':{'fixture':True}})
            parent=dict(schema=plan['schema'],status='passed',model_execution=True,model_frozen=True,training=False,quality_assessed=False,native_control=False,
                predictions=600,solver_updates=300,frames_per_arm=17,limits=a.LIMITS,worker_limits=a.WORKER_LIMITS,plan_sha256=a.PLAN_SHA,
                source_sha256=plan['source_sha256'],elapsed_seconds=20.,admission=admitted,child_reports={},child_terminals={})
            term=dict(status='complete',exit_code=0,cleanup_error=None,plan_sha256=a.PLAN_SHA,elapsed_seconds=20.,finished_at_utc='2026-09-08T20:00:00+00:00')
            parent['terminal_sha256']=save('terminal.json',term)
            for stage in ('core','decode'):
                monitor=dict(status='complete',limits=a.WORKER_LIMITS,elapsed_seconds=1.,sample_count=1,error=None)
                output={'monitor-terminal.json':save(stage+'/result/monitor-terminal.json',monitor)}
                row=dict(status='passed',stage=stage,schema=plan['schema'],model_execution=True,training=False,future_targets_materialized=False,
                    inputs_unchanged=True,sources_unchanged=True,plan_sha256=a.PLAN_SHA,source_sha256=plan['source_sha256'],admission=admitted,
                    combined_limits=a.LIMITS,limits=a.WORKER_LIMITS,elapsed_seconds=2.,output_sha256=output,hardware=trained,hardware_comparison=comparison,
                    runtime_flags={'fixture':True},load_seconds=.2)
                parent['child_reports'][stage]=save(stage+'/result/metrics.json',row)
                terminal=dict(status='complete',exit_code=0,cleanup_error=None,error=None,limits=a.WORKER_LIMITS,elapsed_seconds=2.,peak_combined_rss_bytes=100,minimum_host_available_bytes=8*2**30)
                parent['child_terminals'][stage]=save(stage+'/terminal.json',terminal)
                launch=dict(stage=stage,plan_sha256=a.PLAN_SHA,admission=admitted,deadline=900.,parent_deadline=1800.)
                if stage=='decode':launch['core_metrics_sha256']=parent['child_reports']['core']
                save(stage+'/launch.json',launch)
                (root/stage/'worker.log').write_text('fixture only\n')
                (root/stage/'result/memory.jsonl').write_text(json.dumps(dict(host_rss_bytes=100,cuda_reserved_bytes=100,cuda_allocated_bytes=101,host_available_bytes=8*2**30,cuda_available_bytes=8*2**30))+'\n')
                (root/stage/'parent-memory.jsonl').write_text(json.dumps(dict(combined_rss_bytes=100,host_available_bytes=8*2**30))+'\n')
            save('metrics.json',parent);a.completion(root,plan)
            term['exit_code']=1;parent['terminal_sha256']=save('terminal.json',term);save('metrics.json',parent)
            with self.assertRaises(AssertionError):a.completion(root,plan)

    def test_weight_reports_bind_all_core_and_codec_values(self):
        plan=a.references();base=a.HERE.parent/'factorial-trained-video-prepared-v1/evidence/training/result'
        vae=a.read(a.HERE/'reference/vae-load-reference.json')
        codec={n:dict(shape=r['shape'],dtype='float32',sha256=r['sha256']) for n,r in vae['tensors'].items()}
        records={'core-before.json':a.read(base/'core-before.json'),'core-after.json':a.read(base/'core-after.json'),
                 'core-load':a.read(base/'weight-load.json'),'vae-load':vae,'codec-before.json':codec,'codec-after.json':copy.deepcopy(codec)}
        real=a.read
        def reader(path):
            if path.name=='weight-load.json':return records['core-load' if 'core' in path.parts else 'vae-load']
            return records[path.name] if path.name in records else real(path)
        rows=dict(core=dict(model_frozen=True,rotary_copy_exact=True,block_index=28),decode=dict(all196_unchanged=True,normalization_unchanged=True,normalization_sha256={'0':'a'*64,'1':'b'*64}))
        with patch.object(a,'read',reader):
            a.weights(Path('/fixture'),plan,rows)
            records['codec-after.json'][next(iter(codec))]['sha256']='0'*64
            with self.assertRaises(AssertionError):a.weights(Path('/fixture'),plan,rows)

if __name__=='__main__':unittest.main()
