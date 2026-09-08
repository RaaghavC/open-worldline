"""Focused gate and guarded-parent checks; no actual model or CUDA execution."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import torch
import runner as r
import extension as e

class RunnerTests(unittest.TestCase):
    def test_numeric_gate_requires_both_losses_and_finite_action(self):
        rows={'0':{'closed':2.,'open':3.},'16':{'closed':1.,'open':2.}}
        self.assertTrue(e.scalar_gate(rows,.001,.0002))
        for bad in (0.,float('nan'),float('inf'),True,-.1):
            with self.assertRaises(ValueError):e.scalar_gate(rows,bad,.01)
        rows['16']['open']=3.
        with self.assertRaises(ValueError):e.scalar_gate(rows,.01,.01)

    def test_admission_requires_cfg_review_and_parent_failure_is_single_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary).resolve();(out/'plan.json').write_text('{}')
            plan={'source_sha256':{'test':'a'},'input_identity':{},'profile':'spatial','expected_gpu':'NVIDIA A100-SXM4-80GB','cpu_report_sha256':'b','diagnostic_plan_sha256':'c'}
            measured={'parent_sha256':'p','result_sha256':'r','terminal_sha256':'t','plan_sha256':'c','source_sha256':{},'necessary_numeric_gate_passed':True,'automatic_training_admission':False}
            decision={'schema':'worldline-action-cuda-fixed128-admission-v1','decision':'admit','issued_by':'parent-agent','scope':r.SCOPE,'plan_sha256':r.sha(out/'plan.json'),'source_sha256':plan['source_sha256'],'input_identity':{},'profile':'spatial','limits':r.LIMITS,'expected_gpu':plan['expected_gpu'],'image_generation_admitted':False,'resume_admitted':False,'diagnostic_evidence':measured,'cfg_strategy_unchanged':True,'cpu_report_sha256':'b','reason':'Both fixed losses improved.','cfg_assessment':'No cancellation requiring a strategy change.'}
            path=out/'decision.json';path.write_text(json.dumps(decision));r.admission(path,out,plan,measured)
            decision['cfg_strategy_unchanged']=False;path.write_text(json.dumps(decision))
            with self.assertRaises(ValueError):r.admission(path,out,plan,measured)
            decision['cfg_strategy_unchanged']=True;path.write_text(json.dumps(decision));proc=Mock();proc.returncode=-15
            kwargs=dict(prepared=out,completed_probe=out,cache_run=out,text_directory=out,diagnostic_result=out,weights=out,decision=path)
            with patch.object(r,'read_prepared',return_value=(plan,)),patch.object(e,'validate_diagnostic',return_value=measured),patch.object(r.subprocess,'Popen',return_value=proc),patch.object(r,'supervise',side_effect=RuntimeError('mock watchdog')),patch.object(r,'stop_child') as stop:
                with self.assertRaisesRegex(RuntimeError,'mock watchdog'):r.execute(**kwargs)
                stop.assert_called_once_with(proc)
                self.assertEqual(json.loads((out/'metrics.json').read_text())['status'],'failed')
                self.assertTrue((out/'terminal.json').is_file())
                with self.assertRaises(ValueError):r.execute(**kwargs)

    def test_diagnostic_without_actual_completed_parent_is_rejected(self):
        d=e.diagnostic()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve()
            for name,value in [('plan.json',{}),('metrics.json',{'status':'prepared','model_execution':False}),('terminal.json',{})]:
                (root/name).write_text(json.dumps(value))
            (root/'result').mkdir();(root/'result/metrics.json').write_text('{}')
            with patch.object(d,'read_prepared',return_value=({},None,None,None,None,None)):
                with self.assertRaisesRegex(ValueError,'parent must complete'):e.validate_diagnostic(root,r.sha(root/'plan.json'))

    def test_chunk_coverage_no_large_duplicate_file(self):
        draws={'rng_initial':torch.zeros(3,dtype=torch.uint8)}
        for i in range(128):draws[f'noise_{i:04d}']=torch.tensor([float(i)]);draws[f'rng_after_{i:04d}']=torch.tensor([i],dtype=torch.uint8)
        parts=list(e.chunks(draws));self.assertEqual(len(parts),8)
        self.assertEqual(sum(len(v) for _,v in parts),257)
        self.assertEqual({k for _,v in parts for k in v},set(draws))
        self.assertTrue(all(len(v)<=33 for _,v in parts))

if __name__=='__main__':unittest.main()
