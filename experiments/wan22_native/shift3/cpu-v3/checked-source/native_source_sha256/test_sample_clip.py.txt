# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU contracts for the measured native clip runner; no real weights."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file
from . import sample_clip as s
from .profile_core import prepare_input


def inputs():
    observation = torch.linspace(-.4, .4, 48*18*32).reshape(1,48,1,18,32)
    noise, latent, times = prepare_input(observation, s.make_scheduler().timesteps[0])
    return dict(initial_noise=noise, initial_latent=latent, observation=observation, token_times=times)


def fixture(root):
    """Explicit fake timing metadata around real-shape synthetic input tensors."""
    pair_dir, decode_dir = root/"pair", root/"decode"
    pair_dir.mkdir(); decode_dir.mkdir()
    values = inputs()
    save_file(values, str(pair_dir/"inputs.safetensors"))
    save_file({"placeholder": torch.zeros(1)}, str(pair_dir/"outputs.safetensors"))
    prov = json.loads((s.HERE/"provenance.json").read_text())
    weight = dict(revision=prov["weight_revision"], code_commit=prov["code_commit"], repository=prov["repository"],
        parameter_count=4_999_787_712, storage={"policy":"selective_bf16", "parameters":{"float32":68_573_376,"bfloat16":4_931_214_336}},
        input_files={row["file"]:row for row in prov["weights"]},
        config_sha256=s.sha256(s.HERE/"config.json"), index_sha256=s.sha256(s.HERE/"weights.index.json"))
    (pair_dir/"weight-load.json").write_text(json.dumps(weight))
    text_dir = s.REPO/"experiments/wan_adapter/text_cache/native-results"
    _, _, text = s.load_text(text_dir)
    observation = {"observation_tensor_sha256":s.tensor_sha256(values["observation"]),
        "codec_report_sha256":"codec-report-fixture", "cache_sha256":"codec-cache-fixture"}
    pair = {"status":"passed", "device":"cpu", "sampling_configuration":s.SETTINGS, "seed":s.SEED,
        "runtime_environment":s.runtime_environment("cpu"),
        "latent_shape":list(s.SHAPE), "tokens":720, "observed_tokens":144, "finite_outputs":True,
        "observed_prefix_exact":True, "source_sha256":{name:s.sha256(s.HERE/name) for name in s.CORE_SOURCES},
        "reused_source_sha256":{name:s.sha256(s.REPO/name) for name in s.REUSED_NAMES},
        "output_sha256":{name:s.sha256(pair_dir/name) for name in ("inputs.safetensors","outputs.safetensors","weight-load.json")},
        "weight_load_sha256":s.sha256(pair_dir/"weight-load.json"), "observation":observation, "text":text,
        "pair_seconds":2., "elapsed_seconds":7.3,
        "timings":[{"stage":name,"seconds":seconds} for name,seconds in [
            ("verify and stream official transformer",5.), ("positive native forward",1.),
            ("negative native forward",1.), ("guided velocity transfer to CPU solver",.1),
            ("one official CPU UniPC step",.01)]]}
    (decode_dir/"diagnostic.txt").write_text("Synthetic CPU fixture; no real decoder run.")
    decode = {"status":"passed", "device":"cpu", "finite_output":True, "dtype":"float32",
        "runtime_environment":s.runtime_environment("cpu"),
        "latent_shape":[1,*s.SHAPE], "decoded_shape":[1,3,17,288,512], "cache_clear_after_decode":True,
        "automatic_mps_cpu_fallback":"0", "denoiser_executed":False, "future_rgb_read":False,
        "actions_read":False, "target_read":False, "optional_allocator_cleanup":True,
        "per_convolution_cleanup":False,
        "allocator_chunk_counts":{"encoder":0,"decoder":5},
        "source_sha256":{name:s.sha256(s.HERE/name) for name in s.DECODE_SOURCES},
        "codec":{"weight_sha256":s.WEIGHT_SHA256,"source_sha256":s.SOURCE_SHA256,"config":s.CONFIG},
        "observation":{"observation_tensor_sha256":observation["observation_tensor_sha256"],
            "first_image_metrics_sha256":observation["codec_report_sha256"],"observation_file_sha256":observation["cache_sha256"]},
        "decoder_chunks":[{"first_chunk":i==0,"output_shape":[1,12,1 if i==0 else 4,144,256],"output_dtype":"torch.float32"} for i in range(5)],
        "output_sha256":{"diagnostic.txt":s.sha256(decode_dir/"diagnostic.txt")}, "elapsed_seconds":2.,
        "timings":[{"stage":"verify_and_load_fp32_codec","seconds":.3},{"stage":"decode_full_17_frames","seconds":1.6}]}
    (pair_dir/"metrics.json").write_text(json.dumps(pair)); (pair_dir/"terminal.json").write_text(json.dumps({"status":"complete","exit_code":0}))
    (decode_dir/"metrics.json").write_text(json.dumps(decode))
    return pair_dir, decode_dir, text_dir, pair, decode


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_saved_noise_and_valid_profile_identity_without_rng_regeneration(self):
        with tempfile.TemporaryDirectory() as temp:
            pair_dir,decode_dir,text_dir,_,_ = fixture(Path(temp))
            with mock.patch.object(torch,"randn",side_effect=AssertionError("No seed regeneration")):
                values,evidence,_,_ = s.validate_profiles(pair_dir,decode_dir,text_dir,"cpu")
            self.assertEqual(set(values),s.RUN_KEYS)
            self.assertEqual(evidence["pair_inputs_sha256"],s.sha256(pair_dir/"inputs.safetensors"))
            self.assertEqual(values["positive"].shape,(25,4096));self.assertEqual(values["negative"].shape,(126,4096))

    def test_reader_rejects_targets_before_materialization_and_bad_prefix(self):
        accessed=[]
        class Poisoned:
            def __init__(self,*a,**kw):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def keys(self):return sorted(s.INPUT_KEYS|{"target"})
            def get_tensor(self,key):accessed.append(key);raise AssertionError("Must reject header first")
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"file.safetensors";path.write_bytes(b"header fixture")
            with mock.patch.object(s,"safe_open",Poisoned):
                with self.assertRaises(ValueError):s.read_tensors(path,s.INPUT_KEYS)
        self.assertEqual(accessed,[])
        values=inputs();values["initial_latent"][0,0,0,0] += 1
        with self.assertRaises(ValueError):s.tensor_contract(values)

    def test_completed_profiles_source_timing_and_terminal_required(self):
        with tempfile.TemporaryDirectory() as temp:
            pair_dir,decode_dir,text_dir,pair,decode=fixture(Path(temp))
            for status,code in (("failed",0),("complete",124),("running",None)):
                (pair_dir/"terminal.json").write_text(json.dumps({"status":status,"exit_code":code}))
                with self.assertRaises(ValueError):s.validate_profiles(pair_dir,decode_dir,text_dir,"cpu")
            (pair_dir/"terminal.json").write_text(json.dumps({"status":"complete","exit_code":0}))
            for field,value in (("status","failed"),("sampling_configuration",{"steps":20,"shift":5.,"guidance":5.}),("pair_seconds",float("nan"))):
                changed=copy.deepcopy(pair);changed[field]=value;(pair_dir/"metrics.json").write_text(json.dumps(changed))
                with self.assertRaises(ValueError):s.validate_profiles(pair_dir,decode_dir,text_dir,"cpu")
            changed=copy.deepcopy(pair);changed["source_sha256"]["portable.py"]="stale"
            (pair_dir/"metrics.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):s.validate_profiles(pair_dir,decode_dir,text_dir,"cpu")
            (pair_dir/"metrics.json").write_text(json.dumps(pair))
            (pair_dir/"watchdog-stop.json").write_text("{}");
            with self.assertRaises(ValueError):s.validate_profiles(pair_dir,decode_dir,text_dir,"cpu")
            (pair_dir/"watchdog-stop.json").unlink()
            changed=copy.deepcopy(decode);changed["decoder_chunks"]=changed["decoder_chunks"][:1]
            (decode_dir/"metrics.json").write_text(json.dumps(changed))
            with self.assertRaises(ValueError):s.validate_profiles(pair_dir,decode_dir,text_dir,"cpu")

    def test_budget_includes_all_costs_and_rejects_nonfinite_or_over_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            *_,pair,decode=fixture(Path(temp))
        estimate=s.timing_gate(pair,decode,.2,.5)
        self.assertAlmostEqual(estimate["total_estimate_seconds"],5+50*2+50*.3+.2+2+45+.5)
        for bad in (float("nan"),float("inf"),0,-1):
            with self.assertRaises(ValueError):s.timing_gate(pair,decode,bad,.5)
        expensive=copy.deepcopy(pair);expensive["elapsed_seconds"]=30.
        with self.assertRaisesRegex(ValueError,"exceeds"):s.timing_gate(expensive,decode,.2,.5)
        self.assertEqual((s.STEPS,s.SHIFT,s.GUIDANCE),(50,5.,5.))

    def test_core_and_decoder_environment_are_bound_separately_before_start(self):
        base={name:None for name in s.RUNTIME_VARIABLES};base['PYTORCH_ENABLE_MPS_FALLBACK']='0'
        core={"environment":base,"recommended_max_memory_bytes":None}
        decode=copy.deepcopy(core);decode['environment']['PYTORCH_MPS_LOW_WATERMARK_RATIO']='0.6'
        s.runtime_identity(core,'cpu');s.runtime_identity(decode,'cpu')
        with mock.patch.dict(s.os.environ,{'PYTORCH_MPS_LOW_WATERMARK_RATIO':'1.1','PYTORCH_MPS_FAST_MATH':'1'},clear=True):
            before=dict(s.os.environ);core_env=s.child_environment(core);decode_env=s.child_environment(decode)
            self.assertNotIn('PYTORCH_MPS_LOW_WATERMARK_RATIO',core_env)
            self.assertNotIn('PYTORCH_MPS_FAST_MATH',core_env)
            self.assertEqual(decode_env['PYTORCH_MPS_LOW_WATERMARK_RATIO'],'0.6')
            self.assertEqual(dict(s.os.environ),before)
        bad=copy.deepcopy(core);bad['environment']['PYTORCH_ENABLE_MPS_FALLBACK']='1'
        with self.assertRaises(ValueError):s.runtime_identity(bad,'cpu')

    def test_cleanup_policy_requires_completed_layer_coverage(self):
        report={'per_convolution_cleanup':True,'per_convolution_cleanup_report':{
            'cleanup_calls':2,'completed_cleanups':2,'hooks_removed':True,'registered_layers':1,
            'layers':{'conv2':{'calls':2,'completed_cleanups':2}},'events':[{'status':'passed'},{'status':'passed'}]}}
        self.assertEqual(s.cleanup_identity(report),{'enabled':True,'layer_calls':{'conv2':2}})
        for field,value in (('completed_cleanups',1),('hooks_removed',False),('registered_layers',2)):
            bad=copy.deepcopy(report);bad['per_convolution_cleanup_report'][field]=value
            with self.assertRaises(ValueError):s.cleanup_identity(bad)
        self.assertEqual(s.cleanup_identity({'per_convolution_cleanup':False}),{'enabled':False,'layer_calls':None})

    def test_constant_velocity_sign_all_100_boundaries_and_caller_storage(self):
        values=inputs();noise=values["initial_noise"];obs=values["observation"]
        before_noise=noise.clone();before_obs=obs.clone();calls=[];previous=[None]
        positive=torch.ones(1,1);negative=torch.zeros(1,1)
        schedule=s.make_scheduler()
        def model(latents,times,contexts,seq_len):
            step=len(calls)//2;which=len(calls)%2
            self.assertEqual(seq_len,720);self.assertEqual(times.dtype,torch.int64)
            self.assertTrue(torch.equal(latents[0][:,:1],obs[0]))
            self.assertTrue(torch.equal(times[:,:144],torch.zeros(1,144,dtype=torch.int64)))
            self.assertTrue((times[:,144:]==schedule.timesteps[step]).all())
            self.assertEqual(float(contexts[0][0,0]),1. if which==0 else 0.)
            if which==0:previous[0]=latents[0].clone()
            else:self.assertTrue(torch.equal(previous[0],latents[0]))
            calls.append((step,which))
            return [torch.full_like(latents[0],2. if which==0 else 1.)]
        output,rows=s.integrate(model,noise,obs,positive,negative)
        self.assertEqual(len(calls),100);self.assertEqual(len(rows),50)
        self.assertTrue(all(r["post_step_prefix_exact"] for r in rows))
        self.assertTrue(torch.equal(output[:,:1],obs[0]))
        native_interval=float(schedule.sigmas[0]-schedule.sigmas[-1])
        torch.testing.assert_close(output[:,1:],noise[:,1:]-native_interval*6.,atol=4e-5,rtol=4e-6)
        self.assertTrue(torch.equal(noise,before_noise));self.assertTrue(torch.equal(obs,before_obs))

    def test_state_dependent_solver_matches_independent_projected_loop(self):
        values=inputs();noise=values["initial_noise"];obs=values["observation"]
        positive=torch.ones(1,1);negative=torch.zeros(1,1)
        def model(latents,times,contexts,seq_len):
            return [latents[0]*.125+float(contexts[0][0,0])*.02]
        actual,_=s.integrate(model,noise,obs,positive,negative)
        expected=noise.clone();scheduler=s.make_scheduler()
        for timestep in scheduler.timesteps:
            expected[:,:1]=obs[0]
            pos=expected*.125+.02;neg=expected*.125
            expected=scheduler.step((neg+5*(pos-neg))[None],timestep,expected[None],return_dict=False)[0][0]
            expected[:,:1]=obs[0]
        self.assertTrue(torch.equal(actual,expected))
        with self.assertRaises(FloatingPointError):
            s.integrate(lambda x,*a:[torch.full_like(x[0],float("inf"))],noise,obs,positive,negative)

    def test_interrupt_records_failed_child_and_no_passed_latent(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);values=inputs();values.update(positive=torch.ones(25,4096),negative=torch.zeros(126,4096))
            save_file(values,str(root/"inputs.safetensors"))
            config={"source_sha256":{name:s.sha256(s.HERE/name) for name in s.SOURCES},
                "reused_source_sha256":{name:s.sha256(s.REPO/name) for name in s.REUSED_NAMES},
                "device":"cpu","output":str(root/"result"),"stage":"core","run":str(root),
                "runtime_environment":s.runtime_identity(s.runtime_environment('cpu'),'cpu'),
                "input_sha256":s.sha256(root/"inputs.safetensors"),"deadline":time.monotonic()+900,"weights":"unused"}
            with mock.patch.object(s,"load_core",side_effect=KeyboardInterrupt("test only")):
                with self.assertRaises(KeyboardInterrupt):s.child(config)
            report=json.loads((root/"result/metrics.json").read_text())
            self.assertEqual(report["status"],"failed");self.assertEqual(report["error_type"],"KeyboardInterrupt")
            self.assertFalse((root/"result/latents.safetensors").exists())

    def test_preflight_rejects_stale_source_and_output_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/"report.json"
            report={"status":"passed","tests":8,"source_sha256":{name:s.sha256(s.HERE/name) for name in s.SOURCES},
                "reused_source_sha256":{name:s.sha256(s.REPO/name) for name in s.REUSED_NAMES}}
            path.write_text(json.dumps(report));s.validate_cpu(path)
            report["source_sha256"]["sample_clip.py"]="stale";path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):s.validate_cpu(path)
            with self.assertRaises(FileExistsError):s.new_directory(root)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    if args.output.exists():parser.error("Evidence output must be new")
    args.output.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    report={"status":"passed" if result.wasSuccessful() else "failed","tests":result.testsRun,
        "errors":len(result.errors),"failures":len(result.failures),"seconds":time.monotonic()-started,
        "device":"cpu","external_weight_values_loaded":False,"gpu_executed":False,
        "source_sha256":{name:s.sha256(s.HERE/name) for name in s.SOURCES},
        "reused_source_sha256":{name:s.sha256(s.REPO/name) for name in s.REUSED_NAMES},
        "scope":"Synthetic real-shape tensors, flow oracles, genuine retained text, fake timing metadata and lifecycle failures; not actual model throughput or generated quality"}
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=="__main__":main()
