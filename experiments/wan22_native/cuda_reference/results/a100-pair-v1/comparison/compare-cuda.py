"""Saved CUDA/CPU/MPS pair comparison. NumPy only; no GPU/model/account calls."""
# A neighbouring operational script must not shadow a standard-library module.
import os
import sys
_SCRIPT_DIRECTORY=os.path.dirname(os.path.realpath(__file__))
sys.path[:]=[p for p in sys.path if os.path.realpath(p or os.getcwd())!=_SCRIPT_DIRECTORY]
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import hashlib
import numpy as np

REUSE_PATH='experiments/wan22_native/official_cpu/results/pair-v1/comparison/compare-wan22-official-reference.py'
REUSE_SHA='30c0de0f3b1ecebcfe195ffd616d5763cbee75a43405f2e62d6430af6c382e70'
INPUT_SHA='363ee00416161de834ab8213eb50a386abd42ce9cf3f34ff818e9213b9e9d2b5'
TEXT_SHA='2f00251cd8ffbbd72f8cee232feeac49df8b6645c204dfc07667748cff36c406'
TEXT_MANIFEST_SHA='03936c8122c5a902062fdf6a555552a45d861630d0ee2012a712a6a3caf5c11c'
SHAPE=(48,5,18,32)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reuse(repo):
    path=repo/REUSE_PATH
    if sha(path)!=REUSE_SHA:raise ValueError('Pinned published NumPy comparator v2 differs')
    spec=importlib.util.spec_from_file_location('retained_numpy_comparator_v2',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def names_at(path,requested):
    result={}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name)and target.id in requested:
                    result[target.id]=ast.literal_eval(node.value)
    if set(result)!=set(requested):raise ValueError('Complete literal source lists required')
    return result


def confined(root,name):
    root=Path(root).resolve();path=root/name
    if Path(name).is_absolute()or '..'in Path(name).parts or not path.resolve().is_relative_to(root)or not path.is_file():
        raise ValueError('Evidence path leaves declared run')
    return path


def load_cuda(repo,root,ref):
    root=Path(root);result_root=root/'core/result'
    parent=json.loads((root/'metrics.json').read_text())
    terminal=json.loads((root/'core/terminal.json').read_text())
    report=json.loads((result_root/'metrics.json').read_text())
    if (parent.get('status')!='passed'or parent.get('mode')!='pair'or parent.get('model_execution')is not True
            or terminal.get('status')!='complete'or terminal.get('exit_code')!=0
            or report.get('status')!='passed'or report.get('mode')!='pair'or report.get('stage')!='core'
            or report.get('predictions')!=2 or report.get('solver_updates')!=0
            or report.get('settings')!={'steps':50,'shift':5.,'guidance':5.}
            or report.get('finite_outputs')is not True or report.get('input_tensors_unchanged')is not True
            or report.get('actions_read')is not False or report.get('future_target_read')is not False
            or list(root.rglob('*watchdog-stop*'))):
        raise ValueError('Complete finite CUDA pair with unchanged inputs and no watchdog required')
    if sha(result_root/'metrics.json')!=parent.get('core_report_sha256')or sha(root/'core/terminal.json')!=parent.get('core_terminal_sha256'):
        raise ValueError('CUDA completion report changed')
    package=repo/'experiments/wan22_native/cuda_reference'
    constants=names_at(package/'evidence.py',{'NAMES','SHARED'})
    sources={**{n:sha(confined(package,n))for n in constants['NAMES']},
             **{'shared/'+n:sha(confined(repo,n))for n in constants['SHARED']}}
    if parent.get('source_sha256')!=sources or report.get('source_sha256')!=sources:
        raise ValueError('CUDA source graph differs from frozen local source')
    for name,digest in sources.items():
        if sha(confined(root/'measured-source',name+'.txt'))!=digest:raise ValueError('CUDA executed source snapshot changed')
    expected_files={'inputs.safetensors':INPUT_SHA,'contexts.safetensors':TEXT_SHA,'text-manifest.json':TEXT_MANIFEST_SHA}
    if parent.get('copied_input_sha256')!=expected_files:raise ValueError('CUDA saved input file identities differ')
    for name,digest in expected_files.items():
        if sha(root/name)!=digest:raise ValueError('CUDA fixed input bytes changed')
    if report.get('input_identity')!=parent.get('input_identity'):raise ValueError('CUDA worker and parent input identities differ')
    required={'weight-load.json','outputs.safetensors','completed-positive.safetensors','completed-negative.safetensors'}
    if set(report.get('output_sha256',{}))!=required:raise ValueError('Complete original CUDA pair artifacts required')
    for name,digest in report['output_sha256'].items():
        if sha(confined(result_root,name))!=digest:raise ValueError('CUDA output artifact changed')
    weights=json.loads((result_root/'weight-load.json').read_text())
    expected=json.loads((package/'expected-weights.json').read_text())['tensors']
    if (weights.get('tensor_count')!=825 or weights.get('parameter_count')!=4999787712
            or weights.get('convert_model_dtype')is not False or weights.get('all_shards_verified')is not True
            or weights.get('cuda_copy_exact')is not True or set(weights.get('tensors',{}))!=set(expected)):
        raise ValueError('825 original CUDA parameter records required')
    for name,row in weights['tensors'].items():
        want=expected[name]
        if (row.get('source_sha256')!=want['original_sha256']or row.get('loaded_sha256')!=want['original_sha256']
                or row.get('shape')!=want['shape']or row.get('shard')!=want['shard']
                or row.get('original_dtype')!='float32'or row.get('loaded_dtype')!='float32'or row.get('cuda_copy_exact')is not True):
            raise ValueError('CUDA parameter identity differs: '+name)
    path=result_root/'outputs.safetensors'
    if path.stat().st_size>8*2**20:raise ValueError('Bounded velocity file required')
    values=ref.tensors(path)
    if set(values)!=set(ref.VELOCITIES):raise ValueError('Exactly three CUDA velocity arrays required')
    for value in values.values():
        if value.shape!=SHAPE or value.dtype!=np.dtype('<f4')or not np.isfinite(value).all():raise ValueError('Finite native FP32 velocities required')
    for label,keys in [('positive',{'positive_velocity'}),('negative',{'positive_velocity','negative_velocity'})]:
        partial=ref.tensors(result_root/f'completed-{label}.safetensors')
        if set(partial)!=keys or any(not np.array_equal(partial[k],values[k])for k in keys):raise ValueError('CUDA retained partial outputs differ')
    return values,{'parent_sha256':sha(root/'metrics.json'),'terminal_sha256':sha(root/'core/terminal.json'),
        'worker_sha256':sha(result_root/'metrics.json'),'outputs_sha256':sha(path),'weight_load_sha256':sha(result_root/'weight-load.json'),
        'input_files':expected_files,'input_identity':report['input_identity'],'source_sha256':sources,'hardware':report.get('hardware'),'precision':report.get('precision'),
        'original_fp32_weight_identities_checked':825,'parent_seconds':parent.get('elapsed_seconds'),
        'load_seconds':report.get('load_seconds'),'pair_seconds':report.get('pair_seconds')}


def compare(ref,actual,baseline,official):
    row=ref.statistics(actual,baseline)
    row['comparison_baseline_rms']=row.pop('reference_rms')
    del row['rmse_over_reference_rms']
    normalizer=float(np.sqrt(np.mean(np.asarray(official,dtype=np.float64)**2)))
    row['official_cpu_normalization_rms']=normalizer
    row['rmse_over_official_cpu_rms']=row['rmse']/normalizer if normalizer else None
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--cuda-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise ValueError('Fresh comparison report required')
    ref=reuse(args.repo)
    official_root=args.repo/'experiments/wan22_native/official_cpu/results/pair-v1'
    official,official_identity=ref.load_reference_run(args.repo,official_root)
    for name,digest in [('contexts.safetensors',TEXT_SHA),('text-manifest.json',TEXT_MANIFEST_SHA)]:
        if sha(official_root/name)!=digest:raise ValueError('Official CPU text bytes differ')
    cuda,cuda_identity=load_cuda(args.repo,args.cuda_run,ref)
    if cuda_identity['input_identity']!=json.loads((official_root/'metrics.json').read_text())['input_identity']:
        raise ValueError('CUDA and official CPU declared input identities differ')
    references={'official_cpu':official};identities={'official_cpu':official_identity}
    for label,(directory,digest)in ref.BASELINES.items():
        root=args.repo/'experiments/wan22_native/core-results'/directory
        if sha(root/'inputs.safetensors')!=INPUT_SHA or sha(root/'outputs.safetensors')!=digest:
            raise ValueError('Pinned '+label+' input/output bytes differ')
        references[label]=ref.tensors(root/'outputs.safetensors');identities[label]={'outputs_sha256':digest,'inputs_sha256':INPUT_SHA}
    comparisons={}
    for label,baseline in references.items():
        comparisons[label]={}
        for name in ref.VELOCITIES:
            comparisons[label][name]={region:compare(ref,cuda[name][:,cut],baseline[name][:,cut],official[name][:,cut])
                for region,cut in [('all',slice(None)),('observed',slice(0,1)),('future',slice(1,None))]}
    record={'schema':'worldline-wan22-cuda-pair-comparison-v1','status':'computed','source_sha256':sha(Path(__file__)),
        'reused_comparator_sha256':REUSE_SHA,'numpy_version':np.__version__,
        'scope':'Descriptive saved initial-pair comparison. No new numerical thresholds, model execution, equivalence, quality or failure-cause conclusion.',
        'direction':'CUDA velocity minus each named saved baseline; cosine compares those two arrays',
        'normalization':'Every cross-run relative RMSE divides by the streamed official CPU RMS for that same velocity and region, including comparisons against portable CPU and MPS.',
        'future_slice':'Channels 0:48, latent frames 1:5, height 0:18, width 0:32. Four future latent frames, 110592 values per velocity.',
        'zero_denominators':'Relative RMSE and cosine are null for zero denominators.',
        'cuda_run_identity':cuda_identity,'baseline_identities':identities,'cuda_minus_saved_baseline':comparisons,
        'guidance_equation_residuals':{label:ref.guidance_residual(values)for label,values in {'cuda':cuda,**references}.items()},
        'guidance_residual_denominator':'Its own recomputed eager FP32 guidance, separately labeled as in comparator v2.',
        'saved_files_modified':False}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x')as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    summary={label:{name:row[name]['future']['rmse_over_official_cpu_rms']for name in ref.VELOCITIES}for label,row in comparisons.items()}
    print(json.dumps({'status':'computed','report_sha256':sha(args.output),'future_relative_rmse_official_cpu_denominator':summary},indent=2))


if __name__=='__main__':main()
