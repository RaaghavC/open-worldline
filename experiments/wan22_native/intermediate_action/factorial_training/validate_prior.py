# SPDX-License-Identifier: Apache-2.0
"""Isolated CPU validation of the two exact prior native stages."""
import argparse
import json
from pathlib import Path
import sys
import torch
from safetensors import safe_open


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('cache','profile'),required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    directory=args.repo/('experiments/atrium_factorial/native_cache' if args.kind=='cache' else 'experiments/wan22_native/intermediate_action/cuda_profile')
    sys.path[:0]=[str(directory),str(args.repo)]
    import run as prior
    import packet
    from experiments.wan22_native.official_cpu.streaming import sha,tensor_sha
    read=packet.evidence.read_json if args.kind=='cache' else packet.old.read_json
    relative=packet.evidence.relative_file if args.kind=='cache' else packet.old.relative_file
    root=args.root
    parent=read(root/'metrics.json');terminal=read(root/'terminal.json')
    if parent.get('status')!='passed' or terminal.get('status')!='complete' or terminal.get('exit_code')!=0 or terminal.get('cleanup_error') is not None or list(root.rglob('watchdog-stop.json')):
        raise ValueError('Passed prior parent and terminal required')
    result={'kind':args.kind,'status':'passed','model_execution':False,
            'parent_sha256':sha(root/'metrics.json'),'terminal_sha256':sha(root/'terminal.json'),
            'plan_sha256':sha(root/'plan.json'),'selected_files':{}}
    if args.kind=='cache':
        import cache
        worker=prior.validate_completed(root)
        if parent.get('worker_metrics_sha256')!=sha(root/'worker/metrics.json') or parent.get('terminal_sha256')!=sha(root/'terminal.json'):
            raise ValueError('Cache parent does not bind worker/terminal')
        result.update(worker_sha256=sha(root/'worker/metrics.json'),completion_sha256=sha(root/'result/completion.json'),
                      source_sha256=worker['source_sha256'],all196_unchanged=True,windows={})
        observation=None
        for arm in cache.reader.ARMS:
            values,provenance=cache.read_window(root/'result',arm)
            if not torch.equal(values['target'][:,:,:1],values['observation']) or tensor_sha(values['target'][:,:,:1])!=tensor_sha(values['observation']):
                raise ValueError('New training requires bit-exact raw target/independent prefix')
            current=tensor_sha(values['observation'])
            if observation is not None and observation!=current:raise ValueError('One shared observation required')
            observation=current
            file='result/'+arm+'.safetensors'
            result['selected_files'][file]={'sha256':sha(root/file),'bytes':(root/file).stat().st_size}
            result['windows'][arm]={k:{'shape':list(v.shape),'dtype':'F32','sha256':tensor_sha(v)} for k,v in values.items()}
        result.update(raw_target_prefix_bit_exact=True,observation_sha256=observation)
    else:
        plan,_=packet.read_prepared(root);worker=read(root/'result/metrics.json')
        if (parent.get('result_metrics_sha256')!=sha(root/'result/metrics.json')
                or worker.get('status')!='passed' or worker.get('completed_updates')!=4
                or worker.get('zero_gate_passed') is not True or worker.get('base_unchanged') is not True
                or worker.get('all825_current_value_hashes_verified') is not True
                or worker.get('source_sha256')!=plan['source_sha256'] or len(worker.get('parity',[]))!=16):
            raise ValueError('Completed original placement profile required')
        for name,digest in worker['output_sha256'].items():
            if sha(relative(root/'result',name))!=digest:raise ValueError('Profile output bytes changed')
        expected=prior._original_weights(read(root/'result/weight-load.json'))
        for name in ('before','after-parity','after-block28','after-block29'):
            if read(root/f'result/core-{name}.json')!=expected:raise ValueError('Original825 identities changed')
        for row in worker['parity']:
            if row.get('passed') is not True or row.get('exact_equal') is not True:raise ValueError('Exact profile parity required')
            native=root/f"result/parity/native-{row['context']}.safetensors"
            bridge=root/f"result/parity/block{row['block_index']}-{row['context']}-{row['arm']}-{row['path']}.safetensors"
            arrays=[]
            for path in (native,bridge):
                with safe_open(path,framework='pt',device='cpu') as handle:
                    if set(handle.keys())!={'velocity'} or handle.get_slice('velocity').get_shape()!=[1,48,5,44,78] or handle.get_slice('velocity').get_dtype()!='F32':raise ValueError('Exact spatial velocity schema required')
                    value=handle.get_tensor('velocity')
                    if not torch.isfinite(value).all():raise ValueError('Finite parity required')
                    arrays.append(value)
            if not torch.equal(*arrays):raise ValueError('Saved native profile comparison is not exact')
        result.update(worker_sha256=sha(root/'result/metrics.json'),source_sha256=worker['source_sha256'],
                      exact_zero_comparisons=16,all825_unchanged=True)
    if args.output.exists():raise ValueError('Fresh prior-stage receipt required')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'passed','kind':args.kind,'receipt_sha256':sha(args.output)}))


if __name__=='__main__':main()
