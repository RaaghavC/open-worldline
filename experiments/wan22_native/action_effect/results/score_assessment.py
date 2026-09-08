"""Recompute saved public assessment scores with NumPy, without execution preflight."""
import argparse, hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

# Importing the retained reader must not add files to the verified recovery.
sys.dont_write_bytecode = True

READER_SHA = '487f16b5c39975a70e5f95e92040bc496f6ad1ff68a01cd10665cf80706cece4'
CHECKPOINTS = ('zero', 'old128', 'new128')

def need(ok, message):
    if not ok: raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def close(actual, saved):
    if type(actual) is dict:
        need(type(saved) is dict and set(actual)==set(saved), 'Saved score fields differ')
        for key in actual: close(actual[key], saved[key])
    elif type(actual) is list:
        need(type(saved) is list and len(actual)==len(saved), 'Saved score length differs')
        for a,b in zip(actual,saved): close(a,b)
    elif type(actual) is float:
        need(type(saved) in (int,float) and math.isfinite(actual) and math.isfinite(saved) and math.isclose(actual,saved,rel_tol=1e-12,abs_tol=1e-15), 'Recorded FP64 score differs beyond its existing serialization tolerance')
    else: need(actual==saved, 'Saved score identity differs')

def score(download, expected_index_sha256, output):
    download=Path(download); output=Path(output)
    need(not output.exists(), 'Fresh score output required')
    index_path=download/'index.json'
    need(sha(index_path)==expected_index_sha256, 'Explicit public index hash differs')
    index=json.loads(index_path.read_text()); verified=json.loads((download/'verification.json').read_text())
    need(verified['status']=='verified' and verified['index_sha256']==expected_index_sha256 and verified['files']==320, 'Completed public artifact verification required')
    raw=download/'recovered'
    need({p.relative_to(raw).as_posix() for p in raw.rglob('*') if p.is_file()}==set(index['files']), 'Exact public member inventory required')
    for name,row in index['files'].items():
        path=raw/name
        need(not path.is_symlink() and path.stat().st_size==row['bytes'] and sha(path)==row['sha256'], 'Public file differs: '+name)
    root=raw/'action-results/effect-assessment-v1'; result=root/'result'
    reader_path=root/'source/local/reader.py'
    need(sha(reader_path)==READER_SHA, 'Original NumPy score source differs')
    spec=importlib.util.spec_from_file_location('effect_saved_score_reader', reader_path)
    reader=importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
    worker=reader.read_json(result/'metrics.json',16*1024**2)
    need(worker['status']=='passed' and worker['counts']['head_predictions']==66, 'Complete recorded 66-output assessment required')
    fixed=root/'reference-inputs/original14/original-training-inputs.safetensors'
    need(fixed.stat().st_size<16*1024**2, 'Bounded original target/noise file required')
    arrays=load_file(fixed)
    for key in ('closed_target','open_target','noise'): reader.finite(arrays[key])
    target=np.subtract(arrays['open_target'],arrays['closed_target'],dtype=np.float32)
    files=worker['raw_prediction_sha256']
    expected=[f'heldout-{i:04d}-{cp}-{arm}-{text}.safetensors' for i in range(4) for cp in CHECKPOINTS for arm in reader.ARMS for text in reader.TEXTS]
    expected += [f'seen506-{cp}-{arm}-positive.safetensors' for cp in CHECKPOINTS for arm in reader.ARMS]
    expected += [f'original999-{cp}-{arm}-{text}.safetensors' for cp in CHECKPOINTS for arm in reader.ARMS for text in reader.TEXTS]
    need(set(files)==set(expected) and len(files)==66, 'Exact 66 raw prediction names required')
    held={n:h for n,h in files.items() if n.startswith('heldout-')}
    scores=reader.read_saved_scores(result,held,worker['heldout']['calls'],target)
    close(scores,worker['heldout_scores'])
    def velocity(name):
        return reader.read_tensor_file(result/name,{'velocity':(reader.SHAPE,'F32')},files[name])['velocity']
    seen={};original={}
    for cp in CHECKPOINTS:
        seen[cp]={}; predictions={}
        for arm in reader.ARMS:
            v=velocity(f'seen506-{cp}-{arm}-positive.safetensors')
            truth=arrays['noise']-arrays[arm+'_target']
            mse=float(np.mean((v[:,:,1:].astype(np.float64)-truth[:,:,1:].astype(np.float64))**2))
            seen[cp][arm]={'future_flow_mse_fp64':mse,'seen_training_draw':True}
            for text in reader.TEXTS: predictions[arm+'-'+text]=velocity(f'original999-{cp}-{arm}-{text}.safetensors')
        original[cp]=reader.score(predictions,target)
    close(seen,worker['additional_checks']['seen506']);close(original,worker['additional_checks']['original999'])
    gates=[]
    for i in range(4):
        zero,old,new=scores[i*3:i*3+3]
        need([r['checkpoint'] for r in (zero,old,new)]==list(CHECKPOINTS),'Fixed checkpoint order required')
        row={'noise_index':i,'lower_than_zero':new['future_contrast_mse']<zero['future_contrast_mse'],
             'lower_than_old128':new['future_contrast_mse']<old['future_contrast_mse'],
             'positive_alignment':type(new['cosine']) is float and math.isfinite(new['cosine']) and new['cosine']>0}
        row['passed']=all(row[k] for k in ('lower_than_zero','lower_than_old128','positive_alignment'));gates.append(row)
    need('torch' not in sys.modules,'Offline scorer must not import Torch')
    report={'schema':'worldline-public-action-effect-saved-scores-v1','status':'passed',
        'public_index_sha256':expected_index_sha256,'verified_public_files':len(index['files']),
        'raw_predictions_scored':66,'reader_sha256':READER_SHA,'scorer_sha256':sha(__file__),
        'heldout_scores':scores,'seen506':seen,'original999':original,
        'numerical_gate':{'passed':all(r['passed'] for r in gates),'per_noise':gates},
        'recorded_scores_match':True,'model_execution':False,'torch_imported':False,
        'original_execution_preflight_invoked':False,'original_hashes_restored_or_fabricated':False,
        'limitations':['Public member hashes are checked directly; no original private audit path is reconstructed.',
            'This recomputes saved scores, not model predictions or training.',
            'The existing 1e-12 relative/1e-15 absolute tolerance compares serialized FP64 records only. Strict four-noise inequalities receive no tolerance.',
            'Four noise inputs from one scene do not establish visible door or camera control.']}
    with output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',type=Path,required=True)
    parser.add_argument('--expected-index-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=score(args.download,args.expected_index_sha256,args.output)
    print(json.dumps({'status':report['status'],'raw_predictions_scored':report['raw_predictions_scored'],'numerical_gate':report['numerical_gate']}))
