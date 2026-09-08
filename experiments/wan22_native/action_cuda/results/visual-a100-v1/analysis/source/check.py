# SPDX-License-Identifier: Apache-2.0
import json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import analyze as a

torch.set_num_threads(1)
started=time.monotonic(); source=a.sources();checks=[]
shape=(2,3,3)
zero=np.zeros(shape,dtype=np.float64);one=np.ones(shape,dtype=np.float64)
assert a.score(zero,zero)=={'mae_0_1':0.,'mse_0_1':0.,'psnr_db':None,'exact':True}
assert a.score(zero,one)=={'mae_0_1':1.,'mse_0_1':1.,'psnr_db':-0.,'exact':False}
assert a.score(zero,one,np.zeros((2,3),bool)) is None
checks.append('Exact simple MAE/MSE/PSNR and empty-mask arithmetic')
gt={'closed':np.zeros(shape,np.uint8),'open':np.full(shape,255,np.uint8)}
for predicted,expected in [({'closed':zero,'open':one},True),({'closed':one,'open':zero},False),({'closed':one*.5,'open':one*.5},False)]:
 row=a.paired_frame(predicted,gt)
 assert all(x['strict_both_closer_to_own_truth'] is expected for x in row['regions'].values())
checks.append('Correct, swapped and tied paired-target outcomes')
same=a.paired_frame({'closed':zero,'open':zero},{'closed':gt['closed'],'open':gt['closed']})
assert all(not x['strict_both_closer_to_own_truth'] and not x['informative_truth_contrast'] for x in same['regions'].values())
checks.append('Identical truth gives no informative pair-correctness claim')
try:a.score(zero,np.ones((2,4,3)))
except ValueError:pass
else:raise AssertionError('Malformed shape admitted')
try:a.score(zero,np.full(shape,float('nan')))
except ValueError:pass
else:raise AssertionError('Nonfinite RGB admitted')
checks.append('Shape and finite-value validation')
truth,provenance=a.ground_truth(a.HERE.parent/'atrium-pilot/dense-pair')
assert len(truth['closed'])==len(truth['open'])==17
assert all(x.shape==(704,1248,3) and x.dtype==np.uint8 for arm in a.ARMS for x in truth[arm])
prepared=np.asarray(Image.open(a.REPO/'experiments/wan22_native/spatial_reference/prepared-inputs-v1/spatial.png'))
assert np.array_equal(truth['open'][0],prepared) and np.array_equal(truth['closed'][0],prepared)
assert provenance['preprocessing']=={'resize':[1252,704],'crop':[2,0,1250,704],'resize_applied':True}
assert provenance['arms']['closed']['frames'][1]['incoming_command']=='wait'
assert provenance['arms']['open']['frames'][1]['incoming_command']=='interact'
assert provenance['arms']['open']['frames'][16]['cumulative_requested_left_turn_degrees']==112.5
checks.append('All 34 exact original files, destination commands, 18-value canonicalization and frozen spatial transform')
checks.append('Both processed initial RGB arrays match the actual native spatial packet PNG exactly')
rows=[]
for i in range(17):
 zero_prediction={arm:truth[arm][i].astype(np.float64)/255 for arm in a.ARMS}
 pair=a.paired_frame(zero_prediction,{arm:truth[arm][i] for arm in a.ARMS})
 rows.append({'frame':i,'truth_difference_mae_0_1':pair['truth_difference_mae_0_1'],
  'differing_pixels_any_byte':pair['truth_pixels_differing_any_byte'],
  'differing_pixels_at_least_two_levels':pair['truth_pixels_differing_at_least_two_levels'],
  'repeat_start_mae_0_1':{arm:a.score(truth['open'][0].astype(np.float64)/255,truth[arm][i].astype(np.float64)/255)['mae_0_1'] for arm in a.ARMS}})
# Bind the actual prepared commands to an isolated stand-in completion record.
import tempfile, shutil
from safetensors.numpy import save_file
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder).resolve(); prepared=a.HERE.parent/'wan22-action-cuda-visual-prepared-v2'
    shutil.copyfile(prepared/'plan.json',root/'plan.json')
    shutil.copyfile(prepared/'commands.safetensors',root/'commands.safetensors')
    plan=json.loads((root/'plan.json').read_text());baseline=plan['input_identity']['baseline']
    (root/'core/result').mkdir(parents=True)
    core={'arms':{arm:{'commands_sha256':provenance['arms'][arm]['commands_sha256'],
       'predictions':100,'solver_updates':50,'input_tensor_sha256':baseline['input_tensor_sha256'],
       'text_tensor_sha256':baseline['context_tensor_sha256']} for arm in a.ARMS}}
    (root/'core/result/metrics.json').write_text(json.dumps(core))
    parent={'plan_sha256':a.sha(root/'plan.json'),'source_sha256':plan['source_sha256']}
    assert set(a._bind_conditions(root,parent,provenance)['arms'])==set(a.ARMS)
    core['arms']['open']['commands_sha256']='0'*64
    (root/'core/result/metrics.json').write_text(json.dumps(core))
    try:a._bind_conditions(root,parent,provenance)
    except ValueError:pass
    else:raise AssertionError('Changed measured command hash accepted')
checks.append('Actual saved command byte identities bind correctly; changed sampling command identity fails')
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder).resolve();(root/'closed/rgb').mkdir(parents=True);(root/'closed/frames').mkdir()
    values=np.zeros((1,3,1,704,1248),np.float32)
    save_file({'rgb':values},str(root/'closed/rgb/0000.safetensors'))
    pixels=a.rgb_audit._pixels(values);Image.fromarray(pixels).save(root/'closed/frames/0000.png')
    relative=['closed/rgb/0000.safetensors','closed/frames/0000.png']
    output={'output_sha256':{name:a.sha(root/name) for name in relative}}
    index={'frames':[{'index':0,'file':'0000.safetensors','sha256':a.sha(root/relative[0]),
       'tensor_sha256':a.arr_sha(values),'bytes':(root/relative[0]).stat().st_size}]}
    decoded,image,_=a._read_frame(root,output,index,'closed',0)
    assert decoded.shape==(704,1248,3) and np.all(decoded==.5) and np.all(image==128)
    index['frames'][0]['tensor_sha256']='0'*64
    try:a._read_frame(root,output,index,'closed',0)
    except ValueError:pass
    else:raise AssertionError('Changed tensor identity accepted')
checks.append('Actual-shape raw frame/PNG decoding and conversion pass; corrupted tensor identity fails')
assert source==a.sources()
report={'status':'passed','checks':checks,'elapsed_seconds':time.monotonic()-started,'source_sha256':source,
 'check_source_sha256':a.sha(Path(__file__)),'model_execution':False,'gpu_execution':False,
 'numpy':np.__version__,'pillow':Image.__version__,'torch':torch.__version__,
 'actual_ground_truth':provenance,'target_only_descriptive_diagnostics':rows,
 'generated_result_analysis_executed':False}
(a.HERE/'cpu-check.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':'passed','checks':len(checks),'elapsed_seconds':report['elapsed_seconds'],'report_sha256':a.sha(a.HERE/'cpu-check.json')}))
