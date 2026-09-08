"""Small synthetic comparison checks; no actual effect images or model calls."""
from pathlib import Path
import json,tempfile,time
import numpy as np
import analyze as a
started=time.monotonic();before=a.sources();checks=[]
plan,reports=a.prepared_check();checks.append('Exact effect plan/checkpoint and both original report/input identities')
truth={'closed':np.zeros((2,3,3),np.uint8),'open':np.full((2,3,3),255,np.uint8)}
mid={arm:np.full((2,3,3),0.5,np.float64)for arm in a.base.ARMS}
generated={'effect128':{arm:value.astype(np.float64)/255 for arm,value in truth.items()},'previous128':mid,'previous16':mid}
row=a.compare_frame(generated,truth,np.zeros((2,3,3),np.float64))
assert row['models']['effect128']['pair']['regions']['full_frame']['strict_both_closer_to_own_truth']
for old in a.PRIOR:
 assert not row['models'][old]['pair']['regions']['full_frame']['strict_both_closer_to_own_truth']
 assert all(row['comparisons'][old][arm]['target_mae_delta_effect_minus_previous']==-0.5 for arm in a.base.ARMS)
checks.append('Exact synthetic target-MAE deltas; both old midpoint ties fail')
same={m:mid for m in a.MODELS};row=a.compare_frame(same,truth,np.zeros((2,3,3),np.float64))
assert all(row['comparisons'][old][arm]['effect_vs_previous']['exact'] and row['comparisons'][old][arm]['target_mae_delta_effect_minus_previous']==0 for old in a.PRIOR for arm in a.base.ARMS)
checks.append('Identical model arrays produce zero inter-model and target-error differences')
try:a.compare_frame({'effect128':mid},truth,np.zeros((2,3,3),np.float64))
except ValueError:pass
else:raise AssertionError('Missing historical branches accepted')
checks.append('Missing model/branch data rejected')
images={arm:{i:np.full((2,3,3),i,np.uint8)for i in range(17)}for arm in a.base.ARMS}
with tempfile.TemporaryDirectory() as temporary:
 layout=a.all_frame_sheet(images,Path(temporary)/'all.png')
 assert len(layout)==34 and {(x['arm'],x['frame'])for x in layout}=={(arm,i)for arm in a.base.ARMS for i in range(17)}
 assert len({(x['column'],x['row'])for x in layout})==34
checks.append('Contact placement includes all 34 frames exactly once with unique cells')
assert a.sources()==before
result={'schema':'worldline-effect128-rgb-comparison-cpu-v1','status':'passed','checks':checks,'seconds':time.monotonic()-started,'source_sha256':before,'check_source_sha256':a.sha(__file__),'actual_effect_images_read':False,'models_executed':False,'cloud_calls':0}
with (a.HERE/'cpu-check.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':'passed','checks':len(checks),'report_sha256':a.sha(a.HERE/'cpu-check.json')}))
