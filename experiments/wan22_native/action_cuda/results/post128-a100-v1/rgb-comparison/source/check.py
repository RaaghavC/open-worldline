# SPDX-License-Identifier: Apache-2.0
import json,time
from pathlib import Path
import numpy as np
import analyze as a
start=time.monotonic();before=a.sources();p,r=a.prepared_check();checks=['Exact final128 plan and cp128 identity; prior cp16 source/report and all initial input/text/command file hashes match']
truth={'closed':np.zeros((2,3,3),np.uint8),'open':np.full((2,3,3),255,np.uint8)}
old={k:np.full((2,3,3),.5,np.float64)for k in truth};new={k:v.astype(np.float64)/255 for k,v in truth.items()};r=a.comparison(new,old,truth)
assert r['checkpoint128_pair']['regions']['full_frame']['strict_both_closer_to_own_truth'] and not r['checkpoint16_pair']['regions']['full_frame']['strict_both_closer_to_own_truth']
assert all(v['target_mae_delta_128_minus16']==-.5 for v in r['arms'].values());checks.append('Independent synthetic target-error delta: correct new branches improve by 0.5 and old midpoint ties fail')
r=a.comparison(old,old,truth);assert all(v['checkpoint128_vs_checkpoint16']['exact'] and v['target_mae_delta_128_minus16']==0 for v in r['arms'].values());checks.append('Identical checkpoints give exact zero differences; paired ties remain failures')
try:a.comparison({'closed':new['closed']},old,truth)
except ValueError:pass
else:raise AssertionError('Missing branch accepted')
checks.append('Missing branch rejected');assert a.sources()==before
report={'status':'passed','checks':checks,'seconds':time.monotonic()-start,'source_sha256':before,'fixture_sha256':a.sha(Path(__file__)),'actual_final128_images_read':False,'model_execution':False,'cloud_operations':False}
(a.HERE/'cpu-check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'passed','checks':len(checks),'report_sha256':a.sha(a.HERE/'cpu-check.json')}))
