"""FP64 pixel metrics. No recognition or automatic control verdicts."""
import math
import numpy as np

def score(actual,target):
    if (actual.shape!=target.shape or actual.ndim!=3 or actual.shape[-1]!=3
        or not np.isfinite(actual).all() or not np.isfinite(target).all()):
        raise ValueError('Matched finite HWC RGB arrays required')
    difference=actual.astype(np.float64)-target.astype(np.float64)
    mse=float(np.mean(difference*difference,dtype=np.float64))
    return dict(mae_0_1=float(np.mean(np.abs(difference),dtype=np.float64)),mse_0_1=mse,rmse_0_1=math.sqrt(mse),exact=mse==0)

def average(rows):
    if not rows:raise ValueError('Nonempty equal-size frame scores required')
    mse=float(np.mean([r['mse_0_1'] for r in rows],dtype=np.float64))
    return dict(mae_0_1=float(np.mean([r['mae_0_1'] for r in rows],dtype=np.float64)),mse_0_1=mse,rmse_0_1=math.sqrt(mse),exact=all(r['exact'] for r in rows))

def frame_scores(generated,target,original_first,generated_first,previous_generated,previous_target):
    return dict(generated_vs_target=score(generated,target),original_repeat_vs_target=score(original_first,target),
        own_first_repeat_vs_target=score(generated_first,target),generated_change_from_own_first=score(generated,generated_first),
        target_change_from_original_first=score(target,original_first),generated_change_from_previous=score(generated,previous_generated),
        target_change_from_previous=score(target,previous_target))

def paired_scores(generated,targets,first_generated):
    if set(generated)!={'closed','interact'} or set(targets)!={'closed','interact'} or set(first_generated)!={'closed','interact'}:raise ValueError('Exact ordered door pair required')
    d=generated['interact']-generated['closed'];truth=targets['interact']-targets['closed'];first=first_generated['interact']-first_generated['closed']
    zero=np.zeros_like(d)
    return dict(interact_minus_closed_error=score(d,truth),generated_pair_magnitude=score(d,zero),target_pair_magnitude=score(truth,zero),
        paired_change_from_generated_first=score(d,first),zero_pair_baseline_error=score(zero,truth))

def improvement(actual,baseline):
    # A zero baseline has no strictly better attainable target error.
    if baseline['mse_0_1']==0:return dict(strictly_lower_mae=None,strictly_lower_rmse=None,relative_mse_improvement=None,reason='Original repeat target error is zero; no beat-zero requirement')
    return dict(strictly_lower_mae=actual['mae_0_1']<baseline['mae_0_1'],strictly_lower_rmse=actual['rmse_0_1']<baseline['rmse_0_1'],relative_mse_improvement=1-actual['mse_0_1']/baseline['mse_0_1'],reason='Pixel reproduction comparison only; not a control verdict')
