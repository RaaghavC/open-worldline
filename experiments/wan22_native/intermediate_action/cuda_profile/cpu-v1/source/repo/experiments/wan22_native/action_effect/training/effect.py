"""One additional start0 loss, with no loader, launcher or new model equations."""
import math
import time
import torch
from experiments.wan22_native.action_cuda import probe_math as original

flow_inputs=original.flow_inputs
future_flow_mse=original.future_flow_mse
condition_identity=original.condition_identity
grad_norm=original.grad_norm
require_finite_tree=original.require_finite_tree
compare_velocity=original.compare_velocity
LAMBDA=1.0
ENDPOINT_SCALE=1.0


def endpoint_contrast(positive_closed,negative_closed,positive_open,negative_open):
    """Ideal pure-noise clean contrast, using native CFG5 arithmetic order."""
    values=(positive_closed,negative_closed,positive_open,negative_open)
    if any(v.dtype!=torch.float32 or v.shape!=values[0].shape or not torch.isfinite(v).all() for v in values):
        raise ValueError('Matched finite FP32 head velocities required')
    closed=negative_closed+5*(positive_closed-negative_closed)
    opened=negative_open+5*(positive_open-negative_open)
    return -(opened-closed)


def endpoint_loss(predicted_difference,target_difference):
    if (predicted_difference.ndim!=5 or predicted_difference.shape[0]!=1 or predicted_difference.shape[2]!=5
            or target_difference.shape!=predicted_difference.shape or target_difference.dtype!=torch.float32
            or target_difference.requires_grad or not torch.isfinite(target_difference).all()):
        raise ValueError('B1 five-latent prediction and frozen target difference required')
    return future_flow_mse(predicted_difference,target_difference)


def auxiliary_inputs(windows,noise):
    if len(windows)!=2 or not torch.equal(windows[0]['observation'],windows[1]['observation']):
        raise ValueError('Start0 independent observations must be exactly equal')
    if any(w['target'].shape!=noise.shape for w in windows):raise ValueError('Matched original target shapes required')
    # Reuse the native target/prefix validator; discard its ordinary FM input.
    for window in windows:flow_inputs(window['target'],window['observation'],noise,506)
    difference=windows[1]['commands']-windows[0]['commands']
    expected=torch.zeros_like(difference);expected[0,0,5]=1.
    if not torch.equal(difference,expected):raise ValueError('Only the initial wait/interact command may differ')
    x=noise.clone();x[:,:,:1]=windows[0]['observation']
    prefix=(x.shape[-2]//2)*(x.shape[-1]//2)
    times=torch.full((1,5*prefix),999,dtype=torch.int64);times[:,:prefix]=0
    target_difference=windows[1]['target']-windows[0]['target']
    return x,times,target_difference


def auxiliary_backward(bridge,windows,noise,positive,negative,*,retain,update,check):
    x,times,target=auxiliary_inputs(windows,noise)
    parameters=list(bridge.adapter.parameters());device=parameters[0].device
    if (negative.ndim!=2 or negative.dtype!=torch.float32 or negative.device.type!='cpu'
            or negative.requires_grad or not torch.isfinite(negative).all()):raise ValueError('Genuine fixed negative context required')
    conditions={'pure_noise_sha256':original.tensor_sha(noise),'input_sha256':original.tensor_sha(x),
                'times_sha256':original.tensor_sha(times),'observation_sha256':original.tensor_sha(windows[0]['observation']),
                'target_difference_sha256':original.tensor_sha(target),
                'commands_sha256':{arm:original.tensor_sha(w['commands']) for arm,w in zip(('closed','open'),windows)},
                'context_sha256':{'positive':original.tensor_sha(positive),'negative':original.tensor_sha(negative)}}
    predictions={};extracts=0
    try:
        for label,context in (('positive',positive),('negative',negative)):
            check()
            features=bridge.extract_features(x.to(device),times.to(device),[context.to(device)])
            extracts+=1
            if features.hidden.requires_grad or features.time_embedding.requires_grad:raise RuntimeError('Base features must be frozen constants')
            for arm,window in zip(('closed','open'),windows):
                check()
                prediction=bridge.predict_from_features(features,window['commands'].to(device),window['observation'].to(device),track_grad=True)
                retain(f'auxiliary-{update:04d}-{arm}-{label}',{'velocity':prediction})
                predictions[arm+'-'+label]=prediction
            del features
        predicted=endpoint_contrast(predictions['closed-positive'],predictions['closed-negative'],predictions['open-positive'],predictions['open-negative'])
        loss=endpoint_loss(predicted,target.to(device))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite auxiliary loss')
        value=float(loss.detach().cpu());(LAMBDA*loss).backward()
        if any(p.requires_grad or p.grad is not None for p in bridge.core.parameters()):raise RuntimeError('Auxiliary created foundation gradients')
        return {'enabled':True,'lambda':LAMBDA,'endpoint_scale':ENDPOINT_SCALE,'future_clean_difference_mse':value,
                'weighted_loss':LAMBDA*value,'feature_extracts':extracts,'head_predictions':len(predictions),
                'input_identity':conditions,'prefix_excluded':True,'target_conditioning':False,
                'definition':'Ideal pure-noise endpoint s=1, not literal solver sigma; clean difference=-(Gopen-Gclosed), G=N+5*(P-N)'}
    finally:
        predictions.clear()


def paired_update(bridge, windows, noise, k, context, optimizer, *, check=None, auxiliary=False, negative_context=None, retain=None, update=None):
    """Closed then open, half future loss each, exactly one optimizer step.

    Future target latents are used to form the usual noisy training input and
    FM loss target. The bridge only receives noisy latents, token times, text,
    requested commands and the independent initial observation.
    """
    if not isinstance(windows, (tuple, list)) or len(windows) != 2:
        raise ValueError("Exactly two ordered closed/open windows are required")
    if any(p.requires_grad or p.grad is not None for p in bridge.core.parameters()):
        raise ValueError("Core parameters must remain frozen and gradient-free")
    parameters = list(bridge.adapter.parameters())
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in parameters}:
        raise ValueError("Optimizer must own exactly the adapter parameters")
    device = parameters[0].device
    optimizer.zero_grad(set_to_none=True)
    branches = []
    for label, window in zip(("closed", "open"), windows):
        if check:
            check()
        if not isinstance(window, dict) or set(window) != {"target", "observation", "commands"}:
            raise ValueError("A window must contain only target, observation and commands")
        noisy, times, velocity = flow_inputs(window["target"], window["observation"], noise, k)
        identity = condition_identity(noisy, times, velocity, window)
        if not torch.equal(noisy[:, :, :1], window["observation"]):
            raise RuntimeError("Noisy training input lost its exact observed prefix")
        cuda_noisy = noisy.to(device)
        prediction = bridge(cuda_noisy, times.to(device), [context.to(device)],
                            commands=window["commands"].to(device), observation=window["observation"].to(device))
        loss = future_flow_mse(prediction, velocity.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite branch loss")
        (loss * .5).backward()
        prefix_exact = torch.equal(cuda_noisy[:, :, :1], window["observation"].to(device))
        if not prefix_exact:
            raise RuntimeError("Bridge mutated the clean training prefix")
        branches.append({"branch": label, "future_flow_mse": float(loss.detach().cpu()),
                         "input_sha256": identity, "observed_input_prefix_exact": prefix_exact})
        del prediction, loss, noisy, cuda_noisy, times, velocity
    aux={'enabled':False,'lambda':LAMBDA,'feature_extracts':0,'head_predictions':0,'weighted_loss':0.}
    if auxiliary:
        if retain is None or type(update) is not int or update<1:raise ValueError('Auxiliary raw retention and update number required')
        aux=auxiliary_backward(bridge,windows,noise,context,negative_context,retain=retain,update=update,check=check or (lambda:None))
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError("Every adapter gradient must be present and finite")
    before = grad_norm(parameters)
    recurrent = grad_norm(bridge.adapter.command_gru.parameters())
    output = grad_norm(bridge.adapter.output.parameters())
    if not math.isfinite(before) or before <= 0:
        raise FloatingPointError("A finite positive aggregate adapter gradient is required")
    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    after = grad_norm(parameters)
    optimizer.step()
    require_finite_tree(optimizer.state_dict())
    if any(not torch.isfinite(p).all() for p in parameters):
        raise FloatingPointError("Nonfinite adapter parameters after update")
    if any(p.requires_grad or p.grad is not None for p in bridge.core.parameters()):
        raise RuntimeError("Core gradient ownership changed")
    return {"auxiliary":aux,"total_objective":sum(row["future_flow_mse"] for row in branches)/2+aux["weighted_loss"],"branches": branches, "paired_mean_future_flow_mse": sum(row["future_flow_mse"] for row in branches) / 2,
            "gradient_l2_before_clip": before, "gradient_l2_after_clip": after,
            "command_gru_gradient_l2": recurrent, "output_gradient_l2": output,
            "live_sequential_forwards": 2, "optimizer_updates": 1,
            "loss": "Mean future latent velocity MSE; half each branch; initial latent excluded"}



def execute_steps(bridge, windows, schedule, draws, context, optimizer, *, native_predict,
                  retain, checkpoint, progress, negative_context, check=lambda: None, synchronize=lambda: None):
    """Bounded numerical protocol; callbacks retain evidence and enforce guards.

    The production worker supplies the literal native prediction and CUDA
    synchronizer. CPU fixtures supply explicit stand-ins. Both zero-adapter
    comparisons finish before any optimizer update. No sampling or decoding.
    """
    if (len(schedule) != 2 or [row.get("start") for row in schedule] != [0, 8]
            or [row.get("branches") for row in schedule] != [["closed-0000", "open-0000"], ["closed-0008", "open-0008"]]):
        raise ValueError("Only the prescribed two-update start0/start8 schedule is accepted")
    if torch.count_nonzero(bridge.adapter.output.weight) or torch.count_nonzero(bridge.adapter.output.bias):
        raise ValueError("The numerical probe must start with the exact zero output projection")
    if optimizer.state:
        raise ValueError("The numerical probe must start with a fresh optimizer")
    report = {"completed_updates": 0, "native_comparisons": [], "updates": [],
              "native_reference_predictions": 0, "bridge_predictions": 0,
              "zero_adapter_gate_passed": False, "image_generation": False, "quality_assessed": False}
    checkpoint(0, report)
    progress(report)
    device = next(bridge.adapter.parameters()).device
    row = schedule[0]
    for identity in row["branches"]:
        check()
        window = windows[identity]
        noisy, times, velocity = flow_inputs(window["target"], window["observation"], draws[row["noise_key"]], row["k"])
        hashes = condition_identity(noisy, times, velocity, window)
        synchronize(); begin = time.monotonic()
        reference = native_predict(noisy, times, context)
        synchronize(); native_seconds = time.monotonic() - begin
        report["native_reference_predictions"] += 1
        retain("parity-" + identity + "-native", {"native_velocity": reference})
        report["stage"] = "zero-adapter/" + identity + "/bridge"
        progress(report)
        check()
        synchronize(); begin = time.monotonic()
        bridge_input = noisy.to(device)
        actual = bridge(bridge_input, times.to(device), [context.to(device)],
                        commands=window["commands"].to(device), observation=window["observation"].to(device), track_grad=False)
        synchronize(); bridge_seconds = time.monotonic() - begin
        actual = actual.detach().cpu()
        report["bridge_predictions"] += 1
        retain("parity-" + identity + "-bridge", {"bridged_velocity": actual})
        report["stage"] = "zero-adapter/" + identity + "/compare"
        progress(report)
        comparison = compare_velocity(reference, actual)
        prefix_exact = torch.equal(bridge_input[:, :, :1], window["observation"].to(device))
        unchanged = hashes == condition_identity(noisy, times, velocity, window)
        comparison.update(window_id=identity, input_sha256=hashes, native_seconds=native_seconds,
                          bridge_seconds=bridge_seconds, input_tensors_unchanged=unchanged,
                          observed_input_prefix_exact=prefix_exact)
        report["native_comparisons"].append(comparison)
        progress(report)
        if not prefix_exact or not unchanged:
            raise RuntimeError("Parity call mutated an input or its clean prefix")
        del noisy, times, velocity, reference, actual, bridge_input
    if not all(row["passed"] for row in report["native_comparisons"]):
        raise RuntimeError("Zero-adapter native parity gate failed; no optimizer updates are permitted")
    report["zero_adapter_gate_passed"] = True
    progress(report)
    for index, row in enumerate(schedule, 1):
        check()
        synchronize(); begin = time.monotonic()
        result = paired_update(bridge, [windows[identity] for identity in row["branches"]],
                               draws[row["noise_key"]], row["k"], context, optimizer, check=check, auxiliary=row["start"]==0, negative_context=negative_context, retain=retain, update=index)
        synchronize(); result["seconds"] = time.monotonic() - begin
        result["update"] = index
        result["start"] = row["start"]
        report["bridge_predictions"] += result["live_sequential_forwards"]
        report["updates"].append(result)
        progress(report)
        recurrent = result["command_gru_gradient_l2"]
        if (index == 1 and recurrent != 0.) or (index == 2 and (not math.isfinite(recurrent) or recurrent <= 0.)):
            raise RuntimeError("Expected first-zero then second-positive recurrent gradient did not occur")
        # Only fully validated updates may advance the immutable recovery.
        report["completed_updates"] = index
        checkpoint(index, report)
        progress(report)
    if report["native_reference_predictions"] != 2 or report["bridge_predictions"] != 6:
        raise RuntimeError("The prescribed native/bridge prediction counts differ")
    report["second_update_gru_gradient_nonzero"] = True
    progress(report)
    return report
