"""Reproduce the two original Worldline checkpoints and held-out measurements.

Run from the project directory: python -m worldline.train
The default profile is intentionally small enough for a 24 GB Apple laptop.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import terrain_batch, ecology_states, ecology_teacher
from .models import SpatialFlowNet, EcologyNet, Generator, choose_device, SIZE


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def flow_loss(model, data, labels, noise, times):
    total = 0.
    for i in range(0, len(data), 16):
        target, z, t = data[i:i+16], noise[i:i+16], times[i:i+16]
        value = (1-t[:, None,None,None]) * z + t[:,None,None,None] * target
        total += F.mse_loss(model(value,t,labels[i:i+16]), target-z).item() * len(target)
    return total / len(data)


@torch.inference_mode()
def evaluate_flow_checkpoint(directory, device, seed=314159):
    """Final diagnostics on seeds not used for training or EMA selection."""
    sampler = Generator(directory/"spatial-flow.pt",device=device)
    model = sampler.model
    training, training_labels = terrain_batch(1536,SIZE,seed)
    test, labels = terrain_batch(96,SIZE,seed+900000)
    rng = torch.Generator().manual_seed(seed+900001)
    noise = torch.randn(test.shape,generator=rng).to(device)
    times = torch.rand(len(test),generator=rng).to(device)
    data, labels_device = test.to(device),labels.to(device)
    test_loss = flow_loss(model,data,labels_device,noise,times)
    means = torch.stack([training[training_labels==b].mean(dim=(0,2,3)) for b in range(3)]).to(device)
    variances = torch.stack([training[training_labels==b].var(dim=(0,2,3)) for b in range(3)]).to(device)
    mu,squared_sigma = means[labels_device,:,None,None],variances[labels_device,:,None,None]
    t=times[:,None,None,None]
    x=noise*(1-t)+data*t
    gain=(t*squared_sigma-(1-t))/(t.square()*squared_sigma+(1-t).square())
    independent_baseline=mu+gain*(x-t*mu)
    baseline_loss=F.mse_loss(independent_baseline,data-noise).item()
    samples=np.load(directory/"sample-fields.npz")["fields"]
    nearest=[]
    training_np=training.numpy()
    for index,sample in enumerate(samples):
        same_biome=training_np[training_labels.numpy()==index%3]
        distances=((same_biome-sample[None])**2).mean(axis=(1,2,3))
        nearest.append(float(np.sqrt(distances.min())))
    result={"untouched_test_examples":len(test),"untouched_test_seed":seed+900000,"untouched_test_flow_mse":test_loss,"independent_pixel_gaussian_baseline_test_flow_mse":baseline_loss,"test_loss_reduction_vs_independent_pixel_baseline":1-test_loss/baseline_loss,"sample_nearest_training_rmse":nearest,"sample_nearest_training_rmse_mean":float(np.mean(nearest)),"novelty_diagnostic_limit":"Pixel distance to the nearest training field detects exact copies only; it does not demonstrate novel concepts, generalization beyond the synthetic distribution, or visual superiority."}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource
        figure,axes=plt.subplots(3,4,figsize=(12,8),layout="constrained")
        for biome in range(3):
            a,b=samples[biome],samples[biome+3]
            reference=test[labels==biome][0].numpy()
            for column,field in enumerate([a[0],a[1],b[0],reference[0]]):
                ax=axes[biome,column]
                if column==1:
                    ax.imshow((field+1)/2,vmin=0,vmax=1,cmap="YlGn")
                else:
                    ax.imshow(LightSource(azdeg=315,altdeg=45).shade(field,cmap=plt.get_cmap("terrain"),vert_exag=2,blend_mode="soft"))
                ax.set_xticks([]);ax.set_yticks([])
                if biome==0:
                    ax.set_title(["Generated height A","Generated vegetation A","Generated height B","Unseen teacher example"][column],fontsize=10)
                if column==0:
                    ax.set_ylabel(["Alpine","Desert","Alien"][biome],fontsize=12)
        figure.suptitle("Worldline: locally trained spatial fields",fontsize=16)
        figure.savefig(directory/"generated-fields-overview.png",dpi=140)
        plt.close(figure)
        result["field_overview"]="generated-fields-overview.png"
    except ImportError:
        result["field_overview"]="Plotting skipped: optional matplotlib dependency unavailable"
    return result


def train_flow(args, device, directory):
    torch.manual_seed(args.seed)
    train, labels = terrain_batch(1536, SIZE, args.seed)
    heldout, heldout_labels = terrain_batch(192, SIZE, args.seed+10000)
    train, labels = train.to(device), labels.to(device)
    heldout, heldout_labels = heldout.to(device), heldout_labels.to(device)
    noise = torch.randn_like(heldout)
    times = torch.rand(len(heldout), device=device)
    model = SpatialFlowNet().to(device)
    untrained = flow_loss(model, heldout, heldout_labels, noise, times)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    start, history = time.perf_counter(), []
    for step in range(args.flow_steps):
        indices = torch.randint(0, len(train), (args.batch_size,), device=device)
        target, biome = train[indices], labels[indices]
        z = torch.randn_like(target)
        t = torch.rand(len(target), device=device)
        value = z * (1-t[:,None,None,None]) + target * t[:,None,None,None]
        predicted = model(value, t, biome)
        loss = F.mse_loss(predicted, target-z)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        with torch.no_grad():
            decay = min(.995, (1+step)/(10+step))
            for average, current in zip(ema.parameters(), model.parameters()):
                average.lerp_(current, 1-decay)
        if (step+1) % 100 == 0 or step == 0:
            synchronize(device)
            entry = {"step":step+1,"loss":float(loss.item()),"seconds":round(time.perf_counter()-start,2)}
            history.append(entry)
            print("flow", json.dumps(entry), flush=True)
    model.eval()
    raw_loss = flow_loss(model, heldout, heldout_labels, noise, times)
    ema_loss = flow_loss(ema, heldout, heldout_labels, noise, times)
    chosen = ema if ema_loss <= raw_loss else model
    final_loss = min(raw_loss, ema_loss)
    metadata = {"architecture":"conditional spatial rectified flow","channels":["height","vegetation_density"],"size":SIZE,"biomes":["alpine","desert","alien"],"training_data":"original synthetic terrain, disjoint validation seeds","training_examples":len(train),"heldout_examples":len(heldout),"seed":args.seed,"training_steps":args.flow_steps,"trained_from_scratch":True,"parameters":sum(p.numel() for p in model.parameters())}
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in chosen.state_dict().items()},"metadata":metadata},directory/"spatial-flow.pt")
    result = {**metadata,"device":str(device),"untrained_heldout_flow_mse":untrained,"trained_heldout_flow_mse":final_loss,"relative_loss_reduction":1-final_loss/untrained,"raw_model_loss":raw_loss,"ema_model_loss":ema_loss,"selected":"ema" if chosen is ema else "raw","training_seconds":time.perf_counter()-start,"history":history}
    sampler = Generator(directory/"spatial-flow.pt", device=device)
    sampling_start = time.perf_counter()
    samples = np.stack([sampler.sample(60000+i, i%3, steps=24) for i in range(12)])
    result["sample_seconds_mean"] = (time.perf_counter()-sampling_start)/12
    reference = heldout.cpu().numpy()
    result["field_statistics"] = {}
    for biome in range(3):
        truth = reference[heldout_labels.cpu().numpy()==biome]
        generated = samples[np.arange(12)%3==biome]
        result["field_statistics"][str(biome)] = {"reference_mean":truth.mean(axis=(0,2,3)).tolist(),"generated_mean":generated.mean(axis=(0,2,3)).tolist(),"reference_std":truth.std(axis=(0,2,3)).tolist(),"generated_std":generated.std(axis=(0,2,3)).tolist(),"generated_neighbor_difference":float(np.abs(np.diff(generated[:,0],axis=-1)).mean())}
    np.savez_compressed(directory/"sample-fields.npz",fields=samples,biomes=np.arange(12)%3)
    result.update(evaluate_flow_checkpoint(directory,device,args.seed))
    return result


def train_ecology(args, device, directory):
    torch.manual_seed(args.seed+1)
    states = ecology_states(1024,32,args.seed).to(device)
    heldout = ecology_states(128,32,args.seed+20000).to(device)
    fixed_rain = torch.rand(len(heldout),device=device)
    fixed_heat = torch.rand(len(heldout),device=device)
    fixed_dt = torch.ones(len(heldout),device=device)
    target = ecology_teacher(heldout,fixed_rain,fixed_heat,fixed_dt)
    model = EcologyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    with torch.no_grad():
        untrained = F.mse_loss(model(heldout,fixed_rain,fixed_heat,fixed_dt),target).item()
    start, history = time.perf_counter(), []
    for step in range(args.dynamics_steps):
        indices = torch.randint(0,len(states),(args.batch_size,),device=device)
        state = states[indices]
        rain, heat = torch.rand(args.batch_size,device=device), torch.rand(args.batch_size,device=device)
        dt = torch.rand(args.batch_size,device=device)*1.75+.25
        desired = ecology_teacher(state,rain,heat,dt)
        predicted = model(state,rain,heat,dt)
        loss = F.mse_loss(predicted,desired)*10000
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (step+1)%100==0 or step==0:
            entry={"step":step+1,"scaled_loss":float(loss.item()),"seconds":round(time.perf_counter()-start,2)}
            history.append(entry)
            print("ecology",json.dumps(entry),flush=True)
    model.eval()
    with torch.inference_mode():
        predicted = model(heldout,fixed_rain,fixed_heat,fixed_dt)
        one_step = F.mse_loss(predicted,target).item()
        persistence = F.mse_loss(heldout,target).item()
        per_channel = (predicted-target).square().mean(dim=(0,2,3)).cpu().tolist()
        learned, truth = heldout[:32].clone(),heldout[:32].clone()
        rollout=[]
        for index in range(32):
            truth = ecology_teacher(truth,fixed_rain[:32],fixed_heat[:32],fixed_dt[:32])
            learned = model(learned,fixed_rain[:32],fixed_heat[:32],fixed_dt[:32])
            if index+1 in (1,4,8,16,32):
                rollout.append({"steps":index+1,"model_mse":F.mse_loss(learned,truth).item(),"persistence_mse":F.mse_loss(heldout[:32],truth).item()})
    metadata={"architecture":"residual convolutional field dynamics","channels":["water","vegetation","heat"],"training_data":"original synthetic reaction/diffusion teacher","training_examples":len(states),"heldout_examples":len(heldout),"seed":args.seed+1,"training_steps":args.dynamics_steps,"trained_from_scratch":True,"parameters":sum(p.numel() for p in model.parameters())}
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"metadata":metadata},directory/"ecology.pt")
    return {**metadata,"device":str(device),"untrained_heldout_mse":untrained,"heldout_mse":one_step,"persistence_heldout_mse":persistence,"mse_reduction_vs_persistence":1-one_step/persistence,"mse_by_channel":per_channel,"rollouts":rollout,"training_seconds":time.perf_counter()-start,"history":history}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-steps",type=int,default=1600)
    parser.add_argument("--dynamics-steps",type=int,default=1000)
    parser.add_argument("--batch-size",type=int,default=24)
    parser.add_argument("--seed",type=int,default=314159)
    parser.add_argument("--device",default=None)
    parser.add_argument("--only",choices=["flow","ecology","both"],default="both")
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent.parent/"checkpoints")
    args=parser.parse_args()
    torch.set_num_threads(6)
    args.output.mkdir(parents=True,exist_ok=True)
    device=choose_device(args.device)
    metrics_path=args.output/"training-metrics.json"
    metrics=json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update({"evaluation_scope":"Synthetic held-out fields only. No Genie 3 comparison; no claim of general physical accuracy or photorealistic neural video.","torch_version":str(torch.__version__)})
    if args.only in ("flow","both"):
        metrics["spatial_flow"]=train_flow(args,device,args.output)
        metrics_path.write_text(json.dumps(metrics,indent=2)+"\n")
    if args.only in ("ecology","both"):
        metrics["ecology"]=train_ecology(args,device,args.output)
        metrics_path.write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps({"checkpoint_directory":str(args.output),"completed":args.only}),flush=True)


if __name__ == "__main__":
    main()
