"""Bounded local training CLI for the separate original RGB world experiment."""
import argparse
import copy
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import make_dataset, sample_batch
from .evaluate import evaluate
from .model import RGBModel


def log(record):
    print(json.dumps(record, allow_nan=False), flush=True)


def device_for(name):
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def tensors(batch, device):
    return tuple(torch.from_numpy(value).to(device) for value in batch)


def training_loss(model, history, action, target, noise=None, times=None):
    if model.kind == "flow":
        noise = torch.randn_like(target) if noise is None else noise
        times = torch.rand(len(target), device=target.device) if times is None else times
        noisy = (1-times[:, None, None, None])*noise+times[:, None, None, None]*target
        return F.mse_loss(model(history, action, noisy, times), target-noise)
    prediction = model(history, action)
    horizontal = F.l1_loss(prediction[..., 1:]-prediction[..., :-1], target[..., 1:]-target[..., :-1])
    vertical = F.l1_loss(prediction[..., 1:, :]-prediction[..., :-1, :], target[..., 1:, :]-target[..., :-1, :])
    return F.l1_loss(prediction, target)+.1*(horizontal+vertical)


def run(args):
    if args.steps < 1 or args.batch < 1 or args.max_seconds <= 0 or args.log_every < 1:
        raise ValueError("steps, batch, max-seconds and log-every must be positive")
    if not 0 <= args.context_noise <= .25 or not 1 <= args.sample_steps <= 64:
        raise ValueError("invalid context noise or sampling steps")
    if args.train_scenes < 1 or args.eval_scenes < 1:
        raise ValueError("scene counts must be positive")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty; existing results are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(output)
    device = device_for(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    log({"event": "start", "device": str(device), "kind": args.kind, "output": str(output)})
    started = time.perf_counter()
    def data_log(record):
        if record["completed_scenes"] % 4 == 0 or record["completed_scenes"] == record["scenes"]:
            log(record)
    train = make_dataset("train", args.train_scenes, args.episodes, args.horizon, args.size, data_log)
    validation = make_dataset("validation", args.eval_scenes, args.episodes, args.horizon, args.size, data_log)
    test = make_dataset("test", args.eval_scenes, args.episodes, args.horizon, args.size, data_log)
    data_seconds = time.perf_counter()-started
    manifests = {data["split"]: data["records"] for data in (train, validation, test)}
    scene_sets = [{r["scene_seed"] for r in data["records"]} for data in (train, validation, test)]
    if any(scene_sets[a] & scene_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("scene split overlap")
    hashes = [{r["sha256"] for r in data["records"]} for data in (train, validation, test)]
    if any(hashes[a] & hashes[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("trajectory duplicate across splits")
    (output / "dataset-manifest.json").write_text(json.dumps(manifests, indent=2))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    model = RGBModel(args.kind, args.width).to(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.0001)
    held_batch = tensors(sample_batch(validation, min(64, args.eval_scenes*args.episodes*args.horizon),
                                     np.random.default_rng(300_001)), device)
    val_noise_rng = torch.Generator(device="cpu").manual_seed(300_002)
    held_noise = torch.randn(held_batch[2].shape, generator=val_noise_rng).to(device)
    held_times = torch.rand(len(held_batch[2]), generator=val_noise_rng).to(device)
    def val_loss():
        with torch.inference_mode():
            return float(training_loss(ema, *held_batch, held_noise, held_times).cpu())
    untrained_loss = val_loss()
    best_loss, best_step = untrained_loss, 0
    best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
    records = []
    step = 0
    train_started = time.perf_counter()
    for step in range(1, args.steps+1):
        if time.perf_counter()-train_started >= args.max_seconds:
            step -= 1
            break
        model.train()
        history, action, target = tensors(sample_batch(train, args.batch, np_rng), device)
        if args.context_noise:
            history = (history+torch.randn_like(history)*args.context_noise).clamp(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        loss = training_loss(model, history, action, target)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        with torch.no_grad():
            for averaged, current in zip(ema.parameters(), model.parameters()):
                averaged.lerp_(current, .01)
        if step % args.log_every == 0 or step == args.steps:
            score = val_loss()
            record = {"event": "training", "step": step, "train_loss": float(loss.detach().cpu()),
                      "validation_loss": score, "training_seconds": time.perf_counter()-train_started}
            records.append(record)
            log(record)
            if score < best_loss:
                best_loss, best_step = score, step
                best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
    # A capped run still receives one final validation check, then untouched test evaluation.
    score = val_loss()
    if score < best_loss:
        best_loss, best_step = score, step
        best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
    training_seconds = time.perf_counter()-train_started
    checkpoint = {"state_dict": best_state, "config": config, "selected_step": best_step,
                  "validation_loss": best_loss, "license": "Apache-2.0",
                  "architecture": "original RGB action-conditioned U-Net", "teacher": "original NumPy room simulator"}
    torch.save(checkpoint, output / "model.pt")
    resume = {"state_dict": model.state_dict(), "ema_state_dict": ema.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(), "config": config, "step": step,
              "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np_rng.bit_generator.state}
    if device.type == "mps":
        resume["mps_rng_state"] = torch.mps.get_rng_state()
    if device.type == "cuda":
        resume["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(resume, output / "training-state.pt")
    ema.load_state_dict(best_state)
    log({"event": "evaluation", "selected_step": best_step, "completed_steps": step,
         "training_seconds": training_seconds, "capped": step < args.steps})
    evaluation_started = time.perf_counter()
    test_metrics = evaluate(ema, test, str(device), args.sample_steps, output)
    metrics = {"kind": args.kind, "parameters": sum(p.numel() for p in model.parameters()),
               "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
               "numpy": np.__version__, "data_seconds": data_seconds, "training_seconds": training_seconds,
               "evaluation_seconds": time.perf_counter()-evaluation_started,
               "completed_steps": step, "requested_steps": args.steps, "capped": step < args.steps,
               "selected_step": best_step, "untrained_validation_loss": untrained_loss,
               "selected_validation_loss": best_loss, "training_log": records, "test": test_metrics,
               "claims": "Original small synthetic RGB prediction experiment. No Genie parity or scientific novelty claim."}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    log({"event": "complete", "metrics": str(output / "metrics.json"),
         "one_step": test_metrics["one_step"], "closed_loop": test_metrics["closed_loop"]["horizons"]})
    return metrics


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", required=True, help="New or empty output directory")
    result.add_argument("--kind", choices=["flow", "predictor"], default="flow")
    result.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    result.add_argument("--size", type=int, default=64)
    result.add_argument("--width", type=int, default=24)
    result.add_argument("--steps", type=int, default=1200)
    result.add_argument("--max-seconds", type=float, default=600, help="Training loop wall time cap; data/evaluation timings are separate")
    result.add_argument("--batch", type=int, default=16)
    result.add_argument("--train-scenes", type=int, default=16)
    result.add_argument("--eval-scenes", type=int, default=4)
    result.add_argument("--episodes", type=int, default=2)
    result.add_argument("--horizon", type=int, default=32)
    result.add_argument("--learning-rate", type=float, default=.0003)
    result.add_argument("--context-noise", type=float, default=.02)
    result.add_argument("--sample-steps", type=int, default=8)
    result.add_argument("--log-every", type=int, default=100)
    result.add_argument("--seed", type=int, default=71991)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
