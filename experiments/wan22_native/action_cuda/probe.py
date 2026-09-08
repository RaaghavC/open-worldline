# SPDX-License-Identifier: Apache-2.0
"""Prepare immutable inputs by default; separately admit exactly two CUDA updates."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import torch
from safetensors.torch import save_file

from ..action_adapter.model import PostBlockActionAdapter
from ..action_training.objective import parameter_records, loaded_records, save_checkpoint
from ..cuda_reference.evidence import PRECISION
from ..cuda_reference.guards import LIMITS
from ..official_cpu.streaming import sha, tensor_sha
from ..spatial_reference.guards import atomic, hardware, Monitor, stop_child, supervise
from .bridge import NativeCUDAActionBridge, PROFILES
from . import probe_evidence as evidence
from . import probe_math as numerical


SCHEMA = "worldline-wan22-action-cuda-probe-v1"


def protocol(profile):
    if profile not in PROFILES:
        raise ValueError("Declared baseline or spatial profile required")
    return {"schema": SCHEMA, "scope": evidence.SCOPE, "profile": profile,
            "seed": numerical.SEED, "paired_updates": 2, "starts": [0, 8],
            "branch_order": ["closed", "open"], "batch_per_forward": 1,
            "adapter_parameters": 947712, "adapter_dtype": "float32", "precision": PRECISION,
            "shape": [1, *PROFILES[profile]["shape"]], "native_parity": numerical.PARITY,
            "cross_length_codec_prefix": numerical.PREFIX,
            "objective": "Half each branch's mean future-only latent velocity MSE; positions1..4; no initial loss",
            "optimizer": {**numerical.OPTIMIZER, "betas": list(numerical.OPTIMIZER["betas"])},
            "gradient_clip_l2": 1., "limits": LIMITS,
            "draw_order": "Private CPU generator20260907; randint(50,951) then full FP32 randn for each pair",
            "saved_bytes_are_canonical": True, "noise_regenerated_at_execution": False,
            "warm_start": False, "resume_supported": False,
            "image_generation": False, "fixed16_admitted": False, "quality_assessed": False}


def record_file(path, values):
    return {"sha256": sha(path), "tensors": {name: {"shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."), "sha256": tensor_sha(value)} for name, value in values.items()}}


def _initial_specs():
    with torch.device("meta"):
        adapter = PostBlockActionAdapter()
    return {name: (tuple(value.shape), "F32") for name, value in adapter.state_dict().items()}


def prepare(*, cache_run, text_directory, profile, cpu_report, independent_report, output):
    out = Path(output).absolute()
    if (out.exists() or any(p.is_symlink() for p in (out, *out.parents))
            or out.resolve().is_relative_to(evidence.REPO)
            or out.resolve().is_relative_to(Path(cache_run).resolve())
            or out.resolve().is_relative_to(Path(text_directory).resolve())):
        raise ValueError("A fresh preparation directory outside source and input directories is required")
    out.mkdir(parents=True)
    report = {"schema": SCHEMA, "status": "preparing", "model_execution": False, "quality_assessed": False}
    atomic(out/"metrics.json", report)
    try:
        reviews = {"probe_cpu": evidence.review(cpu_report),
                   "probe_independent": evidence.review(independent_report, independent=True),
                   "bridge": evidence.bridge_reviews()}
        windows, cache_identity = evidence.load_cache(cache_run, profile)
        del windows
        positive, text_identity = evidence.load_positive(text_directory)
        mapping = evidence.snapshot(out)
        adapter = numerical.fresh_adapter()
        initial = {name: value.detach().contiguous().clone() for name, value in adapter.state_dict().items()}
        schedule, draws = numerical.make_draws("probe", shape=(1, *PROFILES[profile]["shape"]))
        draws["rng_initial"] = torch.Generator(device="cpu").manual_seed(numerical.SEED).get_state().clone()
        saved = {"initial-adapter.safetensors": initial, "draws.safetensors": draws,
                 "positive.safetensors": {"atrium": positive},
                 "initial-cpu-rng.safetensors": {"torch_cpu_rng": torch.get_rng_state().clone()}}
        artifacts = {}
        for name, values in saved.items():
            save_file(values, str(out/name))
            artifacts[name] = record_file(out/name, values)
        for name, source in (("cpu-report.json", cpu_report), ("independent-report.json", independent_report)):
            shutil.copyfile(source, out/name)
        plan = {"schema": SCHEMA, "status": "prepared", "profile": profile, "protocol": protocol(profile),
                "source_sha256": mapping, "reviews": reviews, "schedule": schedule,
                "input_identity": {"cache": cache_identity, "text": text_identity, "prepared_artifacts": artifacts},
                "preparation_runtime": {"python": platform.python_version(), "torch": torch.__version__,
                                        "machine": platform.machine(), "platform": platform.platform()},
                "canonical_byte_policy": "Execute saved tensor bytes; never regenerate Gaussian noise or initial parameters on another CPU architecture"}
        if evidence.source_hashes() != mapping:
            raise ValueError("Sources changed during preparation")
        atomic(out/"plan.json", plan)
        report.update(status="prepared", plan_sha256=sha(out/"plan.json"),
                      execute_requires="Separate current-source, exact-plan numerical-only parent admission")
        return plan
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic(out/"metrics.json", report)


def read_prepared(directory, cache_run):
    root = Path(directory)
    plan = evidence.read_json(evidence.relative_file(root, "plan.json"))
    if plan.get("schema") != SCHEMA or plan.get("status") != "prepared" or plan.get("protocol") != protocol(plan.get("profile")):
        raise ValueError("Immutable prepared numerical-probe plan required")
    if str(plan.get("preparation_runtime", {}).get("torch", "")).split("+")[0] != "2.5.1":
        raise ValueError("Canonical preparation must record the declared Torch2.5.1 runtime")
    evidence.validate_sources(root, plan["source_sha256"])
    if evidence.bridge_reviews() != plan["reviews"]["bridge"]:
        raise ValueError("Bridge reviews differ")
    for name, field, independent in (("cpu-report.json", "probe_cpu", False), ("independent-report.json", "probe_independent", True)):
        if evidence.review(evidence.relative_file(root, name), independent=independent) != plan["reviews"][field]:
            raise ValueError("Prepared probe review differs")
    shape = (1, *PROFILES[plan["profile"]]["shape"])
    rng_shape = (torch.get_rng_state().numel(),)
    specs = {"initial-adapter.safetensors": _initial_specs(),
             "draws.safetensors": {"noise_0000": (shape, "F32"), "noise_0001": (shape, "F32"),
                                    "rng_after_0000": (rng_shape, "U8"), "rng_after_0001": (rng_shape, "U8"),
                                    "rng_initial": (rng_shape, "U8")},
             "positive.safetensors": {"atrium": ((25, 4096), "F32")},
             "initial-cpu-rng.safetensors": {"torch_cpu_rng": (rng_shape, "U8")}}
    records = plan["input_identity"]["prepared_artifacts"]
    if set(records) != set(specs):
        raise ValueError("Every exact prepared tensor artifact is required")
    saved = {}
    for name, expected in specs.items():
        saved[name] = evidence.tensors(evidence.relative_file(root, name), expected, records[name]["sha256"])
        if record_file(root/name, saved[name]) != records[name]:
            raise ValueError("Prepared tensor hashes differ")
    initial = saved["initial-adapter.safetensors"]
    if torch.count_nonzero(initial["output.weight"]) or torch.count_nonzero(initial["output.bias"]):
        raise ValueError("Prepared adapter must have a zero output projection, never trained values")
    schedule = plan.get("schedule")
    if not isinstance(schedule, list) or len(schedule) != 2:
        raise ValueError("Exactly two canonical draw records required")
    draws = saved["draws.safetensors"]
    for index, (row, start) in enumerate(zip(schedule, (0, 8))):
        if (row.get("update") != index+1 or row.get("start") != start
                or row.get("branches") != [f"closed-{start:04d}", f"open-{start:04d}"]
                or type(row.get("k")) is not int or not 50 <= row["k"] <= 950
                or row.get("sigma") != row["k"]/1000. or row.get("noise_key") != f"noise_{index:04d}"
                or row.get("noise_sha256") != tensor_sha(draws[f"noise_{index:04d}"])
                or row.get("rng_after_sha256") != tensor_sha(draws[f"rng_after_{index:04d}"])):
            raise ValueError("Canonical schedule or saved draw identity differs")
    positive = saved["positive.safetensors"]["atrium"]
    if tensor_sha(positive) != plan["input_identity"]["text"]["positive_tensor_sha256"]:
        raise ValueError("Prepared positive text tensor differs")
    windows, cache_identity = evidence.load_cache(cache_run, plan["profile"])
    if cache_identity != plan["input_identity"]["cache"]:
        raise ValueError("Native CUDA cache differs from the prepared plan")
    return plan, windows, initial, draws, positive, saved["initial-cpu-rng.safetensors"]["torch_cpu_rng"]


def _runtime_flags():
    from ..cuda_reference.vendor import attention
    return {"matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "flash_attention_2_available": attention.FLASH_ATTN_2_AVAILABLE,
            "flash_attention_3_available": attention.FLASH_ATTN_3_AVAILABLE}


def _original_weights(records):
    expected = evidence.read_json(evidence.HERE.parent/"cuda_reference/expected-weights.json")["tensors"]
    if (records.get("tensor_count") != 825 or records.get("parameter_count") != 4999787712
            or records.get("parameter_bytes") != 19999150848 or records.get("convert_model_dtype") is not False
            or records.get("all_shards_verified") is not True or records.get("cuda_copy_exact") is not True
            or set(records.get("tensors", {})) != set(expected)):
        raise ValueError("Exactly all original FP32 native parameters must be verified")
    for name, row in records["tensors"].items():
        wanted = expected[name]
        if (row.get("source_sha256") != wanted["original_sha256"] or row.get("loaded_sha256") != wanted["original_sha256"]
                or row.get("original_dtype") != "float32" or row.get("loaded_dtype") != "float32"
                or row.get("shape") != wanted["shape"] or row.get("shard") != wanted["shard"]
                or row.get("cuda_copy_exact") is not True or row.get("source_owner_released") is not True):
            raise ValueError("Original native value identity differs: " + name)
    return loaded_records(records)


def native_reference(core, noisy, times, context):
    """Literal native call; caller retains completed values before scoring them.

    No alternate attention or embedding path. The bridge already verifies the
    pinned literal model, FA2 availability and original FP32 storage. Keeping
    finiteness scoring in execute_steps allows a completed nonfinite reference
    to be retained before the numerical gate rejects it.
    """
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        values = core([noisy[0].to("cuda:0")], times.to("cuda:0"), [context.to("cuda:0")], times.shape[1])
    torch.cuda.synchronize()
    if not isinstance(values, (tuple, list)) or len(values) != 1 or not isinstance(values[0], torch.Tensor):
        raise RuntimeError("Native forward must return one velocity tensor")
    return values[0].float().cpu().unsqueeze(0)


def worker(config):
    root = Path(config["prepared_directory"])
    out = root/"result"
    out.mkdir(exist_ok=False)
    began = time.monotonic()
    report = {"schema": SCHEMA, "status": "running", "scope": evidence.SCOPE,
              "model_execution": False, "completed_updates": 0, "limits": LIMITS,
              "quality_assessed": False, "image_generation": False, "fixed16_admitted": False}
    atomic(out/"metrics.json", report)
    try:
        plan, windows, initial, draws, context, cpu_rng = read_prepared(root, config["cache_run"])
        if sha(root/"plan.json") != config["plan_sha256"]:
            raise ValueError("Prepared plan changed after launch")
        admission = evidence.validate_admission(config["admission"], root, plan, config["expected_gpu"])
        if admission != config["admission_record"]:
            raise ValueError("Admission changed after launch")
        report.update(profile=plan["profile"], protocol=plan["protocol"], source_sha256=plan["source_sha256"],
                      input_identity=plan["input_identity"], plan_sha256=config["plan_sha256"],
                      admission=admission, schedule=plan["schedule"], precision=PRECISION)
        report["hardware"] = hardware(config["expected_gpu"])
        if report["hardware"] != plan["input_identity"]["cache"]["hardware"]:
            raise ValueError("CUDA runtime/hardware differs from the completed cache")
        report["runtime_flags"] = _runtime_flags()
        atomic(out/"metrics.json", report)
        with Monitor(out, config["deadline"], "pair") as monitor:
            def check():
                monitor.check()
                if _runtime_flags() != report["runtime_flags"]:
                    raise RuntimeError("CUDA precision/determinism/attention flags changed during the probe")
            from ..cuda_reference.native import load_model
            begin = time.monotonic()
            core, weights = load_model(config["weights"], check)
            torch.cuda.synchronize()
            report["load_seconds"] = time.monotonic()-begin
            atomic(out/"weight-load.json", weights)
            begin = time.monotonic()
            before = parameter_records(core, check=check, expected=_original_weights(weights))
            report["before_hash_seconds"] = time.monotonic()-begin
            atomic(out/"core-before.json", before)
            adapter = PostBlockActionAdapter()
            adapter.load_state_dict(initial, strict=True)
            adapter.to("cuda:0")
            torch.set_rng_state(cpu_rng)
            torch.cuda.manual_seed_all(numerical.SEED)
            for name, value in adapter.state_dict().items():
                verification = value.detach().cpu()
                if tensor_sha(verification) != tensor_sha(initial[name]):
                    raise ValueError("Initial adapter CUDA copy differs from saved canonical values")
                del verification
            bridge = NativeCUDAActionBridge(core, adapter, profile=plan["profile"])
            optimizer = torch.optim.AdamW(adapter.parameters(), **numerical.OPTIMIZER)
            report["model_execution"] = True
            identity = {"schema": SCHEMA, "source_sha256": plan["source_sha256"], "plan_sha256": config["plan_sha256"],
                        "admission_sha256": admission["admission_sha256"], "input_identity": plan["input_identity"],
                        "protocol": plan["protocol"], "hardware": report["hardware"]}
            def retain(name, values):
                path = out/(name+".safetensors")
                if path.exists():
                    raise ValueError("Refuse overwriting numerical evidence")
                save_file({key: value.detach().cpu().contiguous() for key, value in values.items()}, str(path))
            def checkpoint(completed, result):
                check()
                retain(f"cuda-rng-{completed:04d}", {"cuda_rng_0": torch.cuda.get_rng_state(0)})
                private_rng = draws["rng_initial"] if completed == 0 else draws[f"rng_after_{completed-1:04d}"]
                record = save_checkpoint(out, adapter, optimizer, completed, identity=identity, draw_rng_state=private_rng)
                record["cuda_rng_file_sha256"] = sha(out/f"cuda-rng-{completed:04d}.safetensors")
                report["last_checkpoint"] = record
            def progress(values):
                report.update(values)
                atomic(out/"metrics.json", report)
            numerical.execute_steps(bridge, windows, plan["schedule"], draws, context, optimizer,
                native_predict=lambda x, t, c: native_reference(core, x, t, c),
                retain=retain, checkpoint=checkpoint, progress=progress, check=check, synchronize=torch.cuda.synchronize)
            begin = time.monotonic()
            after = parameter_records(core, check=check, expected=before)
            report["after_hash_seconds"] = time.monotonic()-begin
            atomic(out/"core-after.json", after)
            report["base_unchanged"] = True
            report["all825_current_value_hashes_verified"] = True
            check()
        evidence.validate_sources(root, plan["source_sha256"])
        _, cache_identity = evidence.load_cache(config["cache_run"], plan["profile"])
        if cache_identity != plan["input_identity"]["cache"] or time.monotonic() >= config["deadline"]:
            raise ValueError("Inputs changed or deadline reached during final verification")
        report["output_sha256"] = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob("*"))
                                  if path.is_file() and path.name not in ("metrics.json", "memory.jsonl")}
        report.update(status="passed", inputs_unchanged=True, sources_unchanged=True, automatic_promotion=False)
        return report
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-began
        atomic(out/"metrics.json", report)


def execute(*, prepared_directory, cache_run, weights, expected_gpu, admission):
    root = Path(prepared_directory).absolute()
    if (not root.is_dir() or any(path.is_symlink() for path in (root, *root.parents))
            or root.resolve().is_relative_to(evidence.REPO)):
        raise ValueError("Use a regular prepared directory outside the repository")
    if any((root/name).exists() for name in ("launch.json", "terminal.json", "result", "worker.log")):
        raise ValueError("Execution is single-use; retain failures and prepare a fresh probe")
    started = time.monotonic()
    parent = {"schema": SCHEMA, "status": "running", "model_execution": False,
              "scope": evidence.SCOPE, "quality_assessed": False, "limits": LIMITS}
    atomic(root/"metrics.json", parent)
    proc = None
    try:
        plan, *_ = read_prepared(root, cache_run)
        admitted = evidence.validate_admission(admission, root, plan, expected_gpu)
        config = {"prepared_directory": str(root), "cache_run": str(Path(cache_run).absolute()),
                  "weights": str(Path(weights).absolute()), "expected_gpu": expected_gpu,
                  "admission": str(Path(admission).absolute()), "admission_record": admitted,
                  "plan_sha256": sha(root/"plan.json"), "deadline": started+LIMITS["seconds"]}
        atomic(root/"launch.json", config)
        if (root/"launch.json").stat().st_size > 4*2**20:
            raise ValueError("Bounded worker configuration required")
        shutil.copyfile(admission, root/"executed-admission.json")
        parent.update(plan_sha256=config["plan_sha256"], admission=admitted,
                      source_sha256=plan["source_sha256"], input_identity=plan["input_identity"], profile=plan["profile"])
        with (root/"worker.log").open("x") as log:
            if time.monotonic() >= config["deadline"]:
                raise RuntimeError("Probe deadline exhausted before launching the child")
            proc = subprocess.Popen([sys.executable, "-m", __package__+".probe", "--worker-config", str(root/"launch.json")],
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            supervise(proc, root, config["deadline"], "pair")
        terminal = evidence.read_json(root/"terminal.json")
        result = evidence.read_json(root/"result/metrics.json")
        if (terminal.get("status") != "complete" or terminal.get("exit_code") != 0 or terminal.get("cleanup_error") is not None
                or result.get("status") != "passed" or result.get("completed_updates") != 2
                or result.get("zero_adapter_gate_passed") is not True or result.get("base_unchanged") is not True
                or result.get("source_sha256") != plan["source_sha256"] or result.get("input_identity") != plan["input_identity"]
                or list(root.rglob("watchdog-stop.json"))):
            raise RuntimeError("Numerical probe did not complete all gates; retained evidence does not admit training")
        for name, digest in result["output_sha256"].items():
            if sha(evidence.relative_file(root/"result", name)) != digest:
                raise ValueError("Retained numerical output differs")
        if time.monotonic() >= config["deadline"]:
            raise RuntimeError("Probe deadline exhausted during final verification")
        parent.update(status="passed", model_execution=True, result_metrics_sha256=sha(root/"result/metrics.json"),
                      terminal_sha256=sha(root/"terminal.json"), fixed16_admitted=False, image_generation=False)
        return parent
    except BaseException as error:
        cleanup_error = None
        try:
            if proc is not None:
                stop_child(proc)
        except BaseException as cleanup:
            cleanup_error = str(cleanup)
        if not (root/"terminal.json").exists():
            atomic(root/"terminal.json", {"status": "failed", "exit_code": proc.returncode if proc else None,
                   "error": str(error), "cleanup_error": cleanup_error, "mode": "pair"})
        elif cleanup_error:
            atomic(root/"handoff-cleanup-error.json", {"error": str(error), "cleanup_error": cleanup_error})
        if proc is not None:
            parent["model_execution"] = None
            try:
                partial = evidence.read_json(root/"result/metrics.json")
                if type(partial.get("model_execution")) is bool:
                    parent["model_execution"] = partial["model_execution"]
            except (OSError, ValueError):
                pass
        parent.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        parent["elapsed_seconds"] = time.monotonic()-started
        atomic(root/"metrics.json", parent)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-config", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--profile", choices=tuple(PROFILES))
    parser.add_argument("--expected-gpu")
    for name in ("cache-run", "text-directory", "cpu-report", "independent-report", "output", "prepared-directory", "weights", "admission"):
        parser.add_argument("--"+name, type=Path)
    args = parser.parse_args(argv)
    if args.worker_config is not None:
        worker(evidence.read_json(args.worker_config, maximum_bytes=4*2**20))
        return
    names = ("prepared_directory", "cache_run", "weights", "expected_gpu", "admission") if args.execute else (
        "cache_run", "text_directory", "profile", "cpu_report", "independent_report", "output")
    if any(getattr(args, name) is None for name in names):
        parser.error("Supply the complete explicit preparation or execution arguments")
    function = execute if args.execute else prepare
    result = function(**{name: getattr(args, name) for name in names})
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
