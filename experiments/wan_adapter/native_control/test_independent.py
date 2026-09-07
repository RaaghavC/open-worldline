"""Independent CPU-only checks of the native Wan diagnostic contract.

These checks do not load foundation weights or execute a GPU operation.
"""

import hashlib
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from native_control.vendor.fm_solvers_unipc import FlowUniPCMultistepScheduler
from native_control import portable, sampling

torch.set_num_threads(2)


def _scheduler():
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(50, device="cpu", shift=8)
    return scheduler


def test_literal_native_sources_match_independently_fetched_revision():
    expected = {
        "model.py": "c1572ade3bf7345bd4c00d4a19535788fab62f9c08b5aeb15f6d01f4435bb47a",
        "fm_solvers_unipc.py": "0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9",
        "text2video.py.txt": "169f0929b273b27a7bc0e094d15a01d48183adc529a1dbce626b179dced29fe8",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((Path(__file__).parent / "vendor" / name).read_bytes()).hexdigest() == digest


def test_unipc_uses_single_shift_integer_timesteps_and_zero_endpoint():
    scheduler = _scheduler()
    raw = np.linspace(float(torch.tensor(0.999)), 0.0, 51)[:-1]
    shifted = 8.0 * raw / (1.0 + 7.0 * raw)
    expected_sigmas = torch.from_numpy(np.r_[shifted, 0.0].astype(np.float32))
    expected_timesteps = torch.from_numpy((1000.0 * shifted).astype(np.int64))
    assert torch.equal(scheduler.sigmas, expected_sigmas)
    assert torch.equal(scheduler.timesteps, expected_timesteps)
    assert scheduler.timesteps.dtype == torch.int64
    assert scheduler.timesteps[0].item() == 999
    assert scheduler.timesteps[-1].item() == 140
    assert scheduler.sigmas[-1].item() == 0
    assert scheduler.config.solver_order == 2
    assert scheduler.config.prediction_type == "flow_prediction"


def test_unipc_constant_velocity_integrates_every_frame_including_first():
    scheduler = _scheduler()
    initial = torch.randn(1, 2, 5, 3, 4, generator=torch.Generator().manual_seed(42))
    velocity = torch.linspace(-0.5, 0.5, initial.numel()).reshape_as(initial)
    sample = initial.clone()
    for timestep in scheduler.timesteps:
        sample = scheduler.step(velocity, timestep, sample, return_dict=False)[0]
        assert torch.isfinite(sample).all()
        assert sample.dtype == torch.float32
    expected = initial - scheduler.sigmas[0] * velocity
    torch.testing.assert_close(sample, expected, atol=0.00001, rtol=0.00001)
    assert not torch.equal(sample[:, :, 0], initial[:, :, 0])
    assert scheduler.step_index == 50


def test_unipc_noisy_linear_path_reconstructs_known_clean_target():
    scheduler = _scheduler()
    clean = torch.randn(1, 2, 5, 3, 4, generator=torch.Generator().manual_seed(7))
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(8))
    sigma = scheduler.sigmas[0]
    sample = (1 - sigma) * clean + sigma * noise
    for timestep in scheduler.timesteps:
        # The exact flow velocity is noise minus clean at every sigma.
        sample = scheduler.step(noise - clean, timestep, sample, return_dict=False)[0]
    torch.testing.assert_close(sample, clean, atol=0.00001, rtol=0.00001)


def test_portable_attention_matches_explicit_masked_softmax():
    generator = torch.Generator().manual_seed(2026)
    q = torch.randn(2, 5, 2, 8, generator=generator)
    k = torch.randn(2, 7, 2, 8, generator=generator)
    v = torch.randn(2, 7, 2, 8, generator=generator)
    key_lengths, query_lengths = torch.tensor([3, 7]), torch.tensor([4, 5])
    scores = torch.einsum("bqhd,bkhd->bhqk", q * 0.7, k) * 0.3
    scores[0, :, :, 3:] = -torch.inf
    expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v)
    expected[0, 4:] = 0
    actual = portable.attention(q, k, v, q_lens=query_lengths, k_lens=key_lengths,
                                q_scale=0.7, softmax_scale=0.3)
    torch.testing.assert_close(actual, expected, atol=0.000001, rtol=0.00001)


def test_rope_matches_literal_complex_reference_with_head_128_and_padding():
    reference = portable.load_model_module(portable=False)
    device_port = portable.load_model_module(portable=True)
    assert reference is not device_port
    assert reference.rope_apply is not device_port.rope_apply
    grids = torch.tensor([[2, 3, 5], [1, 4, 7]])
    # Wan head 128 splits into 44 temporal, 42 vertical and 42 horizontal dimensions.
    complex_table = torch.cat([reference.rope_params(128, dimension)
                               for dimension in (44, 42, 42)], dim=1)
    real_table = torch.cat([device_port.rope_params(128, dimension)
                            for dimension in (44, 42, 42)], dim=1)
    source = torch.randn(2, 32, 2, 128, generator=torch.Generator().manual_seed(11))
    for dtype in (torch.float32, torch.float16):
        values = source.to(dtype)
        actual = device_port.rope_apply(values, grids, real_table)
        expected = reference.rope_apply(values, grids, complex_table)
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=0.000001, rtol=0.00001)
        assert torch.equal(actual[0, 30:], values[0, 30:].float())
        assert torch.equal(actual[1, 28:], values[1, 28:].float())


def test_timestep_embedding_preserves_literal_double_reference_before_cast():
    reference = portable.load_model_module(portable=False)
    positions = torch.tensor([0, 50, 500, 999, 1000], dtype=torch.int64)
    expected = reference.sinusoidal_embedding_1d(256, positions).float()
    actual = portable.cpu_sinusoidal_embedding(256, positions)
    assert torch.equal(actual, expected)


def test_sampling_is_sequential_cfg_with_identical_noise_and_no_prefix_clamp():
    negative, positive = torch.full((1, 4096), 2.0), torch.full((3, 4096), 3.0)
    calls = []
    previous_latent = [None]

    def denoiser(latents, timestep, contexts, token_count):
        assert token_count == 2880
        assert len(latents) == len(contexts) == 1
        assert timestep.dtype == torch.int64 and timestep.shape == (1,)
        latent = latents[0]
        if len(calls) % 2 == 0:
            assert contexts[0] is negative
            previous_latent[0] = latent.clone()
        else:
            assert contexts[0] is positive
            assert torch.equal(previous_latent[0], latent)
        calls.append(timestep.item())
        return [torch.full_like(latent, float(contexts[0][0, 0]))]

    noise = sampling.initial_noise()
    original = noise.clone()
    events = []
    result = sampling.sample(denoiser, noise, negative, positive,
                             callback=lambda index, timestep, latent: events.append(index))
    native_scheduler = _scheduler()
    assert torch.equal(noise, original)
    assert torch.equal(original, sampling.initial_noise())
    assert len(calls) == 100
    assert events == list(range(50))
    assert calls[::2] == native_scheduler.timesteps.tolist()
    assert calls[1::2] == native_scheduler.timesteps.tolist()
    # negative 2 + guidance 6 * (positive 3 - negative 2) = constant velocity 8.
    expected = original - 8.0 * native_scheduler.sigmas[0]
    torch.testing.assert_close(result, expected, atol=0.00003, rtol=0.00001)
    assert not torch.equal(result[:, 0], original[:, 0])


def test_guided_velocity_overflow_is_rejected_before_a_profile_can_pass():
    contexts = [torch.tensor([[-1.0]]), torch.tensor([[1.0]])]

    def extreme_denoiser(latents, timestep, context, token_count):
        return [torch.full_like(latents[0], float(context[0].item()) * 1e38)]

    with unittest.TestCase().assertRaisesRegex(RuntimeError, "guided"):
        sampling.negative_positive_pair(extreme_denoiser, sampling.initial_noise(),
                                         torch.tensor(999), *contexts)


def test_profile_requires_passed_current_independent_source_report():
    from native_control.profile_pair import validate_independent

    names = ("portable.py", "sampling.py", "test_independent.py",
             "vendor/model.py", "vendor/fm_solvers_unipc.py")
    report = {"status": "passed", "tests_run": 8,
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in names}}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "independent.json"
        path.write_text(json.dumps(report))
        assert validate_independent(path) == hashlib.sha256(path.read_bytes()).hexdigest()
        report["status"] = "failed"
        path.write_text(json.dumps(report))
        with unittest.TestCase().assertRaisesRegex(ValueError, "Completed independent"):
            validate_independent(path)
        report["status"] = "passed"
        report["source_sha256"]["sampling.py"] = "0" * 64
        path.write_text(json.dumps(report))
        with unittest.TestCase().assertRaisesRegex(ValueError, "source changed"):
            validate_independent(path)


def test_cpu_solver_device_boundary_matches_all_cpu_reference_with_changing_velocity():
    from native_control.loop import integrate

    noise = sampling.initial_noise()
    unchanged = noise.clone()
    negative, positive = torch.full((2, 4096), -1.0), torch.full((3, 4096), 1.0)

    def changing_velocity(latents, timestep, context, token_count):
        assert latents[0].device.type == "cpu"
        # Depend on both the evolving state and current timestep, so incorrect
        # solver feedback cannot pass a constant-velocity-only oracle.
        return [latents[0] * 0.03125 + float(context[0][0, 0]) * 0.001
                - timestep.item() * 0.000001]

    reference = sampling.sample(changing_velocity, noise, negative, positive)
    records = []

    def record(index, timestep, latent, timing):
        assert latent.device.type == "cpu" and latent.dtype == torch.float32
        assert all(np.isfinite(value) and value >= 0 for value in timing.values())
        records.append((index, timestep.item()))

    actual = integrate(changing_velocity, noise, negative, positive,
                        device="cpu", callback=record)
    assert torch.equal(actual, reference)
    assert torch.equal(noise, unchanged)
    assert len(records) == 50
    assert [row[1] for row in records] == sampling.make_scheduler().timesteps.tolist()
    assert not torch.equal(actual[:, 0], noise[:, 0])


def test_clip_gate_rejects_bad_estimates_and_uses_actual_decoder_cost():
    from native_control.run_clip import validate_profile
    from fetch_weights import FILES

    here = Path(__file__).parent
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    estimate_key = "non_pair_overhead_plus_50_pairs_plus_decode30s_and_artifacts10s_seconds"
    source_names = ("portable.py", "sampling.py", "profile_pair.py", "vendor/model.py",
                    "vendor/fm_solvers_unipc.py", "../fetch_weights.py", "../text_cache/cache.py")
    scheduler = sampling.make_scheduler()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        text_cache = root / "text"
        text_cache.mkdir()
        (text_cache / "manifest.json").write_text("{}")
        (text_cache / "embeddings.safetensors").write_bytes(b"fixture text file; gate hashes only")
        profile = {
            "status": "passed", "seed": sampling.SEED,
            "steps_for_estimate": sampling.STEPS, "shift": sampling.SHIFT,
            "guidance": sampling.GUIDANCE,
            "runtime_estimate": {"measured_first_pair_seconds": 13.0, estimate_key: 700.0},
            "base": {"weight_sha256": FILES["diffusion_pytorch_model.safetensors"][1],
                     "config_sha256": FILES["config.json"][1]},
            "source_sha256": {name: sha(here / name) for name in source_names},
            "text": {"manifest_sha256": sha(text_cache / "manifest.json"),
                     "embeddings_sha256": sha(text_cache / "embeddings.safetensors")},
            "scheduler": {"timesteps": scheduler.timesteps.tolist(),
                          "sigmas": scheduler.sigmas.tolist()},
        }
        overhead = {"status": "passed", "steps": 50,
                    "loop_sha256": sha(here / "loop.py"), "solver_and_transfers_seconds": 2.0}
        decoder = {
            "status": "passed", "dtype": "float32", "weights_sha256": FILES["Wan2.1_VAE.pth"][1],
            "source_sha256": {name: sha(here.parent / "codec" / relative)
                              for name, relative in (("helper.py", "helper.py"),
                                  ("decode_policy.py", "decode_policy.py"),
                                  ("wan_vae.py", "vendor/wan_vae.py"))},
            "timings": [{"stage": "load_official_vae", "seconds": 5.0},
                        {"stage": "base_decode_fp32", "seconds": 30.0}],
        }
        metrics = root / "metrics.json"
        solver_path, decoder_path = root / "solver.json", root / "decoder.json"

        def write_and_validate():
            metrics.write_text(json.dumps(profile))
            solver_path.write_text(json.dumps(overhead))
            decoder_path.write_text(json.dumps(decoder))
            return validate_profile(root, text_cache, solver_path, decoder_path)

        _, estimate = write_and_validate()
        assert estimate == 707.0  # 700 - old decoder 30 + measured decoder 35 + transfer/solver 2
        for invalid in (float("nan"), float("inf"), -1.0, 0.0, 901.0):
            profile["runtime_estimate"][estimate_key] = invalid
            with unittest.TestCase().assertRaisesRegex(ValueError, "runtime estimate"):
                write_and_validate()
        profile["runtime_estimate"][estimate_key] = 895.0
        with unittest.TestCase().assertRaisesRegex(ValueError, "900-second cap"):
            write_and_validate()
        profile["runtime_estimate"][estimate_key] = 700.0
        profile["scheduler"]["timesteps"][0] = 1000
        with unittest.TestCase().assertRaisesRegex(ValueError, "schedule differs"):
            write_and_validate()
        profile["scheduler"]["timesteps"] = scheduler.timesteps.tolist()
        overhead["loop_sha256"] = "0" * 64
        with unittest.TestCase().assertRaisesRegex(ValueError, "solver/transfer"):
            write_and_validate()
        overhead["loop_sha256"] = sha(here / "loop.py")
        decoder["status"] = "failed"
        with unittest.TestCase().assertRaisesRegex(ValueError, "decoder profile"):
            write_and_validate()


def load_tests(loader, standard_tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value)
                              for name, value in sorted(globals().items())
                              if name.startswith("test_") and callable(value))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON evidence file")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("The requested test evidence file already exists")
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(load_tests(None, None, None))
    if args.output is not None:
        source_files = ["test_independent.py", "portable.py", "sampling.py", "loop.py",
                        "profile_pair.py", "run_clip.py", "evidence.py",
                        "vendor/model.py", "vendor/fm_solvers_unipc.py"]
        report = {
            "status": "passed" if result.wasSuccessful() else "failed",
            "device": "cpu",
            "tests_run": result.testsRun,
            "elapsed_seconds": time.monotonic() - started,
            "torch_version": torch.__version__,
            "failures": [(str(test), trace) for test, trace in result.failures],
            "errors": [(str(test), trace) for test, trace in result.errors],
            "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in source_files},
            "limits": "CPU equation/source checks only; no foundation-weight load, GPU or visual-quality evaluation.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if result.wasSuccessful() else 1)
