# SPDX-License-Identifier: Apache-2.0
"""Read-only data and mocked runtime checks, without native weight values."""
from contextlib import contextmanager
import json
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch
from safetensors.torch import load_file

from . import data, operations, runtime, worker


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_rgb():
    frame_values = np.arange(17, dtype=np.uint8)*8+32
    return np.broadcast_to(frame_values[:, None, None, None], (17, 16, 16, 3)).copy()


class FakeGuard:
    device = "cpu"

    def __init__(self):
        self.report = {}

    def measure(self, name, function):
        return function()

    def save(self):
        pass


@contextmanager
def fake_chunks(*unused):
    yield [{"output_shape": [1, 3, count, 16, 16], "first_chunk": i == 0}
           for i, count in enumerate((1, 4, 4, 4, 4))]


class BoundaryCodec:
    def __init__(self, rgb, mismatched_prefix=False):
        self.rgb = rgb
        self.events = []
        self.mismatched_prefix = mismatched_prefix
        self.target = None

    def encode(self, video):
        assert video.dtype == torch.float32 and video.device.type == "cpu"
        self.events.append(("encode", video.shape[2]))
        expected = torch.from_numpy(self.rgb.transpose(3, 0, 1, 2).copy()[None]).float()/127.5-1
        assert torch.equal(video, expected[:, :, :video.shape[2]])
        if video.shape[2] == 1:
            return torch.full((1, 48, 1, 18, 32), .25)
        assert video.shape[2] == 17 and self.events == [("encode", 1), ("encode", 17)]
        self.target = torch.full((1, 48, 5, 18, 32), .5)
        self.target[:, :, :1] = .75 if self.mismatched_prefix else .25
        return self.target.clone()

    def decode(self, *args, **kwargs):
        assert len(args) == 1 and not kwargs
        assert torch.equal(args[0], self.target)
        self.events.append(("decode", tuple(args[0].shape)))
        expected = torch.from_numpy(self.rgb.transpose(3, 0, 1, 2).copy()[None]).float()/127.5-1
        # Fixed error on all frames allows an independent closed-form score check.
        return expected+.125


def test_whole_roundtrip_orders_independent_encode_target_only_decode_and_scores(tmp_path):
    rgb = tiny_rgb()
    before = rgb.copy()
    codec, guard = BoundaryCodec(rgb), FakeGuard()
    real_score = operations.score_reconstruction

    def score_after_decode(truth, reconstruction):
        assert codec.events == [("encode", 1), ("encode", 17), ("decode", (1, 48, 5, 18, 32))]
        assert np.array_equal(truth, before)
        return real_score(truth, reconstruction)

    with mock.patch.object(data, "load_roundtrip_rgb", return_value=(rgb, {"fixture": True})), \
         mock.patch.object(worker, "chunk_timings", fake_chunks), \
         mock.patch.object(worker, "score_reconstruction", score_after_decode):
        worker.roundtrip(codec, guard, Path("unused-fixture"), tmp_path)
    saved = load_file(str(tmp_path/"encoded.safetensors"))
    assert torch.equal(saved["target"], codec.target)
    assert torch.equal(saved["target"][:, :, :1], saved["observation"])
    assert guard.report["causal_checks"][0]["target_prefix_replaced"] is False
    assert np.array_equal(rgb, before)
    assert guard.report["reconstructed_frames"] == 17 and guard.report["newly_generated_frames"] == 0
    scores = guard.report["reconstruction"]
    assert [row["frame"] for row in scores["per_frame"]] == list(range(17))
    for row in scores["per_frame"]:
        assert row["mae_rgb_0_1"] == pytest.approx(.0625, abs=4e-8)
        assert row["mse_rgb_0_1"] == pytest.approx(.00390625, abs=5e-9)
    assert len(list((tmp_path/"truth").glob("*.png"))) == 17
    assert len(list((tmp_path/"reconstruction").glob("*.png"))) == 17


def test_roundtrip_prefix_mismatch_stops_before_decode_without_repair(tmp_path):
    rgb = tiny_rgb()
    codec, guard = BoundaryCodec(rgb, mismatched_prefix=True), FakeGuard()
    with mock.patch.object(data, "load_roundtrip_rgb", return_value=(rgb, {})):
        with pytest.raises(RuntimeError, match="prefix causality"):
            worker.roundtrip(codec, guard, Path("unused-fixture"), tmp_path)
    assert codec.events == [("encode", 1), ("encode", 17)]
    saved = load_file(str(tmp_path/"encoded.safetensors"))
    assert (saved["target"][:, :, :1] == .75).all()
    assert (saved["observation"] == .25).all()
    assert guard.report["causal_checks"][0]["passed"] is False
    assert not (tmp_path/"reconstruction.safetensors").exists()


def capture_fixture(root, image):
    image.save(root/"frame.png")
    rows = [{"frame": i, "png": "frame.png", "png_sha256": data.sha(root/"frame.png"),
             "action_from_previous": {"never_construct": "command"}} for i in range(17)]
    manifest = root/"manifest.json"
    manifest.write_text(json.dumps({"arms": {"open": rows}, "rgb_transform": "fixture"}))
    return data.sha(manifest)


def test_prescribed_opaque_rgb_bytes_and_rejection_without_command_construction(tmp_path):
    with Image.open(data.HERE/"source-images/open-0000.png") as original:
        rgba = np.array(original)
    assert rgba.shape == (288, 512, 4) and np.all(rgba[:, :, 3] == 255)
    manifest_hash = capture_fixture(tmp_path, Image.fromarray(rgba))
    with mock.patch.object(data, "MANIFEST_SHA256", manifest_hash), \
         mock.patch.object(data.original, "command_vector", side_effect=AssertionError("Command construction")):
        pixels, record = data.load_roundtrip_rgb(tmp_path)
        assert np.array_equal(pixels[0], rgba[:, :, :3])
        assert data.array_sha(pixels[0]) == data.CANONICAL_RGB_SHA256
        assert record["action_values_materialized"] is False
        # A changed file is rejected by its hash before use.
        Image.fromarray(rgba[:, :, :3]).save(tmp_path/"frame.png")
        with pytest.raises(ValueError, match="RGB file changed"):
            data.load_roundtrip_rgb(tmp_path)
    rgba[0, 0, 3] = 254
    manifest_hash = capture_fixture(tmp_path, Image.fromarray(rgba))
    with mock.patch.object(data, "MANIFEST_SHA256", manifest_hash):
        with pytest.raises(ValueError, match="transparent"):
            data.load_roundtrip_rgb(tmp_path)


def test_parent_timeout_terminates_worker_and_retains_failure(tmp_path):
    class Process:
        returncode = None
        terminated = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            self.returncode = -15

    process = Process()
    with mock.patch.object(runtime.subprocess, "Popen", return_value=process) as spawn, \
         mock.patch.object(runtime.time, "monotonic", side_effect=[0., 901., 902.]):
        with pytest.raises(TimeoutError, match="900-second"):
            runtime.run_child(tmp_path, {"mode": "roundtrip"})
    assert process.terminated and process.returncode == -15
    terminal = json.loads((tmp_path/"terminal.json").read_text())
    assert terminal["status"] == "failed" and terminal["error_type"] == "TimeoutError"
    launched = spawn.call_args.kwargs["env"]
    assert {name: launched.get(name) for name in runtime.ENVIRONMENT} == runtime.ENVIRONMENT
    assert json.loads((tmp_path/"launch.json").read_text())["max_memory_gib"] == 18
    assert json.loads((tmp_path/"launch.json").read_text())["minimum_available_gib"] == 2


def test_single_frame_encoder_cannot_read_or_share_future_storage():
    video = torch.arange(1*3*17*4*4, dtype=torch.float32).reshape(1, 3, 17, 4, 4)
    original = video.clone()

    class Encoder:
        def encode(self, passed):
            assert passed.shape == (1, 3, 1, 4, 4)
            assert passed.untyped_storage().nbytes() == passed.numel()*passed.element_size()
            assert torch.equal(passed, original[:, :, :1])
            passed.fill_(0)
            return "one-frame-fixture"

    assert operations.observation_only(Encoder(), video) == "one-frame-fixture"
    assert torch.equal(video, original)
