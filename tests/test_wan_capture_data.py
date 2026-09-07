import hashlib
import json
import math

import numpy as np
from PIL import Image
import pytest

from experiments.wan_adapter.capture_data import ACTION_CHANNELS, command_vector, load_window


@pytest.fixture
def capture(tmp_path):
    Image.new("RGB", (512, 288), (40, 100, 160)).save(tmp_path / "frame.png")
    digest = hashlib.sha256((tmp_path / "frame.png").read_bytes()).hexdigest()
    arms = {}
    for arm in ("closed", "open"):
        commands = [None, "interact" if arm == "open" else "wait"] + ["left"] * 24 + ["wait"] * 16 + ["right"] * 24
        arms[arm] = [{"frame": i, "action_from_previous": command, "png": "frame.png",
                      "png_sha256": digest, "door_open": "must not reach inputs",
                      "world_from_camera": "must not reach inputs", "exr": "not read"}
                     for i, command in enumerate(commands)]
    manifest = {"schema": "worldline-atrium-pilot-v1", "status": "complete", "dense_sequence": True,
                "dataset_license": "CC0-1.0", "scene_family": "single_atrium_layout_v1",
                "scene_seed": 51000, "resolution": [512, 288], "arms": arms}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def edit_manifest(capture, change):
    path = capture / "manifest.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_window_alignment_at_intervention_and_turn_boundaries(capture):
    first = load_window(capture, "open", 0)
    assert first.actions.shape == (16, 6)
    assert first.actions[0, 5] == 1
    np.testing.assert_allclose(first.actions[1:, 3], math.pi / 24)
    np.testing.assert_array_equal(first.actions[:, :3], 0)
    middle = load_window(capture, "closed", 24)
    assert middle.actions[0, 3] == np.float32(math.pi / 24)
    assert not middle.actions[1:].any()
    last = load_window(capture, "closed", 49)
    np.testing.assert_allclose(last.actions[:, 3], -math.pi / 24)


def test_native_pixels_and_no_hidden_state_in_result(capture):
    sample = load_window(capture, "closed", 0)
    assert sample.rgb.shape == (17, 288, 512, 3)
    assert sample.rgb.dtype == np.uint8
    np.testing.assert_array_equal(sample.rgb[0, 0, 0], [40, 100, 160])
    video = sample.video_array()
    assert video.shape == (1, 3, 17, 288, 512) and video.dtype == np.float32
    np.testing.assert_allclose(video[0, :, 0, 0, 0], np.array([40, 100, 160]) / 127.5 - 1, atol=1e-7)
    assert sample.provenance["action_channels"] == list(ACTION_CHANNELS)
    assert sample.provenance["split"] == "development"
    assert sample.provenance["independent_layouts"] == 1
    text = json.dumps(sample.provenance)
    assert "world_from_camera" not in text and "door_open" not in text and "must not reach" not in text


@pytest.mark.parametrize("start,frames", [(-1, 17), (50, 17), (0, 16), (0, 1), (True, 17)])
def test_invalid_window_rejected(capture, start, frames):
    with pytest.raises(ValueError):
        load_window(capture, "open", start, frames)


@pytest.mark.parametrize("change", [
    lambda m: m.update(dense_sequence=False),
    lambda m: m.update(status="partial"),
    lambda m: m.update(dataset_license="unknown"),
    lambda m: m.update(resolution=[256, 144]),
    lambda m: m["arms"]["open"][1].update(action_from_previous="wait"),
    lambda m: m["arms"]["open"][2].update(frame=3),
    lambda m: m["arms"]["open"][0].update(png="../escape.png"),
    lambda m: m["arms"]["open"][0].update(png_sha256="0" * 64),
])
def test_manifest_integrity_rejected(capture, change):
    edit_manifest(capture, change)
    with pytest.raises(ValueError):
        load_window(capture, "open", 0)


def test_unsupported_command_rejected():
    with pytest.raises(ValueError):
        command_vector("guessed forward")


def test_transparent_rgb_is_not_silently_composited(capture):
    Image.new("RGBA", (512, 288), (40, 100, 160, 100)).save(capture / "frame.png")
    digest = hashlib.sha256((capture / "frame.png").read_bytes()).hexdigest()
    edit_manifest(capture, lambda m: [r.update(png_sha256=digest) for arm in m["arms"].values() for r in arm])
    with pytest.raises(ValueError, match="Transparent"):
        load_window(capture, "open", 0)
