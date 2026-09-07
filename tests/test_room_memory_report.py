# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU presentation checks, without trained model execution."""
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.room_world import memory_report as mr


def fixture_tensors():
    rgb = torch.linspace(-1, 1, 66)[None, :, None, None, None].expand(2, 66, 3, 64, 64)
    actions = torch.tensor([[0] + [3] * 24 + [0] * 16 + [4] * 24,
                            [5] + [3] * 24 + [0] * 16 + [4] * 24])
    values = {"uninterrupted": rgb[:, 1:].clone(), "observed_prefix_return": rgb[:, 42:].clone()}
    return rgb, actions, values


@pytest.mark.parametrize("protocol,start,count", [("uninterrupted", 0, 65), ("observed_prefix_return", 41, 24)])
def test_every_frame_action_and_truth_index_is_aligned(protocol, start, count):
    rgb, actions, values = fixture_tensors()
    predictions = {mode: values for mode in ("carry", "reset", "frozen")}
    before = rgb.clone()
    frames = list(mr.aligned_frames(rgb, actions, predictions, protocol))
    assert len(frames) == count
    assert [frame["action_index"] for frame in frames] == list(range(start, start + count))
    assert [frame["observation_index"] for frame in frames] == list(range(start + 1, start + count + 1))
    for frame in frames:
        assert frame["actions"] == actions[:, frame["action_index"]].tolist()
        for mode in mr.MODELS:
            assert torch.equal(frame["frames"][mode], rgb[:, frame["observation_index"]])
    assert torch.equal(rgb, before)
    if protocol == "uninterrupted":
        assert frames[0]["actions"] == [0, 5]


def test_nearest_neighbor_preserves_all_native_pixels_and_both_branches():
    # An RGB checkerboard exposes interpolation, transposition and branch mixups.
    pattern = (torch.arange(64)[:, None] + torch.arange(64)[None, :]).remainder(2).float() * 2 - 1
    rgb = torch.stack([pattern, -pattern, pattern]).unsqueeze(0).repeat(2, 1, 1, 1)
    rgb[1] *= -1
    frame = {"frames": {mode: rgb for mode in mr.MODELS}, "actions": [0, 5], "action_index": 0, "observation_index": 1}
    panel = np.asarray(mr.make_panel(frame, seed=mr.SEEDS[0], scene=300000, protocol="uninterrupted", scale=3))
    _, positions = mr.panel_layout(3)
    for mode in mr.MODELS:
        for branch in (0, 1):
            x, y = positions[(mode, branch)]
            expected = mr.to_rgb8(rgb[branch]).repeat(3, axis=0).repeat(3, axis=1)
            assert np.array_equal(panel[y:y + 192, x:x + 192], expected)
    assert not np.array_equal(mr.to_rgb8(rgb[0]), mr.to_rgb8(rgb[1]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5])
def test_invalid_rgb_cannot_be_hidden_by_display_clipping(bad):
    value = torch.zeros(3, 64, 64)
    value[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite normalized"):
        mr.to_rgb8(value)


def test_confined_input_rejects_traversal_and_hash_changes(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    path = root / "artifact"
    path.write_bytes(b"fixed")
    assert mr.confined_file(root, "artifact", expected=mr.sha256(path)) == path
    with pytest.raises(ValueError, match="hash differs"):
        mr.confined_file(root, "artifact", expected="0" * 64)
    (tmp_path / "outside").write_bytes(b"private")
    (root / "link").symlink_to(tmp_path / "outside")
    for name in ("../outside", "link", str(path)):
        with pytest.raises(ValueError, match="remain inside"):
            mr.confined_file(root, name)


def saved_evaluator_fixture(tmp_path, monkeypatch):
    rgb, actions, values = fixture_tensors()
    root = tmp_path / "evaluation"
    root.mkdir()
    prediction = root / "predictions.pt"
    torch.save(values, prediction)
    tensor_identity, prediction_sha = mr.tensor_hashes(values), mr.sha256(prediction)
    class Dataset:
        manifest_sha256 = "a" * 64
        def __init__(self, *args):
            pass
    monkeypatch.setattr(mr, "DevelopmentDataset", Dataset)
    study = {"schema": "worldline-room-memory-validation-study-v1", "status": "complete",
        "reserved_test_opened": False, "validation": {"manifest_sha256": Dataset.manifest_sha256},
        "sequence_path": "batched", "training_study_sha256": "b" * 64,
        "training_source_sha256": {"fixture_only": "c" * 64}, "runs": []}
    jobs = [(seed, mode) for seed in mr.SEEDS for mode in ("carry", "reset")] + [(mr.SEEDS[0], "frozen")]
    for seed, mode in jobs:
        report = {"schema": "worldline-room-memory-evaluation-v1", "status": "complete",
            "data": study["validation"], "provenance": {"mode": mode, "seed": seed,
                "sequence_path": None if mode == "frozen" else "batched", "training_sequence_path": "batched",
                "training_study_sha256": study["training_study_sha256"], "training_source_sha256": study["training_source_sha256"]},
            "scenes": [{"scene_seed": scene, "first_return_action_index": 41,
                "uninterrupted_generated_history": {"generated_steps": 65},
                "observed_prefix_then_generated_return": {"generated_steps": 24},
                "prediction_file": prediction.name, "prediction_file_sha256": prediction_sha,
                "predictions": tensor_identity} for scene in mr.SCENES["validation"]]}
        path = root / f"{seed}-{mode}.json"
        path.write_text(json.dumps(report))
        study["runs"].append({"seed": seed, "mode": mode, "status": "complete", "terminal_status": "complete",
                             "evaluation": path.name, "evaluation_sha256": mr.sha256(path)})
    (root / "validation.json").write_text(json.dumps(study))
    return root, study


def test_complete_artifact_reader_validates_file_and_tensor_hashes(tmp_path, monkeypatch):
    root, study = saved_evaluator_fixture(tmp_path, monkeypatch)
    reader = mr.PresentationInputs(root, tmp_path)
    values = reader.prediction("20260907-carry", 300000)
    assert tuple(values["uninterrupted"].shape) == (2, 65, 3, 64, 64)
    reader.models["20260907-carry"]["scenes"][300000][1]["predictions"]["uninterrupted"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="tensor keys or hashes"):
        reader.prediction("20260907-carry", 300000)


@pytest.mark.parametrize("change", ["missing_seed", "duplicate", "wrong_truth", "partial"])
def test_incomplete_or_mismatched_evaluation_is_not_presented(tmp_path, monkeypatch, change):
    root, study = saved_evaluator_fixture(tmp_path, monkeypatch)
    if change == "missing_seed":
        study["runs"].pop(0)
    elif change == "duplicate":
        study["runs"].append(study["runs"][0])
    elif change == "wrong_truth":
        study["validation"]["manifest_sha256"] = "0" * 64
    else:
        study["status"] = "running"
    (root / "validation.json").write_text(json.dumps(study))
    with pytest.raises(ValueError):
        mr.PresentationInputs(root, tmp_path)


def test_full_presentation_has_every_scene_seed_protocol_branch_and_frame(tmp_path, monkeypatch):
    rgb, actions, values = fixture_tensors()
    metadata = tmp_path / "validation.json"
    metadata.write_text('{"fixture_only": true}')
    class Dataset:
        manifest_sha256 = "a" * 64
        def load_pair(self, scene):
            return rgb, actions
    class Inputs:
        def __init__(self, *args):
            self.dataset, self.path = Dataset(), metadata
            self.models = {key: {"sha256": "b" * 64, "scenes": {scene: (None, {"prediction_file_sha256": "c" * 64})
                for scene in mr.SCENES["validation"]}} for key in
                [f"{seed}-{mode}" for seed in mr.SEEDS for mode in ("carry", "reset")] + ["frozen"]}
        def prediction(self, *args):
            return values
    seen = []
    def encode_fixture(panels, path, **kwargs):
        count = len(list(panels))
        seen.append(count)
        path.write_bytes(f"synthetic encoder fixture: {count}".encode())
    monkeypatch.setattr(mr, "PresentationInputs", Inputs)
    monkeypatch.setattr(mr.shutil, "which", lambda _: "/fixture/ffmpeg")
    monkeypatch.setattr(mr, "make_panel", lambda *a, **k: Image.new("RGB", (8, 8), "red"))
    monkeypatch.setattr(mr, "write_video", encode_fixture)
    out = tmp_path / "display"
    report = mr.create_report(tmp_path, tmp_path, out)
    assert report["status"] == "complete" and not report["metrics_recomputed"] and not report["models_executed"]
    assert len(report["items"]) == 48 and sorted(seen) == [24] * 24 + [65] * 24
    assert sum(len(item["contact_pages"]) for item in report["items"]) == 288
    expected = {(scene, seed, protocol) for scene in mr.SCENES["validation"] for seed in mr.SEEDS for protocol in mr.PROTOCOLS}
    assert {(item["scene"], item["seed"], item["protocol"]) for item in report["items"]} == expected
    for item in report["items"]:
        assert item["branches"] == [0, 1]
        assert [i for page in item["contact_pages"] for i in page["action_indices"]] == item["action_indices"]
        assert mr.sha256(out / item["video"]["file"]) == item["video"]["sha256"]
    assert (out / "evaluator-validation.json").read_bytes() == metadata.read_bytes()
    assert (out / "index.html").read_text().count('<video controls') == 48
    with pytest.raises(FileExistsError):
        mr.create_report(tmp_path, tmp_path, out)


def test_cpu_video_encoder_retains_frame_count(tmp_path):
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("Optional local ffmpeg/ffprobe are unavailable")
    path = tmp_path / "two-frames.mp4"
    mr.write_video([Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")], path, fps=2, ffmpeg=ffmpeg)
    result = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                            "stream=width,height,nb_read_frames", "-of", "json", str(path)], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    assert (stream["width"], stream["height"], stream["nb_read_frames"]) == (16, 16, "2")
