# SPDX-License-Identifier: Apache-2.0
"""Contract tests with tiny fabricated files, not evidence of rendered quality."""
import json
import math

import numpy as np
import pytest
from PIL import Image

from experiments.atrium_data.validate import check_intrinsics, compare_depth_rays, expected_actions, sha256, validate


def fixture_capture(path, indices=(0,), arms=("closed",)):
    OpenEXR = pytest.importorskip("OpenEXR")
    path.mkdir()
    width, height = 8, 4
    (path / "original-scene.blend").write_bytes(b"Isolated hash-contract fixture; not a Blender scene")
    manifest = {"schema": "worldline-atrium-pilot-v1", "status": "complete", "resolution": [width, height],
                "K": [[width * 24 / 36, 0, width / 2], [0, width * 24 / 36, height / 2], [0, 0, 1]],
                "objects": {"1": "Door leaf"}, "dense_sequence": len(indices) == 66,
                "blend_sha256": sha256(path / "original-scene.blend"), "arms": {}}
    for arm in arms:
        (path / arm).mkdir()
        records, yaw = [], 0.
        for frame, action in enumerate(expected_actions(arm)):
            yaw += math.pi / 24 if action == "left" else -math.pi / 24 if action == "right" else 0
            forward = np.array([-math.sin(yaw), math.cos(yaw), -.07])
            forward /= np.linalg.norm(forward)
            right = np.array([math.cos(yaw), math.sin(yaw), 0.])
            transform = np.eye(4)
            transform[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
            transform[:3, 3] = [.55, -4.1, 1.6]
            record = {"frame": frame, "action_from_previous": action, "door_open": arm == "open" and frame > 0,
                      "yaw_radians": yaw, "world_from_camera": transform.tolist()}
            if frame in indices:
                png, exr = path / arm / f"{frame:04d}.png", path / arm / f"{frame:04d}.exr"
                Image.fromarray(np.full((height, width, 3), 128, dtype=np.uint8)).save(png)
                channels = {"ViewLayer.Depth.Z": np.full((height, width), 2, dtype=np.float32),
                            "ViewLayer.IndexOB.X": np.ones((height, width), dtype=np.float32)}
                for axis in "XYZ":
                    channels[f"ViewLayer.Normal.{axis}"] = np.full((height, width), int(axis == "Z"), dtype=np.float32)
                for axis in "RGB":
                    channels[f"ViewLayer.Combined.{axis}"] = np.full((height, width), .4, dtype=np.float32)
                OpenEXR.File({}, channels).write(str(exr))
                record.update({"png": str(png.relative_to(path)), "exr": str(exr.relative_to(path)),
                               "png_sha256": sha256(png), "exr_sha256": sha256(exr)})
            records.append(record)
        manifest["arms"][arm] = records
    (path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def update_manifest(path, manifest):
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_sparse_capture_does_not_become_dense_training(tmp_path):
    path = tmp_path / "capture"
    fixture_capture(path, indices=(0, 1, 25), arms=("closed", "open"))
    report = validate(path)
    assert report["passed"], report["errors"]
    assert report["capture_kind"] == "sparse_preview"
    assert report["usable_adjacent_rgb_transitions"] == {"closed": 1, "open": 1}
    assert len(report["paired_frames"]) == 3
    assert report["paired_frames"][0]["identical_rgb_pixels"]
    assert "unverified" in report["depth_semantics"]
    assert not validate(path, require_dense=True)["passed"]


def test_dense_capture_checks_actual_files(tmp_path):
    path = tmp_path / "capture"
    fixture_capture(path, indices=tuple(range(66)))
    assert validate(path, require_dense=True)["passed"]
    (path / "closed/0032.exr").unlink()
    assert not validate(path, require_dense=True)["passed"]


@pytest.mark.parametrize("defect, expected", [
    ("hash", "hash mismatch"), ("action", "alignment"), ("yaw", "yaw"),
    ("extrinsics", "orthonormal"), ("K", "K disagrees"), ("door", "door state"),
    ("traversal", "Unsafe file path"), ("false_dense", "dense_sequence"),
])
def test_rejects_corrupted_contract(tmp_path, defect, expected):
    path = tmp_path / "capture"
    manifest = fixture_capture(path)
    records = manifest["arms"]["closed"]
    if defect == "hash": records[0]["png_sha256"] = "0" * 64
    elif defect == "action": records[2]["action_from_previous"] = "right"
    elif defect == "yaw": records[2]["yaw_radians"] = 0
    elif defect == "extrinsics": records[0]["world_from_camera"][0][0] = 2
    elif defect == "K": manifest["K"][0][0] = -1
    elif defect == "door": records[1]["door_open"] = True
    elif defect == "traversal": records[0]["png"] = "../outside.png"
    elif defect == "false_dense": manifest["dense_sequence"] = True
    update_manifest(path, manifest)
    result = validate(path)
    assert not result["passed"]
    assert expected in result["errors"][0]


@pytest.mark.parametrize("channel,value,expected", [
    ("ViewLayer.Depth.Z", float("nan"), "finite"),
    ("ViewLayer.Depth.Z", -1., "negative"),
    ("ViewLayer.Normal.Z", .1, "invalid lengths"),
    ("ViewLayer.IndexOB.X", 2., "absent from manifest"),
])
def test_rejects_bad_native_passes(tmp_path, channel, value, expected):
    OpenEXR = pytest.importorskip("OpenEXR")
    path = tmp_path / "capture"
    manifest = fixture_capture(path)
    exr = path / "closed/0000.exr"
    with OpenEXR.File(str(exr), separate_channels=True) as image:
        channels = {name: item.pixels.copy() for name, item in image.channels().items()}
    channels[channel][:] = value
    OpenEXR.File({}, channels).write(str(exr))
    manifest["arms"]["closed"][0]["exr_sha256"] = sha256(exr)
    update_manifest(path, manifest)
    report = validate(path)
    assert not report["passed"] and expected in report["errors"][0]


def test_depth_convention_requires_off_axis_evidence():
    depths = {("closed", 0): np.full((2, 8), 2.)}
    samples = [{"arm": "closed", "frame": 0, "pixel_xy": [x, 0], "euclidean_range": 2., "axial_z": 1.8} for x in range(8)]
    assert compare_depth_rays(samples, depths)["measured_depth_semantics"] == "euclidean_range"
    for sample in samples:
        sample["axial_z"] = 2.
    assert compare_depth_rays(samples, depths)["measured_depth_semantics"] == "inconclusive"


def test_blender51_multipart_passes_and_filtered_normals(tmp_path):
    OpenEXR = pytest.importorskip("OpenEXR")
    path = tmp_path / "capture"
    manifest = fixture_capture(path)
    exr = path / "closed/0000.exr"
    with OpenEXR.File(str(exr), separate_channels=True) as image:
        channels = {name: item.pixels.copy() for name, item in image.channels().items()}
    channels["ViewLayer.Object Index.X"] = channels.pop("ViewLayer.IndexOB.X")
    channels["ViewLayer.Normal.Z"][0, 0] = .36
    parts = [OpenEXR.Part({"name": name.rsplit(".", 1)[0]}, {name: data}) for name, data in channels.items()]
    # Actual Blender groups Normal XYZ and Combined RGBA, whereas this fixture
    # puts each channel in its own part to independently exercise aggregation.
    for index, part in enumerate(parts):
        part.header["name"] = f"part-{index}"
    OpenEXR.File(parts).write(str(exr))
    manifest["arms"]["closed"][0]["exr_sha256"] = sha256(exr)
    update_manifest(path, manifest)
    report = validate(path)
    assert report["passed"], report["errors"]
    assert report["frames"][0]["nonunit_normal_fraction"] > 0


def test_camera_translation_without_translation_action_is_invalid(tmp_path):
    path = tmp_path / "capture"
    manifest = fixture_capture(path)
    manifest["arms"]["closed"][2]["world_from_camera"][0][3] += .1
    update_manifest(path, manifest)
    report = validate(path)
    assert not report["passed"] and "translation changed" in report["errors"][0]


@pytest.mark.parametrize("lens", [18, 24])
def test_explicit_camera_model_derives_intrinsics(lens):
    focal = 512 * lens / 36
    manifest = {"K": [[focal, 0, 256], [0, focal, 144], [0, 0, 1]],
                "camera_model": {"type": "PERSPECTIVE", "lens_mm": lens, "sensor_width_mm": 36,
                                 "sensor_fit": "HORIZONTAL", "pixel_aspect": [1, 1]}}
    assert check_intrinsics(manifest, 512, 288)[1]["status"] == "matches_declared_camera_model"
    manifest["camera_model"]["lens_mm"] += 1
    with pytest.raises(ValueError, match="explicit camera_model"):
        check_intrinsics(manifest, 512, 288)


def test_legacy_intrinsics_need_native_query_for_physical_calibration(tmp_path):
    path = tmp_path / "capture"
    manifest = fixture_capture(path)
    assert validate(path)["intrinsics_calibration"]["status"] == "structural_checks_only"
    rays = {"blend_sha256": manifest["blend_sha256"], "manifest_sha256": sha256(path / "manifest.json"),
            "native_camera_K": manifest["K"],
            "samples": [{"arm": "closed", "frame": 0, "pixel_xy": [x, 0], "euclidean_range": 2., "axial_z": 1.8} for x in range(8)]}
    ray_path = tmp_path / "rays.json"
    ray_path.write_text(json.dumps(rays))
    result = validate(path, raycast_path=ray_path)
    assert result["passed"] and result["intrinsics_calibration"]["status"] == "matches_independent_native_camera"
    rays["native_camera_K"][0][0] *= 2
    ray_path.write_text(json.dumps(rays))
    result = validate(path, raycast_path=ray_path)
    assert not result["passed"] and "native Blender camera projection" in result["errors"][0]
