# SPDX-License-Identifier: Apache-2.0
# Independent validator. This file uses no Blender API and is licensed under
# the repository's Apache-2.0 license, separately from this directory's renderer.
"""Validate Atrium captures without rendering or exposing state to a model.

python -m experiments.atrium_data.validate CAPTURE_DIR [--require-dense]
Requires numpy, Pillow, and OpenEXR for image inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


SCHEMA = "worldline-atrium-pilot-v1"
FRAME_COUNT = 66


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_file(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("File name must be a nonempty relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe file path: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"Missing file or path outside capture: {relative}")
    return resolved


def finite_array(value, shape, label):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label}: expected finite array of shape {shape}")
    return array


def expected_actions(arm):
    return [None, "interact" if arm == "open" else "wait"] + ["left"] * 24 + ["wait"] * 16 + ["right"] * 24


def check_intrinsics(manifest, width, height):
    intrinsics = finite_array(manifest["K"], (3, 3), "K")
    focal = intrinsics[0, 0]
    expected = [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]
    if focal <= 0 or not np.allclose(intrinsics, expected, atol=1e-5):
        raise ValueError("K disagrees with positive equal focal lengths and centered square pixels")
    model = manifest.get("camera_model")
    if model is None:
        return intrinsics, {"status": "structural_checks_only", "legacy_manifest": True,
                            "limit": "Physical K calibration requires independent native-camera raycast metadata"}
    if not isinstance(model, dict) or model.get("type") != "PERSPECTIVE" or model.get("sensor_fit") != "HORIZONTAL" or model.get("pixel_aspect") != [1, 1]:
        raise ValueError("camera_model must declare PERSPECTIVE, HORIZONTAL fit and [1,1] pixel aspect")
    lens, sensor = model.get("lens_mm"), model.get("sensor_width_mm")
    if any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in (lens, sensor)):
        raise ValueError("camera_model lens and sensor width must be finite positive numbers")
    if not math.isclose(focal, width * lens / sensor, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError("K disagrees with explicit camera_model lens and sensor width")
    return intrinsics, {"status": "matches_declared_camera_model", "legacy_manifest": False,
                        "camera_model": model, "limit": "Manifest consistency; native-camera check requires raycast metadata"}


def check_camera(record, expected_yaw):
    transform = finite_array(record["world_from_camera"], (4, 4), "world_from_camera")
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Camera homogeneous last row is invalid")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=2e-5):
        raise ValueError("Camera rotation is not orthonormal with determinant +1")
    # Independently reconstruct the current v1 camera direction and roll.
    forward = np.array([-math.sin(expected_yaw), math.cos(expected_yaw), -.07])
    forward /= np.linalg.norm(forward)
    right = np.array([math.cos(expected_yaw), math.sin(expected_yaw), 0.])
    expected = np.column_stack((right, np.cross(forward, right), forward))
    if not np.allclose(rotation, expected, atol=2e-5):
        raise ValueError("Camera extrinsics disagree with action-before-frame yaw or OpenCV convention")
    return transform[:3, 3]


def inspect_exr(path, width, height, object_ids):
    import OpenEXR

    with OpenEXR.File(str(path), header_only=True) as header_file:
        if not 1 <= len(header_file.parts) <= 64:
            raise ValueError("EXR part count outside supported range")
        for part in header_file.parts:
            lo, hi = part.header["dataWindow"]
            if tuple(lo) != (0, 0) or tuple(hi - lo + 1) != (width, height):
                raise ValueError("EXR data window does not match capture resolution")
    with OpenEXR.File(str(path), separate_channels=True) as image:
        channels = {}
        for part in image.parts:
            for name, value in part.channels.items():
                if name in channels:
                    raise ValueError(f"Duplicate EXR channel across parts: {name}")
                channels[name] = value.pixels

    def channel(suffix):
        names = [name for name in channels if name.endswith(suffix)]
        if len(names) != 1:
            raise ValueError(f"Expected exactly one EXR channel ending {suffix}; found {names}")
        return finite_array(channels[names[0]], (height, width), names[0])

    depth = channel(".Depth.Z")
    normal = np.stack([channel(f".Normal.{axis}") for axis in "XYZ"], axis=-1)
    # Blender 5.1 writes multipart passes and renamed the Object Index layer.
    ids = channel((".IndexOB.X", ".Object Index.X"))
    color = np.stack([channel(f".Combined.{axis}") for axis in "RGB"], axis=-1)
    if (depth < 0).any():
        raise ValueError("Depth contains negative values")
    if not np.allclose(ids, np.rint(ids), atol=1e-5):
        raise ValueError("Object pass contains noninteger IDs")
    ids = np.rint(ids).astype(np.int64)
    unknown = set(np.unique(ids)) - set(object_ids) - {0}
    if unknown:
        raise ValueError(f"Object IDs absent from manifest: {sorted(unknown)}")
    valid = ids != 0
    if not valid.any():
        raise ValueError("No foreground pixels with object IDs")
    if ((depth[valid] <= 0) | (depth[valid] >= 100)).any():
        raise ValueError("Foreground depth is outside the v1 camera clipping range (0,100m)")
    norms = np.linalg.norm(normal, axis=-1)
    # Cycles filters/averages shading normals. At edges or with material bump,
    # their mean can be shorter than one. Reject impossible lengths and bulk
    # degeneration, and report the nonunit fraction without calling it an error.
    invalid_norm_fraction = float(np.mean(np.abs(norms[valid] - 1) > .08))
    if norms[valid].max() > 1.01 or np.median(norms[valid]) < .5:
        raise ValueError("Foreground normals have invalid lengths or bulk loss of direction")
    result = {
        "channels": sorted(channels), "valid_depth_fraction": float(valid.mean()),
        "foreground_depth_min": float(depth[valid].min()), "foreground_depth_max": float(depth[valid].max()),
        "normal_norm_quantiles": np.quantile(norms[valid], [0, .01, .5, .99, 1]).tolist(),
        "nonunit_normal_fraction": invalid_norm_fraction, "object_ids": sorted(map(int, np.unique(ids))),
        "normal_semantics": "Native filtered shading normals; nonunit values can occur at edges or bump details",
        "linear_rgb_min": float(color.min()), "linear_rgb_max": float(color.max()),
        "background_depth_values": np.unique(depth[~valid]).tolist()[:10],
    }
    return result, ids, depth


def compare_depth_rays(samples, depths, id_images=None):
    """Score independent geometric rays. This does not equate depth with physics."""
    errors = {"euclidean_range": [], "axial_z": []}
    off_axis = []
    per_frame, id_matches = {}, []
    for sample in samples:
        key = (sample["arm"], sample["frame"])
        if key not in depths:
            raise ValueError("Ray sample references an uncaptured frame")
        depth = depths[key]
        x, y = sample["pixel_xy"]
        if type(x) is not int or type(y) is not int or not (0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]):
            raise ValueError("Ray sample pixel is outside image")
        native = float(depth[y, x])
        values = [sample["euclidean_range"], sample["axial_z"]]
        if not all(isinstance(v, (float, int)) and math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Ray distances must be finite and positive")
        for label, value in zip(errors, values):
            errors[label].append(abs(native - value) / value)
        per_frame.setdefault(f"{sample['arm']}/{sample['frame']:04d}", []).append(
            {label: abs(native - value) / value for label, value in zip(errors, values)})
        if id_images is not None and "object_id" in sample:
            id_matches.append(int(id_images[key][y, x]) == sample["object_id"])
        off_axis.append(abs(values[0] - values[1]) / values[0])
    if not samples:
        raise ValueError("No raycast samples")
    medians = {k: float(np.median(v)) for k, v in errors.items()}
    winner = min(medians, key=medians.get)
    loser = max(medians, key=medians.get)
    separated = sum(v >= .03 for v in off_axis)
    # Center rays cannot distinguish these conventions. This threshold requires
    # several useful off-axis queries and a substantial gap between hypotheses.
    conclusive = len(samples) >= 8 and separated >= 6 and medians[winner] < .005 and medians[loser] > .02
    return {"sample_count": len(samples), "off_axis_discriminating_samples": separated,
            "median_relative_errors": medians, "max_relative_errors": {k: max(v) for k, v in errors.items()},
            "measured_depth_semantics": winner if conclusive else "inconclusive",
            "per_frame_median_relative_errors": {key: {label: float(np.median([v[label] for v in values])) for label in errors}
                                                 for key, values in per_frame.items()},
            "ray_object_id_match_fraction": float(np.mean(id_matches)) if id_matches else None,
            "scope": "Native depth agreement with independent scene ray casts; no learned model evaluated"}


def validate(capture_dir, require_dense=False, raycast_path=None):
    root = Path(capture_dir).resolve()
    report = {"schema": "worldline-atrium-validation-v1", "capture": str(root), "errors": [],
              "validator_sha256": sha256(__file__), "validator_license": "Apache-2.0",
              "require_dense": require_dense, "frames": [], "paired_frames": [],
              "depth_semantics": "unverified without independent raycast samples",
              "limits": ["RGB/depth capture validation is not learned video quality evaluation.",
                         "Off-screen doors can change illumination or reflections; visibility alone does not establish identical histories."]}
    pixels, depths, id_images = {}, {}, {}
    try:
        manifest_path = local_file(root, "manifest.json")
        if manifest_path.stat().st_size > 5_000_000:
            raise ValueError("Manifest exceeds 5 MB")
        manifest = json.loads(manifest_path.read_text())
        report["manifest_sha256"] = sha256(manifest_path)
        report["capture_provenance"] = {key: manifest.get(key) for key in (
            "script_sha256", "scene_seed", "blender", "samples", "render_engine", "device",
            "dataset_license", "resolution", "camera_model", "depth_semantics", "rgb_transform",
            "protocol", "elapsed_seconds")}
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
            raise ValueError("Expected a complete worldline-atrium-pilot-v1 capture")
        width, height = manifest["resolution"]
        if any(type(v) is not int or not 1 <= v <= 1280 for v in (width, height)):
            raise ValueError("Resolution must be positive integers up to 1280")
        intrinsics, report["intrinsics_calibration"] = check_intrinsics(manifest, width, height)
        objects = manifest["objects"]
        object_ids = {int(key) for key in objects}
        if len(object_ids) != len(objects) or any(i <= 0 for i in object_ids) or any(not isinstance(n, str) for n in objects.values()):
            raise ValueError("Object manifest must have unique positive integer IDs and string names")
        door_ids = {int(key) for key, name in objects.items() if name.startswith(("Door leaf", "Door handle", "Door inlay"))}
        report["door_object_ids"] = sorted(door_ids)
        blend = local_file(root, "original-scene.blend")
        if sha256(blend) != manifest.get("blend_sha256"):
            raise ValueError("Original scene hash mismatch")
        arms = manifest["arms"]
        if not arms or not set(arms).issubset({"closed", "open"}):
            raise ValueError("Capture arms must be closed and/or open")
        rendered_sets = {}
        positions = []
        for arm, records in arms.items():
            if len(records) != FRAME_COUNT or any(type(r["frame"]) is not int for r in records) or [r["frame"] for r in records] != list(range(FRAME_COUNT)):
                raise ValueError(f"{arm}: expected ordered records 0 through 65, including uncaptured states")
            yaw = 0.
            rendered = set()
            for index, (record, expected_action) in enumerate(zip(records, expected_actions(arm))):
                prefix = f"{arm}/{index:04d}"
                if record.get("action_from_previous") != expected_action:
                    raise ValueError(f"{prefix}: action-before-frame alignment mismatch")
                yaw += math.pi / 24 if expected_action == "left" else -math.pi / 24 if expected_action == "right" else 0
                if not math.isfinite(record["yaw_radians"]) or not math.isclose(record["yaw_radians"], yaw, abs_tol=1e-6):
                    raise ValueError(f"{prefix}: yaw does not match actions applied before this frame")
                if type(record["door_open"]) is not bool or record["door_open"] != (arm == "open" and index > 0):
                    raise ValueError(f"{prefix}: door state does not match preceding interaction")
                positions.append(check_camera(record, yaw))
                file_fields = {"png", "png_sha256", "exr", "exr_sha256"}
                present = file_fields.intersection(record)
                if not present:
                    continue
                if present != file_fields:
                    raise ValueError(f"{prefix}: incomplete RGB/EXR file and hash record")
                rendered.add(index)
                for kind in ("png", "exr"):
                    path = local_file(root, record[kind])
                    if path != (root / arm / f"{index:04d}.{kind}").resolve():
                        raise ValueError(f"{prefix}: file path does not match arm and frame")
                    if sha256(path) != record[f"{kind}_sha256"]:
                        raise ValueError(f"{prefix}: {kind.upper()} hash mismatch")
                with Image.open(root / record["png"]) as png:
                    if png.format != "PNG" or png.size != (width, height) or png.mode not in ("RGB", "RGBA"):
                        raise ValueError(f"{prefix}: invalid RGB PNG dimensions or mode")
                    rgb = np.asarray(png.convert("RGB"))
                metrics, ids, depth = inspect_exr(root / record["exr"], width, height, object_ids)
                door_pixels = int(np.isin(ids, list(door_ids)).sum())
                metrics.update({"arm": arm, "frame": index, "door_visible_pixels": door_pixels,
                                "rgb_pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
                report["frames"].append(metrics)
                pixels[(arm, index)], depths[(arm, index)], id_images[(arm, index)] = rgb, depth, ids
            rendered_sets[arm] = sorted(rendered)
        actual_dense = all(len(indices) == FRAME_COUNT for indices in rendered_sets.values())
        if manifest.get("dense_sequence") is not actual_dense:
            raise ValueError("dense_sequence flag disagrees with actual captured files")
        if not any(rendered_sets.values()):
            raise ValueError("Capture has no rendered observations")
        report["capture_kind"] = "dense" if actual_dense else "sparse_preview"
        report["rendered_indices"] = rendered_sets
        report["transitions_per_arm_in_state_records"] = FRAME_COUNT - 1
        report["camera_translation_span_m"] = np.ptp(np.array(positions), axis=0).tolist()
        if max(report["camera_translation_span_m"]) > 1e-5:
            raise ValueError("Camera translation changed despite v1 actions specifying yaw only")
        report["usable_adjacent_rgb_transitions"] = {arm: sum(i + 1 in indices for i in indices) for arm, indices in rendered_sets.items()}
        if require_dense and not actual_dense:
            raise ValueError("Dense training requested but one or more RGB/EXR frames are missing")
        for index in sorted(set(rendered_sets.get("closed", [])) & set(rendered_sets.get("open", []))):
            closed, opened = pixels[("closed", index)], pixels[("open", index)]
            diff = np.abs(closed.astype(np.float32) - opened.astype(np.float32)) / 255
            report["paired_frames"].append({"frame": index, "rgb_mae_0_1": float(diff.mean()),
                "changed_pixel_fraction": float(np.any(closed != opened, axis=-1).mean()),
                "identical_rgb_pixels": bool(np.array_equal(closed, opened)),
                "closed_door_pixels": int(np.isin(id_images[("closed", index)], list(door_ids)).sum()),
                "open_door_pixels": int(np.isin(id_images[("open", index)], list(door_ids)).sum()),
                "interpretation": "Paired rendered observation only; no conclusion about unrendered history"})
        if raycast_path:
            ray_data = json.loads(Path(raycast_path).read_text())
            if ray_data.get("blend_sha256") != manifest["blend_sha256"] or ray_data.get("manifest_sha256") != report["manifest_sha256"]:
                raise ValueError("Raycast provenance does not match this scene and manifest")
            report["raycast_provenance"] = {key: ray_data.get(key) for key in (
                "schema", "query_sha256", "query_license", "blender", "method", "blend_sha256", "manifest_sha256")}
            report["raycast_provenance"]["raycast_json_sha256"] = sha256(raycast_path)
            native_k = ray_data.get("native_camera_K")
            if native_k is not None:
                native_k = finite_array(native_k, (3, 3), "native_camera_K")
                if not np.allclose(intrinsics, native_k, rtol=1e-5, atol=1e-5):
                    raise ValueError("K disagrees with independently queried native Blender camera projection")
                report["intrinsics_calibration"].update({"status": "matches_independent_native_camera",
                    "native_camera_K": native_k.tolist(), "limit": "Verified for this saved scene and capture resolution"})
            report["depth_semantics"] = compare_depth_rays(ray_data["samples"], depths, id_images)
    except (ValueError, TypeError, KeyError, OSError, ImportError, OverflowError, RuntimeError) as exc:
        report["errors"].append(str(exc))
    report["passed"] = not report["errors"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--require-dense", action="store_true")
    parser.add_argument("--raycast-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.capture_dir, args.require_dense, args.raycast_json)
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        # Reports are new artifacts; avoid silently replacing source evidence.
        with args.output.open("x") as handle:
            handle.write(content)
    print(content, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
