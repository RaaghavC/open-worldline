# SPDX-License-Identifier: Apache-2.0
"""Read original RGB and commanded actions without exposing simulator state.

The single-layout Atrium capture is development data. Multiple windows or
branches do not create independent scenes. No encoder or GPU is used here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


ACTION_CHANNELS = (
    "local_right_m", "local_up_m", "local_forward_m",
    "yaw_left_rad", "pitch_up_rad", "interact_pulse",
)
ACTION_SCHEMA = "worldline-command-deltas-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_vector(command: str) -> np.ndarray:
    result = np.zeros(6, dtype=np.float32)
    if command == "left":
        result[3] = math.pi / 24
    elif command == "right":
        result[3] = -math.pi / 24
    elif command == "interact":
        result[5] = 1
    elif command != "wait":
        raise ValueError(f"Unsupported Atrium command: {command!r}")
    return result


def local_image(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("RGB path must be a nonempty relative string")
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("RGB path must remain inside the capture")
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("RGB file is missing or outside the capture")
    return path


@dataclass(frozen=True)
class RGBActionWindow:
    """Only RGB/actions are model inputs. Provenance is for audit and grouping."""

    rgb: np.ndarray  # uint8 [T,H,W,3], original native PNG values
    actions: np.ndarray  # float32 [T-1,6], raw command units
    provenance: dict

    def video_array(self) -> np.ndarray:
        """Return [1,3,T,H,W] float32 in [-1,1], with no crop or resize."""
        return np.ascontiguousarray(self.rgb.transpose(3, 0, 1, 2)[None], dtype=np.float32) / 127.5 - 1


def load_window(capture_dir: str | Path, arm: str, start: int, frames: int = 17) -> RGBActionWindow:
    if type(start) is not int or start < 0 or type(frames) is not int or frames < 5 or frames % 4 != 1:
        raise ValueError("Window requires a nonnegative integer start and 1+4n frames, n>=1")
    root = Path(capture_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.resolve().is_relative_to(root):
        raise ValueError("Manifest must remain inside the capture")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if (manifest.get("schema") != "worldline-atrium-pilot-v1" or
            manifest.get("status") != "complete" or manifest.get("dense_sequence") is not True):
        raise ValueError("Expected a completed dense Atrium v1 capture")
    if manifest.get("dataset_license") != "CC0-1.0" or manifest.get("scene_family") != "single_atrium_layout_v1":
        raise ValueError("This reader accepts the identified original CC0 Atrium layout only")
    if manifest.get("resolution") != [512, 288]:
        raise ValueError("Expected native 512 by 288 RGB; resizing is not implemented")
    if arm not in ("closed", "open") or arm not in manifest.get("arms", {}):
        raise ValueError("Expected an available closed or open arm")
    records = manifest["arms"][arm]
    if not isinstance(records, list) or len(records) != 66 or start + frames > len(records):
        raise ValueError("Window exceeds the 66-frame capture")
    expected = [None, "interact" if arm == "open" else "wait"] + ["left"] * 24 + ["wait"] * 16 + ["right"] * 24
    for i, record in enumerate(records):
        if record.get("frame") != i or record.get("action_from_previous") != expected[i]:
            raise ValueError("Manifest actions or frame indices disagree with the Atrium protocol")
    pixels, sources = [], []
    for record in records[start:start + frames]:
        path = local_image(root, record.get("png"))
        actual = sha256(path)
        if actual != record.get("png_sha256"):
            raise ValueError(f"RGB hash mismatch at frame {record['frame']}")
        with Image.open(path) as image:
            if image.size != (512, 288) or image.mode not in ("RGB", "RGBA"):
                raise ValueError("RGB file dimensions or mode disagree with the native capture")
            if image.mode == "RGBA" and image.getchannel("A").getextrema() != (255, 255):
                raise ValueError("Transparent input requires an explicit compositing protocol")
            pixels.append(np.array(image.convert("RGB"), dtype=np.uint8))
        sources.append({"frame": record["frame"], "file": record["png"], "sha256": actual})
    # The destination record carries the command for the incoming transition.
    actions = np.stack([command_vector(record["action_from_previous"])
                        for record in records[start + 1:start + frames]])
    provenance = {
        "schema": "worldline-rgb-action-window-v1",
        "capture_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "scene_family": manifest["scene_family"], "scene_seed": manifest["scene_seed"],
        "split": "development", "independent_layouts": 1,
        "arm": arm, "start": start, "frames": frames, "sources": sources,
        "action_schema": ACTION_SCHEMA, "action_channels": list(ACTION_CHANNELS),
        "action_normalization": "none; raw commanded meters/radians/pulse",
        "rgb_transform": manifest.get("rgb_transform"), "data_license": "CC0-1.0",
        "model_inputs": ["rgb", "commanded_action_deltas"],
        "first_observation_index": start,
        "observation_policy": "Encode first RGB alone for conditioning; all later RGB are training targets only",
    }
    return RGBActionWindow(np.stack(pixels), actions, provenance)
