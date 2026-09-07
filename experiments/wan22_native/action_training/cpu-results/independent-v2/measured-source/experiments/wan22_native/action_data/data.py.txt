# SPDX-License-Identifier: Apache-2.0
"""Verified original RGB/commands and the explicit start0 RGB canonicalization."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from experiments.wan_adapter import capture_data as original

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PLAN_SHA256 = "52fead080ba299f89c7bc1aad1f58c1679c93c8ec735c177aa99c5b3916e6a62"
MANIFEST_SHA256 = "942eaf38badb1c2de5489fac59b7727e5c2a3d5699ec44aa415b7862c8440ef0"
READER_SHA256 = "859a1001980a2d0efe99a6369979e5f225c72993fcaa3ef901e130fe60c249ca"
CANONICAL_PNG_SHA256 = "7bdfa121eb2917b837af3ee1faae9e697751cdd3cdee0b21422c1b1f2a53e780"
CANONICAL_RGB_SHA256 = "6754e1dd25290854ddea0052b04154bfc1e750a7b4a7cd408a07e7878b50f705"
CLOSED_RGB_SHA256 = "73532812b5360ce615c527d5f242d88aaf91fd943717aa45bd3ca8f8a93d3dad"
SELECTION = tuple((arm, start) for start in (0, 8, 32, 49) for arm in ("closed", "open"))
CHANNELS = tuple(original.ACTION_CHANNELS)


def sha(path):
    return original.sha256(Path(path))


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True)
class DerivedWindow:
    rgb: np.ndarray
    commands: np.ndarray
    provenance: dict

    def video_array(self):
        return np.ascontiguousarray(self.rgb.transpose(3,0,1,2)[None], dtype=np.float32)/127.5-1.


def canonical_changes(closed, canonical):
    if (closed.shape != (288,512,3) or canonical.shape != closed.shape
            or closed.dtype != np.uint8 or canonical.dtype != np.uint8
            or array_sha(closed) != CLOSED_RGB_SHA256 or array_sha(canonical) != CANONICAL_RGB_SHA256):
        raise ValueError("Exact original closed and canonical open RGB arrays required")
    difference = canonical.astype(np.int16)-closed.astype(np.int16)
    indices = np.argwhere(difference != 0)
    if len(indices) != 18 or int(np.abs(difference).max()) != 1:
        raise ValueError("Prescribed start0 difference must be exactly18 one-level channel values")
    return [{"y":int(y),"x":int(x),"channel":int(c),"raw_closed":int(closed[y,x,c]),
             "derived_canonical":int(canonical[y,x,c])} for y,x,c in indices]


def derive_window(window, canonical):
    p = window.provenance
    identity = (p.get("arm"), p.get("start"))
    if identity not in SELECTION or window.rgb.shape != (17,288,512,3) or window.rgb.dtype != np.uint8:
        raise ValueError("Only the eight declared native17-frame windows are accepted")
    if window.actions.shape != (16,6) or window.actions.dtype != np.float32 or not np.isfinite(window.actions).all():
        raise ValueError("Exactly16 finite real six-channel commands required")
    if array_sha(canonical) != CANONICAL_RGB_SHA256:
        raise ValueError("Canonical first RGB differs")
    rgb = window.rgb.copy()
    changed = identity == ("closed",0)
    differences = canonical_changes(rgb[0], canonical) if changed else []
    if changed:
        rgb[0] = canonical
    elif identity == ("open",0) and array_sha(rgb[0]) != CANONICAL_RGB_SHA256:
        raise ValueError("Open start0 must already contain the canonical RGB")
    provenance = {"raw":p, "raw_rgb_sha256":array_sha(window.rgb), "raw_first_rgb_sha256":array_sha(window.rgb[0]),
        "derived_rgb_sha256":array_sha(rgb), "derived_first_rgb_sha256":array_sha(rgb[0]),
        "commands_sha256":array_sha(window.actions), "derived_shape":list(rgb.shape),
        "changed_first_rgb":changed, "changed_channel_values":len(differences), "channel_changes":differences,
        "canonical_png_sha256":CANONICAL_PNG_SHA256 if identity[1]==0 else None,
        "canonical_rgb_sha256":CANONICAL_RGB_SHA256 if identity[1]==0 else None,
        "target_policy":"Replace only derived closed start0 RGB frame0 before full encoding; all other raw RGB values unchanged",
        "raw_files_modified":False, "old_latent_reused":False}
    return DerivedWindow(rgb, window.actions.copy(), provenance)


def load_selection(capture):
    root = Path(capture).resolve()
    if sha(HERE/"source-plan.md.txt") != PLAN_SHA256 or sha(Path(original.__file__)) != READER_SHA256:
        raise ValueError("Pinned action plan or original command reader changed")
    if sha(root/"manifest.json") != MANIFEST_SHA256:
        raise ValueError("Original dense Atrium manifest differs")
    raw = {(arm,start):original.load_window(root,arm,start,17) for arm,start in SELECTION}
    canonical = raw[("open",0)].rgb[0]
    if raw[("open",0)].provenance["sources"][0]["sha256"] != CANONICAL_PNG_SHA256:
        raise ValueError("Canonical open initial PNG identity differs")
    canonical_changes(raw[("closed",0)].rgb[0], canonical)
    windows = {identity:derive_window(raw[identity],canonical) for identity in SELECTION}
    if not np.array_equal(windows[("closed",0)].rgb[0], windows[("open",0)].rgb[0]):
        raise RuntimeError("Shared start0 observation is not exact")
    return windows


def plan(capture, *, mode="cache"):
    if mode not in ("cache","roundtrip"):
        raise ValueError("Only cache or prescribed open-start0 roundtrip is supported")
    windows = load_selection(capture)
    selected = SELECTION if mode == "cache" else (("open",0),)
    return {"schema":"worldline-wan22-atrium-action-plan-v1","status":"planned","mode":mode,
        "source_plan_sha256":PLAN_SHA256,"capture_manifest_sha256":MANIFEST_SHA256,
        "reader_sha256":READER_SHA256,"split":"development","independent_layouts":1,
        "data_license":"CC0-1.0","action_schema":original.ACTION_SCHEMA,"command_channels":list(CHANNELS),
        "command_units":"Requested local meters, yaw/pitch radians, one-transition interaction pulse; no normalization",
        "command_alignment":"commands[t] is records[t+1].action_from_previous, mapping RGB[t] to RGB[t+1]",
        "selection":[{"id":f"{arm}-{start:04d}","arm":arm,"start":start,"frames":17,
            "source":windows[(arm,start)].provenance} for arm,start in selected],
        "canonical_observation":"One independently encoded open/0000 RGB shared by both start0 branches",
        "canonicalization_changed_channel_values":18,
        "target_shape":[1,48,5,18,32],"observation_shape":[1,48,1,18,32],"commands_shape":[1,16,6],
        "stored_dtype":"float32","causal_check":"Exact tensor equality; no post-encode replacement of target prefix",
        "model_execution":False,"gpu_execution":False,"raw_capture_modified":False,
        "limitations":"One layout; yaw and programmed remote interaction only. Starts8/32/49 have different branch initial images and identical outgoing commands; not isolated action-intervention tests."}


def load_roundtrip_rgb(capture):
    """Read only the prescribed 17 original RGB files; never construct commands."""
    import json
    from PIL import Image
    root = Path(capture).resolve()
    if sha(HERE/'source-plan.md.txt') != PLAN_SHA256 or sha(Path(original.__file__)) != READER_SHA256:
        raise ValueError('Pinned data plan or reader changed')
    manifest_path = root/'manifest.json'
    if not manifest_path.resolve().is_relative_to(root) or sha(manifest_path) != MANIFEST_SHA256:
        raise ValueError('Exact original manifest required')
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['arms']['open'][:17]
    pixels, sources = [], []
    for row in rows:
        path = original.local_image(root, row['png'])
        if sha(path) != row['png_sha256']:
            raise ValueError('Original RGB file changed')
        with Image.open(path) as image:
            if image.size != (512,288) or image.mode not in ('RGB','RGBA'):
                raise ValueError('Native RGB or opaque RGBA required')
            if image.mode == 'RGBA' and image.getchannel('A').getextrema() != (255,255):
                raise ValueError('No transparent pixels or compositing accepted')
            pixels.append(np.array(image.convert('RGB'),dtype=np.uint8))
        sources.append({'frame':row['frame'],'file':row['png'],'sha256':row['png_sha256']})
    rgb = np.stack(pixels)
    if rgb.shape != (17,288,512,3) or array_sha(rgb[0]) != CANONICAL_RGB_SHA256:
        raise ValueError('Incorrect prescribed open-start0 RGB')
    return rgb, {'id':'open-0000','capture_manifest_sha256':MANIFEST_SHA256,'sources':sources,
        'rgb_sha256':array_sha(rgb),'first_rgb_sha256':array_sha(rgb[0]),'raw_rgb_modified':False,
        'data_license':'CC0-1.0','rgb_transform':manifest['rgb_transform'],
        'action_values_materialized':False,'future_latents_read':False}
