"""Read one pinned native RGB/action window without renderer or model imports."""
from dataclasses import dataclass, fields
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

WIDTH, HEIGHT = 1248, 704
CAPTURE_SHA = '725b739fd801c497ef8024f126cbf79a27e7dbbd38fd0311e57d9932fd9bc4d8'
SCENE_SHA = 'cf794ac98a01209be8ea70b3849f5f87174f0750b2990159cb052002836f7a56'
ARMS = tuple(m+'_'+d for m in ('stationary', 'left', 'right') for d in ('closed', 'interact'))
CHANNELS = ['local_right_m', 'local_up_m', 'local_forward_m', 'yaw_left_rad', 'pitch_up_rad', 'interact_pulse']
MAX_MANIFEST_BYTES = 2*1024*1024
MAX_IMAGE_BYTES = 16*1024*1024


@dataclass(frozen=True)
class RGBActionWindow:
    """Only model-facing arrays; all are independently owned contiguous FP32.

    initial_rgb: [3,704,1248], values in [-1,1].
    future_rgb_targets: [16,3,704,1248], values in [-1,1].
    commands: [16,6], recorded destination-aligned deltas and interaction pulse.
    """
    initial_rgb: np.ndarray
    future_rgb_targets: np.ndarray
    commands: np.ndarray


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_hash(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def bounded_bytes(path, maximum):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Regular file required: '+path.name)
    require(0 < path.stat().st_size <= maximum, 'File byte bound failed: '+path.name)
    with path.open('rb') as stream:
        content = stream.read(maximum+1)
    require(0 < len(content) <= maximum, 'File grew past byte bound: '+path.name)
    return content


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: '+key)
        result[key] = value
    return result


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def checked_manifest(capture_root, arm, expected_manifest_sha256):
    require(is_hash(expected_manifest_sha256), 'Caller must provide the exact manifest SHA256')
    require(type(arm) is str and arm in ARMS, 'Unknown factorial arm')
    supplied = Path(capture_root)
    require(supplied.is_dir() and not supplied.is_symlink(), 'Regular capture root required')
    root = supplied.resolve(strict=True)
    raw = bounded_bytes(root/'manifest.json', MAX_MANIFEST_BYTES)
    require(hashlib.sha256(raw).hexdigest() == expected_manifest_sha256, 'Manifest identity differs')
    manifest = json.loads(raw, object_pairs_hook=no_duplicates,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON constant')))
    require(type(manifest) is dict, 'Manifest object required')
    required = {
        'schema': 'worldline-atrium-factorial-v1', 'status': 'complete', 'split': 'development',
        'scene_family': 'single_atrium_layout_v1', 'scene_seed': 51000, 'independent_layouts': 1,
        'resolution': [WIDTH, HEIGHT], 'frames_per_arm': 17,
        'capture_source_sha256': CAPTURE_SHA, 'scene_source_sha256': SCENE_SHA,
        'neural_model_execution': False, 'action_channels': CHANNELS,
        'command_alignment': 'record t contains the command applied from observation t-1 to t; frame0 has no incoming command',
        'interaction_semantics': 'Programmed instantaneous remote door toggle to102degrees; no reach/contact/collision simulation',
        'model_inputs': ['initial_rgb', 'requested_command_deltas'], 'training_targets': ['future_rgb'],
        'extra_metadata_not_model_inputs': ['door_open', 'yaw_radians', 'world_from_camera'],
        'rendered_unique_states': 66, 'copied_identical_states': 36,
    }
    for key, value in required.items():
        require(type(manifest.get(key)) is type(value) and manifest[key] == value, 'Manifest protocol differs: '+key)
    require(finite(manifest.get('yaw_step_radians')) and abs(manifest['yaw_step_radians']-math.pi/120) <= 1e-12, 'Yaw step differs')
    require(type(manifest.get('arms')) is dict and set(manifest['arms']) == set(ARMS), 'Complete six-arm metadata required')
    first_state, known_hashes = {}, {}
    # Validate all102 cheap metadata rows, but open image bytes only for selected arm.
    for name in ARMS:
        motion, door = name.split('_')
        direction = {'stationary': 0, 'left': 1, 'right': -1}[motion]
        rows = manifest['arms'][name]
        require(type(rows) is list and len(rows) == 17, 'Each arm must declare17 ordered frames')
        for frame, row in enumerate(rows):
            require(type(row) is dict and type(row.get('frame')) is int and row['frame'] == frame, 'Frame ordering differs')
            path = name+'/'+f'{frame:04d}.png'
            require(row.get('png') == path and is_hash(row.get('png_sha256')), 'Image path/hash differs')
            opened = door == 'interact' and frame > 0
            require(type(row.get('door_open')) is bool and row['door_open'] == opened, 'Door protocol label differs')
            yaw = direction*frame*math.pi/120
            require(finite(row.get('yaw_radians')) and abs(row['yaw_radians']-yaw) <= 1e-12, 'Recorded yaw differs')
            command = row.get('command_from_previous')
            if frame == 0:
                require(command is None, 'Initial frame must have no incoming command')
            else:
                expected = [0., 0., 0., direction*math.pi/120, 0., float(door == 'interact' and frame == 1)]
                require(type(command) is list and len(command) == 6 and all(finite(x) for x in command), 'Six finite command channels required')
                require(all(abs(x-y) <= 1e-12 for x, y in zip(command, expected)), 'Destination-aligned command differs')
            state = (direction*frame, opened)
            origin = first_state.get(state)
            require('reused_from' in row and row['reused_from'] == origin, 'Reuse identity differs from first equivalent state')
            if origin is None:
                first_state[state] = path
            else:
                require(row['png_sha256'] == known_hashes[origin], 'Declared reused image hash differs')
            known_hashes[path] = row['png_sha256']
    require(len(first_state) == 66, 'Unique declared state count differs')
    return root, manifest['arms'][arm], raw


def image_path(root, arm, frame):
    folder = root/arm
    require(folder.is_dir() and not folder.is_symlink(), 'Regular arm directory required')
    return folder/f'{frame:04d}.png'


def read_rgb(path, expected_sha256):
    content = bounded_bytes(path, MAX_IMAGE_BYTES)
    require(hashlib.sha256(content).hexdigest() == expected_sha256, 'Consumed PNG hash differs: '+path.name)
    require(len(content) >= 33 and content[:8] == b'\x89PNG\r\n\x1a\n' and content[12:16] == b'IHDR'
            and content[24] == 8 and content[25] in (2, 6), '8-bit truecolor PNG header required')
    with Image.open(io.BytesIO(content)) as image:
        require(image.format == 'PNG' and image.size == (WIDTH, HEIGHT), 'Native1248x704 PNG required')
        require(image.mode in ('RGB', 'RGBA') and getattr(image, 'n_frames', 1) == 1, 'Single8-bit RGB or opaque RGBA image required')
        image.load()
        pixels = np.array(image)
    require(pixels.dtype == np.uint8, '8-bit pixels required')
    if pixels.shape[2] == 4:
        require(bool((pixels[:, :, 3] == 255).all()), 'Nonopaque alpha is not supported')
        pixels = pixels[:, :, :3]
    require(pixels.shape == (HEIGHT, WIDTH, 3), 'Native RGB shape differs')
    result = np.array(pixels.transpose(2, 0, 1), dtype=np.float32, order='C', copy=True)
    result /= np.float32(127.5)
    result -= np.float32(1.)
    require(np.isfinite(result).all() and result.min() >= -1 and result.max() <= 1, 'Normalized RGB bounds failed')
    return result, len(content)


def _read(capture_root, arm, expected_manifest_sha256):
    root, rows, manifest_bytes = checked_manifest(capture_root, arm, expected_manifest_sha256)
    initial, initial_bytes = read_rgb(image_path(root, arm, 0), rows[0]['png_sha256'])
    future = np.empty((16, 3, HEIGHT, WIDTH), dtype=np.float32)
    image_identities = {rows[0]['png']: {'sha256': rows[0]['png_sha256'], 'bytes': initial_bytes}}
    for frame in range(1, 17):
        pixels, size = read_rgb(image_path(root, arm, frame), rows[frame]['png_sha256'])
        future[frame-1] = pixels
        image_identities[rows[frame]['png']] = {'sha256': rows[frame]['png_sha256'], 'bytes': size}
        del pixels
    # Consume recorded deltas after validation; do not synthesize commands from pose/state.
    commands = np.array([r['command_from_previous'] for r in rows[1:]], dtype=np.float32, order='C')
    require(bounded_bytes(root/'manifest.json', MAX_MANIFEST_BYTES) == manifest_bytes, 'Manifest changed during window read')
    window = RGBActionWindow(initial, future, commands)
    for item in fields(window):
        value = getattr(window, item.name)
        require(value.dtype == np.float32 and value.flags.c_contiguous and value.flags.owndata, 'Owned contiguous FP32 arrays required')
    return window, {'schema': 'worldline-atrium-factorial-window-verification-v1', 'status': 'verified',
        'manifest_sha256': expected_manifest_sha256, 'arm': arm, 'image_files_read': 17,
        'images': image_identities, 'capture_source_sha256': CAPTURE_SHA, 'scene_source_sha256': SCENE_SHA,
        'model_facing_fields': [f.name for f in fields(window)], 'pixel_operation': 'uint8 to FP32; divide127.5 then subtract1; no crop/resize/compositing',
        'validation_scope': 'All six arms metadata checked; only the requested17 images were opened. No full102-image or native-camera audit is claimed.',
        'model_execution': False, 'renderer_execution': False}


def read_window(capture_root, arm, *, expected_manifest_sha256):
    """Return only initial RGB, future RGB targets and16 recorded commands."""
    return _read(capture_root, arm, expected_manifest_sha256)[0]


def verify_window(capture_root, arm, *, expected_manifest_sha256):
    """Separate provenance receipt for logging; never a model-conditioning object."""
    window, receipt = _read(capture_root, arm, expected_manifest_sha256)
    receipt['arrays'] = {field.name: {'shape': list(getattr(window, field.name).shape), 'dtype': 'float32',
        'sha256': hashlib.sha256(memoryview(getattr(window, field.name)).cast('B')).hexdigest()}
        for field in fields(window)}
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--expected-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh receipt path required')
    receipt = verify_window(args.capture_root, args.arm, expected_manifest_sha256=args.expected_manifest_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': receipt['status'], 'image_files_read': 17, 'arm': args.arm}, indent=2))
