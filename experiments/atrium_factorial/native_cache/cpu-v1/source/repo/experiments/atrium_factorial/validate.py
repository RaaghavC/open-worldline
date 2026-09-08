"""Independent saved capture validation. No Blender, Torch or model execution."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

CAPTURE_SHA = '725b739fd801c497ef8024f126cbf79a27e7dbbd38fd0311e57d9932fd9bc4d8'
SCENE_SHA = 'cf794ac98a01209be8ea70b3849f5f87174f0750b2990159cb052002836f7a56'
MOTIONS = ('stationary', 'left', 'right')
DOORS = ('closed', 'interact')
ARMS = tuple(m + '_' + d for m in MOTIONS for d in DOORS)
CHANNELS = ['local_right_m', 'local_up_m', 'local_forward_m', 'yaw_left_rad', 'pitch_up_rad', 'interact_pulse']
WIDTH, HEIGHT, COUNT = 1248, 704, 17
STEP = math.pi / 120
AXIS_ATOL = 2e-5  # Native Blender camera matrices are stored from FP32 values.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def regular(path, maximum):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Require a regular file: ' + path.name)
    require(0 < path.stat().st_size <= maximum, 'File size outside bound: ' + path.name)
    return path


def finite_number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), label + ': finite numeric value required')
    return value


def array(value, shape, label):
    def leaves(x):
        if isinstance(x, list):
            return all(leaves(v) for v in x)
        return type(x) in (int, float) and math.isfinite(x)
    require(isinstance(value, list) and leaves(value), label + ': malformed numeric array')
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), label + ': shape or finite check failed')
    return result


def close(actual, expected, label, atol=1e-12):
    require(abs(finite_number(actual, label) - expected) <= atol, label + ': differs from independently expected value')


def camera(record, yaw, label):
    transform = array(record['world_from_camera'], (4, 4), label + '/camera')
    rotation = transform[:3, :3]
    require(np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7, rtol=0), label + ': invalid homogeneous row')
    require(np.allclose(transform[:3, 3], [-.75, -4.4, 1.6], atol=AXIS_ATOL, rtol=0), label + ': camera translation changed')
    require(np.allclose(rotation.T @ rotation, np.eye(3), atol=AXIS_ATOL, rtol=0), label + ': camera axes not orthonormal')
    require(abs(np.linalg.det(rotation) - 1) <= AXIS_ATOL, label + ': camera axes not right handed')
    # Closed-form OpenCV camera axes for fixed downward pitch atan(.07).
    # Derived independently of mathutils.to_track_quat and the old validator.
    c, s = math.cos(yaw), math.sin(yaw)
    cp, sp = 1 / math.sqrt(1 + .07**2), .07 / math.sqrt(1 + .07**2)
    expected = np.array([[c, s*sp, -s*cp], [s, -c*sp, c*cp], [0, -cp, -sp]])
    error = float(np.abs(rotation - expected).max())
    require(error <= AXIS_ATOL, label + ': camera axes disagree with command yaw and fixed pitch')
    recovered_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    require(abs(recovered_yaw - yaw) <= AXIS_ATOL, label + ': recovered camera yaw disagrees')
    return {'maximum_axis_error': error, 'recovered_yaw_degrees': math.degrees(recovered_yaw)}


def validate_metadata(manifest):
    require(type(manifest) is dict and manifest.get('schema') == 'worldline-atrium-factorial-v1', 'Capture schema differs')
    require(manifest.get('status') == 'complete', 'Capture is not complete')
    exact = {
        'split': 'development', 'scene_family': 'single_atrium_layout_v1', 'scene_seed': 51000,
        'independent_layouts': 1, 'dataset_license': 'CC0-1.0', 'resolution': [WIDTH, HEIGHT],
        'render_engine': 'Cycles', 'device': 'METAL', 'frames_per_arm': COUNT,
        'scene_source_sha256': SCENE_SHA, 'capture_source_sha256': CAPTURE_SHA,
        'action_channels': CHANNELS, 'neural_model_execution': False,
        'model_inputs': ['initial_rgb', 'requested_command_deltas'],
        'training_targets': ['future_rgb'],
        'extra_metadata_not_model_inputs': ['door_open', 'yaw_radians', 'world_from_camera'],
        'camera_convention': 'world_from_camera uses OpenCV x right, y down, z forward; world z up; meters',
        'command_alignment': 'record t contains the command applied from observation t-1 to t; frame0 has no incoming command',
        'interaction_semantics': 'Programmed instantaneous remote door toggle to102degrees; no reach/contact/collision simulation',
        'repeated_states': 'Identical camera and door states reuse the first saved PNG for that state; every reuse is identified',
    }
    for key, expected in exact.items():
        require(type(manifest.get(key)) is type(expected) and manifest[key] == expected, 'Capture metadata differs: ' + key)
    require(type(manifest.get('samples')) is int and manifest['samples'] > 0, 'Positive integer render samples required')
    require(type(manifest.get('blender')) is str and manifest['blender'], 'Blender version missing')
    require(finite_number(manifest.get('elapsed_seconds'), 'capture elapsed seconds') >= 0, 'Negative elapsed seconds')
    transform = manifest.get('rgb_transform')
    require(type(transform) is dict and set(transform) == {'view', 'look', 'exposure', 'gamma', 'display'}, 'RGB transform fields differ')
    require(transform['view'] == 'AgX' and all(type(transform[k]) is str for k in ('view', 'look', 'display')), 'RGB transform strings differ')
    finite_number(transform['exposure'], 'RGB exposure')
    require(finite_number(transform['gamma'], 'RGB gamma') > 0, 'RGB gamma must be positive')
    close(manifest.get('yaw_step_radians'), STEP, 'yaw step')
    require(np.array_equal(array(manifest['K'], (3, 3), 'K'), [[624., 0., 624.], [0., 624., 352.], [0., 0., 1.]]), 'Intrinsic matrix differs')
    require(type(manifest.get('arms')) is dict and set(manifest['arms']) == set(ARMS), 'Require exactly six factorial arms')
    require(type(manifest.get('blend_sha256')) is str and len(manifest['blend_sha256']) == 64 and all(c in '0123456789abcdef' for c in manifest['blend_sha256']), 'Blend hash malformed')
    rows, first_state, endpoints = {}, {}, {}
    max_axis_error, rendered, reused = 0., 0, 0
    for motion in MOTIONS:
        direction = {'stationary': 0, 'left': 1, 'right': -1}[motion]
        for door in DOORS:
            arm = motion + '_' + door
            records = manifest['arms'][arm]
            require(type(records) is list and len(records) == COUNT, arm + ': require17 records')
            command_sum = 0.
            for frame, record in enumerate(records):
                label = arm + '/' + f'{frame:04d}'
                require(type(record) is dict and type(record.get('frame')) is int and record['frame'] == frame, label + ': frame ordering differs')
                expected_yaw = direction * frame * STEP
                close(record.get('yaw_radians'), expected_yaw, label + '/yaw')
                expected_open = door == 'interact' and frame > 0
                require(type(record.get('door_open')) is bool and record['door_open'] == expected_open, label + ': door state differs')
                if frame == 0:
                    require(record.get('command_from_previous') is None, label + ': frame0 has no incoming command')
                else:
                    expected_command = np.array([0., 0., 0., direction*STEP, 0., float(door == 'interact' and frame == 1)])
                    actual_command = array(record.get('command_from_previous'), (6,), label + '/command')
                    require(np.allclose(actual_command, expected_command, atol=1e-12, rtol=0), label + ': command label differs')
                    command_sum += float(actual_command[3])
                close(command_sum, expected_yaw, label + '/accumulated yaw')
                matrix = camera(record, expected_yaw, label)
                max_axis_error = max(max_axis_error, matrix['maximum_axis_error'])
                if frame == COUNT-1:
                    endpoints[arm] = {'expected_yaw_degrees': direction*24., 'recorded_yaw_degrees': math.degrees(record['yaw_radians']), **matrix}
                name = arm + '/' + f'{frame:04d}.png'
                require(record.get('png') == name, label + ': image path differs')
                digest = record.get('png_sha256')
                require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), label + ': image hash malformed')
                key = (direction*frame, expected_open)
                expected_reuse = first_state.get(key)
                require('reused_from' in record and record['reused_from'] == expected_reuse, label + ': render reuse does not match first identical state')
                if expected_reuse is None:
                    first_state[key] = name
                    rendered += 1
                else:
                    require(digest == rows[expected_reuse]['png_sha256'], label + ': reused image hash differs')
                    reused += 1
                require(finite_number(record.get('render_or_copy_seconds'), label + '/seconds') >= 0, label + ': negative duration')
                rows[name] = record
    require(len(rows) == 102 and rendered == 66 and reused == 36, 'Expected102 images,66 unique renders and36 copies')
    require(type(manifest.get('rendered_unique_states')) is int and manifest['rendered_unique_states'] == rendered, 'Rendered counter differs')
    require(type(manifest.get('copied_identical_states')) is int and manifest['copied_identical_states'] == reused, 'Copied counter differs')
    return {'rows': rows, 'endpoints': endpoints, 'maximum_camera_axis_error': max_axis_error,
            'rendered_unique_states': rendered, 'copied_identical_states': reused,
            'command_vectors': 96, 'frame0_without_command': 6}


def read_rgb(path, digest):
    path = regular(path, 16*1024*1024)
    content = path.read_bytes()
    require(hashlib.sha256(content).hexdigest() == digest, 'PNG file hash differs: ' + path.name)
    with Image.open(io.BytesIO(content)) as image:
        require(image.format == 'PNG' and image.size == (WIDTH, HEIGHT), 'PNG dimensions/format differ: ' + path.name)
        require(image.mode in ('RGB', 'RGBA') and getattr(image, 'n_frames', 1) == 1, 'Require a single8-bit RGB/opaque RGBA PNG')
        image.load()
        pixels = np.array(image, dtype=np.uint8)
    if pixels.shape[2] == 4:
        require(bool((pixels[:, :, 3] == 255).all()), 'PNG alpha is not opaque')
        rgb = np.ascontiguousarray(pixels[:, :, :3])
    else:
        rgb = pixels
    require(rgb.shape == (HEIGHT, WIDTH, 3), 'RGB shape differs')
    return rgb, {'bytes': len(content), 'sha256': digest, 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(), 'mode': 'RGBA' if pixels.shape[2] == 4 else 'RGB'}


def difference(left, right):
    diff = left.astype(np.int16) - right.astype(np.int16)
    diff64 = diff.astype(np.float64) / 255.
    return {'mae_0_1': float(np.abs(diff64).mean()), 'rmse_0_1': float(np.sqrt(np.square(diff64).mean())),
            'maximum_channel_difference_0_1': float(np.abs(diff64).max()),
            'changed_pixels': int(np.any(diff != 0, axis=2).sum()), 'pixels': HEIGHT*WIDTH,
            'meaning': 'Descriptive captured-image difference; not a neural quality or causal localization score'}


def audit_capture(capture_root, capture_source, scene_source, expected_manifest_sha256=None):
    started = time.perf_counter()
    root = Path(capture_root).absolute()
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), 'Regular capture directory required')
    source = regular(capture_source, 1024*1024)
    scene = regular(scene_source, 1024*1024)
    require(sha(source) == CAPTURE_SHA and sha(scene) == SCENE_SHA, 'Reviewed producer source bytes differ')
    manifest_path = regular(root/'manifest.json', 2*1024*1024)
    raw = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(raw).hexdigest()
    if expected_manifest_sha256 is not None:
        require(manifest_hash == expected_manifest_sha256, 'Expected manifest hash differs')
    manifest = json.loads(raw)
    metadata = validate_metadata(manifest)
    rows = metadata.pop('rows')
    actual_pngs = {p.relative_to(root).as_posix() for p in root.rglob('*.png')}
    require(actual_pngs == set(rows), 'Actual PNG inventory differs from102 declared files')
    blend = regular(root/'original-scene.blend', 2*1024**3)
    require(sha(blend) == manifest['blend_sha256'], 'Saved Blender file hash differs')
    image_records, endpoints = {}, {}
    first_hash, first_rgb_hash = None, None
    for name, record in rows.items():
        rgb, identity = read_rgb(root/name, record['png_sha256'])
        image_records[name] = identity
        if record['frame'] == 0:
            if first_hash is None:
                first_hash, first_rgb_hash = identity['sha256'], identity['rgb_sha256']
            require(identity['sha256'] == first_hash and identity['rgb_sha256'] == first_rgb_hash, 'All six initial images must be byte exact')
        if record['frame'] == COUNT-1:
            endpoints[name.split('/')[0]] = rgb
    comparisons = {}
    for door in DOORS:
        for motion in ('left', 'right'):
            a, b = motion+'_'+door, 'stationary_'+door
            comparisons[a+'_vs_'+b] = difference(endpoints[a], endpoints[b])
    for motion in MOTIONS:
        a, b = motion+'_interact', motion+'_closed'
        comparisons[a+'_vs_'+b] = difference(endpoints[a], endpoints[b])
    require(sha(manifest_path) == manifest_hash and sha(source) == CAPTURE_SHA and sha(scene) == SCENE_SHA, 'Manifest or producer source changed during audit')
    return {'schema': 'worldline-atrium-factorial-independent-v1', 'status': 'passed',
            'manifest_sha256': manifest_hash, 'capture_source_sha256': CAPTURE_SHA, 'scene_source_sha256': SCENE_SHA,
            'validator_source_sha256': sha(__file__), 'images': image_records, 'image_count': 102,
            'first_image_file_sha256': first_hash, 'first_image_rgb_sha256': first_rgb_hash,
            'all_initial_images_byte_exact': True, 'metadata': metadata,
            'original_scene': {'bytes': blend.stat().st_size, 'sha256': manifest['blend_sha256'], 'loaded_or_deserialized': False},
            'end_frame_differences': comparisons, 'source_and_manifest_unchanged': True,
            'scope': 'One previously seen procedural room, development capture. Programmed remote toggle and camera targets; no generalization or learned action-control claim.',
            'limitations': ['Matrices and K are checked against declared source equations, not a new native Blender ray calibration.',
                           'Door state is a checked producer label; this audit does not infer the visible door angle from pixels.',
                           'PNG differences include lighting, view and object effects; they do not isolate a spatial door mask.',
                           'No Blender, Torch, renderer or neural model executed. Only saved files were read.'],
            'blender_imported': 'bpy' in sys.modules, 'torch_imported': 'torch' in sys.modules,
            'seconds': time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('capture-root', 'capture-source', 'scene-source', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--expected-manifest-sha256')
    args = parser.parse_args()
    require(not args.output.exists() and not any(p.is_symlink() for p in (args.output.absolute(), *args.output.absolute().parents)), 'Fresh regular output required')
    try:
        report = audit_capture(args.capture_root, args.capture_source, args.scene_source, args.expected_manifest_sha256)
    except BaseException as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'schema': 'worldline-atrium-factorial-independent-v1', 'status': 'failed',
            'error_type': type(error).__name__, 'error': str(error), 'validator_source_sha256': sha(__file__),
            'model_execution': False, 'renderer_execution': False}, indent=2)+'\n')
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report[k] for k in ('status', 'image_count', 'manifest_sha256', 'all_initial_images_byte_exact', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()
