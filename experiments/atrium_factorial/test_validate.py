"""Small synthetic metadata and PNG fixtures; no producer imports or models."""
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from experiments.atrium_factorial import validate as v

REPO = Path(__file__).resolve().parents[2]


def metadata():
    result = {
        'schema': 'worldline-atrium-factorial-v1', 'status': 'complete', 'split': 'development',
        'scene_family': 'single_atrium_layout_v1', 'scene_seed': 51000, 'independent_layouts': 1,
        'dataset_license': 'CC0-1.0', 'resolution': [1248, 704], 'render_engine': 'Cycles',
        'device': 'METAL', 'samples': 32, 'frames_per_arm': 17, 'blender': 'synthetic-fixture',
        'scene_source_sha256': v.SCENE_SHA, 'capture_source_sha256': v.CAPTURE_SHA,
        'action_channels': list(v.CHANNELS), 'neural_model_execution': False,
        'model_inputs': ['initial_rgb', 'requested_command_deltas'], 'training_targets': ['future_rgb'],
        'extra_metadata_not_model_inputs': ['door_open', 'yaw_radians', 'world_from_camera'],
        'camera_convention': 'world_from_camera uses OpenCV x right, y down, z forward; world z up; meters',
        'command_alignment': 'record t contains the command applied from observation t-1 to t; frame0 has no incoming command',
        'interaction_semantics': 'Programmed instantaneous remote door toggle to102degrees; no reach/contact/collision simulation',
        'repeated_states': 'Identical camera and door states reuse the first saved PNG for that state; every reuse is identified',
        'rgb_transform': {'view': 'AgX', 'look': 'fixture', 'exposure': 0., 'gamma': 1., 'display': 'sRGB'},
        'yaw_step_radians': math.radians(1.5), 'elapsed_seconds': 1.,
        'K': [[624., 0., 624.], [0., 624., 352.], [0., 0., 1.]],
        'blend_sha256': hashlib.sha256(b'synthetic blend identity; never deserialized').hexdigest(),
        'rendered_unique_states': 66, 'copied_identical_states': 36, 'arms': {},
    }
    initial_forward = np.array([0., 1., -.07])
    initial_forward /= np.linalg.norm(initial_forward)
    initial_right = np.array([1., 0., 0.])
    initial_down = np.cross(initial_forward, initial_right)
    initial = np.column_stack((initial_right, initial_down, initial_forward))
    first = {}
    for motion, direction in (('stationary', 0), ('left', 1), ('right', -1)):
        for door in ('closed', 'interact'):
            arm = motion+'_'+door
            result['arms'][arm] = []
            yaw = 0.
            for frame in range(17):
                command = None if frame == 0 else [0., 0., 0., direction*math.radians(1.5), 0., float(door == 'interact' and frame == 1)]
                if command is not None:
                    yaw += command[3]
                rotate_world = np.array([[math.cos(yaw), -math.sin(yaw), 0.], [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])
                pose = np.eye(4)
                pose[:3, :3] = rotate_world @ initial
                pose[:3, 3] = [-.75, -4.4, 1.6]
                opened = door == 'interact' and frame > 0
                state = (direction*frame, opened)
                name = f'{arm}/{frame:04d}.png'
                reuse = first.get(state)
                if reuse is None:
                    first[state] = name
                result['arms'][arm].append({'frame': frame, 'command_from_previous': command,
                    'door_open': opened, 'yaw_radians': yaw, 'world_from_camera': pose.astype(np.float32).tolist(),
                    'png': name, 'png_sha256': hashlib.sha256(repr(state).encode()).hexdigest(),
                    'reused_from': reuse, 'render_or_copy_seconds': 0.001})
    return result


def png_bytes(color, size=(1248, 704), mode='RGB'):
    stream = io.BytesIO()
    Image.new(mode, size, color).save(stream, format='PNG')
    return stream.getvalue()


def capture_fixture(root):
    manifest = metadata()
    state_data = {}
    for records in manifest['arms'].values():
        for row in records:
            key = (round(math.degrees(row['yaw_radians'])/1.5), row['door_open'])
            if key not in state_data:
                state_data[key] = png_bytes((30+key[0]+16, 110 if key[1] else 50, 70))
            data = state_data[key]
            path = root/row['png']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            row['png_sha256'] = hashlib.sha256(data).hexdigest()
    (root/'original-scene.blend').write_bytes(b'synthetic blend identity; never deserialized')
    (root/'manifest.json').write_text(json.dumps(manifest)+'\n')
    return manifest


class Checks(unittest.TestCase):
    def test_complete_factorial_metadata_and_cumulative_yaw(self):
        result = v.validate_metadata(metadata())
        self.assertEqual(result['command_vectors'], 96)
        self.assertEqual(result['rendered_unique_states'], 66)
        self.assertEqual(result['copied_identical_states'], 36)
        self.assertAlmostEqual(result['endpoints']['left_closed']['recovered_yaw_degrees'], 24., places=5)
        self.assertAlmostEqual(result['endpoints']['right_interact']['recovered_yaw_degrees'], -24., places=5)

    def test_changed_labels_sources_and_counts_reject(self):
        def changed_frame0(m): m['arms']['stationary_closed'][0]['command_from_previous'] = [0.]*6
        def changed_camera(m): m['arms']['left_closed'][1]['command_from_previous'][3] *= -1
        def late_interaction(m): m['arms']['left_interact'][2]['command_from_previous'][5] = 1.
        def wrong_door(m): m['arms']['right_interact'][0]['door_open'] = True
        def missing_frame(m): m['arms']['stationary_interact'].pop()
        def bad_source(m): m['capture_source_sha256'] = 'a'*64
        def source_status(m): m['status'] = 'running'
        def bool_number(m): m['arms']['stationary_closed'][1]['command_from_previous'][0] = False
        for mutation in (changed_frame0, changed_camera, late_interaction, wrong_door, missing_frame, bad_source, source_status, bool_number):
            with self.subTest(mutation=mutation.__name__):
                data = metadata(); mutation(data)
                with self.assertRaises(ValueError): v.validate_metadata(data)

    def test_camera_translation_reflection_yaw_and_nonfinite_reject(self):
        for field, delta in (('translation', .01), ('reflection', -1.), ('yaw', .04), ('nan', float('nan'))):
            with self.subTest(field=field):
                data = metadata()
                pose = data['arms']['left_closed'][3]['world_from_camera']
                if field == 'translation': pose[0][3] += delta
                elif field == 'reflection':
                    for row in pose[:3]: row[0] *= delta
                elif field == 'yaw': pose[0][0] += delta
                else: pose[2][2] = delta
                with self.assertRaises(ValueError): v.validate_metadata(data)

    def test_render_reuse_requires_first_identical_state_and_bytes(self):
        mutations = [('stationary_closed', 1, 'reused_from', 'left_closed/0016.png'),
                     ('stationary_interact', 1, 'reused_from', 'stationary_closed/0000.png'),
                     ('left_closed', 0, 'png_sha256', 'f'*64)]
        for arm, frame, key, value in mutations:
            with self.subTest(key=key, arm=arm):
                data = metadata(); data['arms'][arm][frame][key] = value
                with self.assertRaises(ValueError): v.validate_metadata(data)
        data = metadata(); data['rendered_unique_states'] = 65
        with self.assertRaises(ValueError): v.validate_metadata(data)

    def test_all102_actual_png_reads_and_descriptive_endpoint_math(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            capture_fixture(root)
            report = v.audit_capture(root,
                REPO/'experiments/atrium_factorial/render.py', REPO/'experiments/atrium_data/render.py',
                v.sha(root/'manifest.json'))
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(len(report['images']), 102)
            self.assertTrue(report['all_initial_images_byte_exact'])
            self.assertFalse(report['torch_imported'])
            self.assertFalse(report['blender_imported'])
            measured = report['end_frame_differences']['left_closed_vs_stationary_closed']
            self.assertAlmostEqual(measured['mae_0_1'], 16/(3*255), places=12)
            self.assertAlmostEqual(measured['rmse_0_1'], 16/(255*math.sqrt(3)), places=12)
            self.assertEqual(measured['changed_pixels'], 1248*704)
            with self.assertRaises(ValueError):
                v.audit_capture(root, REPO/'experiments/atrium_factorial/render.py', REPO/'experiments/atrium_data/render.py', '0'*64)

    def test_png_hash_dimensions_alpha_and_rgb_bytes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory)/'image.png'
            content = png_bytes((12, 23, 34, 255), mode='RGBA')
            path.write_bytes(content)
            rgb, record = v.read_rgb(path, hashlib.sha256(content).hexdigest())
            self.assertTrue(np.all(rgb == [12, 23, 34]))
            self.assertEqual(record['rgb_sha256'], hashlib.sha256(bytes([12, 23, 34])*(1248*704)).hexdigest())
            with self.assertRaises(ValueError): v.read_rgb(path, '0'*64)
            for content in (png_bytes((0, 0, 0), size=(1247, 704)), png_bytes((1, 2, 3, 254), mode='RGBA')):
                path.write_bytes(content)
                with self.assertRaises(ValueError): v.read_rgb(path, hashlib.sha256(content).hexdigest())


if __name__ == '__main__':
    unittest.main()
