"""Synthetic protocol, pixel and data-boundary checks with no real capture dependency."""
import copy
from dataclasses import fields
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from experiments.atrium_factorial import reader as r


def png(color, *, size=(1248, 704), mode='RGB'):
    stream = io.BytesIO()
    Image.new(mode, size, color).save(stream, format='PNG')
    return stream.getvalue()


def write_manifest(root, manifest):
    raw = (json.dumps(manifest, indent=2)+'\n').encode()
    (root/'manifest.json').write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fixture(root):
    manifest = {
        'schema': 'worldline-atrium-factorial-v1', 'status': 'complete', 'split': 'development',
        'scene_family': 'single_atrium_layout_v1', 'scene_seed': 51000, 'independent_layouts': 1,
        'resolution': [1248, 704], 'frames_per_arm': 17,
        'capture_source_sha256': r.CAPTURE_SHA, 'scene_source_sha256': r.SCENE_SHA,
        'neural_model_execution': False, 'action_channels': list(r.CHANNELS),
        'command_alignment': 'record t contains the command applied from observation t-1 to t; frame0 has no incoming command',
        'interaction_semantics': 'Programmed instantaneous remote door toggle to102degrees; no reach/contact/collision simulation',
        'model_inputs': ['initial_rgb', 'requested_command_deltas'], 'training_targets': ['future_rgb'],
        'extra_metadata_not_model_inputs': ['door_open', 'yaw_radians', 'world_from_camera'],
        'rendered_unique_states': 66, 'copied_identical_states': 36,
        'yaw_step_radians': math.radians(1.5), 'arms': {},
    }
    state_files = {}
    for motion, sign in (('stationary', 0), ('left', 1), ('right', -1)):
        for door in ('closed', 'interact'):
            name = motion+'_'+door
            (root/name).mkdir()
            manifest['arms'][name] = []
            for frame in range(17):
                opened = door == 'interact' and frame > 0
                state = (sign*frame, opened)
                path = f'{name}/{frame:04d}.png'
                first = state_files.get(state)
                if first is None:
                    data = png((128+sign*frame, 255 if opened else 0, 64))
                    state_files[state] = (path, data)
                else:
                    data = first[1]
                (root/path).write_bytes(data)
                command = None if frame == 0 else [0., 0., 0., sign*math.radians(1.5), 0., float(door == 'interact' and frame == 1)]
                manifest['arms'][name].append({'frame': frame, 'png': path,
                    'png_sha256': hashlib.sha256(data).hexdigest(), 'reused_from': None if first is None else first[0],
                    'command_from_previous': command, 'yaw_radians': sign*math.radians(frame*1.5),
                    'door_open': opened,
                    'world_from_camera': 'synthetic non-model metadata; reader does not use camera pose'})
    return manifest, write_manifest(root, manifest)


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name).resolve()
        cls.manifest, cls.digest = fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def tearDown(self):
        write_manifest(self.root, self.manifest)

    def test_all_six_arm_shapes_units_endpoint_values_and_boundary(self):
        for arm in r.ARMS:
            with self.subTest(arm=arm):
                result = r.read_window(self.root, arm, expected_manifest_sha256=self.digest)
                self.assertEqual([f.name for f in fields(result)], ['initial_rgb', 'future_rgb_targets', 'commands'])
                self.assertEqual(result.initial_rgb.shape, (3, 704, 1248))
                self.assertEqual(result.future_rgb_targets.shape, (16, 3, 704, 1248))
                self.assertEqual(result.commands.shape, (16, 6))
                sign = 1 if arm.startswith('left') else -1 if arm.startswith('right') else 0
                expected = np.array(self.manifest['arms'][arm][1]['command_from_previous'], dtype=np.float32)
                self.assertTrue(np.array_equal(result.commands[0], expected))
                self.assertAlmostEqual(float(result.commands[:, 3].astype(np.float64).sum()), sign*math.radians(24), places=7)
                self.assertEqual(int(result.commands[:, 5].sum()), int(arm.endswith('interact')))
                self.assertEqual(float(result.initial_rgb[1, 0, 0]), -1.)
                self.assertEqual(float(result.future_rgb_targets[-1, 1, 0, 0]), 1. if arm.endswith('interact') else -1.)
                self.assertEqual(float(result.future_rgb_targets[-1, 0, 0, 0]), float(np.float32(128+sign*16)/np.float32(127.5)-np.float32(1.)))
                for value in (result.initial_rgb, result.future_rgb_targets, result.commands):
                    self.assertEqual(value.dtype, np.float32)
                    self.assertTrue(value.flags.owndata and value.flags.c_contiguous)
                del result

    def test_future_change_preserves_initial_and_commands_with_new_manifest_pin(self):
        result = r.read_window(self.root, 'left_closed', expected_manifest_sha256=self.digest)
        initial, commands = result.initial_rgb.copy(), result.commands.copy()
        old_target = result.future_rgb_targets[7].copy()
        result.future_rgb_targets[0].fill(0.75)
        self.assertTrue(np.array_equal(result.initial_rgb, initial))
        self.assertFalse(np.shares_memory(result.initial_rgb, result.future_rgb_targets))
        del result
        path = self.root/'left_closed/0008.png'
        original = path.read_bytes()
        data = png((240, 1, 2))
        changed = copy.deepcopy(self.manifest)
        changed['arms']['left_closed'][8]['png_sha256'] = hashlib.sha256(data).hexdigest()
        try:
            path.write_bytes(data)
            digest = write_manifest(self.root, changed)
            later = r.read_window(self.root, 'left_closed', expected_manifest_sha256=digest)
            self.assertTrue(np.array_equal(later.initial_rgb, initial))
            self.assertTrue(np.array_equal(later.commands, commands))
            self.assertFalse(np.array_equal(later.future_rgb_targets[7], old_target))
        finally:
            path.write_bytes(original)

    def test_exactly17_selected_images_read_and_unconsumed_arm_is_not_opened(self):
        with mock.patch.object(r, 'read_rgb', wraps=r.read_rgb) as read:
            result = r.read_window(self.root, 'right_interact', expected_manifest_sha256=self.digest)
            paths = [call.args[0].relative_to(self.root).as_posix() for call in read.call_args_list]
            self.assertEqual(paths, [f'right_interact/{i:04d}.png' for i in range(17)])
            del result

    def test_manifest_pin_complete_labels_paths_and_finite_values_fail_closed(self):
        with self.assertRaises(ValueError): r.read_window(self.root, 'left_closed', expected_manifest_sha256='0'*64)
        mutations = [('status', 'running'), ('frames_per_arm', 16), ('capture_source_sha256', '1'*64)]
        for key, value in mutations:
            with self.subTest(key=key):
                data = copy.deepcopy(self.manifest); data[key] = value
                digest = write_manifest(self.root, data)
                with self.assertRaises(ValueError): r.read_window(self.root, 'left_closed', expected_manifest_sha256=digest)
        for key, value in [('command_from_previous', [0.,0.,0.,-math.pi/120,0.,0.]),
                           ('command_from_previous', [0.,0.,0.,float('nan'),0.,0.]),
                           ('command_from_previous', [False,0.,0.,math.pi/120,0.,0.]),
                           ('png', '../other.png'), ('door_open', True), ('reused_from', 'right_closed/0001.png')]:
            with self.subTest(key=key, value=value):
                data = copy.deepcopy(self.manifest); data['arms']['left_closed'][1][key] = value
                digest = write_manifest(self.root, data)
                with self.assertRaises(ValueError): r.read_window(self.root, 'left_closed', expected_manifest_sha256=digest)
        data = copy.deepcopy(self.manifest); data['arms']['stationary_interact'].pop()
        digest = write_manifest(self.root, data)
        with self.assertRaises(ValueError): r.read_window(self.root, 'left_closed', expected_manifest_sha256=digest)

    def test_changed_image_hash_dimension_alpha_and_pixel_mapping(self):
        path = self.root/'left_closed/0001.png'
        original = path.read_bytes()
        try:
            for data in (png((0,0,0)), png((0,0,0), size=(1247,704)), png((0,0,0,254), mode='RGBA')):
                path.write_bytes(data)
                with self.assertRaises(ValueError): r.read_window(self.root, 'left_closed', expected_manifest_sha256=self.digest)
                changed = copy.deepcopy(self.manifest)
                changed['arms']['left_closed'][1]['png_sha256'] = hashlib.sha256(data).hexdigest()
                digest = write_manifest(self.root, changed)
                if data != png((0,0,0)):
                    with self.assertRaises(ValueError): r.read_window(self.root, 'left_closed', expected_manifest_sha256=digest)
                write_manifest(self.root, self.manifest)
            data = png((0,127,255,255), mode='RGBA');path.write_bytes(data)
            rgb, _ = r.read_rgb(path, hashlib.sha256(data).hexdigest())
            expected = np.array([0,127,255], dtype=np.float32)/np.float32(127.5)-np.float32(1.)
            self.assertTrue(np.array_equal(rgb[:, 0, 0], expected))
            self.assertEqual(rgb.min(), -1.)
            self.assertEqual(rgb.max(), 1.)
        finally:
            path.write_bytes(original)

    def test_duplicate_json_bounds_and_separate_receipt(self):
        original = (self.root/'manifest.json').read_bytes()
        duplicate = original[:-2]+b', "status": "complete"}\n'
        try:
            (self.root/'manifest.json').write_bytes(duplicate)
            with self.assertRaises(ValueError):
                r.read_window(self.root, 'left_closed', expected_manifest_sha256=hashlib.sha256(duplicate).hexdigest())
        finally:
            (self.root/'manifest.json').write_bytes(original)
        with mock.patch.object(r, 'MAX_MANIFEST_BYTES', 100):
            with self.assertRaises(ValueError): r.read_window(self.root, 'left_closed', expected_manifest_sha256=self.digest)
        with mock.patch.object(r, 'MAX_IMAGE_BYTES', 100):
            with self.assertRaises(ValueError): r.read_window(self.root, 'left_closed', expected_manifest_sha256=self.digest)
        receipt = r.verify_window(self.root, 'stationary_closed', expected_manifest_sha256=self.digest)
        self.assertEqual(receipt['image_files_read'], 17)
        self.assertEqual(set(receipt['arrays']), {'initial_rgb', 'future_rgb_targets', 'commands'})
        self.assertFalse(receipt['model_execution'])


if __name__ == '__main__':
    unittest.main()
