# SPDX-License-Identifier: Apache-2.0
"""Small inert metadata/storage corruption tests; no models or original weights."""
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import struct
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import warnings
import zipfile
from . import weights as w


class WeightTests(unittest.TestCase):
    def test_rebuild_requires_exact_contiguous_storage(self):
        storage = {'storage_key': '0', 'elements': 6}
        self.assertEqual(w._rebuilt(storage, 0, (2, 3), (3, 1), False, None)['shape'], [2, 3])
        for offset, shape, stride in ((1, (2, 3), (3, 1)), (0, (2, 3), (1, 2)),
                                     (0, (2, 2), (2, 1)), (0, (2, 3), (3, True)),
                                     (0, (0, 3), (3, 1))):
            with self.assertRaises(ValueError):
                w._rebuilt(storage, offset, shape, stride, False, None)

    def test_metadata_unpickler_rejects_unapproved_globals(self):
        raw = pickle.dumps(math.sqrt)
        with self.assertRaisesRegex(ValueError, 'Unapproved pickle global'):
            w.MetadataOnly(io.BytesIO(raw)).load()
        reader = w.MetadataOnly(io.BytesIO(b''))
        self.assertEqual(reader.persistent_load(('storage', 'FLOAT32_STORAGE_TAG', '12', 'cpu', 6)),
                         {'storage_key': '12', 'elements': 6})
        for value in [('storage', 'HALF_STORAGE_TAG', '12', 'cpu', 6),
                      ('storage', 'FLOAT32_STORAGE_TAG', '../12', 'cpu', 6),
                      ('storage', 'FLOAT32_STORAGE_TAG', '12', 'cpu', True),
                      ('storage', 'FLOAT32_STORAGE_TAG', '12', 'cpu', w.PARAMETERS + 1)]:
            with self.assertRaises(ValueError):
                reader.persistent_load(value)

    def test_metadata_archive_duplicates_and_trailing_data(self):
        def archive(contents):
            stream = io.BytesIO()
            with warnings.catch_warnings(), zipfile.ZipFile(stream, 'w') as output:
                warnings.simplefilter('ignore', UserWarning)
                for name, raw in contents:
                    output.writestr(name, raw)
            stream.seek(0)
            return zipfile.ZipFile(stream)
        data = pickle.dumps({})
        with archive([('archive/data.pkl', data)]) as source:
            self.assertEqual(w.read_metadata(source), ('archive/', {}))
        for contents in [[('archive/data.pkl', data), ('archive/data.pkl', data)],
                         [('archive/data.pkl', data + b'x')], [('unrelated', data)],
                         [('archive/data.pkl', b'x' * (2**20 + 1))]]:
            with archive(contents) as source, self.assertRaises(ValueError):
                w.read_metadata(source)

    def test_streamed_storage_match_and_corruption(self):
        raw = struct.pack('<6f', 1, 2, 3, 4, 5, 6)
        metadata = {'layer.weight': {'storage_key': '7', 'elements': 6, 'shape': [2, 3]}}
        rows = {'layer.weight': {'shape': [2, 3], 'sha256': hashlib.sha256(raw).hexdigest(), 'cuda_copy_exact': True}}
        for payload, passed in ((raw, True), (raw[:-1], False), (bytes([raw[0] ^ 1]) + raw[1:], False)):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, 'w') as archive:
                archive.writestr('archive/data/7', payload)
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive:
                if passed:
                    result, total = w.compare_storages(archive, 'archive/', metadata, rows)
                    self.assertEqual(result, {'layer.weight': hashlib.sha256(raw).hexdigest()})
                    self.assertEqual(total, 6)
                    with self.assertRaises(ValueError):
                        w.compare_storages(archive, 'archive/', metadata, {})
                else:
                    with self.assertRaises(ValueError):
                        w.compare_storages(archive, 'archive/', metadata, rows)

    def test_load_record_bounds_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); target = root / 'record.json'
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{}', ' ' * (2**20 + 1)):
                target.write_text(text)
                with self.assertRaises(ValueError):
                    w.record(target)
            target.write_text('{}'); link = root / 'link'; link.symlink_to(target)
            with self.assertRaises(ValueError):
                w.record(link)

    def test_record_mutation_during_parse_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); checkpoint = root / 'checkpoint'; checkpoint.write_bytes(b'cp')
            record = root / 'record'; record.write_bytes(b'before')
            def changed(path):
                path.write_bytes(b'after'); return {'tensors': {}}
            with patch.object(w, 'record', side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'changed while parsing'):
                    w.audit_vae_weights(checkpoint, [record])

    def test_checkpoint_mutation_after_initial_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); checkpoint = root / 'checkpoint'; checkpoint.write_bytes(b'safe')
            record = root / 'record'; record.write_bytes(b'record')
            def changed(*args):
                checkpoint.write_bytes(b'evil'); return {}, w.PARAMETERS
            with (patch.object(w, 'VAE_BYTES', 4), patch.object(w, 'VAE_SHA', w.sha(checkpoint)),
                  patch.object(w, 'record', return_value={'tensors': {}}),
                  patch.object(w.zipfile, 'ZipFile', return_value=MagicMock()),
                  patch.object(w, 'read_metadata', return_value=('archive/', {str(i): {} for i in range(196)})),
                  patch.object(w, 'compare_storages', side_effect=changed)):
                with self.assertRaisesRegex(ValueError, 'unchanged checkpoint'):
                    w.audit_vae_weights(checkpoint, [record])


if __name__ == '__main__':
    unittest.main()
