"""Five bounded offline cases; synthetic URLs never make an HTTP request."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('public_fetcher', HERE / 'fetch_artifacts.py')
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


def fixture(root, malformed=None):
    assets = root / 'assets'; assets.mkdir()
    values = {'spatial-results/frames/0000.safetensors': bytes(range(256)) * 4097,
              'spatial-results/worker.log': b''}
    if malformed == 'traversal':
        values['../escape'] = values.pop('spatial-results/worker.log')
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name, value in values.items():
            member = tarfile.TarInfo(name); member.size = len(value)
            if malformed == 'link' and name.endswith('worker.log'):
                member.type = tarfile.SYMTYPE; member.linkname = '/tmp/escape'
            archive.addfile(member, io.BytesIO(value))
            if malformed == 'duplicate' and name.endswith('worker.log'):
                archive.addfile(member, io.BytesIO(value))
    data = stream.getvalue(); parts = []
    for i, start in enumerate(range(0, len(data), 211)):
        part = data[start:start + 211]; name = f'evidence.tar.gz.part-{i:03d}'
        (assets / name).write_bytes(part)
        parts.append({'name': name, 'bytes': len(part), 'sha256': hashlib.sha256(part).hexdigest()})
    files = {name: {'bytes': len(value), 'sha256': hashlib.sha256(value).hexdigest()} for name, value in values.items()}
    index = {'schema': 'worldline-action-recovery-v1', 'label': 'synthetic', 'files': files, 'parts': parts,
             'file_bytes': sum(row['bytes'] for row in files.values()), 'transport_bytes': len(data),
             'stream_sha256': hashlib.sha256(data).hexdigest()}
    index_bytes = (json.dumps(index) + '\n').encode(); (assets / 'index.json').write_bytes(index_bytes)
    url = 'https://github.com/RaaghavC/open-worldline/releases/download/offline-fixture/'
    metadata = {'schema': fetcher.SCHEMA,
                'index': {'name': 'index.json', 'bytes': len(index_bytes),
                          'sha256': hashlib.sha256(index_bytes).hexdigest(), 'url': url + 'index.json'},
                'parts': [dict(row, url=url + row['name']) for row in parts],
                'stream_sha256': index['stream_sha256']}
    path = root / 'metadata.json'; path.write_text(json.dumps(metadata))
    return path, assets, values, metadata


class FetchTests(unittest.TestCase):
    def test_exact_multipart_bytes_empty_log_and_nonfresh_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); metadata, assets, values, _ = fixture(root)
            report = fetcher.recover(metadata, root / 'out', assets)
            self.assertEqual(report['status'], 'verified')
            self.assertEqual(report['files'], 2)
            for name, value in values.items():
                self.assertEqual((root / 'out/recovered' / name).read_bytes(), value)
            before = (root / 'out/verification.json').read_bytes()
            with self.assertRaises(ValueError):
                fetcher.recover(metadata, root / 'out', assets)
            self.assertEqual((root / 'out/verification.json').read_bytes(), before)

    def test_corrupt_or_truncated_asset_rejected(self):
        for mutation in ('flip', 'truncate'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); metadata, assets, _, _ = fixture(root)
                path = assets / 'evidence.tar.gz.part-000'; value = path.read_bytes()
                path.write_bytes(bytes([value[0] ^ 1]) + value[1:] if mutation == 'flip' else value[:-1])
                with self.assertRaises(ValueError):
                    fetcher.recover(metadata, root / 'out', assets)
                self.assertFalse((root / 'out/verification.json').exists())

    def test_traversal_links_and_duplicate_members_rejected(self):
        for malformed in ('traversal', 'link', 'duplicate'):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); metadata, assets, _, _ = fixture(root, malformed)
                with self.assertRaises(ValueError):
                    fetcher.recover(metadata, root / 'out', assets)
                self.assertFalse((root / 'escape').exists())
                self.assertFalse((root / 'out/verification.json').exists())

    def test_pin_order_url_and_index_mismatch_rejected(self):
        for mutation in ('order', 'url', 'size-bool', 'index'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                root = Path(temp); path, assets, _, value = fixture(root)
                if mutation == 'order':
                    value['parts'][0], value['parts'][1] = value['parts'][1], value['parts'][0]
                elif mutation == 'url':
                    value['parts'][0]['url'] = 'https://example.com/' + value['parts'][0]['name']
                elif mutation == 'size-bool':
                    value['parts'][0]['bytes'] = True
                else:
                    value['stream_sha256'] = '0' * 64
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    fetcher.recover(path, root / 'out', assets)
                self.assertFalse((root / 'out/verification.json').exists())

    def test_simulated_public_url_download_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path, assets, _, metadata = fixture(root); calls = []
            replies = {row['url']: (assets / row['name']).read_bytes() for row in [metadata['index'], *metadata['parts']]}

            def offline_url(url):
                calls.append(url)
                return io.BytesIO(replies[url])

            with patch.object(fetcher.urllib.request, 'build_opener', side_effect=AssertionError('Network forbidden')):
                report = fetcher.recover(path, root / 'out', opener=offline_url)
            self.assertEqual(report['status'], 'verified')
            self.assertEqual(report['transport'], 'public-https')
            self.assertEqual(calls, [row['url'] for row in [metadata['index'], *metadata['parts']]])


if __name__ == '__main__':
    unittest.main()
