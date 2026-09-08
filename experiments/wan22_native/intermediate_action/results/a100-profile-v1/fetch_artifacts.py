# SPDX-License-Identifier: Apache-2.0
"""Download or verify pinned public fixed16 action-CUDA artifacts without a provider account."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import urllib.error
import urllib.parse
import urllib.request

SCHEMA = 'worldline-action-release-download-v1'
HEX = re.compile(r'[0-9a-f]{64}')
MAX_PART = 90_000_000
MAX_INDEX = 16_000_000
RELEASE_PREFIX = '/RaaghavC/open-worldline/releases/download/'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def positive_int(value, maximum):
    return type(value) is int and 0 < value <= maximum


def release_record(record, name, maximum):
    require(isinstance(record, dict) and set(record) == {'name', 'bytes', 'sha256', 'url'},
            'Expected one exact release asset record')
    require(record['name'] == name and positive_int(record['bytes'], maximum), 'Invalid asset name/size')
    require(isinstance(record['sha256'], str) and HEX.fullmatch(record['sha256']), 'Invalid asset SHA256')
    require(isinstance(record['url'], str), 'Expected a public release URL')
    url = urllib.parse.urlsplit(record['url'])
    require(url.scheme == 'https' and url.netloc == 'github.com' and not url.query and not url.fragment,
            'Expected an HTTPS public GitHub release URL without credentials or query parameters')
    require(url.path.startswith(RELEASE_PREFIX), 'Expected this repository release path')
    suffix = url.path[len(RELEASE_PREFIX):].split('/')
    require(len(suffix) == 2 and suffix[0] and urllib.parse.unquote(suffix[1]) == name,
            'Expected a release tag and matching asset filename')
    tag = urllib.parse.unquote(suffix[0])
    require(tag not in {'.', '..'} and '/' not in tag and '\\' not in tag, 'Invalid release tag')
    return suffix[0]


def validate_metadata(value):
    require(isinstance(value, dict) and set(value) == {'schema', 'index', 'parts', 'stream_sha256'},
            'Unexpected download metadata fields')
    require(value['schema'] == SCHEMA, 'Unknown download metadata schema')
    tag = release_record(value['index'], 'index.json', MAX_INDEX)
    require(isinstance(value['parts'], list) and 0 < len(value['parts']) <= 1000, 'Invalid part count')
    for i, part in enumerate(value['parts']):
        require(release_record(part, f'evidence.tar.gz.part-{i:03d}', MAX_PART) == tag,
                'Every asset must belong to the same exact release tag')
    require(isinstance(value['stream_sha256'], str) and HEX.fullmatch(value['stream_sha256']),
            'Invalid full-stream SHA256')


class HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        require(parsed.scheme == 'https' and parsed.username is None and parsed.password is None,
                'Release redirect must remain HTTPS without credentials')
        return super().redirect_request(request, response, code, message, headers, newurl)


def https_open(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Worldline-artifact-verifier'})
    return urllib.request.build_opener(HTTPSRedirect()).open(request, timeout=60)


def obtain(record, output, parts_directory, opener):
    target = output / record['name']
    try:
        if parts_directory is None:
            source = opener(record['url'])
        else:
            path = parts_directory / record['name']
            require(path.is_file() and not path.is_symlink(), 'Missing local regular asset: ' + record['name'])
            source = path.open('rb')
        with source, target.open('xb') as handle:
            total = 0; checksum = hashlib.sha256()
            while True:
                chunk = source.read(min(1 << 20, record['bytes'] + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                require(total <= record['bytes'], 'Asset exceeds declared length: ' + record['name'])
                checksum.update(chunk); handle.write(chunk)
        require(total == record['bytes'] and checksum.hexdigest() == record['sha256'],
                'Asset length or SHA256 differs: ' + record['name'])
    except (OSError, urllib.error.URLError):
        # Do not print a redirected, signed download URL in an exception.
        raise ValueError('Unable to obtain release asset: ' + record['name']) from None
    return target


class Parts(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.handle = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.handle is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.handle = path.open('rb')
            count = self.handle.readinto(buffer)
            if count:
                return count
            self.handle.close(); self.handle = None

    def close(self):
        if self.handle is not None:
            self.handle.close()
        super().close()


def verify_index(index, metadata):
    require(index.get('schema') == 'worldline-action-recovery-v1', 'Unknown recovery index')
    expected_parts = [{key: part[key] for key in ('name', 'bytes', 'sha256')} for part in metadata['parts']]
    require(index.get('parts') == expected_parts and index.get('stream_sha256') == metadata['stream_sha256'],
            'Pinned metadata and recovery index disagree')
    files = index.get('files')
    require(isinstance(files, dict) and files, 'Expected a nonempty original-file index')
    for name, row in files.items():
        relative = PurePosixPath(name)
        require(isinstance(name, str) and relative.parts and name == str(relative) and not relative.is_absolute()
                and '..' not in relative.parts and '\\' not in name and '\x00' not in name,
                'Unsafe original file path')
        require(isinstance(row, dict) and set(row) == {'bytes', 'sha256'}, 'Invalid original-file record')
        require(type(row['bytes']) is int and 0 <= row['bytes'] < 100_000_000, 'Invalid original-file size')
        require(isinstance(row['sha256'], str) and HEX.fullmatch(row['sha256']), 'Invalid original-file hash')
    require(index.get('file_bytes') == sum(row['bytes'] for row in files.values()), 'Original-file total differs')
    require(index.get('transport_bytes') == sum(row['bytes'] for row in expected_parts), 'Transport total differs')


def recover(metadata_path, output, parts_directory=None, *, opener=https_open):
    metadata_path = Path(metadata_path); output = Path(output)
    require(metadata_path.stat().st_size <= MAX_INDEX, 'Metadata is unexpectedly large')
    metadata_bytes = metadata_path.read_bytes(); metadata = json.loads(metadata_bytes)
    validate_metadata(metadata)
    require(not output.exists() and not output.is_symlink(), 'Output must be a fresh directory')
    if parts_directory is not None:
        parts_directory = Path(parts_directory)
    output.mkdir(parents=True)
    (output / 'download-metadata.json').write_bytes(metadata_bytes)
    index_path = obtain(metadata['index'], output, parts_directory, opener)
    index = json.loads(index_path.read_text()); verify_index(index, metadata)
    paths = []; stream = hashlib.sha256()
    for part in metadata['parts']:
        path = obtain(part, output, parts_directory, opener); paths.append(path)
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                stream.update(chunk)
    require(stream.hexdigest() == metadata['stream_sha256'], 'Complete stream SHA256 differs')
    recovered = output / 'recovered'; recovered.mkdir(); seen = set()
    with io.BufferedReader(Parts(paths)) as stream_file, tarfile.open(fileobj=stream_file, mode='r|gz') as archive:
        for member in archive:
            name = member.name
            require(member.isfile() and name in index['files'] and name not in seen,
                    'Unexpected, linked or duplicate archive member')
            expected = index['files'][name]
            require(member.size == expected['bytes'], 'Archive member length differs')
            target = recovered.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True); checksum = hashlib.sha256()
            with archive.extractfile(member) as source, target.open('xb') as dest:
                for chunk in iter(lambda: source.read(1 << 20), b''):
                    checksum.update(chunk); dest.write(chunk)
            require(checksum.hexdigest() == expected['sha256'], 'Original file SHA256 differs: ' + name)
            seen.add(name)
    require(seen == set(index['files']), 'Original files are missing from the archive')
    result = {'status': 'verified', 'files': len(seen), 'file_bytes': index['file_bytes'],
              'transport_bytes': index['transport_bytes'], 'stream_sha256': metadata['stream_sha256'],
              'index_sha256': metadata['index']['sha256'],
              'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
              'transport': 'local-files' if parts_directory is not None else 'public-https',
              'model_execution': False, 'provider_account_required': False}
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parts-directory', type=Path, help='Verify already downloaded assets without network access')
    args = parser.parse_args()
    print(json.dumps(recover(args.metadata, args.output, args.parts_directory)))


if __name__ == '__main__':
    main()
