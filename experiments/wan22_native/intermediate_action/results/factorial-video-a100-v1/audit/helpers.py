"""Reused pure NumPy/Pillow checks from the prior visual artifact auditor."""
import hashlib,json,math,struct
from pathlib import Path
import numpy as np
from PIL import Image
import rgb

def need(value, label):
    if not value:
        raise AssertionError(label)


def read(path):
    return rgb._json(Path(path))


def sha(path):
    return rgb._sha(path)


def tensor_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def array_file(path, shapes, maximum=16*2**20):
    """Check the header before reading the small FP32/int64 input payloads."""
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and 8 < path.stat().st_size <= maximum,
         'bounded regular tensor file: ' + path.name)
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        need(2 <= size <= 65536, 'bounded tensor header')
        header = rgb._parse(stream.read(size))
        header.pop('__metadata__', None)
        need(set(header) == set(shapes), 'exact tensor names: ' + path.name)
        spans = []
        for name, (shape, dtype) in shapes.items():
            row = header[name]
            need(row['shape'] == list(shape) and row['dtype'] == dtype, 'tensor shape/dtype: ' + name)
            a, b = row['data_offsets']
            need(type(a) is int and type(b) is int and 0 <= a < b
                 and b-a == math.prod(shape)*(4 if dtype == 'F32' else 8), 'exact tensor byte span')
            spans.append((a, b))
        end = 0
        for a, b in sorted(spans):
            need(a == end, 'contiguous nonoverlapping payload'); end = b
        need(8+size+end == path.stat().st_size, 'exact tensor payload length')
        values = {}
        for name, (shape, dtype) in shapes.items():
            a, b = header[name]['data_offsets']; stream.seek(8+size+a)
            value = np.frombuffer(stream.read(b-a), dtype='<f4' if dtype == 'F32' else '<i8').reshape(shape)
            need(np.isfinite(value).all(), 'finite tensor: ' + name)
            values[name] = value
    return values


def inventory(root):
    result = {}
    for path in sorted(root.rglob('*')):
        need(not path.is_symlink(), 'no symlink artifacts')
        if path.is_file():
            result[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    return result


def jsonlines(path):
    return [rgb._parse(line) for line in Path(path).read_bytes().splitlines() if line]


def check_rgb(result, arm, report):
    """Reuse the independent bounded shard reader, adapting only path prefixes."""
    height, width = 704, 1248
    row = report['arms'][arm]; images = row['images']
    need(images['frames'] == 17 and images['height'] == height and images['width'] == width
         and images['conditioned_initial_frames'] == 1 and images['new_future_frames'] == 16
         and images['contact_frame_indices'] == list(rgb.CONTACT)
         and images['contact_images_resized'] is False, 'native presentation contract')
    need({p.name for p in (result/arm/'frames').iterdir()} == {f'{i:04d}.png' for i in range(17)},
         'exact authoritative PNG frame set')
    contact_path = rgb._artifact(result, arm+'/comparison.png', report, rgb.IMAGE_LIMIT)
    with Image.open(contact_path) as opened:
        need(opened.format == 'PNG' and opened.mode == 'RGB'
             and opened.size == (2*width, 5*(height+28)), 'contact sheet dimensions')
        contact = opened.copy()
    def inspect(number, value):
        expected = rgb._pixels(value)
        png = rgb._artifact(result, f'{arm}/frames/{number:04d}.png', report, rgb.IMAGE_LIMIT)
        need(np.array_equal(rgb._png(png, height, width), expected), 'exact raw-to-PNG rounded pixels')
        if number in rgb.CONTACT:
            slot = rgb.CONTACT.index(number); x, y = slot%2*width, slot//2*(height+28)
            need(np.array_equal(np.asarray(contact.crop((x,y+28,x+width,y+28+height))), expected),
                 'unresized contact frame pixels')
            label = np.asarray(contact.crop((x+8,y+6,x+width,y+28)))
            need(np.any(np.max(label,axis=-1)<128), 'dark label ink in contact header')
    try:
        raw = rgb._frames(result, arm+'/rgb', 'spatial', 'generated_clip', report, inspect)
    finally:
        contact.close()
    with Image.open(rgb._artifact(result,arm+'/preview.gif',report,rgb.IMAGE_LIMIT)) as preview:
        need(preview.format == 'GIF' and preview.size == (width,height)
             and 1 <= preview.n_frames <= 17 and preview.info.get('loop') == 0, 'GIF contract')
        durations = []
        for frame in range(preview.n_frames):
            preview.seek(frame); duration = preview.info.get('duration')
            need(type(duration) is int and duration > 0 and duration%120 == 0, 'GIF timing unit')
            durations.append(duration)
    need(sum(durations)==2040 and images['preview_requested_frame_duration_ms']==120
         and images['preview_encoded_frames']==len(durations)
         and images['preview_encoded_frame_durations_ms']==durations
         and images['preview_encoded_duration_ms']==sum(durations), 'actual GIF timing metadata')
    rgb._close(images['preview_requested_playback_fps'],1000/120,'GIF requested playback FPS')
    return dict(raw_rgb=raw, png_frames_exact=17, contact_frame_regions_exact=10,
                contact_label_check='Dark header ink only; no font or glyph byte claim.',
                gif_frame_durations_ms=durations, gif_color_equality_checked=False,
                initial_conditioned_frames=1, new_future_frames=16, cache_clear=row['cache_clear'],
                decode_seconds=row['decode_seconds'])

