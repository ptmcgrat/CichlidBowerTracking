"""Depth arrays as PNG files the browser can read back exactly.

A height map is not a picture: the page needs the centimetre value under the
cursor, and it needs to re-threshold without asking the server. So the value is
carried losslessly in the pixels rather than rendered to colour here — red is
the high byte, green the low byte, blue is 255 where the pixel is valid and 0
where it was missing. Alpha stays opaque so the browser never premultiplies the
payload away.

Written with zlib and struct rather than an imaging library. A PNG of this
shape is about forty lines, and the alternative was a dependency on OpenCV or
Pillow for one function.

Quantisation is 0.01 cm by default, against a sensor whose own noise floor is
0.05 cm, so nothing real is lost. Raw frames carry no-return pixels thousands
of centimetres away, so the range is clamped before scaling: without it one bad
pixel exhausts the 16 bits and the whole frame quantises coarsely.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

DEFAULT_SCALE = 0.01
MAX_COUNT = 65000


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack('>I', len(data)) + kind + data +
            struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF))


def write_png(path: Path, rgba: np.ndarray) -> int:
    """An RGBA array to a PNG file. Returns the size in bytes."""
    height, width = rgba.shape[:2]
    raw = bytearray()
    for row in range(height):
        raw.append(0)                       # filter type 0: none
        raw.extend(rgba[row].tobytes())
    body = (b'\x89PNG\r\n\x1a\n' +
            _chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)) +
            _chunk(b'IDAT', zlib.compress(bytes(raw), 6)) +
            _chunk(b'IEND', b''))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return len(body)


def encode_depth(array: np.ndarray, scale: float = DEFAULT_SCALE,
                 clip: Optional[Tuple[float, float]] = None,
                 clip_percentile: float = 0.2) -> Tuple[np.ndarray, dict]:
    """A height map as RGBA, with the metadata needed to decode it.

    clip is the physical window to keep. Pass one when it is known — the range
    the interpolated data occupies, say — and anything outside is a no-return
    pixel rather than a reading. Without one a percentile clamp is used, and if
    the span still will not fit the scale is coarsened rather than the encode
    failing.
    """
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape + (4,), np.uint8), {
            'scale': scale, 'offset': 0.0, 'width': array.shape[1],
            'height': array.shape[0], 'empty': True}

    low, high = float(finite.min()), float(finite.max())
    clipped = 0
    if clip is not None:
        low, high = float(clip[0]), float(clip[1])
        clipped = int(np.count_nonzero((finite < low) | (finite > high)))
        array = np.clip(array, low, high)
    elif clip_percentile and finite.size > 100:
        lo_p = float(np.percentile(finite, clip_percentile))
        hi_p = float(np.percentile(finite, 100 - clip_percentile))
        pad = max(0.5, 0.05 * (hi_p - lo_p))
        lo_p, hi_p = lo_p - pad, hi_p + pad
        if hi_p > lo_p and (lo_p > low or hi_p < high):
            clipped = int(np.count_nonzero((finite < lo_p) | (finite > hi_p)))
            array = np.clip(array, lo_p, hi_p)
            low, high = lo_p, hi_p

    while (high - low) / scale > MAX_COUNT and scale < 10:
        scale *= 2

    offset = float(np.floor(low / scale) - 1) * scale
    counts = np.round((array - offset) / scale)
    counts = np.where(np.isfinite(counts), counts, 0)
    counts = np.clip(counts, 0, 65535).astype(np.uint16)

    valid = np.isfinite(array)
    height, width = array.shape
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[:, :, 0] = (counts >> 8).astype(np.uint8)
    rgba[:, :, 1] = (counts & 0xFF).astype(np.uint8)
    rgba[:, :, 2] = np.where(valid, 255, 0)
    rgba[:, :, 3] = 255

    meta = {'scale': scale, 'offset': offset, 'width': width, 'height': height}
    if clipped:
        meta['clipped'] = clipped
    return rgba, meta


def decode_depth(rgba: np.ndarray, meta: dict) -> np.ndarray:
    """The inverse, for tests and for anything server-side that needs values."""
    counts = (rgba[:, :, 0].astype(np.uint16) << 8) | rgba[:, :, 1]
    values = counts * meta['scale'] + meta['offset']
    return np.where(rgba[:, :, 2] == 0, np.nan, values)


def read_png(path: Path) -> np.ndarray:
    """Read back a PNG written by write_png. Filter type 0 only."""
    data = Path(path).read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', 'not a PNG'
    pos, width, height, idat = 8, 0, 0, b''
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos + 4])[0]
        kind = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        if kind == b'IHDR':
            width, height = struct.unpack('>II', payload[:8])
        elif kind == b'IDAT':
            idat += payload
        pos += 12 + length
    raw = zlib.decompress(idat)
    stride = width * 4
    out = np.zeros((height, width, 4), np.uint8)
    for row in range(height):
        start = row * (stride + 1)
        assert raw[start] == 0, 'only filter type 0 is written'
        out[row] = np.frombuffer(raw[start + 1:start + 1 + stride],
                                 np.uint8).reshape(width, 4)
    return out


def write_depth(path: Path, array: np.ndarray, **kwargs) -> dict:
    """Encode a height map and write it. Returns its metadata plus the size."""
    rgba, meta = encode_depth(array, **kwargs)
    meta['bytes'] = write_png(path, rgba)
    meta['name'] = Path(path).name
    return meta