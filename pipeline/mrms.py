"""Minimal decoder for MRMS GRIB2 files (PNG packing, template 5.41). No native dependencies."""
import gzip, io, struct
import numpy as np
from PIL import Image

NX, NY = 7000, 3500
LAT_N = 54.995     # centre of the top row
LON_W = -129.995   # centre of the left column (230.005E)
STEP = 0.01


def decode(path_or_bytes):
    """Return a (3500, 7000) float32 array of the field in native units, NaN where missing or negative."""
    if isinstance(path_or_bytes, (bytes, bytearray)):
        raw = path_or_bytes
    else:
        with open(path_or_bytes, 'rb') as f:
            raw = f.read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    if raw[:4] != b'GRIB' or raw[7] != 2:
        raise ValueError('not GRIB2')
    i, R, E, D, png = 16, None, None, None, None
    while i < len(raw) - 4 and raw[i:i + 4] != b'7777':
        length = struct.unpack('>I', raw[i:i + 4])[0]
        section = raw[i + 4]
        if section == 3:
            nx, ny = struct.unpack('>II', raw[i + 30:i + 38])
            if (nx, ny) != (NX, NY):
                raise ValueError(f'unexpected grid {nx}x{ny}')
        elif section == 5:
            if struct.unpack('>H', raw[i + 9:i + 11])[0] != 41:
                raise ValueError('not PNG packing')
            R = struct.unpack('>f', raw[i + 11:i + 15])[0]
            E = _signed(raw[i + 15:i + 17])
            D = _signed(raw[i + 17:i + 19])
        elif section == 7:
            png = raw[i + 5:i + length]
        i += length
    if png is None or R is None:
        raise ValueError('incomplete GRIB2 message')
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(io.BytesIO(png))
    x = np.asarray(img, dtype=np.float32)
    if x.shape != (NY, NX):
        raise ValueError(f'unexpected image {x.shape}')
    y = (R + x * (2.0 ** E)) / (10.0 ** D)
    y[y < 0] = np.nan
    return y


def _signed(b):
    v = struct.unpack('>H', b)[0]
    return -(v & 0x7fff) if v & 0x8000 else v


def cell(lat, lon):
    row = int(round((LAT_N - lat) / STEP))
    col = int(round((lon - LON_W) / STEP))
    return row, col
