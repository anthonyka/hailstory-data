"""Turn the daily radar grids into per-tile binary files, national images and summary grids."""
import datetime as dt, glob, gzip, json, math, os, struct, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage

import mrms
import qc

MIN_MM = 12.7          # half an inch
MIN_CLUSTER = 3        # drop isolated specks smaller than this many cells
DAY0 = dt.date(2020, 10, 14)
EIGHT = np.ones((3, 3), dtype=bool)
IMG_W = 3500
MERC_TOP = 55.0
MERC_BOTTOM = 20.0

# Colour ramp by hail size. Transparent below half an inch.
RAMP = [(12.7, (96, 196, 214)), (19.0, (70, 220, 170)), (25.4, (250, 226, 90)),
        (38.1, (255, 158, 50)), (50.8, (240, 70, 60)), (76.2, (226, 60, 200)), (101.6, (255, 255, 255))]


def merc_y(lat):
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def image_rows():
    """For each output row, the source row in the 2 km grid, so the image is true Web Mercator."""
    top, bottom = merc_y(MERC_TOP), merc_y(MERC_BOTTOM)
    width_rad = math.radians(70.0)
    height = int(round(IMG_W * (top - bottom) / width_rad))
    ys = top - (np.arange(height) + 0.5) / height * (top - bottom)
    lats = np.degrees(2 * np.arctan(np.exp(ys)) - math.pi / 2)
    rows = np.clip(((MERC_TOP - lats) / 0.02).astype(int), 0, 1749)
    return rows


ROWS = image_rows()


def palette():
    pal = [0, 0, 0] * 256
    for i, (_, c) in enumerate(RAMP):
        pal[(i + 1) * 3:(i + 1) * 3 + 3] = list(c)
    return pal


def render(grid, path):
    """Max-pool 1 km to 2 km, map to the ramp, reproject to Mercator and save as a small PNG."""
    g = np.nan_to_num(grid, nan=0.0)
    pooled = g.reshape(1750, 2, 3500, 2).max(axis=(1, 3))
    merc = pooled[ROWS]
    idx = np.zeros(merc.shape, dtype=np.uint8)
    for i, (mm, _) in enumerate(RAMP):
        idx[merc >= mm] = i + 1
    im = Image.fromarray(idx, mode='P')
    im.putpalette(palette())
    im.save(path, optimize=True, transparency=0)


