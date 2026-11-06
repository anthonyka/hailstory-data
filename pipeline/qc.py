"""Quality control for daily radar hail. Used by the full build and the daily update, so both follow the same rules."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

MIN_MM = 12.7
MIN_CLUSTER = 3
EIGHT = np.ones((3, 3), dtype=bool)
RECORD_MM = 200.0        # the largest US hailstone on record is 8 inches
COLD_MONTHS = {10, 11, 12, 1, 2, 3}


def cluster_table(grid, us_mask, report_points):
    """Group hail cells into connected clusters and describe each one.
    report_points holds (lat, lon*0.8) for reports on this storm day and the days either side."""
    g = np.nan_to_num(grid, nan=0.0)
    g[~us_mask] = 0
    m = g >= MIN_MM
    labels, n = ndimage.label(m, structure=EIGHT)
    if not n:
        return None
    rows, cols = np.nonzero(m)
    lab = labels[rows, cols]
    vals = g[rows, cols]
    size = np.bincount(lab, minlength=n + 1).astype(np.int64)
    a25 = np.bincount(lab, weights=vals >= 24.5, minlength=n + 1).astype(np.int64)
    a76 = np.bincount(lab, weights=vals >= 75.5, minlength=n + 1).astype(np.int64)
    top = np.zeros(n + 1, dtype=np.float32)
    np.maximum.at(top, lab, vals)
    dist = np.full(n + 1, 9999.0, dtype=np.float32)
    if len(report_points):
        tree = cKDTree(report_points)
        step = 1 if len(rows) < 60000 else 3
        la = 54.995 - rows[::step] * 0.01
        lo = (-129.995 + cols[::step] * 0.01) * 0.8
        d, _ = tree.query(np.stack([la, lo], 1), workers=1)
        np.minimum.at(dist, lab[::step], (d * 111.0).astype(np.float32))
    sum_r = np.bincount(lab, weights=rows, minlength=n + 1)
    sum_c = np.bincount(lab, weights=cols, minlength=n + 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        lat = 54.995 - sum_r / size * 0.01
        lon = -129.995 + sum_c / size * 0.01
    return {'g': g, 'rows': rows, 'cols': cols, 'lab': lab, 'vals': vals, 'size': size, 'a25': a25, 'a76': a76, 'top': top, 'dist': dist, 'lat': lat, 'lon': lon}


def zone_key(lat, lon):
    return (np.floor(lat * 4).astype(np.int32), np.floor(lon * 4).astype(np.int32))


def reject_reasons(t, month, zones):
    """Return an array of reason codes per cluster: 0 keep, 1 speck, 2 impossible, 3 implausible, 4 cold, 5 zone."""
    n = len(t['size'])
    reason = np.zeros(n, dtype=np.uint8)
    zone_hit = np.zeros(n, dtype=bool)
    if zones:
        a, b = zone_key(np.nan_to_num(t['lat']), np.nan_to_num(t['lon']))
        zone_hit = np.array([(int(x), int(y)) in zones for x, y in zip(a, b)])
    ratio = t['a76'] / np.maximum(t['a25'], 1)
    reason[zone_hit & (t['dist'] >= 50)] = 5
    if month in COLD_MONTHS:
        reason[t['dist'] >= 100] = 4
    reason[(t['a25'] >= 20) & (ratio >= 0.3) & (t['dist'] >= 50)] = 3
    reason[t['top'] >= RECORD_MM] = 2
    reason[t['size'] < MIN_CLUSTER] = 1
    reason[0] = 0
    return reason


def clean(grid, us_mask, report_points, month, zones):
    """Return rows, cols, sizeMm for the cells that pass, plus counts of what was removed."""
    t = cluster_table(grid, us_mask, report_points)
    if t is None:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.uint8), {}
    reason = reject_reasons(t, month, zones)
    keep = reason[t['lab']] == 0
    removed = {}
    for code in (2, 3, 4, 5):
        bad = reason == code
        if bad.any():
            removed[code] = {'clusters': int(bad.sum()), 'cells1in': int(t['a25'][bad].sum())}
    vals = np.clip(np.rint(t['vals'][keep]), 0, 255).astype(np.uint8)
    return t['rows'][keep].astype(np.int32), t['cols'][keep].astype(np.int32), vals, removed
