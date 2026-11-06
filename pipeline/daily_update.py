"""Add each new storm day to the delta files. Runs once a day on GitHub Actions and needs no secrets."""
import csv, datetime as dt, gzip, io, json, math, os, struct, sys, time, urllib.error, urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mrms
import qc
from build_tiles import render

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA = os.path.join(ROOT, 'delta')
DAY0 = dt.date(2020, 10, 14)
BASE = 'https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MESH_Max_1440min_00.50'
SPC = 'https://www.spc.noaa.gov/climo/reports/'
TIMES = ['120000', '123000', '113000', '130000', '110000', '133000']
UA = {'User-Agent': 'hailstory-data updater'}


class Unavailable(Exception):
    """A source failed. The run stops without saving, so the same day is tried again next time."""


def get(url, timeout=60):
    """Return the file, or None if it does not exist yet. Any other failure raises."""
    req = urllib.request.Request(url, headers=UA)
    problem = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            problem = e
        except Exception as e:
            problem = e
        time.sleep(5 * (attempt + 1))
    raise Unavailable(f'{url}: {problem}')


def load_mask():
    with gzip.open(os.path.join(ROOT, 'pipeline', 'usmask.bin.gz'), 'rb') as f:
        return np.unpackbits(np.frombuffer(f.read(), dtype=np.uint8), count=24500000).reshape(3500, 7000).astype(bool)


def reports(storm_day):
    """SPC daily reports. The file for YYMMDD covers the storm day that starts at 12:00 UTC on that date."""
    tag = storm_day.strftime('%y%m%d')
    ymd = storm_day.year * 10000 + storm_day.month * 100 + storm_day.day
    out = {'hail': [], 'wind': [], 'torn': []}
    for kind in out:
        blob = get(f'{SPC}{tag}_rpts_{kind}.csv')
        if blob is None:
            age = (dt.datetime.now(dt.timezone.utc).date() - storm_day).days
            if age <= 4:
                return None      # reports are not posted yet, so wait for them
            print('no report file, continuing without it:', tag, kind)
            continue
        for row in csv.reader(io.StringIO(blob.decode('latin1'))):
            if len(row) < 7 or row[0] == 'Time':
                continue
            try:
                lat, lon = float(row[5]), float(row[6])
            except ValueError:
                continue
            if kind == 'hail':
                try:
                    out[kind].append([ymd, lat, lon, float(row[1]) / 100])
                except ValueError:
                    pass
            elif kind == 'wind':
                out[kind].append([ymd, lat, lon, round(int(row[1]) / 1.151) if row[1].isdigit() else 0])
            else:
                mag = int(row[1]) if row[1].isdigit() else -9
                out[kind].append([ymd, lat, lon, lat, lon, mag, 0.0, 0, 0, 0])
    return out


def append_radar(day_index, rows, cols, vals):
    lat_idx = 5499 - rows
    lon_idx = -13000 + cols
    tile_lat = np.floor_divide(lat_idx, 100)
    tile_lon = np.floor_divide(lon_idx, 100)
    cell = np.mod(lat_idx, 100) * 100 + np.mod(lon_idx, 100)
    key = (tile_lat + 100).astype(np.int64) * 1000 + (tile_lon + 200)
    for k in np.unique(key):
        sel = key == k
        order = np.argsort(cell[sel])
        c = cell[sel][order].astype('<u2')
        v = vals[sel][order]
        lat, lon = int(k) // 1000 - 100, int(k) % 1000 - 200
        path = os.path.join(DELTA, 't', f'{lat}_{lon}.bin.gz')
        old = gzip.open(path, 'rb').read() if os.path.exists(path) else struct.pack('<H', 0)
        n_days = struct.unpack('<H', old[:2])[0]
        block = struct.pack('<HH', day_index, len(c)) + c.tobytes() + v.tobytes()
        with open(path, 'wb') as f:
            f.write(gzip.compress(struct.pack('<H', n_days + 1) + old[2:] + block, 9, mtime=0))


def append_ground(rep):
    tiles = {}

    def add(lat, lon, field, row):
        tiles.setdefault(f'{math.floor(lat)}_{math.floor(lon)}', {'h': [], 'w': [], 't': []})[field].append(row)

    for r in rep['hail']:
        add(r[1], r[2], 'h', r)
    for r in rep['wind']:
        add(r[1], r[2], 'w', r)
    for i, r in enumerate(rep['torn']):
        add(r[1], r[2], 't', [r[0] * 1000 + i] + r)
    for key, new in tiles.items():
        path = os.path.join(DELTA, 'g', key + '.json.gz')
        old = json.load(gzip.open(path, 'rt')) if os.path.exists(path) else {'h': [], 'w': [], 't': []}
        for field in new:
            old[field] += new[field]
        with open(path, 'wb') as f:
            f.write(gzip.compress(json.dumps(old, separators=(',', ':')).encode(), 9, mtime=0))


def main():
    for sub in ('t', 'g', 'day'):
        os.makedirs(os.path.join(DELTA, sub), exist_ok=True)
    state_path = os.path.join(ROOT, 'state.json')
    state = json.load(open(state_path))
    days_path = os.path.join(DELTA, 'days.json')
    index = json.load(open(days_path)) if os.path.exists(days_path) else {'days': []}
    with open(os.path.join(ROOT, 'pipeline', 'qc_zones.json')) as f:
        zones = {tuple(z) for z in json.load(f)['zones']}
    mask = load_mask()
    next_file_day = dt.date.fromisoformat(state['lastFileDay']) + dt.timedelta(1)
    today = dt.datetime.now(dt.timezone.utc)
    added = 0
    while next_file_day <= today.date() and added < 14:
        tag = next_file_day.strftime('%Y%m%d')
        blob = None
        for t in TIMES:
            blob = get(f'{BASE}/{tag}/MRMS_MESH_Max_1440min_00.50_{tag}-{t}.grib2.gz')
            if blob:
                break
        if not blob:
            if (today.date() - next_file_day).days <= 3:
                print('radar not available yet', tag)
                break
            print('radar missing, skipping day', tag)
            state['skipped'] = state.get('skipped', []) + [tag]
            state['lastFileDay'] = next_file_day.isoformat()
            next_file_day += dt.timedelta(1)
            continue
        storm_day = next_file_day - dt.timedelta(1)
        day_index = (storm_day - DAY0).days
        today_rep = reports(storm_day)
        before = reports(storm_day - dt.timedelta(1))
        if today_rep is None or before is None:
            print('reports not available yet', tag)
            break
        pts = [(r[1], r[2] * 0.8) for r in today_rep['hail'] + before['hail']]
        rows, cols, vals, removed = qc.clean(mrms.decode(blob), mask, np.array(pts) if pts else [], storm_day.month, zones)
        if len(rows):
            append_radar(day_index, rows, cols, vals)
            grid = np.zeros((3500, 7000), dtype=np.float32)
            grid[rows, cols] = vals
            render(grid, os.path.join(DELTA, 'day', f'{day_index}.png'))
            index['days'].append({'d': day_index, 'km1': int((vals >= 25).sum()), 'mm': int(vals.max())})
        append_ground(today_rep)
        state['lastFileDay'] = next_file_day.isoformat()
        print(storm_day, 'cells', len(rows), 'removed', removed, 'reports', {k: len(v) for k, v in today_rep.items()})
        next_file_day += dt.timedelta(1)
        added += 1
    json.dump(state, open(state_path, 'w'))
    json.dump(index, open(days_path, 'w'), separators=(',', ':'))
    print('days added', added)


if __name__ == '__main__':
    main()
