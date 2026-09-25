import argparse, base64, datetime, glob, json, os, re, shutil, subprocess, sys, warnings

import cv2
import numpy as np
import pandas as pd
from skimage import morphology

from helper_modules.file_manager import FileManager as FM

"""
Build the static HTML pages that back the bower dashboard, one project at a time.

    python createServer.py Prep YH_MC_Parentals MC_920_t001_tr1

Reads the outputs the pipeline has already produced, checks the project has
actually completed the requested stage, and writes a single self-contained HTML
file (no external assets, no server) into

    __AnalysisStates/<analysisID>/WebServer/<projectID>/

which is then uploaded with the FileManager. Only Prep is implemented so far;
Depth, Cluster and IntegratedData are stubbed as subparsers so the command
surface stays stable as they land.

Depth arrays are embedded as lossless PNGs with the value packed into the red
and green channels (16-bit) and validity in blue, so the browser can recover
real centimetre values under the cursor rather than just showing a picture.
"""

DEPTH_SCALE = 0.01          # cm per count in the packed PNGs
JPEG_QUALITY = 82
TRIAL_SUFFIXES = ['FirstDepth.jpg', 'FirstDepth.npy', 'FirstPi.jpg',
                  'LastDepth.jpg', 'LastDepth.npy', 'LastPi.jpg',
                  'ResetDepth.npy']


# ---------------------------------------------------------------- small helpers

def parse_points(path):
    """DepthCrop.txt / VideoCrop.txt are written as ','.join(str(tuple)), which
    puts commas inside each point, so split on the parentheses instead."""
    with open(path) as f:
        text = f.read()
    pts = [(int(x), int(y)) for x, y in re.findall(r'\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)', text)]
    if len(pts) != 4:
        raise ValueError('Expected 4 points in ' + path + ', found ' + str(len(pts)))
    return pts


def b64(data, mime):
    return 'data:' + mime + ';base64,' + base64.b64encode(data).decode('ascii')


def encode_image(img, max_width=760):
    """BGR array -> base64 JPEG."""
    if img is None:
        return None
    if img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        img = cv2.resize(img, (max_width, int(round(img.shape[0] * scale))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return b64(buf.tobytes(), 'image/jpeg')


def encode_depth(arr, scale=DEPTH_SCALE, clip_percentile=0.2, clip_range=None):
    """Float array in cm -> base64 PNG carrying exact values.

    Value is quantized to int16 counts and split across the red (high byte) and
    green (low byte) channels; blue is 255 where the pixel is valid and 0 where
    it was NaN. Alpha is left fully opaque throughout so the browser never
    premultiplies away the payload.

    Raw sensor frames are not range-filtered the way the interpolated ones are,
    so a handful of pixels with no return can hold values thousands of cm away.
    Those are clamped to a percentile window first — they are meaningless for
    display — and if the remaining span still will not fit in 16 bits the scale
    is coarsened rather than the encode failing.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None

    lo, hi = float(finite.min()), float(finite.max())
    clipped = 0
    if clip_range is not None:
        # caller knows the physical window — used for raw frames, where anything
        # outside the range the sand occupies is a no-return pixel, not a reading
        c_lo, c_hi = clip_range
        clipped = int(np.count_nonzero((finite < c_lo) | (finite > c_hi)))
        arr = np.clip(arr, c_lo, c_hi)
        lo, hi = c_lo, c_hi
    elif clip_percentile and finite.size > 100:
        p_lo = float(np.percentile(finite, clip_percentile))
        p_hi = float(np.percentile(finite, 100 - clip_percentile))
        pad = max(0.5, 0.05 * (p_hi - p_lo))
        p_lo, p_hi = p_lo - pad, p_hi + pad
        if p_hi > p_lo and (p_lo > lo or p_hi < hi):
            clipped = int(np.count_nonzero((finite < p_lo) | (finite > p_hi)))
            arr = np.clip(arr, p_lo, p_hi)
            lo, hi = p_lo, p_hi

    # coarsen the step until the span fits, rather than raising
    while (hi - lo) / scale > 65000 and scale < 10:
        scale *= 2

    offset = float(np.floor(lo / scale) - 1) * scale
    counts = np.round((arr - offset) / scale)
    counts = np.where(np.isfinite(counts), counts, 0)
    counts = np.clip(counts, 0, 65535).astype(np.uint16)

    valid = np.isfinite(arr)
    h, w = arr.shape
    bgra = np.zeros((h, w, 4), np.uint8)
    bgra[:, :, 0] = np.where(valid, 255, 0)           # blue: validity flag
    bgra[:, :, 1] = (counts & 0xFF).astype(np.uint8)   # green: low byte
    bgra[:, :, 2] = (counts >> 8).astype(np.uint8)     # red: high byte
    bgra[:, :, 3] = 255
    ok, buf = cv2.imencode('.png', bgra)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    meta = {'scale': scale, 'offset': offset, 'width': w, 'height': h}
    if clipped:
        meta['clipped'] = clipped
    return b64(buf.tobytes(), 'image/png'), meta


def polygon_area(pts):
    """Shoelace area of a simple polygon."""
    if len(pts) < 3:
        return 0.0
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def clip_polygon(subject, clip):
    """Sutherland-Hodgman clip of one convex polygon by another.

    Both crops are quadrilaterals, so this is enough and avoids a dependency on
    shapely for what is a few lines of arithmetic.
    """
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p, q, a, b):
        x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
        x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return q
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    # orient the clip polygon counter-clockwise so `inside` has a fixed sense
    clip = list(clip)
    a = 0.0
    for i in range(len(clip)):
        x1, y1 = clip[i]
        x2, y2 = clip[(i + 1) % len(clip)]
        a += x1 * y2 - x2 * y1
    if a < 0:
        clip = clip[::-1]

    out = [tuple(p) for p in subject]
    for i in range(len(clip)):
        if not out:
            return []
        a_pt, b_pt = clip[i], clip[(i + 1) % len(clip)]
        prev, out = out, []
        for j, cur in enumerate(prev):
            last = prev[j - 1]
            if inside(cur, a_pt, b_pt):
                if not inside(last, a_pt, b_pt):
                    out.append(intersect(last, cur, a_pt, b_pt))
                out.append(cur)
            elif inside(last, a_pt, b_pt):
                out.append(intersect(last, cur, a_pt, b_pt))
    return out


def polygon_mask(shape, points):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.array(points, np.int32)], 1)
    return mask.astype(bool)


def depth_stats(arr, crop_mask):
    """Numbers a viewer needs to judge whether a frame is usable."""
    inside = arr[crop_mask]
    valid = np.isfinite(inside)
    out = {'valid_fraction': float(valid.mean()) if inside.size else 0.0,
           'n_pixels': int(inside.size)}
    if valid.any():
        good = inside[valid]
        out['median'] = float(np.median(good))
        out['p1'] = float(np.percentile(good, 1))
        out['p99'] = float(np.percentile(good, 99))
    return out


def filtered_change(first, last):
    """Height change between two depth frames, with out-of-tray pixels removed.

    The stored arrays are distance from the sensor, so sand that has been built
    up reads *smaller*. Subtracting later from earlier therefore gives height
    gained: positive is a castle, negative is a pit. This matches
    DepthAnalyzer.returnHeightChange."""
    difference = first - last
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        average = np.nanmean([last, first], axis=0)
        median_height = np.nanmedian(first)
    bad = (average > median_height + 4) | (average < median_height - 8)
    difference = np.where(bad, np.nan, difference)
    return difference, float(median_height)


def load_npy(path):
    return np.load(path).astype(np.float64)


# ------------------------------------------------------------------ prep page

class PrepFiles2Missing(Exception):
    """PrepFiles2 holds the time-matched Pi/depth pairs the pages are built on."""
    pass


TRIAL_SIDES = [('First', 'start'), ('Last', 'end')]


def project_order(s_dt):
    """The projects a sweep covers, in the order the pages link them."""
    return s_dt[(s_dt.Prep == True) & (s_dt.RunAnalysis == True)].index.sort_values().to_list()


def registration_info_path(fm):
    return fm.localAnalysisDir + 'RegistrationInfo.json'


def read_registration_info(fm):
    path = registration_info_path(fm)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_prepfiles2(fm, lp):
    """Read the corrected image pairs. Raises if they have not been built.

    createPrepFiles2.py writes one Pi still and one depth still per trial side,
    matched on capture time. The old PrepFiles pairing could be hours apart, so
    the pages are built on these instead."""
    d = fm.localProjectDir + 'PrepFiles2/'
    manifest_path = d + 'pairs.json'
    if not os.path.exists(manifest_path):
        raise PrepFiles2Missing('no PrepFiles2/pairs.json')
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except Exception as e:
        raise PrepFiles2Missing('pairs.json is unreadable (' + repr(e) + ')')

    pairs, missing = [], []
    for i in range(1, len(lp.trials) + 1):
        for side, word in TRIAL_SIDES:
            stem = d + 'Trial_' + str(i) + side
            needed = {'depth_jpg': stem + 'Depth.jpg', 'pi_jpg': stem + 'Pi.jpg',
                      'depth_npy': stem + 'Depth.npy'}
            absent = [os.path.basename(p) for p in needed.values() if not os.path.exists(p)]
            if absent:
                missing.extend(absent)
                continue
            meta = manifest.get('trials', {}).get(str(i), {}).get(side, {})
            pairs.append({
                'key': 'trial' + str(i) + '_' + word,
                'label': 'Trial ' + str(i) + ' ' + word,
                'trial': i, 'side': side,
                'gapMinutes': meta.get('gapMinutes'),
                'piTime': meta.get('piTime', ''), 'depthTime': meta.get('depthTime', ''),
                'piSource': meta.get('piFile', ''), 'depthSource': meta.get('depthPic', ''),
                **needed,
            })
    if not pairs:
        raise PrepFiles2Missing('PrepFiles2 has no complete trial pairs' +
                                (' (missing ' + ', '.join(missing[:6]) + ')' if missing else ''))
    return pairs, missing


def check_prep_files(fm, lp):
    """The AnalysisStates check for StartingFiles never looks at the per-trial
    prep files, so a project can read Prep == True while missing them."""
    problems = []

    for path in [fm.localFirstFrame, fm.localLastFrame, fm.localPiRGB,
                 fm.localFirstDepthRGB, fm.localLastDepthRGB,
                 fm.localDepthCropFile, fm.localVideoCropFile, fm.localTransMFile]:
        if not os.path.exists(path):
            problems.append('missing ' + os.path.basename(path))

    trial_status = []
    for i in range(len(lp.trials)):
        missing = []
        for suffix in TRIAL_SUFFIXES:
            name = 'Trial_' + str(i + 1) + suffix
            if not os.path.exists(fm.localPrepDir + name):
                missing.append(name)
        trial_status.append(missing)
        if missing:
            problems.append('trial ' + str(i + 1) + ' missing ' + ', '.join(missing))

    return problems, trial_status


def build_quality_payload(fm, lp, raw, smooth, meta):
    """Just what the pixel-quality tab needs, so the Prep page can carry it.

    Deliberately smaller than the Depth payload: the endpoint raw frames, the
    residual map and the day still, at half resolution.
    """
    days_meta = meta['days']
    extras = meta.get('_extras', {})
    depth_dir = fm.localProjectDir + 'DepthFiles/'

    s_finite = smooth[np.isfinite(smooth)]
    s_med = float(np.median(s_finite)) if s_finite.size else 60.0
    window = (s_med - 30.0, s_med + 30.0)

    trials = sorted({d['trial'] for d in days_meta})
    baselines = {}
    for t in trials:
        idx = [i for i, d in enumerate(days_meta) if d['trial'] == t]
        if idx:
            baselines[str(t)] = idx[0]

    spans = [d['n_frames'] for d in days_meta]
    typical = float(np.median(spans)) if spans else 0

    days, frames = [], []
    for i, d in enumerate(days_meta):
        entry = {'day': i, 'trial': d['trial'], 'date': d['first_time'][:10],
                 'firstTime': d['first_time'][11:16], 'lastTime': d['last_time'][11:16],
                 'nFrames': d['n_frames'],
                 'partial': bool(d['n_frames'] < 0.75 * typical)}
        res = extras.get('residual')
        if res is not None:
            q = res[i]
            qv = q[np.isfinite(q) & (q > 0)]
            if qv.size > 100:
                lq = np.log(qv)
                lm = float(np.median(lq))
                la = float(np.median(np.abs(lq - lm))) * 1.4826
                entry['quality'] = {'residual': {
                    'median': round(float(np.exp(lm)), 4),
                    'madFactor': round(float(np.exp(la)), 4),
                    'p99': round(float(np.percentile(qv, 99)), 4),
                    'max': round(float(qv.max()), 3)}}
        days.append(entry)

        f = {}
        for key, arr in (('rawFirst', raw[i, 0]), ('rawLast', raw[i, 1])):
            s, m = encode_depth(half(arr), clip_range=window)
            f[key] = {'src': s, 'meta': m}
        if res is not None:
            s, m = encode_depth(half(res[i], 'max'))
            f['residual'] = {'src': s, 'meta': m}
        name = d.get('first_jpg') or d.get('last_jpg')
        if name and os.path.exists(depth_dir + name):
            f['jpg'] = encode_image(cv2.imread(depth_dir + name), max_width=lp.width // 2)
        frames.append(f)

    return {'frameSize': [lp.width // 2, lp.height // 2],
            'days': days, 'frames': frames, 'baselines': baselines,
            'trials': [{'trial': t} for t in trials]}


def build_prep_payload(fm, lp, trial_status, pairs, neighbours=None):
    """Three views of the same thing: the depth crop, the video crop, and the
    registration between them, each with one row per trial."""
    depth_points = parse_points(fm.localDepthCropFile)
    video_points = parse_points(fm.localVideoCropFile)
    transM = np.load(fm.localTransMFile)
    crop_mask = polygon_mask((lp.height, lp.width), depth_points)

    by_trial = {}
    for p in pairs:
        by_trial.setdefault(p['trial'], {})[p['side']] = p

    trials, pi_size = [], None
    for i, trial in enumerate(lp.trials, 1):
        sides = by_trial.get(i, {})
        entry = {
            'number': i,
            'start': str(trial.startTime), 'stop': str(trial.stopTime),
            'numDays': int(trial.num_days),
            'nDaylightFrames': len(trial.daylight_frames), 'nFrames': len(trial.frames),
            'movies': [int(m.index) for m in trial.movies],
            'complete': len(sides) == 2,
        }
        if len(sides) != 2:
            trials.append(entry)
            continue

        first, last = sides['First'], sides['Last']
        entry['gaps'] = {'first': first['gapMinutes'], 'last': last['gapMinutes']}
        entry['pairKeys'] = {'first': first['key'], 'last': last['key']}

        d_first = cv2.imread(first['depth_jpg'])
        d_last = cv2.imread(last['depth_jpg'])
        p_first = cv2.imread(first['pi_jpg'])
        p_last = cv2.imread(last['pi_jpg'])
        if pi_size is None and p_first is not None:
            pi_size = [int(p_first.shape[1]), int(p_first.shape[0])]

        entry['depthFirst'] = encode_image(d_first, max_width=lp.width)
        entry['depthLast'] = encode_image(d_last, max_width=lp.width)
        entry['piFirst'] = encode_image(p_first, max_width=min(p_first.shape[1], 900))
        entry['piLast'] = encode_image(p_last, max_width=min(p_last.shape[1], 900))
        entry['piFirstWarped'] = encode_image(
            cv2.warpPerspective(p_first, transM, (lp.width, lp.height)), max_width=lp.width)
        entry['piLastWarped'] = encode_image(
            cv2.warpPerspective(p_last, transM, (lp.width, lp.height)), max_width=lp.width)

        change, _ = filtered_change(load_npy(first['depth_npy']), load_npy(last['depth_npy']))
        c_src, c_meta = encode_depth(change)
        entry['change'] = {'src': c_src, 'meta': c_meta,
                           'stats': depth_stats(change, crop_mask),
                           'label': 'Trial ' + str(i) + ' \u2014 height change, positive is sand added'}
        trials.append(entry)

    # the depth crop inverse-warped into Pi coordinates, so how much of the
    # camera's view the depth data actually covers can be seen and measured
    depth_in_video, coverage = None, None
    try:
        inv = np.linalg.inv(transM)
        pts = cv2.perspectiveTransform(
            np.array(depth_points, np.float32).reshape(-1, 1, 2), inv).reshape(-1, 2)
        depth_in_video = [[int(round(x)), int(round(y))] for x, y in pts]

        w, h = pi_size if pi_size else (1296, 972)
        frame = [(0, 0), (w, 0), (w, h), (0, h)]
        v_area = polygon_area(video_points)
        f_area = polygon_area(frame)
        if v_area > 0 and f_area > 0:
            coverage = {
                'ofVideoCrop': round(
                    100.0 * polygon_area(clip_polygon(depth_in_video, video_points)) / v_area, 1),
                'ofFrame': round(
                    100.0 * polygon_area(clip_polygon(depth_in_video, frame)) / f_area, 1),
                'videoOfFrame': round(
                    100.0 * polygon_area(clip_polygon(video_points, frame)) / f_area, 1),
            }
    except Exception as e:
        print('    could not compute depth coverage: ' + repr(e))

    prep_log = ''
    if os.path.exists(fm.localPrepLogfile):
        with open(fm.localPrepLogfile) as f:
            prep_log = ''.join([line for line in f if 'DateAnalyzed' in line
                                or 'Username' in line or 'Nodename' in line]).strip()

    # the pixel-quality tab, if the depth bundle is there. Optional: a project
    # that has not run the Depth stage still gets the rest of the page.
    quality = None
    try:
        q_raw, q_smooth, q_meta = load_depth_endpoints(fm)
        quality = build_quality_payload(fm, lp, q_raw, q_smooth, q_meta)
    except DepthFilesMissing:
        pass
    except Exception as e:
        print('    pixel-quality tab skipped: ' + repr(e))

    return {
        'projectID': lp.projectID, 'tankID': lp.tankID, 'analysisID': fm.analysisID,
        'device': getattr(lp, 'device', 'unknown'),
        'masterStart': str(lp.master_start), 'masterStop': str(getattr(lp, 'master_stop', '')),
        'nFrames': len(lp.frames), 'nMovies': len(lp.movies),
        'depthPoints': depth_points, 'videoPoints': video_points,
        'depthInVideo': depth_in_video, 'coverage': coverage,
        'frameSize': [lp.width, lp.height], 'piSize': pi_size or [1296, 972],
        'trials': trials,
        'logIssues': lp.malformed_file, 'prepLog': prep_log,
        'quality': quality,
        'registration': read_registration_info(fm),
        'neighbours': neighbours or {},
        'built': str(datetime.datetime.now().replace(microsecond=0)),
        'branch': fm.branch_name,
    }


# ---------------------------------------------------------------------- markup

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --video: #6fb2e8;
    --ok: #5aa87a;
    --ink: #e8ecf1;
    --ink-dim: #93a0b0;
    --bg: #0e1116;
    --panel: #171c24;
    --line: #262d38;
    --tray: #f2a33c;
    --warn: #e0693f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.55 "Inter", "Helvetica Neue", Arial, sans-serif;
  }
  .wrap { max-width: 1680px; margin: 0 auto; padding: 28px 24px 72px; }
  h1 { font-size: 25px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
  h2 { font-size: 16px; font-weight: 600; margin: 30px 0 12px; }
  .sub { color: var(--ink-dim); margin: 0 0 22px; }
  .meta { display: flex; flex-wrap: wrap; gap: 10px 26px; padding: 14px 0 18px;
          border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
  .meta div { font-size: 13px; color: var(--ink-dim); }
  .meta b { display: block; color: var(--ink); font-weight: 500;
            font-variant-numeric: tabular-nums; }
  .tabs { display: flex; gap: 4px; margin: 22px 0 18px; flex-wrap: wrap; }
  .tabs button {
    background: none; border: 1px solid var(--line); color: var(--ink-dim);
    padding: 7px 15px; border-radius: 999px; cursor: pointer; font: inherit; font-size: 14px;
  }
  .tabs button[aria-selected="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink); }
  .tabs button:focus-visible { outline: 2px solid var(--tray); outline-offset: 2px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 18px; }
  /* the depth crop row is four views of one thing, so keep them side by side */
  .grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
  @media (max-width: 1400px) { .grid.cols-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  @media (max-width: 760px) { .grid.cols-4 { grid-template-columns: 1fr; } }
  figure { margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
           overflow: hidden; }
  figcaption { padding: 10px 13px; font-size: 13px; color: var(--ink-dim);
               border-top: 1px solid var(--line); }
  figcaption b { color: var(--ink); font-weight: 500; }
  .stage { position: relative; line-height: 0; background: #000; }
  .stage img, .stage canvas { width: 100%; display: block; }
  .stage svg { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
  .stage polygon.video { stroke: var(--video); }
  .stage polygon { fill: none; stroke: var(--tray); stroke-width: 3; vector-effect: non-scaling-stroke; }
  .stage polygon.derived { stroke-width: 2; stroke-dasharray: 8 6; opacity: .65; }
  .grid.cols-6 { grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
  @media (max-width: 1600px) { .grid.cols-6 { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
  @media (max-width: 900px) { .grid.cols-6 { grid-template-columns: 1fr; } }
  .scalebar { height: 9px; border-radius: 2px; margin: 0 13px 4px; }
  .scalelab { display: flex; justify-content: space-between; padding: 0 13px 10px;
              font-size: 11px; color: var(--ink-dim); font-variant-numeric: tabular-nums; }
  .readout { position: absolute; left: 8px; bottom: 8px; background: rgba(6,9,13,.82);
             padding: 3px 8px; border-radius: 4px; font-size: 12px; color: var(--ink);
             font-variant-numeric: tabular-nums; pointer-events: none; opacity: 0;
             transition: opacity .12s; }
  .stage:hover .readout { opacity: 1; }
  .stage canvas { width: 100%; display: block; }
  .trial { margin-bottom: 26px; }
  .trial h2 { display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap;
              border-bottom: 1px solid var(--line); padding-bottom: 8px; margin: 0 0 14px; }
  .trial h2 span { font-weight: 400; font-size: 13px; color: var(--ink-dim);
                   font-variant-numeric: tabular-nums; }
  .readout { position: absolute; left: 8px; bottom: 8px; background: rgba(6,9,13,.82);
             padding: 3px 8px; border-radius: 4px; font-size: 12px;
             font-variant-numeric: tabular-nums; color: var(--ink); pointer-events: none;
             opacity: 0; transition: opacity .12s; }
  .stage:hover .readout { opacity: 1; }
  .controls { display: flex; align-items: center; gap: 12px; padding: 10px 13px;
              border-top: 1px solid var(--line); font-size: 13px; color: var(--ink-dim); }
  .controls input[type=range] { flex: 1; accent-color: var(--tray); }
  .scalebar { height: 9px; border-radius: 2px; margin: 0 13px 11px; }
  .swipe { position: absolute; inset: 0; cursor: col-resize; }
  .swipe .over { position: absolute; top: 0; bottom: 0; left: 0; overflow: hidden; }
  .swipe .over img { position: absolute; top: 0; left: 0; max-width: none; }
  .handle { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--ink);
            pointer-events: none; }
  .handle::after { content: ""; position: absolute; top: 50%; left: 50%; width: 26px; height: 26px;
                   transform: translate(-50%,-50%); border: 2px solid var(--ink); border-radius: 50%;
                   background: rgba(14,17,22,.65); }
  .note { background: rgba(224,105,63,.11); border: 1px solid rgba(224,105,63,.4);
          color: #f0c3ae; padding: 11px 14px; border-radius: 7px; margin: 0 0 18px; font-size: 14px; }
  .note ul { margin: 6px 0 0 18px; padding: 0; }
  .foot { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--line);
          color: var(--ink-dim); font-size: 12px; }
  a { color: var(--tray); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .back { display: inline-block; margin-bottom: 14px; font-size: 13px; color: var(--ink-dim); }
  .bar { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; padding: 12px 0;
         border-bottom: 1px solid var(--line); margin-bottom: 18px; font-size: 14px;
         color: var(--ink-dim); }
  .bar select { background: var(--panel); border: 1px solid var(--line); color: var(--ink);
                padding: 7px 10px; border-radius: 6px; font: inherit; font-size: 14px; }
  .btn { margin-left: auto; border: 1px solid var(--line); color: var(--ink);
         padding: 7px 14px; border-radius: 6px; font-size: 14px; }
  .btn:hover { border-color: var(--tray); text-decoration: none; }
  .btn.off { opacity: .35; pointer-events: none; }
  .fresh { color: var(--ok); }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<div class="wrap">
  <a class="back" href="../index.html">All projects in __ANALYSIS__</a>
  <h1 id="title"></h1>
  <p class="sub" id="subtitle"></p>
  <div class="meta" id="meta"></div>
  <div class="bar">
    <a class="btn" id="prevLink" style="margin-left:0">&larr; Previous</a>
    <a class="btn" id="nextLink" style="margin-left:0">Next &rarr;</a>
    <span class="stat" id="position"></span>
    <span class="stat" id="regInfo"></span>
    <a class="btn" id="fixLink" href="Register.html">Fix registration or crops</a>
  </div>
  <div class="tabs" id="tabs" role="tablist"></div>
  <div id="body"></div>
  <p class="foot" id="foot"></p>
</div>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);

function jet(t){t=Math.min(1,Math.max(0,t));return[Math.max(0,Math.min(1,1.5-Math.abs(4*t-3)))*255,
  Math.max(0,Math.min(1,1.5-Math.abs(4*t-2)))*255,Math.max(0,Math.min(1,1.5-Math.abs(4*t-1)))*255];}

function decodeDepth(img, meta) {
  const c = document.createElement('canvas');
  c.width = meta.width; c.height = meta.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, meta.width, meta.height).data;
  const out = new Float32Array(meta.width * meta.height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4)
    out[i] = px[p+2] === 0 ? NaN : ((px[p] << 8) | px[p+1]) * meta.scale + meta.offset;
  return out;
}

function polySVG(polys, w, h) {
  let s = '<svg viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none" aria-hidden="true">';
  polys.forEach(p => {
    if (!p.points) return;
    s += '<polygon class="crop ' + (p.cls || '') + '" points="' +
         p.points.map(q => q.join(',')).join(' ') + '"/>';
  });
  return s + '</svg>';
}

function imagePanel(src, caption, polys, size) {
  const fig = document.createElement('figure');
  const w = (size || D.frameSize)[0], h = (size || D.frameSize)[1];
  fig.innerHTML = '<div class="stage">' +
    (src ? '<img src="' + src + '" alt="">' : '<div style="aspect-ratio:4/3"></div>') +
    (polys ? polySVG(polys, w, h) : '') + '</div>' +
    '<figcaption>' + caption + '</figcaption>';
  return fig;
}

function statLine(s) {
  if (!s || s.median === undefined) return 'No valid pixels inside the tray.';
  return 'Median ' + s.median.toFixed(2) + ' cm \u00b7 1\u201399% ' + s.p1.toFixed(2) + ' to ' +
         s.p99.toFixed(2) + ' cm \u00b7 ' + (100*s.valid_fraction).toFixed(1) + '% of tray pixels valid';
}

function depthPanel(layer, caption, polys) {
  const fig = document.createElement('figure');
  fig.innerHTML =
    '<div class="stage"><canvas></canvas>' + (polys ? polySVG(polys, D.frameSize[0], D.frameSize[1]) : '') +
    '<span class="readout">\u2014</span></div>' +
    '<div class="scalebar"></div>' +
    '<div class="controls"><label>Range \u00b1<span class="rangeval"></span> cm</label>' +
    '<input type="range" min="0.5" max="10" step="0.5" value="2"></div>' +
    '<figcaption>' + caption + '<br>' + statLine(layer.stats) + '</figcaption>';
  const canvas = fig.querySelector('canvas'), readout = fig.querySelector('.readout');
  const slider = fig.querySelector('input'), rangeval = fig.querySelector('.rangeval');
  const meta = layer.meta;
  canvas.width = meta.width; canvas.height = meta.height;
  const ctx = canvas.getContext('2d');
  let values = null;
  function paint() {
    if (!values) return;
    const range = parseFloat(slider.value);
    rangeval.textContent = range.toFixed(1);
    const out = ctx.createImageData(meta.width, meta.height);
    for (let i = 0, p = 0; i < values.length; i++, p += 4) {
      const v = values[i];
      if (Number.isNaN(v)) { out.data[p]=out.data[p+1]=out.data[p+2]=0; out.data[p+3]=255; continue; }
      const c = jet((v + range) / (2*range));
      out.data[p]=c[0]; out.data[p+1]=c[1]; out.data[p+2]=c[2]; out.data[p+3]=255;
    }
    ctx.putImageData(out, 0, 0);
    let stops=[]; for(let i=0;i<=10;i++){const c=jet(i/10);stops.push('rgb('+[c[0]|0,c[1]|0,c[2]|0]+') '+(i*10)+'%');}
    fig.querySelector('.scalebar').style.background='linear-gradient(to right,'+stops.join(',')+')';
  }
  const img = new Image();
  img.onload = () => { values = decodeDepth(img, meta); paint(); };
  img.src = layer.src;
  slider.addEventListener('input', paint);
  fig.querySelector('.stage').addEventListener('mousemove', ev => {
    if (!values) return;
    const r = canvas.getBoundingClientRect();
    const col = Math.floor((ev.clientX-r.left)/r.width*meta.width);
    const row = Math.floor((ev.clientY-r.top)/r.height*meta.height);
    if (col<0||row<0||col>=meta.width||row>=meta.height) return;
    const v = values[row*meta.width+col];
    readout.textContent = (Number.isNaN(v) ? 'masked' : (v>=0?'+':'') + v.toFixed(2) + ' cm') +
                          '  \u00b7  row ' + row + ', col ' + col;
  });
  return fig;
}

function swipePanel(base, overSrc, caption) {
  const fig = document.createElement('figure');
  fig.innerHTML =
    '<div class="stage"><img class="base" src="' + base + '" alt="">' +
    '<div class="swipe"><div class="over"><img src="' + overSrc + '" alt=""></div>' +
    '<div class="handle"></div></div></div>' +
    '<div class="controls"><label>Divider</label>' +
    '<input type="range" min="0" max="100" step="1" value="50" aria-label="Divider position"></div>' +
    '<figcaption>' + caption + '</figcaption>';
  const stage = fig.querySelector('.stage'), baseImg = fig.querySelector('img.base');
  const swipe = fig.querySelector('.swipe'), over = fig.querySelector('.over');
  const overImg = over.querySelector('img'), handle = fig.querySelector('.handle');
  const slider = fig.querySelector('input');
  let frac = 0.5;
  function sync() {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w) return;
    overImg.style.width = w + 'px'; overImg.style.height = h + 'px';
    over.style.width = (frac*100) + '%'; handle.style.left = (frac*100) + '%';
  }
  function set(f){ frac=Math.min(1,Math.max(0,f)); slider.value=frac*100; sync(); }
  swipe.addEventListener('mousemove', ev => {
    const r = swipe.getBoundingClientRect(); set((ev.clientX-r.left)/r.width);
  });
  slider.addEventListener('input', () => set(slider.value/100));
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(sync).observe(stage);
  if (baseImg.complete) sync(); else baseImg.addEventListener('load', sync);
  return fig;
}

const DEPTH_POLY = [{ points: D.depthPoints }];
const VIDEO_POLY = [{ points: D.videoPoints, cls: 'video' },
                    { points: D.depthInVideo, cls: 'derived' }];
const REACH_POLY = [{ points: D.videoPoints, cls: 'video derived' },
                    { points: D.depthInVideo }];

function trialRow(t, build, cls) {
  const sec = document.createElement('section');
  sec.className = 'trial';
  const gaps = t.gaps ? ' \u00b7 pairs matched to ' + t.gaps.first + ' and ' + t.gaps.last + ' min'
                      : '';
  sec.innerHTML = '<h2>Trial ' + t.number + '<span>' + t.start.slice(0,16) + ' to ' +
    t.stop.slice(0,16) + ' \u00b7 ' + t.numDays + ' days \u00b7 ' + t.nDaylightFrames +
    ' daylight frames' + gaps + '</span></h2>';
  if (!t.complete) {
    sec.innerHTML += '<div class="note">No complete PrepFiles2 pair for this trial.</div>';
    return sec;
  }
  const grid = document.createElement('div');
  grid.className = 'grid' + (cls ? ' ' + cls : '');
  build(grid, t);
  sec.appendChild(grid);
  return sec;
}

function reachCaption() {
  const c = D.coverage;
  if (!c) return '<b>Depth reach</b> \u2014 the tray crop mapped onto the Pi camera. ' +
                 'Coverage could not be computed.';
  return '<b>Depth reach</b> \u2014 the tray crop (orange) mapped into Pi coordinates, ' +
         'against the video crop (blue). Depth covers <b>' + c.ofVideoCrop +
         '%</b> of the video crop and <b>' + c.ofFrame + '%</b> of the whole frame; the ' +
         'video crop itself is ' + c.videoOfFrame + '% of the frame. The rest is video the ' +
         'depth camera never sees.';
}

function viewDepthCrop() {
  const box = document.createElement('div');
  box.innerHTML = '<p class="sub">The polygon should follow the tray walls. Change that ' +
    'reaches the boundary means the crop is clipping part of the bower. The fourth panel ' +
    'shows the same crop in Pi coordinates, so how much of the camera view carries depth ' +
    'data is visible.</p>';
  D.trials.forEach(t => box.appendChild(trialRow(t, (grid, t) => {
    grid.appendChild(imagePanel(t.depthFirst, '<b>Start</b> \u2014 depth camera', DEPTH_POLY));
    grid.appendChild(imagePanel(t.depthLast, '<b>Stop</b> \u2014 depth camera', DEPTH_POLY));
    grid.appendChild(depthPanel(t.change, '<b>Total change over the trial</b>', DEPTH_POLY));
    grid.appendChild(imagePanel(t.piFirst, reachCaption(), REACH_POLY, D.piSize));
  }, 'cols-4')));
  return box;
}

function viewVideoCrop() {
  const box = document.createElement('div');
  box.innerHTML = '<p class="sub">Blue is the video crop. The faint orange outline is the ' +
    'depth crop mapped into Pi coordinates \u2014 the video crop should contain it.</p>';
  D.trials.forEach(t => box.appendChild(trialRow(t, (grid, t) => {
    grid.appendChild(imagePanel(t.piFirst, '<b>Start</b> \u2014 Pi camera', VIDEO_POLY, D.piSize));
    grid.appendChild(imagePanel(t.piLast, '<b>Stop</b> \u2014 Pi camera', VIDEO_POLY, D.piSize));
  })));
  return box;
}

function viewRegistration() {
  const box = document.createElement('div');
  box.innerHTML = '<p class="sub">Move the pointer across to wipe the warped Pi frame over the ' +
    'depth frame. Tray edges should stay continuous across the divider. The cameras are fixed, ' +
    'so a start that looks right and a stop that looks wrong means the sand moved, not the ' +
    'registration.</p>';
  D.trials.forEach(t => box.appendChild(trialRow(t, (grid, t) => {
    grid.appendChild(swipePanel(t.depthFirst, t.piFirstWarped, '<b>Start</b>'));
    grid.appendChild(swipePanel(t.depthLast, t.piLastWarped, '<b>Stop</b>'));
  })));
  return box;
}

// ------------------------------------------------------------ pixel quality
const QCACHE = {};
function qDecode(layer) {
  if (QCACHE[layer.src]) return QCACHE[layer.src];
  const m = layer.meta;
  const c = document.createElement('canvas');
  c.width = m.width; c.height = m.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(layer.img, 0, 0);
  const px = ctx.getImageData(0, 0, m.width, m.height).data;
  const out = new Float32Array(m.width * m.height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4)
    out[i] = px[p+2] === 0 ? NaN : ((px[p] << 8) | px[p+1]) * m.scale + m.offset;
  QCACHE[layer.src] = out;
  return out;
}

function qLoad(layers) {
  return Promise.all(layers.filter(Boolean).map(l => new Promise(res => {
    if (l.img) return res();
    const im = new Image();
    im.onload = () => { l.img = im; res(); };
    im.onerror = () => res();
    im.src = l.src;
  })));
}

function qDiff(a, b) {
  const o = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) o[i] = a[i] - b[i];
  return o;
}

// filtered_change, in the browser: drop any pixel whose mean height sits more
// than 4 cm above or 8 cm below the tray median. This is what the Depth Crop
// panel applies, and without it the tank walls are drawn alongside the tray.
function qFilteredDiff(first, last) {
  const fin = [];
  for (let i = 0; i < first.length; i++) if (!Number.isNaN(first[i])) fin.push(first[i]);
  fin.sort((a, b) => a - b);
  const med = fin.length ? fin[fin.length >> 1] : 0;
  const o = new Float32Array(first.length);
  for (let i = 0; i < first.length; i++) {
    const avg = (first[i] + last[i]) / 2;
    o[i] = (avg > med + 4 || avg < med - 8) ? NaN : first[i] - last[i];
  }
  return o;
}

function qMap(values, caption, opts) {
  opts = opts || {};
  const Q = D.quality, QW = Q.frameSize[0], QH = Q.frameSize[1];
  const range = opts.range === undefined ? 2 : opts.range;
  const mark = opts.mark, mc = opts.markColour || [255, 0, 200];
  const fig = document.createElement('figure');
  fig.innerHTML = '<div class="stage"><canvas width="' + QW + '" height="' + QH +
    '"></canvas><span class="readout">\u2014</span></div>' +
    '<div class="scalebar"></div>' +
    '<div class="scalelab"><span>-' + range + ' cm</span><span>pit \u2190 0 \u2192 castle</span>' +
    '<span>+' + range + ' cm</span></div>' +
    '<figcaption>' + caption + '</figcaption>';
  const canvas = fig.querySelector('canvas'), ctx = canvas.getContext('2d');
  const img = ctx.createImageData(QW, QH);
  for (let i = 0, p = 0; i < values.length; i++, p += 4) {
    if (mark && mark[i]) {
      img.data[p] = mc[0]; img.data[p+1] = mc[1]; img.data[p+2] = mc[2]; img.data[p+3] = 255;
      continue;
    }
    const v = values[i];
    if (Number.isNaN(v)) { img.data[p]=img.data[p+1]=img.data[p+2]=0; img.data[p+3]=255; continue; }
    const c = jet((v + range) / (2 * range));
    img.data[p]=c[0]; img.data[p+1]=c[1]; img.data[p+2]=c[2]; img.data[p+3]=255;
  }
  ctx.putImageData(img, 0, 0);
  let stops = [];
  for (let i = 0; i <= 10; i++) { const c = jet(i/10); stops.push('rgb('+[c[0]|0,c[1]|0,c[2]|0]+') '+(i*10)+'%'); }
  fig.querySelector('.scalebar').style.background = 'linear-gradient(to right,' + stops.join(',') + ')';
  const ro = fig.querySelector('.readout');
  fig.querySelector('.stage').addEventListener('mousemove', ev => {
    const r = canvas.getBoundingClientRect();
    const x = Math.floor((ev.clientX - r.left) / r.width * QW);
    const y = Math.floor((ev.clientY - r.top) / r.height * QH);
    if (x < 0 || y < 0 || x >= QW || y >= QH) return;
    const v = values[y * QW + x];
    ro.textContent = (Number.isNaN(v) ? 'no data' : (v >= 0 ? '+' : '') + v.toFixed(2) + ' cm') +
                     '  \u00b7  ' + x + ', ' + y;
  });
  return fig;
}

function qPhoto(src, caption) {
  const fig = document.createElement('figure');
  fig.innerHTML = (src ? '<div class="stage"><img src="' + src + '" alt=""></div>'
                       : '<div class="stage" style="aspect-ratio:4/3"></div>') +
                  '<figcaption>' + caption + (src ? '' : ' \u2014 not available') + '</figcaption>';
  return fig;
}

let QCROP;
function qOutsideCrop() {
  // the crop is in full-resolution depth coordinates; these maps are half
  if (QCROP !== undefined) return QCROP;
  const pts = D.depthPoints, Q = D.quality;
  if (!pts || pts.length < 3 || !Q) { QCROP = null; return QCROP; }
  const QW = Q.frameSize[0], QH = Q.frameSize[1];
  const m = new Uint8Array(QW * QH);
  for (let y = 0; y < QH; y++) {
    const py = y * 2 + 0.5;
    for (let x = 0; x < QW; x++) {
      const px = x * 2 + 0.5;
      let inside = false;
      for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
        if ((yi > py) !== (yj > py) &&
            px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside;
      }
      if (!inside) m[y * QW + x] = 1;
    }
  }
  QCROP = m;
  return QCROP;
}

function viewQuality() {
  const box = document.createElement('div');
  const Q = D.quality;
  if (!Q) {
    box.innerHTML = '<div class="note">No DepthFiles bundle for this project, so there is ' +
      'nothing to assess. Run the Depth stage, then rebuild this page.</div>';
    return box;
  }

  let trial = Q.trials.length ? Q.trials[0].trial : 1;
  let k = 4;
  let range = 2;

  const sub = document.createElement('div');
  sub.className = 'tabs';
  box.appendChild(sub);

  const bar = document.createElement('div');
  bar.className = 'bar';
  bar.innerHTML =
    '<label>Residual k = <b id="qk">' + k + '</b></label>' +
    '<input type="range" id="qks" min="1" max="8" step="0.5" value="' + k + '">' +
    '<label>Range \u00b1<b id="qr">' + range.toFixed(1) + '</b> cm</label>' +
    '<input type="range" id="qrs" min="0.5" max="10" step="0.5" value="' + range + '">' +
    '<span class="stat" id="qinfo"></span>';
  box.appendChild(bar);

  const note = document.createElement('p');
  note.className = 'sub';
  note.innerHTML = 'Each row is one day. Residual is the RMS departure from a straight line ' +
    'fitted through that day: a steadily building pixel scores near zero because the fit ' +
    'absorbs the build, and it does not grow with the length of the day, so partial days stay ' +
    'comparable. The cut is that day\u2019s own median times its MAD factor to the power k, ' +
    'and the filtered columns are drawn on total change from the trial start. Use this to ' +
    'judge whether the tray crop is excluding the right pixels. Change maps use the same ' +
    'out-of-tray filter and the same default \u00b12 cm scale as the Depth Crop tab, so the ' +
    'two are directly comparable.';
  box.appendChild(note);

  const body = document.createElement('div');
  box.appendChild(body);

  function cutAt(st) { return st ? st.median * Math.pow(st.madFactor, k) : Infinity; }

  function dayRow(d, baseline) {
    const sec = document.createElement('section');
    sec.className = 'trial';
    const st = (d.quality || {}).residual;
    sec.innerHTML = '<h2>' + d.date + '<span>' + d.firstTime + ' to ' + d.lastTime + ' \u00b7 ' +
      d.nFrames + ' frames' + (d.partial ? ' \u00b7 partial day' : '') +
      (st ? ' \u00b7 residual median ' + st.median : '') + '</span></h2>';
    const grid = document.createElement('div');
    grid.className = 'grid cols-6';
    sec.appendChild(grid);

    const f = Q.frames[d.day];
    qLoad([f.rawFirst, f.rawLast, f.residual, baseline.rawFirst]).then(() => {
      const rF = qDecode(f.rawFirst), rL = qDecode(f.rawLast);
      const daily = qFilteredDiff(rF, rL);
      const total = qFilteredDiff(qDecode(baseline.rawFirst), rL);
      // the filter columns are always shown on total change
      const basis = total;
      const bOpts = { range: range };

      grid.textContent = '';
      grid.appendChild(qMap(total, '<b>Total change, raw</b> \u2014 trial start to the end ' +
        'of this day', { range: range }));
      grid.appendChild(qMap(daily, '<b>Daily change, raw</b>', { range: range }));

      let resid = null, nres = 0, nval = 0;
      if (f.residual && st) {
        const thr = cutAt(st), m = qDecode(f.residual);
        resid = new Uint8Array(m.length);
        for (let i = 0; i < m.length; i++) {
          if (Number.isNaN(m[i])) continue;
          nval++;
          if (m[i] > thr) { resid[i] = 1; nres++; }
        }
        grid.appendChild(qMap(basis, '<b>Residual on total change</b> \u2014 k = ' + k +
          ', cut at ' + thr.toFixed(3) + ' cm. <span style="color:#ff00c8">Magenta</span> is ' +
          'what it removes: ' + nres + ' pixels, ' +
          (100 * nres / Math.max(1, nval)).toFixed(2) + '%.',
          Object.assign({ mark: resid }, bOpts)));
      } else {
        grid.appendChild(qPhoto(null, '<b>Residual</b>'));
      }

      const oc = qOutsideCrop();
      let nout = 0, nvalid = 0;
      for (let i = 0; i < basis.length; i++) {
        if (Number.isNaN(basis[i])) continue;
        nvalid++;
        if (oc && oc[i]) nout++;
      }
      grid.appendChild(qMap(basis, '<b>Crop on total change</b> \u2014 ' +
        (oc ? '<span style="color:#ff00c8">Magenta</span> is what the four-point crop ' +
              'excludes: ' + nout + ' pixels, ' +
              (100 * nout / Math.max(1, nvalid)).toFixed(2) + '%.'
            : 'No DepthCrop.txt was available.'),
        Object.assign({ mark: oc }, bOpts)));

      const union = new Uint8Array(basis.length);
      let nunion = 0, nboth = 0;
      for (let i = 0; i < union.length; i++) {
        const a = resid ? resid[i] : 0, b = oc ? oc[i] : 0;
        if (a || b) { union[i] = 1; if (!Number.isNaN(basis[i])) nunion++; }
        if (a && b && !Number.isNaN(basis[i])) nboth++;
      }
      grid.appendChild(qMap(basis, '<b>Both filters on total change</b> \u2014 together they ' +
        'remove ' + nunion + ' pixels, ' + (100 * nunion / Math.max(1, nvalid)).toFixed(2) +
        '%. ' + nboth + ' of those the crop already excluded, so residual adds ' +
        (nunion - nout) + ' beyond it. Removed pixels are blacked out, so what remains in ' +
        'colour is what the analysis would use.',
        Object.assign({ mark: union, markColour: [0, 0, 0] }, bOpts)));

      grid.appendChild(qPhoto(f.jpg, '<b>Depth camera</b> \u2014 ' + d.firstTime));
    });
    return sec;
  }

  function draw() {
    Array.from(sub.children).forEach(b =>
      b.setAttribute('aria-selected', String(+b.dataset.trial === trial)));
    const days = Q.days.filter(d => d.trial === trial);
    const baseline = Q.frames[+Q.baselines[trial]];
    bar.querySelector('#qinfo').innerHTML = '<b>' + days.length + '</b> days in trial ' + trial;
    body.textContent = '';
    if (!baseline) return;

    // rows are built as they scroll in: a long trial is a great many canvases
    const io = typeof IntersectionObserver !== 'undefined'
      ? new IntersectionObserver(entries => {
          entries.forEach(e => {
            if (!e.isIntersecting) return;
            io.unobserve(e.target);
            e.target.replaceWith(dayRow(days[+e.target.dataset.i], baseline));
          });
        }, { rootMargin: '400px' })
      : null;

    days.forEach((d, i) => {
      if (!io) { body.appendChild(dayRow(d, baseline)); return; }
      const ph = document.createElement('section');
      ph.className = 'trial';
      ph.dataset.i = i;
      ph.innerHTML = '<h2>' + d.date + '<span>loading\u2026</span></h2>';
      body.appendChild(ph);
      io.observe(ph);
    });
  }

  Q.trials.forEach(t => {
    const b = document.createElement('button');
    b.textContent = 'Trial ' + t.trial;
    b.dataset.trial = t.trial;
    b.setAttribute('role', 'tab');
    b.addEventListener('click', () => { trial = t.trial; draw(); });
    sub.appendChild(b);
  });
  bar.querySelector('#qrs').addEventListener('input', e => {
    range = parseFloat(e.target.value);
    bar.querySelector('#qr').textContent = range.toFixed(1);
    draw();
  });
  bar.querySelector('#qks').addEventListener('input', e => {
    k = parseFloat(e.target.value);
    bar.querySelector('#qk').textContent = k;
    draw();
  });
  draw();
  return box;
}

function build() {
  document.getElementById('title').textContent = D.projectID;
  document.getElementById('subtitle').textContent =
    'Prep \u2014 tray crop, video crop and camera registration.';
  document.getElementById('meta').innerHTML =
    [['Tank', D.tankID], ['Analysis', D.analysisID], ['Sensor', D.device],
     ['Recording', D.masterStart.slice(0,16) + ' to ' + D.masterStop.slice(0,16)],
     ['Depth frames', D.nFrames], ['Videos', D.nMovies], ['Trials', D.trials.length]]
    .map(([k,v]) => '<div>' + k + '<b>' + v + '</b></div>').join('');

  const firstKey = (D.trials.find(t => t.pairKeys) || {}).pairKeys;
  document.getElementById('fixLink').href =
    'Register.html' + (firstKey ? '?pair=' + encodeURIComponent(firstKey.first) : '');

  // step between projects without going back to the index
  const nb = D.neighbours || {};
  const prev = document.getElementById('prevLink'), next = document.getElementById('nextLink');
  if (nb.prev) prev.href = '../' + nb.prev + '/Prep.html'; else prev.className = 'btn off';
  if (nb.next) next.href = '../' + nb.next + '/Prep.html'; else next.className = 'btn off';
  if (nb.position && nb.total)
    document.getElementById('position').textContent =
      nb.position + ' of ' + nb.total + ' in this analysis';

  const reg = D.registration;
  const el = document.getElementById('regInfo');
  if (reg) {
    const who = (reg.who || reg.initials || 'unknown').split('@')[0];
    el.innerHTML = 'Registration updated <b>' + reg.appliedAt.slice(0, 16) + '</b> by <b>' +
      who + '</b> \u00b7 ' + reg.nPairs + ' pairs, ' + reg.rms_px + ' px' +
      (reg.note ? ' \u00b7 ' + reg.note : '');
    el.className = 'stat fresh';
  } else {
    el.textContent = 'Registration has not been revised since the pipeline set it.';
  }

  if (D.logIssues && D.logIssues.length) {
    const n = document.createElement('div');
    n.className = 'note';
    n.innerHTML = 'The log parser flagged this project:<ul>' +
      D.logIssues.map(x => '<li>' + x + '</li>').join('') + '</ul>';
    document.getElementById('tabs').before(n);
  }

  const views = [['Depth Crop', viewDepthCrop], ['Video Crop', viewVideoCrop],
                 ['Registration', viewRegistration], ['Pixel Quality', viewQuality]];
  const tabs = document.getElementById('tabs'), body = document.getElementById('body');
  const cache = [];
  const buttons = views.map((v, i) => {
    const b = document.createElement('button');
    b.textContent = v[0]; b.setAttribute('role','tab');
    b.addEventListener('click', () => show(i));
    tabs.appendChild(b);
    return b;
  });
  function show(i) {
    buttons.forEach((b, j) => b.setAttribute('aria-selected', String(j === i)));
    body.textContent = '';
    if (!cache[i]) cache[i] = views[i][1]();
    body.appendChild(cache[i]);
  }
  show(0);

  document.getElementById('foot').textContent =
    'Built ' + D.built + ' from branch ' + D.branch + '. ' + D.prepLog.replace(/\n/g, ' \u00b7 ');
}
build();
</script>
</body>
</html>
"""


def write_page(path, title, payload):
    html = PAGE.replace('__TITLE__', title)
    html = html.replace('__ANALYSIS__', payload['analysisID'])
    html = html.replace('__PAYLOAD__', json.dumps(payload).replace('</', '<\\/'))
    with open(path, 'w') as f:
        f.write(html)
    return os.path.getsize(path)


def thumbnail(change, width=260, limit=2.0):
    """Small jet-coloured depth-change image for the project index."""
    valid = np.isfinite(change)
    norm = np.clip((np.where(valid, change, 0) + limit) / (2 * limit), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[~valid] = 0
    scale = width / img.shape[1]
    img = cv2.resize(img, (width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    return b64(buf.tobytes(), 'image/jpeg') if ok else None


INDEX_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --ink:#e8ecf1; --ink-dim:#93a0b0; --bg:#0e1116; --panel:#171c24;
          --line:#262d38; --tray:#f2a33c; --warn:#e0693f; --ok:#5aa87a; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 "Inter","Helvetica Neue",Arial,sans-serif; }
  .wrap { max-width:1640px; margin:0 auto; padding:28px 24px 72px; }
  h1 { font-size:25px; font-weight:600; margin:0 0 4px; letter-spacing:-0.01em; }
  .sub { color:var(--ink-dim); margin:0 0 20px; }
  a { color:var(--tray); text-decoration:none; }
  a:hover { text-decoration:underline; }
  .bar { display:flex; gap:12px; align-items:center; flex-wrap:wrap;
         padding:14px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line);
         margin-bottom:22px; }
  .bar input { flex:1; min-width:200px; background:var(--panel); border:1px solid var(--line);
               color:var(--ink); padding:8px 12px; border-radius:6px; font:inherit; font-size:14px; }
  .bar select, .bar button { background:var(--panel); border:1px solid var(--line);
                color:var(--ink); padding:8px 10px; border-radius:6px; font:inherit; font-size:14px; }
  .bar button { cursor:pointer; }
  .bar button[aria-pressed="true"] { background:var(--ink); color:var(--bg); border-color:var(--ink); }
  .bar span { color:var(--ink-dim); font-size:13px; font-variant-numeric:tabular-nums; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); gap:14px; }
  .group { margin-bottom:30px; }
  .group h2 { font-size:14px; font-weight:600; margin:0 0 11px; padding-bottom:7px;
              border-bottom:1px solid var(--line); display:flex; gap:9px; align-items:baseline; }
  .group h2 span { color:var(--ink-dim); font-weight:400; font-size:13px;
                   font-variant-numeric:tabular-nums; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden;
          display:flex; flex-direction:column; }
  .card img { width:100%; display:block; background:#000; }
  .card .noimg { aspect-ratio:4/3; background:#000; display:flex; align-items:center;
                 justify-content:center; color:var(--ink-dim); font-size:13px; }
  .card .body { padding:11px 12px; flex:1; }
  .card h2 { font-size:13px; font-weight:600; margin:0 0 3px; line-height:1.3;
             word-break:break-word; }
  .card p { margin:0; font-size:12px; color:var(--ink-dim); font-variant-numeric:tabular-nums;
            line-height:1.4; }
  .pages { display:flex; gap:3px; flex-wrap:wrap; padding:0 10px 10px; }
  .pages a, .pages em { font-size:11px; padding:3px 8px; border-radius:999px;
                        border:1px solid var(--line); font-style:normal; }
  .pages a { color:var(--ink); border-color:#3a4553; }
  .pages a:hover { background:var(--ink); color:var(--bg); text-decoration:none; }
  .pages em { color:#5c6675; }
  .flag { display:inline-block; margin-top:7px; font-size:11px; padding:2px 8px;
          border-radius:999px; border:1px solid; }
  .flag.warn { color:var(--warn); border-color:rgba(224,105,63,.5); }
  .flag.fail { color:#ef7a7a; border-color:rgba(239,122,122,.5); }
  .foot { margin-top:40px; padding-top:14px; border-top:1px solid var(--line);
          color:var(--ink-dim); font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__ANALYSIS__</h1>
  <p class="sub" id="sub"></p>
  <div class="bar">
    <input id="q" type="search" placeholder="Filter by project, tank or category" aria-label="Filter projects">
    <select id="sort">
      <option value="name">Sort by project</option>
      <option value="tank">Sort by tank</option>
      <option value="start">Sort by start date</option>
      <option value="status">Sort by status</option>
    </select>
    <button id="flat" aria-pressed="false">Ungroup</button>
    <span id="count"></span>
  </div>
  <div id="grid"></div>
  <p class="foot" id="foot"></p>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const PAGES = ['Prep', 'Depth', 'Cluster', 'IntegratedData'];
const grid = document.getElementById('grid');
const q = document.getElementById('q');
const sortBy = document.getElementById('sort');
const flatBtn = document.getElementById('flat');

document.getElementById('sub').textContent =
  D.projects.length + ' projects with Prep complete. Pick one to review its tray crop and registration.';
document.getElementById('foot').textContent = 'Built ' + D.built + ' from branch ' + D.branch + '.';

function card(p) {
  const el = document.createElement('div');
  el.className = 'card';
  const pages = PAGES.map(name => p.pages[name]
      ? '<a href="' + p.id + '/' + name + '.html">' + name + '</a>'
      : '<em>' + name + '</em>').join('');
  let flag = '';
  if (p.status === 'needs_prepfiles2')
    flag = '<span class="flag warn">Needs PrepFiles2</span>';
  else if (p.status === 'needs_depthfiles')
    flag = '<span class="flag warn">Needs DepthFiles</span>';
  else if (p.status === 'needs_clusters')
    flag = '<span class="flag warn">Needs cluster file</span>';
  else if (p.status === 'failed') flag = '<span class="flag fail">Build failed</span>';
  else if (p.missing) flag = '<span class="flag warn">' + p.missing + ' incomplete</span>';
  el.innerHTML =
    (p.thumb ? '<img src="' + p.thumb + '" alt="Depth change for ' + p.id + '">'
             : '<div class="noimg">No depth preview</div>') +
    '<div class="body"><h2>' + p.id + '</h2>' +
    '<p>' + [p.tank, p.trials + ' trial' + (p.trials === 1 ? '' : 's'),
             p.start ? p.start.slice(0, 10) : ''].filter(Boolean).join(' \u00b7 ') + '</p>' +
    flag + '</div><div class="pages">' + pages + '</div>';
  return el;
}

function rank(x) { return x.status === 'failed' ? 0 : (x.status.slice(0,6) === 'needs_' ? 1 : (x.missing ? 2 : 3)); }

function sorted(rows) {
  const key = sortBy.value;
  return rows.slice().sort((a, b) => {
    if (key === 'status') {
      if (rank(a) !== rank(b)) return rank(a) - rank(b);
      return a.id.localeCompare(b.id);
    }
    const va = (key === 'tank' ? a.tank : key === 'start' ? a.start : a.id) || '';
    const vb = (key === 'tank' ? b.tank : key === 'start' ? b.start : b.id) || '';
    return String(va).localeCompare(String(vb)) || a.id.localeCompare(b.id);
  });
}

function gridOf(rows) {
  const g = document.createElement('div');
  g.className = 'grid';
  rows.forEach(p => g.appendChild(card(p)));
  return g;
}

function render() {
  const term = q.value.trim().toLowerCase();
  const rows = D.projects.filter(p =>
    !term || p.id.toLowerCase().includes(term) || (p.tank || '').toLowerCase().includes(term) ||
    (p.category || '').toLowerCase().includes(term));
  grid.textContent = '';

  if (flatBtn.getAttribute('aria-pressed') === 'true') {
    grid.appendChild(gridOf(sorted(rows)));
  } else {
    const groups = new Map();
    rows.forEach(p => {
      const k = p.category || 'Uncategorised';
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push(p);
    });
    // named categories alphabetically, anything uncategorised last
    const keys = Array.from(groups.keys()).sort((a, b) => {
      if (a === 'Uncategorised') return 1;
      if (b === 'Uncategorised') return -1;
      return a.localeCompare(b);
    });
    keys.forEach(k => {
      const sec = document.createElement('section');
      sec.className = 'group';
      const rowsK = groups.get(k);
      const bad = rowsK.filter(p => rank(p) < 3).length;
      sec.innerHTML = '<h2>' + k + '<span>' + rowsK.length + ' project' +
                      (rowsK.length === 1 ? '' : 's') +
                      (bad ? ' \u00b7 ' + bad + ' needing attention' : '') + '</span></h2>';
      sec.appendChild(gridOf(sorted(rowsK)));
      grid.appendChild(sec);
    });
  }
  document.getElementById('count').textContent =
    rows.length + ' of ' + D.projects.length + ' shown';
}
flatBtn.addEventListener('click', () => {
  flatBtn.setAttribute('aria-pressed', flatBtn.getAttribute('aria-pressed') !== 'true');
  render();
});
q.addEventListener('input', render);
sortBy.addEventListener('change', render);
render();
</script>
</body>
</html>
"""


INDEX_CACHE = '_index.json'


def load_index_entries(out_root):
    """Entries from the last sweep, so one project can be refreshed alone."""
    path = out_root + INDEX_CACHE
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def save_index_entries(out_root, entries):
    with open(out_root + INDEX_CACHE, 'w') as f:
        json.dump(entries, f)


def refresh_index(fm_obj, out_root, entry, branch):
    """Replace one project's entry and rewrite index.html.

    Keeps the landing page current as projects are fixed one at a time, without
    rebuilding every project to do it.
    """
    entries = load_index_entries(out_root)
    replaced = False
    for i, e in enumerate(entries):
        if e.get('id') == entry.get('id'):
            entries[i] = entry
            replaced = True
            break
    if not replaced:
        entries.append(entry)
    entries.sort(key=lambda e: e.get('id', ''))
    save_index_entries(out_root, entries)
    return write_index(out_root + 'index.html', fm_obj.analysisID, entries, branch)


def write_index(path, analysis_id, projects, branch):
    payload = {'analysisID': analysis_id, 'projects': projects, 'branch': branch,
               'built': str(datetime.datetime.now().replace(microsecond=0))}
    html = INDEX_PAGE.replace('__TITLE__', analysis_id + ' · Bower dashboard')
    html = html.replace('__ANALYSIS__', analysis_id)
    html = html.replace('__PAYLOAD__', json.dumps(payload).replace('</', '<\\/'))
    with open(path, 'w') as f:
        f.write(html)
    return os.path.getsize(path)


# ============================================================== depth viewer

class DepthFilesMissing(Exception):
    """DepthFiles/daily_endpoints.npz is what the Depth page is built on."""
    pass


# thresholds and minimum bower size, mirroring FileManager
DAILY_THRESHOLD = 0.4
TOTAL_THRESHOLD = 1.0
MIN_PIXELS = 100
PIXEL_LENGTH = 0.1030168618          # cm per pixel
THRESHOLD_SWEEP = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]


DIAGNOSTIC_KEYS = ('travel', 'residual', 'stdMean', 'stdMax')


def load_depth_endpoints(fm):
    """Read DepthFiles/daily_endpoints.{npz,json}. Raises if absent."""
    d = fm.localProjectDir + 'DepthFiles/'
    npz_path, json_path = d + 'daily_endpoints.npz', d + 'daily_endpoints.json'
    if not (os.path.exists(npz_path) and os.path.exists(json_path)):
        raise DepthFilesMissing('no DepthFiles/daily_endpoints.npz')
    with open(json_path) as f:
        meta = json.load(f)
    extras = {}
    with np.load(npz_path) as z:
        raw = z['raw'].astype(np.float64)
        smooth = z['smooth'].astype(np.float64)
        for k in DIAGNOSTIC_KEYS + ('trend',):
            if k in z:
                extras[k] = z[k].astype(np.float64)
    meta['_extras'] = extras
    if raw.shape != smooth.shape:
        raise DepthFilesMissing('raw and smooth arrays disagree in shape')
    if raw.shape[0] != len(meta.get('days', [])):
        raise DepthFilesMissing('array has %d days, metadata lists %d'
                                % (raw.shape[0], len(meta.get('days', []))))
    return raw, smooth, meta


def bower_locations(change, threshold, min_pixels=MIN_PIXELS):
    """+1 castle, -1 pit, 0 neither, NaN outside. Mirrors
    DepthAnalyzer.returnBowerLocations."""
    castle = np.where(change >= threshold, True, False)
    pit = np.where(change <= -threshold, True, False)
    castle = remove_small(castle, min_pixels)
    pit = remove_small(pit, min_pixels)
    out = (castle.astype(int) - pit.astype(int)).astype(float)
    out[np.isnan(change)] = np.nan
    return out


def remove_small(mask, min_pixels):
    """skimage renamed min_size to max_size; call whichever exists."""
    try:
        return morphology.remove_small_objects(mask, max_size=min_pixels)
    except TypeError:
        return morphology.remove_small_objects(mask, min_size=min_pixels)


def volume_summary(change, threshold):
    """Castle, pit and total volumes in cm3, computed at full resolution."""
    loc = bower_locations(change, threshold)
    area = PIXEL_LENGTH ** 2
    castle = float(np.nansum(np.where(loc == 1, change, 0), dtype=np.float64)) * area
    pit = -float(np.nansum(np.where(loc == -1, change, 0), dtype=np.float64)) * area
    return {
        'threshold': threshold,
        'castleVolume': round(castle, 2),
        'pitVolume': round(pit, 2),
        'bowerVolume': round(castle + pit, 2),
        'castleArea': round(float(np.count_nonzero(loc == 1)) * area, 2),
        'pitArea': round(float(np.count_nonzero(loc == -1)) * area, 2),
    }


def half(frame, how='mean'):
    """Downsample by 2 for display, ignoring NaN.

    Use how='max' for one-sided diagnostic layers. Averaging a lone bad pixel
    with three good neighbours divides it by four, so on a mean-downsampled map
    the hover value understates exactly the isolated pixels a threshold is meant
    to catch — and then disagrees with a histogram computed at full resolution.
    """
    h, w = frame.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    blocks = frame[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        if how == 'max':
            return np.nanmax(blocks, axis=(1, 3))
        return np.nanmean(blocks, axis=(1, 3))


def build_depth_payload(fm, lp, raw, smooth, meta):
    """Everything the Depth page needs.

    Height maps are shipped at half resolution as packed PNGs and the browser
    subtracts them to make daily, overnight and cumulative changes. Volumes are
    computed here at full resolution, so the numbers never depend on the
    downsampling done for display.
    """
    days = meta['days']
    n_days = len(days)

    # --- flag the days that are not comparable to a normal one --------------
    spans = [d['n_frames'] for d in days]
    typical = float(np.median(spans)) if spans else 0
    for i, d in enumerate(days):
        d['partial'] = bool(d['n_frames'] < 0.75 * typical)
        d['overnightOK'] = False
        d['overnightNote'] = ''
        if i + 1 < n_days:
            nxt = days[i + 1]
            gap_h = (datetime.datetime.fromisoformat(nxt['first_time']) -
                     datetime.datetime.fromisoformat(d['last_time'])).total_seconds() / 3600.0
            d['gapHours'] = round(gap_h, 2)
            if nxt['trial'] != d['trial']:
                d['overnightNote'] = 'spans the tank reset'
            elif gap_h <= 0:
                d['overnightNote'] = 'the next day overlaps this one in time'
            elif gap_h < 6:
                d['overnightNote'] = 'gap of only %.1f h — recording restarted' % gap_h
            elif gap_h > 20:
                d['overnightNote'] = 'gap of %.1f h — a recording break' % gap_h
            else:
                d['overnightOK'] = True
        else:
            d['gapHours'] = None

    # --- per-trial baselines ------------------------------------------------
    trials = sorted({d['trial'] for d in days})
    baseline_index = {}
    for t in trials:
        idx = [i for i, d in enumerate(days) if d['trial'] == t]
        baseline_index[t] = idx[0]

    extras = meta.get('_extras', {})

    # --- volumes, at full resolution ---------------------------------------
    day_stats = []
    for i, d in enumerate(days):
        first_s, last_s = smooth[i, 0], smooth[i, 1]
        daily = first_s - last_s                      # positive: sand added
        entry = {
            'day': i, 'trial': d['trial'],
            'date': d['first_time'][:10],
            'firstTime': d['first_time'][11:16], 'lastTime': d['last_time'][11:16],
            'nFrames': d['n_frames'], 'partial': d['partial'],
            'gapHours': d['gapHours'], 'overnightOK': d['overnightOK'],
            'overnightNote': d['overnightNote'],
            'rawValid': [round(d['raw_valid_first'], 4), round(d['raw_valid_last'], 4)],
            'daily': volume_summary(daily, DAILY_THRESHOLD),
        }
        b = baseline_index[d['trial']]
        cumulative = smooth[b, 0] - last_s
        entry['cumulative'] = volume_summary(cumulative, TOTAL_THRESHOLD)

        # travel rate for this day alone, at full resolution. The trial-level
        # threshold assumes one noise floor for the whole trial; if conditions
        # change day to day that is either too loose or too tight, so the same
        # statistic is computed per day and can be compared.
        t0 = datetime.datetime.fromisoformat(d['first_time'])
        t1 = datetime.datetime.fromisoformat(d['last_time'])
        hours = max(0.01, (t1 - t0).total_seconds() / 3600.0)
        entry['hours'] = round(hours, 2)
        # log-median and MAD for each metric on its own, so a threshold can be
        # set per day per metric rather than inherited from the trial
        quality = {}
        for name in ('travel', 'residual'):
            arr = extras.get(name)
            if arr is None:
                continue
            q = arr[i]
            qv = q[np.isfinite(q) & (q > 0)]
            if qv.size < 100:
                continue
            lq = np.log(qv)
            lm = float(np.median(lq))
            la = float(np.median(np.abs(lq - lm))) * 1.4826
            quality[name] = {
                'median': round(float(np.exp(lm)), 4),
                'madFactor': round(float(np.exp(la)), 4),
                'p99': round(float(np.percentile(qv, 99)), 4),
                'max': round(float(qv.max()), 3),
            }
        if quality:
            entry['quality'] = quality

        if 'travel' in extras:
            ex = (extras['travel'][i] - np.abs(daily)) / hours
            v = ex[np.isfinite(ex) & (ex > 0)]
            if v.size > 100:
                lv = np.log(v)
                lmed = float(np.median(lv))
                lmad = float(np.median(np.abs(lv - lmed))) * 1.4826
                entry['travelStats'] = {
                    'median': round(float(np.exp(lmed)), 4),
                    'madFactor': round(float(np.exp(lmad)), 3),
                    'p99': round(float(np.percentile(v, 99)), 4),
                    'max': round(float(v.max()), 3),
                    'cuts': [{'k': k,
                              'value': round(float(np.exp(lmed + k * lmad)), 4),
                              'masked': round(100.0 * float(np.count_nonzero(
                                  v > np.exp(lmed + k * lmad))) / v.size, 2)}
                             for k in (1, 2, 3, 4, 5, 6)],
                }
        if d['overnightOK']:
            overnight = last_s - smooth[i + 1, 0]
            entry['overnight'] = volume_summary(overnight, DAILY_THRESHOLD)
        else:
            entry['overnight'] = None
        day_stats.append(entry)

    def trial_travel(idx, total_net):
        """Travel summed over a trial's days, per lights-on hour.

        Travel from sensor noise grows linearly with frame count, so dividing by
        hours gives a rate that is comparable across trials of different length.
        Subtracting |net| removes the part of the movement that actually went
        somewhere, so a pixel that genuinely built a lot cannot be flagged."""
        travel = extras.get('travel')
        if travel is None:
            return None
        total = np.nansum(np.stack([travel[i] for i in idx]), axis=0)
        hours = 0.0
        for i in idx:
            t0 = datetime.datetime.fromisoformat(days[i]['first_time'])
            t1 = datetime.datetime.fromisoformat(days[i]['last_time'])
            hours += (t1 - t0).total_seconds() / 3600.0
        if hours <= 0:
            return None
        rate = total / hours
        excess = (total - np.abs(total_net)) / hours

        v = excess[np.isfinite(excess) & (excess > 0)]
        stats = {'hours': round(hours, 1), 'nDays': len(idx)}
        if v.size:
            lv = np.log(v)
            lmed = float(np.median(lv))
            lmad = float(np.median(np.abs(lv - lmed))) * 1.4826
            stats.update({
                'median': round(float(np.exp(lmed)), 4),
                'madFactor': round(float(np.exp(lmad)), 3),
                'p50': round(float(np.percentile(v, 50)), 4),
                'p90': round(float(np.percentile(v, 90)), 4),
                'p99': round(float(np.percentile(v, 99)), 4),
                'max': round(float(v.max()), 3),
            })
            # what a cut at each k would remove, so the choice can be made by eye
            stats['cuts'] = [{'k': k,
                              'value': round(float(np.exp(lmed + k * lmad)), 4),
                              'masked': round(100.0 * float(np.count_nonzero(v > np.exp(lmed + k * lmad))) / v.size, 2)}
                             for k in (1, 1.5, 2, 2.5, 3, 4, 5, 6)]
            counts, edges = np.histogram(lv, bins=48)
            stats['hist'] = {'counts': counts.tolist(),
                             'edges': [round(float(np.exp(e)), 4) for e in edges]}
        # max, not mean: these maps exist to find isolated outliers, so the
        # hover value has to mean the same thing as the histogram
        return {'rate': dict(zip(('src', 'meta'), encode_depth(half(rate, 'max')))),
                'excess': dict(zip(('src', 'meta'), encode_depth(half(excess, 'max')))),
                'stats': stats}

    # --- per-trial totals and the threshold sweep ---------------------------
    trial_stats = []
    for t in trials:
        idx = [i for i, d in enumerate(days) if d['trial'] == t]
        b, e = idx[0], idx[-1]
        total = smooth[b, 0] - smooth[e, 1]
        sweep = [volume_summary(total, th) for th in THRESHOLD_SWEEP]
        reset_change = None
        reset_path = fm.localPrepDir + 'Trial_' + str(t) + 'ResetDepth.npy'
        if os.path.exists(reset_path):
            reset = load_npy(reset_path)
            rc = reset - smooth[e, 1]
            reset_change = {'map': encode_depth(half(rc))[0],
                            'meta': encode_depth(half(rc))[1],
                            'volumes': volume_summary(rc, TOTAL_THRESHOLD)}
        trial_stats.append({
            'travel': trial_travel(idx, total),
            'trial': t, 'firstDay': b, 'lastDay': e, 'nDays': len(idx),
            'start': days[b]['first_time'], 'stop': days[e]['last_time'],
            'total': volume_summary(total, TOTAL_THRESHOLD),
            'sweep': sweep,
            'totalMap': dict(zip(('src', 'meta'), encode_depth(half(total)))),
            'resetChange': reset_change,
        })

    # --- half-resolution height maps for the browser ------------------------
    depth_dir = fm.localProjectDir + 'DepthFiles/'
    # the interpolated array is range-filtered, so it defines what counts as sand
    s_finite = smooth[np.isfinite(smooth)]
    s_med = float(np.median(s_finite)) if s_finite.size else 60.0
    height_window = (s_med - 30.0, s_med + 30.0)
    frames = []
    for i in range(n_days):
        entry = {}
        for key, arr in (('smoothFirst', smooth[i, 0]), ('smoothLast', smooth[i, 1]),
                         ('rawFirst', raw[i, 0]), ('rawLast', raw[i, 1])):
            s, m = encode_depth(half(arr), clip_range=height_window)
            entry[key] = {'src': s, 'meta': m}
        if 'travel' in extras:
            hrs = max(0.01, day_stats[i].get('hours', 1) if i < len(day_stats) else 1)
            ex = (extras['travel'][i] - np.abs(smooth[i, 0] - smooth[i, 1])) / hrs
            s, m = encode_depth(half(ex, 'max'))
            entry['excess'] = {'src': s, 'meta': m}
        for key, arr in (('travel', extras.get('travel')),
                         ('residual', extras.get('residual')),
                         ('stdMean', extras.get('stdMean')),
                         ('stdMax', extras.get('stdMax'))):
            if arr is not None:
                s, m = encode_depth(half(arr[i], 'max'))
                entry[key] = {'src': s, 'meta': m}
        # the depth camera stills copied out by depth_endpoints
        for key, name in (('jpgFirst', days[i].get('first_jpg')),
                          ('jpgLast', days[i].get('last_jpg'))):
            if name and os.path.exists(depth_dir + name):
                entry[key] = encode_image(cv2.imread(depth_dir + name), max_width=lp.width // 2)
        frames.append(entry)

    depth_points = None
    if os.path.exists(fm.localDepthCropFile):
        try:
            depth_points = parse_points(fm.localDepthCropFile)
        except Exception:
            depth_points = None

    return {
        'projectID': lp.projectID, 'tankID': lp.tankID, 'analysisID': fm.analysisID,
        'frameSize': [lp.width // 2, lp.height // 2],
        'depthPoints': depth_points,
        'pixelLength': PIXEL_LENGTH,
        'defaultThreshold': TOTAL_THRESHOLD,
        'dailyThreshold': DAILY_THRESHOLD,
        'thresholdSweep': THRESHOLD_SWEEP,
        'days': day_stats, 'trials': trial_stats, 'frames': frames,
        'hasStd': bool(meta.get('hasStd')), 'hasTravel': bool(meta.get('hasTravel')),
        'baselines': {str(k): v for k, v in baseline_index.items()},
        'built': str(datetime.datetime.now().replace(microsecond=0)),
        'branch': fm.branch_name,
    }


DEPTH_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --ink:#e8ecf1; --ink-dim:#93a0b0; --bg:#0e1116; --panel:#171c24;
          --line:#262d38; --tray:#f2a33c; --ok:#5aa87a; --warn:#e0693f; --cool:#6fb2e8; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 "Inter","Helvetica Neue",Arial,sans-serif; }
  .wrap { max-width:1760px; margin:0 auto; padding:24px 22px 80px; }
  h1 { font-size:24px; font-weight:600; margin:0 0 3px; }
  h2 { font-size:15px; font-weight:600; margin:26px 0 10px; }
  .sub { color:var(--ink-dim); margin:0 0 18px; }
  a { color:var(--tray); text-decoration:none; } a:hover { text-decoration:underline; }
  .back { display:inline-block; margin-bottom:12px; font-size:13px; color:var(--ink-dim); }
  .tabs { display:flex; gap:4px; margin:16px 0 18px; flex-wrap:wrap; }
  .tabs button { background:none; border:1px solid var(--line); color:var(--ink-dim);
                 padding:8px 17px; border-radius:999px; cursor:pointer; font:inherit; font-size:14px; }
  .tabs button[aria-selected="true"] { background:var(--ink); color:var(--bg); border-color:var(--ink); }
  .bar { display:flex; gap:14px; align-items:center; flex-wrap:wrap; padding:13px 0;
         border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin-bottom:16px; }
  .bar label { font-size:13px; color:var(--ink-dim); }
  .bar input[type=range] { width:190px; accent-color:var(--tray); }
  .bar button { background:var(--panel); border:1px solid var(--line); color:var(--ink);
                padding:7px 13px; border-radius:6px; font:inherit; font-size:13px; cursor:pointer; }
  .bar button[aria-pressed="true"] { background:var(--tray); color:#141414; border-color:var(--tray); }
  .spacer { flex:1; }
  .stat { font-size:13px; color:var(--ink-dim); font-variant-numeric:tabular-nums; }
  .stat b { color:var(--ink); font-weight:500; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; }
  .grid.cols-6 { grid-template-columns:repeat(6,minmax(0,1fr)); gap:10px; }
  @media (max-width:1600px) { .grid.cols-6 { grid-template-columns:repeat(3,minmax(0,1fr)); } }
  @media (max-width:900px) { .grid.cols-6 { grid-template-columns:1fr; } }
  .trial { margin-bottom:26px; }
  .trial h2 { display:flex; gap:12px; align-items:baseline; flex-wrap:wrap;
              border-bottom:1px solid var(--line); padding-bottom:8px; margin:0 0 12px; }
  .trial h2 span { font-weight:400; font-size:13px; color:var(--ink-dim);
                   font-variant-numeric:tabular-nums; }
  figure { margin:0; background:var(--panel); border:1px solid var(--line);
           border-radius:8px; overflow:hidden; }
  figcaption { padding:10px 13px; font-size:13px; color:var(--ink-dim); border-top:1px solid var(--line); }
  figcaption b { color:var(--ink); font-weight:500; }
  .stage { position:relative; line-height:0; background:#000; }
  .stage canvas, .stage img { width:100%; display:block; }
  .readout { position:absolute; left:8px; bottom:8px; background:rgba(6,9,13,.82);
             padding:3px 8px; border-radius:4px; font-size:12px; color:var(--ink);
             font-variant-numeric:tabular-nums; pointer-events:none; opacity:0; transition:opacity .12s; }
  .stage:hover .readout { opacity:1; }
  .scalebar { height:9px; border-radius:2px; margin:0 13px 4px; }
  .scalelab { display:flex; justify-content:space-between; padding:0 13px 10px;
              font-size:11px; color:var(--ink-dim); font-variant-numeric:tabular-nums; }
  .cards { display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; }
  .block { margin-bottom:26px; }
  .matrix { display:grid; gap:6px; align-items:start; }
  .matrix .colhead { font-size:12px; color:var(--ink-dim); padding:0 0 4px;
                     font-variant-numeric:tabular-nums; }
  .matrix .colhead b { display:block; color:var(--ink); font-weight:600; font-size:13px; }
  .rowlab { font-size:12px; color:var(--ink); padding:4px 8px 0 0; }
  .rowlab b { display:block; font-weight:600; }
  .rowlab span { color:var(--ink-dim); font-size:11px; }
  .cell { background:var(--panel); border:1px solid var(--line); border-radius:5px;
          overflow:hidden; }
  .cellnote { padding:14px 8px; font-size:11px; color:var(--ink-dim); text-align:center; }
  .cellfoot { padding:3px 6px; font-size:10px; color:var(--ink-dim);
              border-top:1px solid var(--line); font-variant-numeric:tabular-nums; }
  .blank { }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden;
          cursor:pointer; }
  .card:hover { border-color:var(--tray); }
  .card .body { padding:12px 14px; }
  .card h3 { margin:0 0 6px; font-size:15px; font-weight:600; }
  .card table { width:100%; font-size:12px; border-collapse:collapse;
                font-variant-numeric:tabular-nums; }
  .card td { padding:2px 0; color:var(--ink-dim); }
  .card td:last-child { text-align:right; color:var(--ink); }
  .days { display:flex; gap:3px; flex-wrap:wrap; margin-bottom:14px; }
  .days button { background:var(--panel); border:1px solid var(--line); color:var(--ink-dim);
                 padding:5px 9px; border-radius:5px; font:inherit; font-size:12px; cursor:pointer;
                 font-variant-numeric:tabular-nums; }
  .days button[aria-pressed="true"] { background:var(--ink); color:var(--bg); border-color:var(--ink); }
  .days button.partial { border-style:dashed; }
  .note { background:rgba(224,105,63,.1); border:1px solid rgba(224,105,63,.35);
          color:#f0c3ae; padding:10px 13px; border-radius:7px; font-size:13px; margin:0 0 14px; }
  table.vol { width:100%; border-collapse:collapse; font-size:13px;
              font-variant-numeric:tabular-nums; }
  table.vol th, table.vol td { text-align:right; padding:5px 9px; border-bottom:1px solid var(--line); }
  table.vol th:first-child, table.vol td:first-child { text-align:left; }
  table.vol th { color:var(--ink-dim); font-weight:500; }
  .chart { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
  .chart svg { width:100%; height:230px; display:block; }
  .foot { margin-top:40px; padding-top:14px; border-top:1px solid var(--line);
          color:var(--ink-dim); font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <a class="back" href="../index.html">All projects in __ANALYSIS__</a>
  <h1 id="title"></h1>
  <p class="sub" id="sub"></p>
  <div class="tabs" id="tabs" role="tablist"></div>
  <div id="body"></div>
  <p class="foot" id="foot"></p>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const W = D.frameSize[0], H = D.frameSize[1];
const cache = {};

function jet(t) {
  t = Math.min(1, Math.max(0, t));
  return [Math.max(0,Math.min(1,1.5-Math.abs(4*t-3)))*255,
          Math.max(0,Math.min(1,1.5-Math.abs(4*t-2)))*255,
          Math.max(0,Math.min(1,1.5-Math.abs(4*t-1)))*255];
}

// Height maps arrive as PNGs with the value packed into red and green and
// validity in blue, so the browser recovers real centimetres and can subtract
// two frames to make any change map it needs.
function decode(layer) {
  if (cache[layer.src]) return cache[layer.src];
  const m = layer.meta;
  const c = document.createElement('canvas');
  c.width = m.width; c.height = m.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(layer.img, 0, 0);
  const px = ctx.getImageData(0, 0, m.width, m.height).data;
  const out = new Float32Array(m.width * m.height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4)
    out[i] = px[p+2] === 0 ? NaN : ((px[p] << 8) | px[p+1]) * m.scale + m.offset;
  cache[layer.src] = out;
  return out;
}

function loadAll(layers) {
  return Promise.all(layers.map(l => new Promise(res => {
    if (l.img) return res();
    const im = new Image();
    im.onload = () => { l.img = im; res(); };
    im.onerror = () => res();
    im.src = l.src;
  })));
}

function diff(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

function mapPanel(values, caption, opts) {
  opts = opts || {};
  // three scales: signed change about zero, absolute height, and one-sided
  // diagnostics like variability where only the magnitude means anything
  let lo, hi, label;
  if (opts.fixed) {
    lo = opts.fixed[0]; hi = opts.fixed[1];
    label = ['%L cm', 'sensor distance', '%H cm'];
  } else if (opts.absolute) {
    const fin = Array.from(values).filter(v => !Number.isNaN(v));
    fin.sort((a,b) => a-b);
    lo = fin.length ? fin[Math.floor(fin.length*0.02)] : 0;
    hi = fin.length ? fin[Math.floor(fin.length*0.98)] : 1;
    label = ['%L cm', 'sensor distance', '%H cm'];
  } else if (opts.log) {
    lo = opts.log[0]; hi = opts.log[1];
    label = ['%L cm/h', 'log scale', '%H cm/h'];
  } else if (opts.positive !== undefined) {
    lo = 0; hi = opts.positive;
    label = ['0', 'higher is worse', '%H cm'];
  } else {
    hi = opts.range === undefined ? 2 : opts.range; lo = -hi;
    label = ['%L cm', 'pit \u2190 0 \u2192 castle', '+%H cm'];
  }
  const range = hi;
  const fig = document.createElement('figure');
  const fmt = s => s.replace('%L', lo.toFixed(2)).replace('%H', hi.toFixed(2));
  fig.innerHTML = '<div class="stage"><canvas width="' + W + '" height="' + H + '"></canvas>' +
    '<span class="readout">—</span></div><div class="scalebar"></div>' +
    '<div class="scalelab"><span>' + fmt(label[0]) + '</span><span>' + label[1] +
    '</span><span>' + fmt(label[2]) + '</span></div>' +
    '<figcaption>' + caption + '</figcaption>';
  const canvas = fig.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(W, H);
  const thr = (opts.absolute || opts.fixed || opts.log || opts.positive !== undefined) ? 0 : (opts.threshold || 0);
  const mark = opts.mark;
  const mc = opts.markColour || [255, 0, 200];
  for (let i = 0, p = 0; i < values.length; i++, p += 4) {
    // pixels a filter removed: magenta, so a 0.5% mask is still visible
    if (mark && mark[i]) { img.data[p]=mc[0]; img.data[p+1]=mc[1]; img.data[p+2]=mc[2]; img.data[p+3]=255; continue; }
    const v = values[i];
    if (Number.isNaN(v)) { img.data[p]=img.data[p+1]=img.data[p+2]=0; img.data[p+3]=255; continue; }
    if (thr && Math.abs(v) < thr) {           // below threshold: grey, not coloured
      img.data[p]=img.data[p+1]=img.data[p+2]=44; img.data[p+3]=255; continue;
    }
    const frac = opts.log
      ? (Math.log(Math.max(v, 1e-6)) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))
      : (v - lo) / (hi - lo);
    const [r,g,b] = jet(frac);
    img.data[p]=r; img.data[p+1]=g; img.data[p+2]=b; img.data[p+3]=255;
  }
  ctx.putImageData(img, 0, 0);
  let stops = [];
  for (let i=0;i<=10;i++){ const c=jet(i/10); stops.push('rgb('+[c[0]|0,c[1]|0,c[2]|0]+') '+(i*10)+'%'); }
  fig.querySelector('.scalebar').style.background='linear-gradient(to right,'+stops.join(',')+')';
  const ro = fig.querySelector('.readout');
  fig.querySelector('.stage').addEventListener('mousemove', ev => {
    const r = canvas.getBoundingClientRect();
    const x = Math.floor((ev.clientX-r.left)/r.width*W), y = Math.floor((ev.clientY-r.top)/r.height*H);
    if (x<0||y<0||x>=W||y>=H) return;
    const v = values[y*W+x];
    ro.textContent = (Number.isNaN(v) ? 'no data' : (v>=0?'+':'') + v.toFixed(2) + ' cm') +
                     '  ·  ' + x + ', ' + y;
  });
  return fig;
}

function photoPanel(src, caption) {
  const fig = document.createElement('figure');
  fig.innerHTML = (src ? '<div class="stage"><img src="' + src + '" alt=""></div>'
                       : '<div class="stage" style="aspect-ratio:4/3"></div>') +
                  '<figcaption>' + caption + (src ? '' : ' — not in this bundle') + '</figcaption>';
  return fig;
}

function volTable(rows) {
  const t = document.createElement('table');
  t.className = 'vol';
  t.innerHTML = '<tr><th></th><th>castle cm³</th><th>pit cm³</th><th>total cm³</th>' +
                '<th>castle cm²</th><th>pit cm²</th></tr>' +
    rows.map(([lab, v]) => v ? '<tr><td>' + lab + '</td><td>' + v.castleVolume +
      '</td><td>' + v.pitVolume + '</td><td>' + v.bowerVolume + '</td><td>' +
      v.castleArea + '</td><td>' + v.pitArea + '</td></tr>'
      : '<tr><td>' + lab + '</td><td colspan="5" style="text-align:left;color:#93a0b0">not available</td></tr>').join('');
  return t;
}

// ------------------------------------------------------------------ landing
function renderLanding() {
  const box = document.createElement('div');
  const cards = document.createElement('div');
  cards.className = 'cards';
  D.trials.forEach(t => {
    const c = document.createElement('div');
    c.className = 'card';
    c.innerHTML = '<div class="stage"><canvas width="' + W + '" height="' + H + '"></canvas></div>' +
      '<div class="body"><h3>Trial ' + t.trial + '</h3>' +
      '<table><tr><td>days</td><td>' + t.nDays + '</td></tr>' +
      '<tr><td>' + t.start.slice(0,10) + ' to ' + t.stop.slice(0,10) + '</td><td></td></tr>' +
      '<tr><td>bower volume</td><td>' + t.total.bowerVolume + ' cm³</td></tr>' +
      '<tr><td>castle</td><td>' + t.total.castleVolume + ' cm³</td></tr>' +
      '<tr><td>pit</td><td>' + t.total.pitVolume + ' cm³</td></tr></table></div>';
    c.addEventListener('click', () => show(D.trials.indexOf(t) + 1));
    cards.appendChild(c);
    loadAll([t.totalMap]).then(() => {
      const v = decode(t.totalMap);
      const ctx = c.querySelector('canvas').getContext('2d');
      const img = ctx.createImageData(W, H);
      for (let i=0,p=0;i<v.length;i++,p+=4){
        const x=v[i];
        if (Number.isNaN(x)){img.data[p]=img.data[p+1]=img.data[p+2]=0;img.data[p+3]=255;continue;}
        const c2=jet((x+2)/4); img.data[p]=c2[0];img.data[p+1]=c2[1];img.data[p+2]=c2[2];img.data[p+3]=255;
      }
      ctx.putImageData(img,0,0);
    });
  });
  box.appendChild(cards);

  const h = document.createElement('h2');
  h.textContent = 'Volume built per day';
  box.appendChild(h);
  box.appendChild(dailyChart());
  return box;
}

function trialBand(run, pad, bw, hgt) {
  const x0 = pad + run.from*bw, x1 = pad + (run.to+1)*bw;
  const shade = run.trial % 2 ? 'rgba(242,163,60,.07)' : 'rgba(111,178,232,.07)';
  return '<rect x="' + x0 + '" y="' + (pad-6) + '" width="' + (x1-x0) + '" height="' +
         (hgt-pad-(pad-6)) + '" fill="' + shade + '"/>' +
         '<text x="' + ((x0+x1)/2) + '" y="' + (pad+8) + '" fill="#93a0b0" font-size="12" ' +
         'text-anchor="middle">Trial ' + run.trial + '</text>';
}

function dailyChart() {
  const host = document.createElement('div');
  host.className = 'chart';
  const days = D.days;
  const vals = days.map(d => d.daily.bowerVolume);
  const over = days.map(d => d.overnight ? d.overnight.bowerVolume : null);
  const max = Math.max(1, ...vals, ...over.filter(v => v !== null));
  const w = 1000, hgt = 230, pad = 34, bw = (w - 2*pad) / days.length;
  let s = '';
  // a band and a label per trial, so each day's trial is obvious
  let run = null;
  days.forEach((d, i) => {
    if (!run || run.trial !== d.trial) {
      if (run) s += trialBand(run, pad, bw, hgt);
      run = { trial: d.trial, from: i, to: i };
    } else run.to = i;
  });
  if (run) s += trialBand(run, pad, bw, hgt);
  days.forEach((d, i) => {
    const x = pad + i*bw;
    const hh = (vals[i]/max) * (hgt - 2*pad);
    s += '<rect x="' + (x+1) + '" y="' + (hgt-pad-hh) + '" width="' + (bw-2) + '" height="' + hh +
         '" fill="' + (d.partial ? '#6b5334' : 'var(--tray)') + '"><title>Trial ' + d.trial +
         ', day ' + i + ' (' + d.date + ') daily ' + vals[i] + ' cm³' +
         (d.partial ? ', partial day' : '') + '</title></rect>';
    if (over[i] !== null) {
      const oh = (over[i]/max) * (hgt - 2*pad);
      s += '<rect x="' + (x+bw*0.3) + '" y="' + (hgt-pad-oh) + '" width="' + (bw*0.4) +
           '" height="' + oh + '" fill="var(--cool)" opacity="0.85"><title>Day ' + i +
           ' overnight ' + over[i] + ' cm³</title></rect>';
    } else if (d.overnightNote) {
      s += '<text x="' + (x+bw/2) + '" y="' + (hgt-pad+12) + '" fill="#e0693f" font-size="11" ' +
           'text-anchor="middle">✕</text><title>' + d.overnightNote + '</title>';
    }
  });
  s += '<line x1="' + pad + '" y1="' + (hgt-pad) + '" x2="' + (w-pad) + '" y2="' + (hgt-pad) +
       '" stroke="#262d38"/>';
  s += '<text x="' + pad + '" y="' + (pad-12) + '" fill="#93a0b0" font-size="12">cm³, ' +
       'orange = daytime, blue = overnight, dashed bars are partial days, ✕ marks a gap that is not a night</text>';
  for (let k=0;k<=2;k++){
    const y = hgt-pad-(k/2)*(hgt-2*pad);
    s += '<text x="4" y="' + (y+4) + '" fill="#93a0b0" font-size="11">' + Math.round(max*k/2) + '</text>';
  }
  host.innerHTML = '<svg viewBox="0 0 ' + w + ' ' + hgt + '" preserveAspectRatio="none">' + s + '</svg>';
  return host;
}

// -------------------------------------------------------------- trial view
let DEPTH_CROP;
function cropOutside() {
  // the crop is in full-resolution coordinates; these maps are half
  if (DEPTH_CROP !== undefined) return DEPTH_CROP;
  const pts = D.depthPoints;
  if (!pts || pts.length < 3) { DEPTH_CROP = null; return DEPTH_CROP; }
  const W = D.frameSize[0], H = D.frameSize[1];
  const m = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const py = y * 2 + 0.5;
    for (let x = 0; x < W; x++) {
      const px = x * 2 + 0.5;
      let inside = false;
      for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
        if ((yi > py) !== (yj > py) &&
            px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside;
      }
      if (!inside) m[y * W + x] = 1;
    }
  }
  DEPTH_CROP = m;
  return DEPTH_CROP;
}

// everything outside the tray is dropped before it is drawn or measured, so it
// cannot join a bower region or contribute to an area
function cropped(values) {
  const oc = cropOutside();
  if (!oc) return values;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = oc[i] ? NaN : values[i];
  return out;
}

function labelCell(text, sub) {
  const d = document.createElement('div');
  d.className = 'rowlab';
  d.innerHTML = '<b>' + text + '</b>' + (sub ? '<span>' + sub + '</span>' : '');
  return d;
}

function blankCell() {
  const d = document.createElement('div');
  d.className = 'blank';
  return d;
}

// connected components over the thresholded change, castle and pit separately,
// dropping anything smaller than minPixels. Mirrors returnBowerLocations.
function bowerRegions(values, W, H, thr, minPixels, scale) {
  const sign = new Int8Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isNaN(v)) continue;
    if (v >= thr) sign[i] = 1;
    else if (v <= -thr) sign[i] = -1;
  }
  const keep = new Uint8Array(values.length);
  const seen = new Uint8Array(values.length);
  const stack = new Int32Array(values.length);
  for (let start = 0; start < sign.length; start++) {
    if (!sign[start] || seen[start]) continue;
    const s = sign[start];
    let top = 0, n = 0;
    const members = [];
    stack[top++] = start;
    seen[start] = 1;
    while (top > 0) {
      const p = stack[--top];
      members.push(p);
      n++;
      const x = p % W, y = (p / W) | 0;
      if (x > 0 && sign[p-1] === s && !seen[p-1]) { seen[p-1] = 1; stack[top++] = p-1; }
      if (x < W-1 && sign[p+1] === s && !seen[p+1]) { seen[p+1] = 1; stack[top++] = p+1; }
      if (y > 0 && sign[p-W] === s && !seen[p-W]) { seen[p-W] = 1; stack[top++] = p-W; }
      if (y < H-1 && sign[p+W] === s && !seen[p+W]) { seen[p+W] = 1; stack[top++] = p+W; }
    }
    // the maps are half resolution, so a component covers 4x this many pixels
    if (n * scale >= minPixels) for (const p of members) keep[p] = 1;
  }
  const notBower = new Uint8Array(values.length);
  let area = 0, vol = 0;
  for (let i = 0; i < keep.length; i++) {
    if (keep[i]) { area++; vol += Math.abs(values[i]); }
    else notBower[i] = 1;
  }
  return { notBower, area: area * scale, volume: vol * scale };
}

function renderTrial(t) {
  const box = document.createElement('div');
  const days = D.days.filter(d => d.trial === t.trial);
  let thr = D.defaultThreshold, minPx = 100;

  const bar = document.createElement('div');
  bar.className = 'bar';
  bar.innerHTML =
    '<label>Bower height \u2265 <b id="bh">' + thr.toFixed(2) + '</b> cm</label>' +
    '<input type="range" id="bhs" min="0.1" max="3" step="0.05" value="' + thr + '">' +
    '<label>Minimum region <b id="bp">' + minPx + '</b> px</label>' +
    '<input type="range" id="bps" min="0" max="2000" step="25" value="' + minPx + '">' +
    '<span class="spacer"></span><span class="stat" id="tinfo"></span>';
  box.appendChild(bar);

  const note = document.createElement('p');
  note.className = 'sub';
  note.innerHTML = 'One column per day, eight to a block. All six rows come from the ' +
    'interpolated array, with everything outside the tray crop removed. Daylight change is ' +
    'that day\u2019s first lights-on frame to its ' +
    'last; night is that evening to the next morning; 24 hours is the two together. The ' +
    'bower rows keep only regions that clear the height threshold and are large enough ' +
    'once connected, with everything else greyed.';
  box.appendChild(note);

  const body = document.createElement('div');
  box.appendChild(body);

  const ROWS = [
    ['Total change', 'trial start to end of day', 'total', 4],
    ['24 hour change', 'morning to next morning', 'full', 2],
    ['Daylight change', 'morning to evening', 'daylight', 2],
    ['Night change', 'evening to next morning', 'night', 2],
    ['Bower, total', 'regions in total change', 'bowerTotal', 4],
    ['Bower, daylight', 'regions in daylight change', 'bowerDaily', 2],
  ];

  function buildBlock(chunk, host) {
    // always eight columns, so a short final block keeps the same day width
    const COLS = 8;
    const pad = COLS - chunk.length;
    const grid = document.createElement('div');
    grid.className = 'matrix';
    grid.style.gridTemplateColumns = '132px repeat(' + COLS + ', minmax(0, 1fr))';
    host.textContent = '';
    host.appendChild(grid);

    grid.appendChild(blankCell());
    chunk.forEach(d => {
      const h = document.createElement('div');
      h.className = 'colhead';
      h.innerHTML = '<b>' + d.date.slice(5) + '</b><span>' + d.nFrames + ' frames' +
        (d.partial ? ', partial' : '') + '</span>';
      grid.appendChild(h);
    });
    for (let i = 0; i < pad; i++) grid.appendChild(blankCell());

    const cells = {};
    ROWS.forEach(([name, sub, key]) => {
      grid.appendChild(labelCell(name, sub));
      chunk.forEach(d => {
        const slot = document.createElement('div');
        slot.className = 'cell';
        grid.appendChild(slot);
        cells[key + ':' + d.day] = slot;
      });
      for (let i = 0; i < pad; i++) grid.appendChild(blankCell());
    });

    const need = [];
    chunk.forEach(d => {
      const f = D.frames[d.day];
      need.push(f.smoothFirst, f.smoothLast);
      const nxt = D.frames[d.day + 1];
      if (nxt && D.days[d.day + 1] && D.days[d.day + 1].trial === t.trial)
        need.push(nxt.smoothFirst);
    });
    need.push(D.frames[+D.baselines[t.trial]].smoothFirst);

    loadAll(need).then(() => {
      const base = decode(D.frames[+D.baselines[t.trial]].smoothFirst);
      const scale = 4;                       // half-resolution pixels to full
      chunk.forEach(d => {
        const f = D.frames[d.day];
        const mF = decode(f.smoothFirst), mL = decode(f.smoothLast);
        const nextDay = D.days[d.day + 1];
        const sameTrial = nextDay && nextDay.trial === t.trial;
        const nF = sameTrial ? decode(D.frames[d.day + 1].smoothFirst) : null;

        const total = cropped(diff(base, mL));
        const daylight = cropped(diff(mF, mL));
        const night = nF ? cropped(diff(mL, nF)) : null;
        const full = nF ? cropped(diff(mF, nF)) : null;
        const W = D.frameSize[0], H = D.frameSize[1];

        const put = (key, values, range, extra) => {
          const slot = cells[key + ':' + d.day];
          if (!slot) return;
          slot.textContent = '';
          if (!values) {
            slot.appendChild(qNote(key === 'night' || key === 'full'
              ? 'no following day in this trial' : 'not available'));
            return;
          }
          slot.appendChild(cellMap(values, range, extra));
        };

        put('total', total, 4);
        put('full', full, 2);
        put('daylight', daylight, 2);
        put('night', night, 2);

        const bT = bowerRegions(total, W, H, thr, minPx, scale);
        put('bowerTotal', total, 4, { mark: bT.notBower, markColour: [30, 34, 40] });
        cells['bowerTotal:' + d.day].appendChild(
          qFoot(bT.area.toLocaleString() + ' px \u00b7 ' + bT.volume.toFixed(0) + ' cm'));

        const bD = bowerRegions(daylight, W, H, thr, minPx, scale);
        put('bowerDaily', daylight, 2, { mark: bD.notBower, markColour: [30, 34, 40] });
        cells['bowerDaily:' + d.day].appendChild(
          qFoot(bD.area.toLocaleString() + ' px \u00b7 ' + bD.volume.toFixed(0) + ' cm'));
      });
    });
  }

  function qNote(text) {
    const d = document.createElement('div');
    d.className = 'cellnote';
    d.textContent = text;
    return d;
  }

  function qFoot(text) {
    const d = document.createElement('div');
    d.className = 'cellfoot';
    d.textContent = text;
    return d;
  }

  // a bare canvas: at eight columns there is no room for captions
  function cellMap(values, range, opts) {
    opts = opts || {};
    const W = D.frameSize[0], H = D.frameSize[1];
    const wrap = document.createElement('div');
    wrap.className = 'stage';
    const canvas = document.createElement('canvas');
    canvas.width = W; canvas.height = H;
    wrap.appendChild(canvas);
    const ro = document.createElement('span');
    ro.className = 'readout';
    ro.textContent = '\u2014';
    wrap.appendChild(ro);
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(W, H);
    const mark = opts.mark, mc = opts.markColour || [255, 0, 200];
    for (let i = 0, p = 0; i < values.length; i++, p += 4) {
      if (mark && mark[i]) {
        img.data[p]=mc[0]; img.data[p+1]=mc[1]; img.data[p+2]=mc[2]; img.data[p+3]=255;
        continue;
      }
      const v = values[i];
      if (Number.isNaN(v)) { img.data[p]=img.data[p+1]=img.data[p+2]=0; img.data[p+3]=255; continue; }
      const c = jet((v + range) / (2 * range));
      img.data[p]=c[0]; img.data[p+1]=c[1]; img.data[p+2]=c[2]; img.data[p+3]=255;
    }
    ctx.putImageData(img, 0, 0);
    wrap.addEventListener('mousemove', ev => {
      const r = canvas.getBoundingClientRect();
      const x = Math.floor((ev.clientX - r.left) / r.width * W);
      const y = Math.floor((ev.clientY - r.top) / r.height * H);
      if (x < 0 || y < 0 || x >= W || y >= H) return;
      const v = values[y * W + x];
      ro.textContent = Number.isNaN(v) ? 'no data' : (v >= 0 ? '+' : '') + v.toFixed(2) + ' cm';
    });
    return wrap;
  }

  function draw() {
    body.textContent = '';
    bar.querySelector('#tinfo').innerHTML = '<b>' + days.length + '</b> days \u00b7 ' +
      t.start.slice(0, 10) + ' to ' + t.stop.slice(0, 10);
    const blocks = [];
    for (let i = 0; i < days.length; i += 8) blocks.push(days.slice(i, i + 8));

    const io = typeof IntersectionObserver !== 'undefined'
      ? new IntersectionObserver(entries => {
          entries.forEach(e => {
            if (!e.isIntersecting) return;
            io.unobserve(e.target);
            buildBlock(blocks[+e.target.dataset.b], e.target);
          });
        }, { rootMargin: '300px' })
      : null;

    blocks.forEach((chunk, i) => {
      const host = document.createElement('div');
      host.className = 'block';
      host.dataset.b = i;
      host.innerHTML = '<p class="stat">' + chunk[0].date + ' to ' +
        chunk[chunk.length - 1].date + '\u2026</p>';
      body.appendChild(host);
      if (io) io.observe(host); else buildBlock(chunk, host);
    });
  }

  bar.querySelector('#bhs').addEventListener('input', e => {
    thr = parseFloat(e.target.value);
    bar.querySelector('#bh').textContent = thr.toFixed(2);
    draw();
  });
  bar.querySelector('#bps').addEventListener('input', e => {
    minPx = parseInt(e.target.value, 10);
    bar.querySelector('#bp').textContent = minPx;
    draw();
  });
  draw();
  return box;
}

function sweepChart(t) {
  const host = document.createElement('div');
  host.className = 'chart';
  host.style.marginTop = '16px';
  const pts = t.sweep;
  const max = Math.max(1, ...pts.map(p => p.bowerVolume));
  const w = 1000, hgt = 230, pad = 36;
  const x = i => pad + (i/(pts.length-1))*(w-2*pad);
  const y = v => hgt - pad - (v/max)*(hgt-2*pad);
  const line = key => pts.map((p,i) => (i?'L':'M') + x(i) + ' ' + y(p[key])).join(' ');
  let s = '<path d="' + line('bowerVolume') + '" fill="none" stroke="var(--tray)" stroke-width="2"/>' +
          '<path d="' + line('castleVolume') + '" fill="none" stroke="var(--ok)" stroke-width="1.5"/>' +
          '<path d="' + line('pitVolume') + '" fill="none" stroke="var(--cool)" stroke-width="1.5"/>';
  pts.forEach((p,i) => {
    s += '<circle cx="' + x(i) + '" cy="' + y(p.bowerVolume) + '" r="3" fill="var(--tray)">' +
         '<title>threshold ' + p.threshold + ' cm: total ' + p.bowerVolume + ' cm³, castle ' +
         p.castleVolume + ', pit ' + p.pitVolume + '</title></circle>';
    s += '<text x="' + x(i) + '" y="' + (hgt-pad+14) + '" fill="#93a0b0" font-size="11" ' +
         'text-anchor="middle">' + p.threshold + '</text>';
  });
  s += '<text x="' + pad + '" y="' + (pad-12) + '" fill="#93a0b0" font-size="12">' +
       'Whole-trial volume against threshold (cm³) — orange total, green castle, blue pit</text>';
  host.innerHTML = '<svg viewBox="0 0 ' + w + ' ' + hgt + '" preserveAspectRatio="none">' + s + '</svg>';
  return host;
}

// ------------------------------------------------------------------- shell
function show(i) {
  Array.from(document.getElementById('tabs').children).forEach((b, j) =>
    b.setAttribute('aria-selected', String(j === i)));
  const body = document.getElementById('body');
  body.textContent = '';
  body.appendChild(i === 0 ? renderLanding() : renderTrial(D.trials[i-1]));
}

function build() {
  document.getElementById('title').textContent = D.projectID + ' — depth';
  const nd = D.days.length, bad = D.days.filter(d => d.partial).length;
  document.getElementById('sub').textContent =
    nd + ' days across ' + D.trials.length + ' trial' + (D.trials.length===1?'':'s') +
    (bad ? ', ' + bad + ' of them partial' : '') +
    '. Maps are at half resolution; volumes are computed at full resolution.';
  const tabs = document.getElementById('tabs');
  ['Overview'].concat(D.trials.map(t => 'Trial ' + t.trial)).forEach((name, i) => {
    const b = document.createElement('button');
    b.textContent = name; b.setAttribute('role','tab');
    b.addEventListener('click', () => show(i));
    tabs.appendChild(b);
  });
  document.getElementById('foot').textContent = 'Built ' + D.built + ' from branch ' + D.branch + '.';
  show(0);
}
build();
</script>
</body>
</html>
"""


def write_depth(path, analysis_id, payload):
    html = DEPTH_PAGE.replace('__TITLE__', payload['projectID'] + ' \u00b7 Depth')
    html = html.replace('__ANALYSIS__', analysis_id)
    html = html.replace('__PAYLOAD__', json.dumps(payload).replace('</', '<\\/'))
    with open(path, 'w') as f:
        f.write(html)
    return os.path.getsize(path)


# ============================================================= cluster viewer

class ClusterFileMissing(Exception):
    """AllLabeledClusters.csv is what the Cluster page is built on."""
    pass


# the ten codes, their labels and colours, from DepthAnalyzer
BID_LABELS = {'c': 'bower scoop', 'p': 'bower spit', 'b': 'bower multiple',
              'f': 'feed scoop', 't': 'feed spit', 'm': 'feed multiple',
              's': 'spawn', 'd': 'drop sand', 'o': 'fish other', 'x': 'no fish other'}
BID_COLORS = {'c': '#4f7fe8', 'p': '#4f7fe8', 'b': '#4f7fe8',
              'f': '#f2a33c', 't': '#f2a33c', 'm': '#f2a33c',
              's': '#e884a8', 'd': '#5aa87a', 'o': '#5aa87a', 'x': '#8a9099'}
BID_GROUPS = [('Building', ['c', 'p', 'b'], '#4f7fe8'),
              ('Feeding', ['f', 't', 'm'], '#f2a33c'),
              ('Spawning', ['s'], '#e884a8'),
              ('Other', ['d', 'o'], '#5aa87a'),
              ('No fish', ['x'], '#8a9099')]
CONFIDENCE = 0.67


def load_clusters(fm):
    """Read AllLabeledClusters.csv. Raises if it is not there."""
    path = fm.localAllLabeledClustersFile
    if not os.path.exists(path):
        raise ClusterFileMissing('no AllLabeledClusters.csv')
    dt = pd.read_csv(path, index_col='TimeStamp', parse_dates=True)
    for col in ('X', 'Y', 'Prediction'):
        if col not in dt.columns:
            raise ClusterFileMissing('column ' + col + ' is missing from the cluster file')
    return dt.sort_index()


def in_crop(dt, video_points, buffer_px=20):
    """Which events fall inside the video crop.

    Recomputed here rather than trusting the InFrame column, because the crop
    can be redrawn after the cluster analysis ran and the point of this view is
    to see what the *current* crop would discard. Note the coordinate order:
    cluster X is the row index and Y the column, so the point is (Y, X).
    """
    from shapely.geometry import Point, Polygon
    poly = Polygon(video_points).buffer(buffer_px, join_style=2)
    return dt.apply(lambda r: poly.contains(Point(r['Y'], r['X'])), axis=1)


def pack(arr, dtype):
    """A numeric column as base64, so the browser gets a typed array."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode('ascii')


def day_photos(fm, lp, max_width=760):
    """One Pi still per day, from the video that started that day.

    The Pi camera only writes a still when a video starts, and videos are
    roughly daily, so this is the natural per-day image. They live beside the
    mp4s in Videos/, so they are fetched by name rather than by directory.
    """
    out = {}
    for movie in getattr(lp, 'movies', []):
        pic = getattr(movie, 'pic_file', None)
        if not pic:
            continue
        local = fm.localProjectDir + pic
        if not os.path.exists(local):
            fetch_optional(local.replace(fm.localMasterDir, fm.cloudMasterDir), local)
        if not os.path.exists(local):
            continue
        img = cv2.imread(local)
        if img is None:
            continue
        day = str(movie.startTime.date())
        if day in out:
            continue                      # first video of the day wins
        out[day] = {'src': encode_image(img, max_width=max_width),
                    'time': str(movie.startTime),
                    'size': [int(img.shape[1]), int(img.shape[0])],
                    'file': pic.split('/')[-1]}
    return out


def build_cluster_payload(fm, lp, dt, video_points, depth_points, transM):
    """Every event, packed, so the page can re-filter without a rebuild.

    The confidence cut, the crop test and the category grouping all happen in the
    browser. That keeps the slider live, lets the density map bin whatever is
    currently selected, and means the spatial statistics are computed on exactly
    the events on screen.
    """
    inside = in_crop(dt, video_points)
    has_clip = (dt['ClipCreated'] == 'Yes') if 'ClipCreated' in dt.columns \
        else pd.Series(True, index=dt.index)
    prob = dt['Probability'] if 'Probability' in dt.columns \
        else pd.Series(1.0, index=dt.index)

    bids = list(BID_LABELS)
    bid_index = dt['Prediction'].map({b: i for i, b in enumerate(bids)})
    bid_index = bid_index.fillna(255).astype(np.uint8)      # 255: no prediction

    # depth-space coordinates, so distances come out in cm. Cluster X is the row
    # index and Y the column, so the transform takes (Y, X).
    yy = dt['Y'].to_numpy(float)
    xx = dt['X'].to_numpy(float)
    w = transM[2][0] * yy + transM[2][1] * xx + transM[2][2]
    xd = (transM[0][0] * yy + transM[0][1] * xx + transM[0][2]) / w
    yd = (transM[1][0] * yy + transM[1][1] * xx + transM[1][2]) / w

    times = dt.index
    day0 = times.normalize().min()
    day_index = (times.normalize() - day0).days.to_numpy()

    trial_index = np.zeros(len(dt), np.uint8)
    trials = []
    for i, trial in enumerate(lp.trials, 1):
        sel = (times >= trial.startTime) & (times <= trial.stopTime)
        trial_index[sel] = i
        trials.append({'number': i, 'start': str(trial.startTime),
                       'stop': str(trial.stopTime), 'n': int(sel.sum())})

    flags = (inside.to_numpy().astype(np.uint8) |
             (has_clip.to_numpy().astype(np.uint8) << 1))

    photos = day_photos(fm, lp)

    # Cluster coordinates are in Pi camera space, which is not the depth
    # camera's lp.width/lp.height. Take the real frame size from a Pi still, and
    # fall back to the data itself if none is available.
    pi_size = None
    for p in photos.values():
        pi_size = p['size']
        break
    if pi_size is None:
        pi_size = [int(max(1296, yy.max() + 1)), int(max(972, xx.max() + 1))]

    days = []
    for d in range(int(day_index.max()) + 1 if len(dt) else 0):
        date = str((day0 + pd.Timedelta(days=d)).date())
        days.append({'index': d, 'date': date,
                     'n': int((day_index == d).sum()),
                     'trial': int(trial_index[day_index == d][0])
                              if (day_index == d).any() else 0,
                     'photo': photos.get(date)})

    events = {
        'n': int(len(dt)),
        'y': pack(np.clip(yy, 0, 65535), np.uint16),
        'x': pack(np.clip(xx, 0, 65535), np.uint16),
        'xd': pack(np.clip(xd, -32000, 32000), np.int16),
        'yd': pack(np.clip(yd, -32000, 32000), np.int16),
        'bid': pack(bid_index, np.uint8),
        'prob': pack(np.clip(prob.fillna(0) * 255, 0, 255), np.uint8),
        'flags': pack(flags, np.uint8),
        'trial': pack(trial_index, np.uint8),
        'hour': pack(times.hour.to_numpy(), np.uint8),
        'day': pack(np.clip(day_index, 0, 65535), np.uint16),
    }

    return {
        'projectID': lp.projectID, 'tankID': lp.tankID, 'analysisID': fm.analysisID,
        'piSize': pi_size, 'depthSize': [int(lp.width), int(lp.height)],
        'videoPoints': video_points, 'depthPoints': depth_points,
        'days': days,
        'labels': BID_LABELS, 'colours': BID_COLORS, 'bids': bids,
        'groups': [{'name': n, 'bids': b, 'colour': c} for n, b, c in BID_GROUPS],
        'confidence': CONFIDENCE, 'pixelLength': PIXEL_LENGTH,
        'nTotal': int(len(dt)),
        'nDays': int(day_index.max()) + 1 if len(dt) else 0,
        'trials': trials, 'events': events,
        'built': str(datetime.datetime.now().replace(microsecond=0)),
        'branch': fm.branch_name,
    }


CLUSTER_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --ink:#e8ecf1; --ink-dim:#93a0b0; --bg:#0e1116; --panel:#171c24;
          --line:#262d38; --tray:#f2a33c; --video:#6fb2e8; --warn:#e0693f; --ok:#5aa87a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 "Inter","Helvetica Neue",Arial,sans-serif; }
  .wrap { max-width:1760px; margin:0 auto; padding:24px 22px 80px; }
  h1 { font-size:24px; font-weight:600; margin:0 0 3px; }
  h2 { font-size:15px; font-weight:600; margin:26px 0 12px; }
  .sub { color:var(--ink-dim); margin:0 0 18px; }
  a { color:var(--tray); text-decoration:none; } a:hover { text-decoration:underline; }
  .back { display:inline-block; margin-bottom:12px; font-size:13px; color:var(--ink-dim); }
  .meta { display:flex; flex-wrap:wrap; gap:10px 26px; padding:14px 0 18px;
          border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
  .meta div { font-size:13px; color:var(--ink-dim); }
  .meta b { display:block; color:var(--ink); font-weight:500; font-variant-numeric:tabular-nums; }
  .tabs { display:flex; gap:4px; margin:18px 0; flex-wrap:wrap; }
  .tabs button { background:none; border:1px solid var(--line); color:var(--ink-dim);
                 padding:8px 17px; border-radius:999px; cursor:pointer; font:inherit; font-size:14px; }
  .tabs button[aria-selected="true"] { background:var(--ink); color:var(--bg); border-color:var(--ink); }
  .bar { display:flex; gap:14px; align-items:center; flex-wrap:wrap; padding:12px 0;
         border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin-bottom:16px; }
  .bar label { font-size:13px; color:var(--ink-dim); }
  .bar input[type=range] { width:170px; accent-color:var(--tray); }
  .bar b { color:var(--ink); font-variant-numeric:tabular-nums; }
  .bar button { background:var(--panel); border:1px solid var(--line); color:var(--ink);
                padding:7px 13px; border-radius:6px; font:inherit; font-size:13px; cursor:pointer; }
  .bar button[aria-pressed="true"] { background:var(--tray); color:#141414; border-color:var(--tray); }
  .spacer { flex:1; }
  .stat { font-size:13px; color:var(--ink-dim); font-variant-numeric:tabular-nums; }
  .stat b { color:var(--ink); font-weight:500; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); gap:18px; }
  figure { margin:0; background:var(--panel); border:1px solid var(--line);
           border-radius:8px; overflow:hidden; }
  figcaption { padding:10px 13px; font-size:13px; color:var(--ink-dim); border-top:1px solid var(--line); }
  figcaption b { color:var(--ink); font-weight:500; }
  .stage { position:relative; line-height:0; background:#0b0d11; }
  .stage canvas { width:100%; display:block; }
  .chart { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }
  .chart svg { width:100%; height:220px; display:block; }
  table.t { width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }
  table.t th, table.t td { text-align:right; padding:5px 9px; border-bottom:1px solid var(--line); }
  table.t th:first-child, table.t td:first-child { text-align:left; }
  table.t th { color:var(--ink-dim); font-weight:500; }
  .swatch { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; }
  .days { display:flex; gap:3px; flex-wrap:wrap; margin-bottom:14px; }
  .days button { background:var(--panel); border:1px solid var(--line); color:var(--ink-dim);
                 padding:5px 9px; border-radius:5px; font:inherit; font-size:12px; cursor:pointer;
                 font-variant-numeric:tabular-nums; }
  .days button[aria-pressed="true"] { background:var(--ink); color:var(--bg); border-color:var(--ink); }
  .days button.partial { border-style:dashed; }
  .stage img { width:100%; display:block; }
  .note { background:rgba(224,105,63,.1); border:1px solid rgba(224,105,63,.35);
          color:#f0c3ae; padding:10px 13px; border-radius:7px; font-size:13px; margin:0 0 14px; }
  .foot { margin-top:40px; padding-top:14px; border-top:1px solid var(--line);
          color:var(--ink-dim); font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <a class="back" href="../index.html">All projects in __ANALYSIS__</a>
  <h1 id="title"></h1>
  <p class="sub" id="sub"></p>
  <div class="meta" id="meta"></div>
  <div class="tabs" id="tabs" role="tablist"></div>
  <div id="body"></div>
  <p class="foot" id="foot"></p>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const PW = D.piSize[0], PH = D.piSize[1];
const CW = 620, CH = Math.round(620 * PH / PW);

function unpack(b64, Type) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Type(buf);
}
const E = {
  n: D.events.n,
  y: unpack(D.events.y, Uint16Array), x: unpack(D.events.x, Uint16Array),
  xd: unpack(D.events.xd, Int16Array), yd: unpack(D.events.yd, Int16Array),
  bid: unpack(D.events.bid, Uint8Array), prob: unpack(D.events.prob, Uint8Array),
  flags: unpack(D.events.flags, Uint8Array), trial: unpack(D.events.trial, Uint8Array),
  hour: unpack(D.events.hour, Uint8Array), day: unpack(D.events.day, Uint16Array),
};
const BID_OF = {}; D.bids.forEach((b, i) => BID_OF[b] = i);
const NO_PRED = 255;

const state = { confidence: D.confidence, density: true, requireClip: true, requireCrop: true };

function select(trial, bids) {
  const want = bids ? new Set(bids.map(b => BID_OF[b])) : null;
  const cut = Math.round(state.confidence * 255);
  const idx = [];
  for (let i = 0; i < E.n; i++) {
    if (trial && E.trial[i] !== trial) continue;
    if (E.bid[i] === NO_PRED) continue;
    if (want && !want.has(E.bid[i])) continue;
    if (E.prob[i] < cut) continue;
    if (state.requireClip && !(E.flags[i] & 2)) continue;
    if (state.requireCrop && !(E.flags[i] & 1)) continue;
    idx.push(i);
  }
  return idx;
}

function excluded(trial, kind) {
  const cut = Math.round(state.confidence * 255);
  const idx = [];
  for (let i = 0; i < E.n; i++) {
    if (trial && E.trial[i] !== trial) continue;
    if (kind === 'noClip') { if (!(E.flags[i] & 2)) idx.push(i); continue; }
    if (kind === 'outside') { if (!(E.flags[i] & 1)) idx.push(i); continue; }
    if (kind === 'lowConf') {
      if (E.bid[i] !== NO_PRED && E.prob[i] < cut && (E.flags[i] & 3) === 3) idx.push(i);
      continue;
    }
    if (kind === 'noPred') { if (E.bid[i] === NO_PRED) idx.push(i); continue; }
  }
  return idx;
}

// ------------------------------------------------------------ spatial spread
function stats(idx) {
  const n = idx.length;
  if (n < 3) return { n: n };
  const L = D.pixelLength;
  let mx = 0, my = 0;
  for (const i of idx) { mx += E.xd[i]; my += E.yd[i]; }
  mx /= n; my /= n;
  let sxx = 0, syy = 0, sxy = 0;
  for (const i of idx) {
    const dx = E.xd[i] - mx, dy = E.yd[i] - my;
    sxx += dx*dx; syy += dy*dy; sxy += dx*dy;
  }
  sxx /= n; syy /= n; sxy /= n;
  const sd = Math.sqrt(sxx + syy) * L;              // RMS distance from the centroid
  const tr = sxx + syy, det = sxx*syy - sxy*sxy;    // dispersion ellipse
  const root = Math.sqrt(Math.max(0, tr*tr/4 - det));
  const major = Math.sqrt(Math.max(0, tr/2 + root)) * L;
  const minor = Math.sqrt(Math.max(0, tr/2 - root)) * L;
  const angle = 0.5 * Math.atan2(2*sxy, sxx - syy) * 180 / Math.PI;
  return { n, cx: mx*L, cy: my*L, sd, major, minor, angle,
           anisotropy: major > 0 ? minor/major : 0 };
}

function separation(a, b) {
  if (!a.sd || !b.sd) return null;
  const d = Math.hypot(a.cx - b.cx, a.cy - b.cy);
  return { distance: d, index: d / ((a.sd + b.sd) / 2) };
}

// ------------------------------------------------------------------ drawing
function panel(sets, caption, opts) {
  opts = opts || {};
  if (!Array.isArray(sets)) sets = [{ idx: sets, colour: opts.colour || '#f2a33c' }];
  const idx = sets.length === 1 ? sets[0].idx : [].concat(...sets.map(s => s.idx));
  const fig = document.createElement('figure');
  fig.innerHTML = '<div class="stage"><canvas width="' + CW + '" height="' + CH +
                  '"></canvas></div><figcaption>' + caption + '</figcaption>';
  const canvas = fig.querySelector('canvas'), ctx = canvas.getContext('2d');
  const sx = CW / PW, sy = CH / PH;

  ctx.fillStyle = '#0b0d11'; ctx.fillRect(0, 0, CW, CH);
  if (opts.background) {
    const im = new Image();
    im.onload = () => {
      ctx.globalAlpha = 0.45;
      ctx.drawImage(im, 0, 0, CW, CH);
      ctx.globalAlpha = 1;
      paintPoints();
    };
    im.src = opts.background;
    var deferred = true;
  }
  ctx.save();
  ctx.strokeStyle = 'rgba(111,178,232,.7)'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  D.videoPoints.forEach((p, i) =>
    i ? ctx.lineTo(p[0]*sx, p[1]*sy) : ctx.moveTo(p[0]*sx, p[1]*sy));
  ctx.closePath(); ctx.stroke(); ctx.restore();

  function paintPoints() {
  sets.forEach(set => paintSet(set.idx, set.colour));
  }

  function paintSet(idx, colour) {
  if (state.density && idx.length) {
    // binning keeps this constant-time however many events there are, which
    // matters once a trial runs to hundreds of thousands
    const BX = 124, BY = Math.max(1, Math.round(124 * PH / PW));
    const bins = new Float32Array(BX * BY);
    for (const i of idx) {
      const bx = Math.min(BX-1, Math.floor(E.y[i] / PW * BX));
      const by = Math.min(BY-1, Math.floor(E.x[i] / PH * BY));
      if (bx >= 0 && by >= 0) bins[by*BX + bx]++;
    }
    const sm = new Float32Array(bins.length);
    for (let by = 0; by < BY; by++) for (let bx = 0; bx < BX; bx++) {
      let s = 0, c = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const ny = by+dy, nx = bx+dx;
        if (ny < 0 || nx < 0 || ny >= BY || nx >= BX) continue;
        s += bins[ny*BX + nx]; c++;
      }
      sm[by*BX + bx] = s / c;
    }
    let max = 0;
    for (let i = 0; i < sm.length; i++) if (sm[i] > max) max = sm[i];
    if (max > 0) {
      const cw = CW/BX, ch = CH/BY;
      ctx.fillStyle = colour;
      for (let by = 0; by < BY; by++) for (let bx = 0; bx < BX; bx++) {
        const v = sm[by*BX + bx];
        if (v <= 0) continue;
        // log scale: a bower core is orders of magnitude denser than its edge
        ctx.globalAlpha = Math.min(1, 0.08 + 0.92 * Math.log1p(v) / Math.log1p(max));
        ctx.fillRect(bx*cw, by*ch, cw + 0.6, ch + 0.6);
      }
      ctx.globalAlpha = 1;
    }
  } else {
    ctx.fillStyle = colour;
    ctx.globalAlpha = idx.length > 20000 ? 0.15 : (idx.length > 4000 ? 0.3 : 0.55);
    const r = idx.length > 20000 ? 0.9 : 1.6;
    for (const i of idx) {
      ctx.beginPath(); ctx.arc(E.y[i]*sx, E.x[i]*sy, r, 0, 6.2832); ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  }
  if (!opts.background) paintPoints();

  (opts.ellipses || []).forEach(e => drawEllipse(e.stats, e.idx));

  function drawEllipse(st, idx) {
    if (!st || st.n < 3 || !idx.length) return;
    const L = D.pixelLength;
    let mx = 0, my = 0;
    for (const i of idx) { mx += E.y[i]; my += E.x[i]; }
    mx /= idx.length; my /= idx.length;
    ctx.save();
    ctx.translate(mx*sx, my*sy);
    ctx.rotate(-st.angle * Math.PI / 180);
    ctx.strokeStyle = '#fff'; ctx.globalAlpha = 0.8; ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.ellipse(0, 0, (st.major/L)*sx, (st.minor/L)*sy, 0, 0, 6.2832);
    ctx.stroke();
    ctx.restore();
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(mx*sx, my*sy, 3, 0, 6.2832); ctx.fill();
  }
  return fig;
}

function legendOf(sets) {
  return sets.map(s => '<span class="swatch" style="background:' + s.colour + '"></span>' +
                       s.name + ' ' + s.idx.length).join(' &nbsp; ');
}

function statTable(rows) {
  const t = document.createElement('table');
  t.className = 't';
  t.innerHTML = '<tr><th>set</th><th>events</th><th>spread cm</th><th>major cm</th>' +
    '<th>minor cm</th><th>minor/major</th><th>angle</th></tr>' +
    rows.map(([name, s]) => s.n >= 3
      ? '<tr><td>' + name + '</td><td>' + s.n + '</td><td>' + s.sd.toFixed(2) + '</td><td>' +
        s.major.toFixed(2) + '</td><td>' + s.minor.toFixed(2) + '</td><td>' +
        s.anisotropy.toFixed(2) + '</td><td>' + s.angle.toFixed(0) + '\u00b0</td></tr>'
      : '<tr><td>' + name + '</td><td>' + s.n +
        '</td><td colspan="5" style="text-align:left;color:#93a0b0">too few events</td></tr>')
      .join('');
  return t;
}

function renderTrial(tnum) {
  const box = document.createElement('div');
  const bar = document.createElement('div');
  bar.className = 'bar';
  bar.innerHTML =
    '<label>Confidence \u2265 <b id="cv">' + state.confidence.toFixed(2) + '</b></label>' +
    '<input type="range" id="conf" min="0" max="0.99" step="0.01" value="' +
      state.confidence + '">' +
    '<button id="mode" aria-pressed="' + state.density + '">' +
      (state.density ? 'Density' : 'Scatter') + '</button>' +
    '<button id="clip" aria-pressed="' + state.requireClip + '">Clip required</button>' +
    '<button id="crop" aria-pressed="' + state.requireCrop + '">Inside crop</button>' +
    '<span class="spacer"></span><span class="stat" id="kept"></span>';
  box.appendChild(bar);

  const row1 = document.createElement('div'); row1.className = 'grid';
  const h2 = document.createElement('h2'); h2.textContent = 'What was set aside';
  const row2 = document.createElement('div'); row2.className = 'grid';
  const h3 = document.createElement('h2'); h3.textContent = 'Spatial spread';
  const tables = document.createElement('div');
  box.appendChild(row1);
  box.appendChild(h2); box.appendChild(row2);
  box.appendChild(h3); box.appendChild(tables);

  // spits blue, scoops orange, multiple green — the same three for feeding
  const BLUE = '#6fb2e8', ORANGE = '#f2a33c', GREEN = '#5aa87a';
  const RED = '#e0693f';

  function draw() {
    row1.textContent = ''; row2.textContent = ''; tables.textContent = '';

    const spit = select(tnum, ['p']), scoop = select(tnum, ['c']), multi = select(tnum, ['b']);
    const sP = stats(spit), sS = stats(scoop);
    const building = [{ name: 'spit', colour: BLUE, idx: spit },
                      { name: 'scoop', colour: ORANGE, idx: scoop },
                      { name: 'multiple', colour: GREEN, idx: multi }];
    row1.appendChild(panel(building,
      '<b>Building</b> \u2014 ' + (spit.length + scoop.length + multi.length) + ' events<br>' +
      legendOf(building) + '<br>Ellipses mark the spread of spits and scoops.',
      { ellipses: [{ stats: sP, idx: spit }, { stats: sS, idx: scoop }] }));

    const feeding = [{ name: 'feed spit', colour: BLUE, idx: select(tnum, ['t']) },
                     { name: 'feed scoop', colour: ORANGE, idx: select(tnum, ['f']) },
                     { name: 'feed multiple', colour: GREEN, idx: select(tnum, ['m']) }];
    row1.appendChild(panel(feeding,
      '<b>Feeding</b> \u2014 ' + feeding.reduce((n, s) => n + s.idx.length, 0) +
      ' events<br>' + legendOf(feeding)));

    const other = [{ name: 'spawn', colour: RED, idx: select(tnum, ['s']) },
                   { name: 'other', colour: BLUE, idx: select(tnum, ['d', 'o']) }];
    row1.appendChild(panel(other,
      '<b>Spawning and other</b> \u2014 ' + other.reduce((n, s) => n + s.idx.length, 0) +
      ' events<br>' + legendOf(other)));

    [['noClip', 'No clip created', 'at the frame border, so no clip could be cut'],
     ['outside', 'Cropped out', 'outside the video crop as it stands now'],
     ['lowConf', 'Low confidence', 'classified, but under ' + state.confidence.toFixed(2)]
    ].forEach(([kind, name, why]) => {
      const idx = excluded(tnum, kind);
      row2.appendChild(panel([{ name: name, colour: RED, idx: idx }],
        '<b>' + name + '</b> \u2014 ' + idx.length + '<br>' + why));
    });

    tables.appendChild(statTable([
      ['bower spit (p)', sP], ['bower scoop (c)', sS],
      ['bower, all (c p b)', stats(select(tnum, ['c','p','b']))],
      ['feeding (f t m)', stats(select(tnum, ['f','t','m']))],
      ['spawning (s)', stats(select(tnum, ['s']))],
    ]));
    const sep = separation(sS, sP);
    const p = document.createElement('p');
    p.className = 'stat'; p.style.marginTop = '8px';
    p.innerHTML = sep
      ? 'Spits and scoops sit <b>' + sep.distance.toFixed(2) + ' cm</b> apart at the centroid, ' +
        '<b>' + sep.index.toFixed(2) + '</b> times their mean spread. Above about 1 they occupy ' +
        'distinguishable places; near 0 they are intermixed. Spread is the RMS distance from the ' +
        'centroid; major and minor are the axes of the dispersion ellipse, so minor/major near 1 ' +
        'is a circular scatter and near 0 is a line. Distances are in tray centimetres.'
      : 'Not enough spits or scoops at this confidence to compare.';
    tables.appendChild(p);

    const h4 = document.createElement('h2');
    h4.textContent = 'Counts by behaviour';
    tables.appendChild(h4);
    const ct = document.createElement('table');
    ct.className = 't';
    ct.innerHTML = '<tr><th>behaviour</th><th>code</th><th>events used</th></tr>' +
      D.bids.map(b => '<tr><td>' + D.labels[b] + '</td><td>' + b + '</td><td>' +
        select(tnum, [b]).length + '</td></tr>').join('') +
      '<tr><th>set aside</th><th></th><th></th></tr>' +
      [['no clip created', 'noClip'], ['cropped out', 'outside'],
       ['low confidence', 'lowConf'], ['no prediction', 'noPred']]
      .map(([n, k]) => '<tr><td>' + n + '</td><td></td><td>' +
        excluded(tnum, k).length + '</td></tr>').join('');
    tables.appendChild(ct);

    let kept = 0;
    D.bids.forEach(b => { kept += select(tnum, [b]).length; });
    const total = (D.trials.find(t => t.number === tnum) || { n: 0 }).n;
    bar.querySelector('#kept').innerHTML = '<b>' + kept + '</b> of ' + total + ' used';
  }

  bar.querySelector('#conf').addEventListener('input', e => {
    state.confidence = parseFloat(e.target.value);
    bar.querySelector('#cv').textContent = state.confidence.toFixed(2);
    draw();
  });
  const modeBtn = bar.querySelector('#mode');
  modeBtn.addEventListener('click', () => {
    state.density = !state.density;
    modeBtn.textContent = state.density ? 'Density' : 'Scatter';
    modeBtn.setAttribute('aria-pressed', String(state.density));
    draw();
  });
  [['clip', 'requireClip'], ['crop', 'requireCrop']].forEach(([id, key]) => {
    const b = bar.querySelector('#' + id);
    b.addEventListener('click', () => {
      state[key] = !state[key];
      b.setAttribute('aria-pressed', String(state[key]));
      draw();
    });
  });
  draw();
  return box;
}

function renderDays() {
  const box = document.createElement('div');
  const bar = document.createElement('div');
  bar.className = 'bar';
  bar.innerHTML =
    '<label>Confidence \u2265 <b id="cv2">' + state.confidence.toFixed(2) + '</b></label>' +
    '<input type="range" id="conf2" min="0" max="0.99" step="0.01" value="' +
      state.confidence + '">' +
    '<button id="mode2" aria-pressed="' + state.density + '">' +
      (state.density ? 'Density' : 'Scatter') + '</button>' +
    '<span class="spacer"></span><span class="stat" id="dayInfo"></span>';
  box.appendChild(bar);

  const strip = document.createElement('div');
  strip.className = 'days';
  box.appendChild(strip);
  const grid = document.createElement('div');
  grid.className = 'grid';
  box.appendChild(grid);

  let selected = (D.days.find(d => d.n > 0) || D.days[0] || {}).index || 0;

  function selectDay(day, bids) {
    const want = bids ? new Set(bids.map(b => BID_OF[b])) : null;
    const cut = Math.round(state.confidence * 255);
    const idx = [];
    for (let i = 0; i < E.n; i++) {
      if (E.day[i] !== day) continue;
      if (E.bid[i] === NO_PRED) continue;
      if (want && !want.has(E.bid[i])) continue;
      if (E.prob[i] < cut) continue;
      if (state.requireClip && !(E.flags[i] & 2)) continue;
      if (state.requireCrop && !(E.flags[i] & 1)) continue;
      idx.push(i);
    }
    return idx;
  }

  function draw() {
    strip.textContent = '';
    D.days.forEach(d => {
      const b = document.createElement('button');
      b.textContent = d.date.slice(5);
      b.title = d.n + ' clusters, trial ' + d.trial + (d.photo ? '' : ', no still');
      b.setAttribute('aria-pressed', String(d.index === selected));
      if (!d.photo) b.className = 'partial';
      b.addEventListener('click', () => { selected = d.index; draw(); });
      strip.appendChild(b);
    });

    const d = D.days[selected] || {};
    grid.textContent = '';
    const all = selectDay(selected, null);
    bar.querySelector('#dayInfo').innerHTML = d.date + ' \u00b7 trial ' + d.trial +
      ' \u00b7 <b>' + all.length + '</b> events used of ' + d.n + ' detected';

    if (d.photo) {
      const fig = document.createElement('figure');
      fig.innerHTML = '<div class="stage"><img src="' + d.photo.src + '" alt=""></div>' +
        '<figcaption><b>Pi camera</b> \u2014 ' + d.photo.time.slice(11, 16) + ', ' +
        d.photo.file + '</figcaption>';
      grid.appendChild(fig);
      grid.appendChild(panel([{ name: 'all', colour: '#f2a33c', idx: all }],
        '<b>All events this day</b> \u2014 ' + all.length + ', over the still',
        { background: d.photo.src }));
    } else {
      grid.appendChild(panel([{ name: 'all', colour: '#f2a33c', idx: all }],
        '<b>All events this day</b> \u2014 ' + all.length + '<br>No Pi still for this date.'));
    }
    D.groups.forEach(g => {
      const idx = selectDay(selected, g.bids);
      grid.appendChild(panel([{ name: g.name, colour: g.colour, idx: idx }],
        '<b>' + g.name + '</b> \u2014 ' + idx.length + ' events',
        d.photo ? { background: d.photo.src } : {}));
    });
  }

  bar.querySelector('#conf2').addEventListener('input', e => {
    state.confidence = parseFloat(e.target.value);
    bar.querySelector('#cv2').textContent = state.confidence.toFixed(2);
    draw();
  });
  const mb = bar.querySelector('#mode2');
  mb.addEventListener('click', () => {
    state.density = !state.density;
    mb.textContent = state.density ? 'Density' : 'Scatter';
    mb.setAttribute('aria-pressed', String(state.density));
    draw();
  });
  draw();
  return box;
}

function renderOverview() {
  const box = document.createElement('div');
  const t = document.createElement('table');
  t.className = 't';
  t.innerHTML = '<tr><th>trial</th><th>detected</th><th>used</th>' +
    D.groups.map(g => '<th>' + g.name + '</th>').join('') + '</tr>' +
    D.trials.map(tr => '<tr><td>Trial ' + tr.number + '</td><td>' + tr.n + '</td><td>' +
      select(tr.number, null).length + '</td>' +
      D.groups.map(g => '<td>' + select(tr.number, g.bids).length + '</td>').join('') +
      '</tr>').join('');
  box.appendChild(t);
  const p = document.createElement('p');
  p.className = 'sub'; p.style.marginTop = '14px';
  p.textContent = 'Counts use the filters set on a trial tab, currently confidence \u2265 ' +
    state.confidence.toFixed(2) + '.';
  box.appendChild(p);
  return box;
}

function build() {
  document.getElementById('title').textContent = D.projectID + ' \u2014 clusters';
  document.getElementById('sub').textContent =
    'Sand manipulation events by category, with the ones the analysis sets aside shown ' +
    'separately. Every event is in the page, so the filters re-apply live.';
  document.getElementById('meta').innerHTML =
    [['Tank', D.tankID], ['Analysis', D.analysisID], ['Clusters', D.nTotal],
     ['Trials', D.trials.length], ['Days', D.nDays],
     ['Pixel size', D.pixelLength.toFixed(4) + ' cm']]
    .map(([k, v]) => '<div>' + k + '<b>' + v + '</b></div>').join('');

  const views = [['Overview', renderOverview], ['By day', renderDays]].concat(
    D.trials.map(tr => ['Trial ' + tr.number, () => renderTrial(tr.number)]));
  const tabs = document.getElementById('tabs'), body = document.getElementById('body');
  const buttons = views.map((v, i) => {
    const b = document.createElement('button');
    b.textContent = v[0]; b.setAttribute('role', 'tab');
    b.addEventListener('click', () => show(i));
    tabs.appendChild(b);
    return b;
  });
  function show(i) {
    buttons.forEach((b, j) => b.setAttribute('aria-selected', String(j === i)));
    body.textContent = '';
    body.appendChild(views[i][1]());     // rebuilt each time, so filter state is current
  }
  show(0);
  document.getElementById('foot').textContent =
    'Built ' + D.built + ' from branch ' + D.branch + '.';
}
build();
</script>
</body>
</html>
"""


def write_cluster(path, analysis_id, payload):
    html = CLUSTER_PAGE.replace('__TITLE__', payload['projectID'] + ' \u00b7 Clusters')
    html = html.replace('__ANALYSIS__', analysis_id)
    html = html.replace('__PAYLOAD__', json.dumps(payload).replace('</', '<\\/'))
    with open(path, 'w') as f:
        f.write(html)
    return os.path.getsize(path)


# ------------------------------------------------------------------------ main

def fetch_optional(cloud_path, local_path, directory=False):
    """Fetch something that may not be there yet.

    FileManager.downloadData runs `rclone lsf` on the parent directory first and
    drops into pdb.set_trace() if that fails, and allow_errors does not cover it
    — so a path whose parent does not exist in the cloud yet hangs at a debugger
    prompt. rclone copy just returns non-zero."""
    cmd = ['rclone', 'copy' if directory else 'copyto', cloud_path, local_path]
    out = subprocess.run(cmd, capture_output=True, encoding='utf-8')
    return out.returncode == 0 and os.path.exists(local_path)


def upload(fm_obj, path, quiet=False):
    """Upload one file and say where it went. uploadData raises on failure, so
    anything printed here actually landed."""
    fm_obj.uploadData(path)
    if not quiet:
        cloud = path.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir)
        print('    uploaded -> ' + cloud)


def build_register_payload(fm, lp, pairs):
    """Full-resolution stills for every corrected pair, plus the current crops."""
    out_pairs = []
    for p in pairs:
        depth_img = cv2.imread(p['depth_jpg'])
        pi_img = cv2.imread(p['pi_jpg'])
        if depth_img is None or pi_img is None:
            continue
        out_pairs.append({
            'key': p['key'], 'label': p['label'], 'gapMinutes': p['gapMinutes'],
            'piTime': p['piTime'], 'depthTime': p['depthTime'],
            'depthFile': os.path.basename(p['depth_jpg']),
            'piFile': os.path.basename(p['pi_jpg']),
            'depth': encode_image(depth_img, max_width=depth_img.shape[1]),
            'pi': encode_image(pi_img, max_width=min(pi_img.shape[1], 1296)),
            'depthSize': [int(depth_img.shape[1]), int(depth_img.shape[0])],
            'piSize': [int(pi_img.shape[1]), int(pi_img.shape[0])],
        })
    if not out_pairs:
        raise PrepFiles2Missing('no readable PrepFiles2 images')

    current = {}
    if os.path.exists(fm.localTransMFile):
        current['transM'] = np.load(fm.localTransMFile).tolist()
    if os.path.exists(fm.localDepthCropFile):
        current['depthPoints'] = parse_points(fm.localDepthCropFile)
    if os.path.exists(fm.localVideoCropFile):
        current['videoPoints'] = parse_points(fm.localVideoCropFile)

    return {
        'schema': 'cichlid-registration/1',
        'projectID': lp.projectID, 'tankID': lp.tankID, 'analysisID': fm.analysisID,
        'frameSize': [lp.width, lp.height],
        'pairs': out_pairs, 'current': current,
        'built': str(datetime.datetime.now().replace(microsecond=0)),
    }


REGISTER_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --ink:#e8ecf1; --ink-dim:#93a0b0; --bg:#0e1116; --panel:#171c24;
          --line:#262d38; --tray:#f2a33c; --video:#6fb2e8; --warn:#e0693f; --ok:#5aa87a; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 "Inter","Helvetica Neue",Arial,sans-serif; }
  .wrap { max-width:1760px; margin:0 auto; padding:24px 22px 80px; }
  h1 { font-size:24px; font-weight:600; margin:0 0 3px; }
  h2 { font-size:15px; font-weight:600; margin:26px 0 10px; }
  .sub { color:var(--ink-dim); margin:0 0 16px; }
  a { color:var(--tray); text-decoration:none; }
  a:hover { text-decoration:underline; }
  .back { display:inline-block; margin-bottom:12px; font-size:13px; color:var(--ink-dim); }
  .mtabs { display:flex; gap:4px; margin:14px 0 16px; }
  .mtabs button { background:none; border:1px solid var(--line); color:var(--ink-dim);
                  padding:9px 20px; border-radius:999px; cursor:pointer; font:inherit; font-size:14px; }
  .mtabs button[aria-selected="true"] { background:var(--ink); color:var(--bg); border-color:var(--ink); }
  .bar { display:flex; gap:13px; align-items:center; flex-wrap:wrap; padding:13px 0;
         border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin-bottom:16px; }
  .bar label { font-size:13px; color:var(--ink-dim); }
  .bar select, .save input { background:var(--panel); border:1px solid var(--line); color:var(--ink);
            padding:8px 11px; border-radius:6px; font:inherit; font-size:14px; }
  .bar button, .save button { background:var(--panel); border:1px solid var(--line); color:var(--ink);
            padding:8px 14px; border-radius:6px; font:inherit; font-size:14px; cursor:pointer; }
  .bar button:hover, .save button:hover:enabled { border-color:var(--tray); }
  .bar button[aria-pressed="true"] { background:var(--tray); color:#141414; border-color:var(--tray); }
  .spacer { flex:1; }
  .stat { font-size:13px; color:var(--ink-dim); font-variant-numeric:tabular-nums; }
  .stat b { color:var(--ink); font-weight:500; }
  .stat.good b { color:var(--ok); } .stat.bad b { color:var(--warn); }
  .panes { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  .pane { background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
  .pane header { padding:9px 13px; font-size:13px; color:var(--ink-dim);
                 border-bottom:1px solid var(--line); display:flex; justify-content:space-between;
                 gap:10px; align-items:center; }
  .pane header .tools { display:flex; gap:5px; }
  .pane header button { background:none; border:1px solid var(--line); color:var(--ink-dim);
                        padding:3px 9px; border-radius:5px; font:inherit; font-size:12px; cursor:pointer; }
  .pane header button[aria-pressed="true"] { background:var(--ink); color:var(--bg); border-color:var(--ink); }
  .canvas-host { position:relative; line-height:0; background:#000; cursor:crosshair; }
  .canvas-host img { width:100%; display:block; }
  .canvas-host svg { position:absolute; inset:0; width:100%; height:100%; }
  .loupe { position:absolute; width:132px; height:132px; border-radius:50%; border:2px solid var(--ink);
           pointer-events:none; display:none; background-repeat:no-repeat; z-index:5;
           box-shadow:0 3px 14px rgba(0,0,0,.6); }
  .loupe::after { content:""; position:absolute; inset:0; background:
      linear-gradient(var(--tray),var(--tray)) center/1px 100% no-repeat,
      linear-gradient(var(--tray),var(--tray)) center/100% 1px no-repeat; opacity:.75; }
  table { width:100%; border-collapse:collapse; font-size:13px; font-variant-numeric:tabular-nums; }
  th, td { text-align:left; padding:5px 9px; border-bottom:1px solid var(--line); }
  th { color:var(--ink-dim); font-weight:500; }
  td button { background:none; border:none; color:var(--warn); cursor:pointer; font:inherit; }
  .swatch { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:7px; }
  .preview { position:relative; overflow:hidden; background:#000; line-height:0; }
  .preview .base { width:100%; display:block; }
  .preview .over { position:absolute; top:0; left:0; bottom:0; overflow:hidden; }
  .preview .over .warpbox { position:absolute; top:0; left:0; transform-origin:0 0; }
  .preview .over img { display:block; max-width:none; }
  .preview svg { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
  .preview .handle { position:absolute; top:0; bottom:0; width:2px; background:var(--ink);
                     pointer-events:none; }
  .save { display:flex; gap:11px; align-items:center; flex-wrap:wrap; margin-top:20px;
          padding-top:16px; border-top:1px solid var(--line); }
  .save input#initials { width:110px; text-transform:uppercase; }
  .save input#note { flex:1; min-width:220px; }
  .save button:disabled { opacity:.42; cursor:not-allowed; }
  .msg { padding:11px 14px; border-radius:7px; font-size:14px; margin:14px 0 0; }
  .msg.info { background:rgba(242,163,60,.1); border:1px solid rgba(242,163,60,.35); color:#f0d4ab; }
  .msg.done { background:rgba(90,168,122,.12); border:1px solid rgba(90,168,122,.4); color:#a9dcc0; }
  polygon.crop { fill:none; stroke:var(--tray); stroke-width:2; vector-effect:non-scaling-stroke; }
  polygon.crop.video { stroke:var(--video); }
  circle.corner { fill:var(--tray); stroke:#141414; stroke-width:1.5; cursor:grab; }
  circle.corner.video { fill:var(--video); }
  @media (max-width:920px) { .panes { grid-template-columns:1fr; } }
</style>
</head>
<body>
<div class="wrap">
  <a class="back" href="../index.html">All projects in __ANALYSIS__</a>
  <h1 id="title"></h1>
  <p class="sub" id="sub"></p>

  <div class="mtabs" role="tablist">
    <button id="tabPoints" role="tab" aria-selected="true">Match points</button>
    <button id="tabCrop" role="tab" aria-selected="false">Adjust tray crop</button>
  </div>

  <!-- ------------------------------------------------------------ points -->
  <section id="viewPoints">
    <div class="bar">
      <label for="pickPair">Pick points on</label>
      <select id="pickPair"></select>
      <button id="undo">Remove last pair</button>
      <button id="clear">Clear all pairs</button>
      <span class="spacer"></span>
      <span class="stat" id="npairs"></span>
      <span class="stat" id="rms"></span>
    </div>
    <div class="panes">
      <div class="pane">
        <header><span>Depth camera</span><span id="depthName"></span></header>
        <div class="canvas-host" id="depthHost">
          <img id="depthImg" alt="Depth camera still">
          <svg id="depthSvg" preserveAspectRatio="none"></svg>
          <div class="loupe" id="depthLoupe"></div>
        </div>
      </div>
      <div class="pane">
        <header><span>Pi camera</span><span id="piName"></span></header>
        <div class="canvas-host" id="piHost">
          <img id="piImg" alt="Pi camera still">
          <svg id="piSvg" preserveAspectRatio="none"></svg>
          <div class="loupe" id="piLoupe"></div>
        </div>
      </div>
    </div>

    <h2>Result</h2>
    <div class="bar">
      <label for="viewPair">Check the fit against</label>
      <select id="viewPair"></select>
      <span class="spacer"></span>
      <span class="stat" id="fitNote"></span>
    </div>
    <div class="panes">
      <div class="pane">
        <header><span>Registration preview</span><span>move the pointer to wipe</span></header>
        <div class="preview" id="preview">
          <img class="base" id="previewBase" alt="">
          <div class="over" id="previewOver">
            <div class="warpbox" id="warpBox"><img id="warpImg" alt=""></div>
          </div>
          <div class="handle" id="previewHandle"></div>
          <svg id="previewSvg" preserveAspectRatio="none"></svg>
        </div>
      </div>
      <div class="pane">
        <header><span>Point pairs</span><span id="pairCount"></span></header>
        <div style="max-height:340px;overflow:auto"><table id="table"><tbody></tbody></table></div>
      </div>
    </div>
  </section>

  <!-- -------------------------------------------------------------- crop -->
  <section id="viewCrop" hidden>
    <div class="bar">
      <label for="cropPair">Show</label>
      <select id="cropPair"></select>
      <span class="spacer"></span>
      <span class="stat">The two crops are independent — the depth camera does not
        always see the whole video field of view.</span>
    </div>
    <div class="panes">
      <div class="pane">
        <header>
          <span>Tray crop, depth camera</span>
          <span class="tools">
            <button data-mode="drag" data-which="depth" aria-pressed="true">Drag corners</button>
            <button data-mode="place" data-which="depth" aria-pressed="false">Place 4 new</button>
            <button data-reset="depth">Reset</button>
          </span>
        </header>
        <div class="canvas-host" id="cropDepthHost">
          <img id="cropDepthImg" alt="Depth camera still">
          <svg id="cropDepthSvg" preserveAspectRatio="none"></svg>
          <div class="loupe" id="cropDepthLoupe"></div>
        </div>
      </div>
      <div class="pane">
        <header>
          <span>Video crop, Pi camera</span>
          <span class="tools">
            <button data-mode="drag" data-which="video" aria-pressed="true">Drag corners</button>
            <button data-mode="place" data-which="video" aria-pressed="false">Place 4 new</button>
            <button data-reset="video">Reset</button>
          </span>
        </header>
        <div class="canvas-host" id="cropVideoHost">
          <img id="cropVideoImg" alt="Pi camera still">
          <svg id="cropVideoSvg" preserveAspectRatio="none"></svg>
          <div class="loupe" id="cropVideoLoupe"></div>
        </div>
      </div>
    </div>
  </section>

  <div class="save">
    <input id="initials" maxlength="4" placeholder="Initials" aria-label="Your initials">
    <input id="note" placeholder="Optional note (what was wrong with the old registration?)">
    <button id="download" disabled>Save registration file</button>
    <span class="stat" id="saveHint"></span>
  </div>
  <div id="msg"></div>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);
const COLORS = ['#f2a33c','#5aa87a','#6fb2e8','#e0693f','#c48ce0','#e8d45a',
                '#57c9c1','#e884a8','#9ad45a','#b0916a','#7f8ce0','#d9d9d9'];

let pairs = [];      // {depth:[x,y], pi:[x,y]} in native pixel coords
let pending = null;
let H = null, residuals = [], rms = null;
let pickPair = null, viewPair = null, cropPair = null;
const ORIGINAL = {
  depth: (D.current.depthPoints || [[80,60],[560,60],[560,420],[80,420]]).map(p => p.slice()),
  video: (D.current.videoPoints || [[200,150],[1100,150],[1100,860],[200,860]]).map(p => p.slice()),
};
let crops = { depth: ORIGINAL.depth.map(p => p.slice()),
              video: ORIGINAL.video.map(p => p.slice()) };
let cropMode = { depth: 'drag', video: 'drag' };
let placing = { depth: [], video: [] };

// ---- homography from N correspondences (normalized DLT, h33 fixed to 1) ------
function normalize(pts) {
  const n = pts.length;
  let cx = 0, cy = 0;
  pts.forEach(p => { cx += p[0]; cy += p[1]; });
  cx /= n; cy /= n;
  let d = 0;
  pts.forEach(p => { d += Math.hypot(p[0]-cx, p[1]-cy); });
  d /= n;
  const s = d > 1e-9 ? Math.SQRT2 / d : 1;
  return { T: [[s,0,-s*cx],[0,s,-s*cy],[0,0,1]],
           pts: pts.map(p => [(p[0]-cx)*s, (p[1]-cy)*s]) };
}
function solveLin(A, b) {
  const n = b.length;
  const M = A.map((row, i) => row.concat([b[i]]));
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c+1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-12) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / M[i][i]);
}
function mul(A, B) {
  return A.map((row, i) => B[0].map((_, j) => row.reduce((s, v, k) => s + v * B[k][j], 0)));
}
function homography(src, dst) {
  if (src.length < 4) return null;
  const S = normalize(src), Dn = normalize(dst);
  const A = [], b = [];
  for (let i = 0; i < src.length; i++) {
    const [x, y] = S.pts[i], [u, v] = Dn.pts[i];
    A.push([x, y, 1, 0, 0, 0, -u*x, -u*y]); b.push(u);
    A.push([0, 0, 0, x, y, 1, -v*x, -v*y]); b.push(v);
  }
  const n = 8, ATA = Array.from({length:n}, () => new Array(n).fill(0)), ATb = new Array(n).fill(0);
  for (let r = 0; r < A.length; r++)
    for (let i = 0; i < n; i++) {
      ATb[i] += A[r][i] * b[r];
      for (let j = 0; j < n; j++) ATA[i][j] += A[r][i] * A[r][j];
    }
  const h = solveLin(ATA, ATb);
  if (!h) return null;
  const Hn = [[h[0],h[1],h[2]],[h[3],h[4],h[5]],[h[6],h[7],1]];
  const inv = m => { const s = m[0][0];
    return [[1/s,0,-m[0][2]/s],[0,1/s,-m[1][2]/s],[0,0,1]]; };
  return mul(mul(inv(Dn.T), Hn), S.T);
}
function apply(M, p) {
  const w = M[2][0]*p[0] + M[2][1]*p[1] + M[2][2];
  return [(M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / w,
          (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / w];
}

// -------------------------------------------------------------------- shared
function hostPoint(host, ev, size) {
  const img = host.querySelector('img');
  const r = img.getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width * size[0],
          (ev.clientY - r.top) / r.height * size[1]];
}
function attachLoupe(host, loupe, getSrc) {
  const R = 66, OFFSET = 74, ZOOM = 5;
  host.addEventListener('mousemove', ev => {
    const img = host.querySelector('img');
    const r = img.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    loupe.style.display = 'block';
    // Sit above the cursor, but flip below near the top edge or the pane clips it.
    const above = y > 2 * R + OFFSET;
    loupe.style.left = Math.min(Math.max(x - R, 4), r.width - 2 * R - 4) + 'px';
    loupe.style.top = (above ? y - 2 * R - OFFSET : y + OFFSET) + 'px';
    loupe.style.backgroundImage = 'url(' + getSrc() + ')';
    loupe.style.backgroundSize = (r.width * ZOOM) + 'px ' + (r.height * ZOOM) + 'px';
    // The point under the cursor sits at (x*ZOOM, y*ZOOM) in the scaled image and
    // has to land at the loupe's centre. That depends only on the cursor, not on
    // where the loupe was placed — deriving it from left/top instead puts the
    // magnified view out by the flip offset.
    loupe.style.backgroundPosition =
      (R - x * ZOOM) + 'px ' + (R - y * ZOOM) + 'px';
  });
  host.addEventListener('mouseleave', () => { loupe.style.display = 'none'; });
}
function markerSVG(pts, size, extra) {
  const r = Math.max(4, size[0] / 150);
  let s = '';
  pts.forEach((p, i) => {
    if (!p) return;
    const c = COLORS[i % COLORS.length];
    s += '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="' + r + '" fill="none" stroke="' + c +
         '" stroke-width="' + (r/2.2) + '"/>' +
         '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="' + (r/4) + '" fill="' + c + '"/>' +
         '<text x="' + (p[0] + r*1.6) + '" y="' + (p[1] - r*0.7) + '" fill="' + c +
         '" font-size="' + (r*2.6) + '">' + (i+1) + '</text>';
  });
  return s + (extra || '');
}
function paint(svg, size, inner) {
  svg.setAttribute('viewBox', '0 0 ' + size[0] + ' ' + size[1]);
  svg.innerHTML = inner;
}
function option(p) { return '<option value="' + p.key + '">' + p.label + '</option>'; }

// -------------------------------------------------------------- points view
function redrawPoints() {
  if (!pickPair) return;
  const dPts = pairs.map(p => p.depth).concat(pending ? [pending] : []);
  paint(document.getElementById('depthSvg'), pickPair.depthSize, markerSVG(dPts, pickPair.depthSize));
  paint(document.getElementById('piSvg'), pickPair.piSize,
        markerSVG(pairs.map(p => p.pi), pickPair.piSize));
  fit();
  table();
  preview();
}

function fit() {
  H = pairs.length >= 4 ? homography(pairs.map(p => p.pi), pairs.map(p => p.depth)) : null;
  residuals = []; rms = null;
  if (H) {
    let sum = 0;
    pairs.forEach(p => {
      const q = apply(H, p.pi);
      const e = Math.hypot(q[0] - p.depth[0], q[1] - p.depth[1]);
      residuals.push(e); sum += e * e;
    });
    rms = Math.sqrt(sum / pairs.length);
  }
  document.getElementById('npairs').innerHTML =
    'Pairs <b>' + pairs.length + '</b>' + (pairs.length < 6 ? ' (six or more preferred)' : '');
  const el = document.getElementById('rms');
  if (rms === null) { el.className = 'stat'; el.innerHTML = 'Fit error <b>&mdash;</b>'; }
  else {
    el.className = 'stat ' + (rms < 3 ? 'good' : 'bad');
    el.innerHTML = 'Fit error <b>' + rms.toFixed(2) + ' px</b>' +
                   (rms < 3 ? '' : ' &mdash; check the worst pair below');
  }
  document.getElementById('fitNote').textContent =
    H ? 'worst point ' + Math.max(...residuals).toFixed(2) + ' px' : 'need four pairs';
  document.getElementById('pairCount').textContent = pairs.length + ' pairs';
  const ok = pairs.length >= 6 && rms !== null && document.getElementById('initials').value.trim();
  document.getElementById('download').disabled = !ok;
  document.getElementById('saveHint').textContent = ok ? '' :
    (pairs.length < 6 ? 'Add ' + (6 - pairs.length) + ' more pair(s)' : 'Enter your initials');
}

function table() {
  const rows = pairs.map((p, i) =>
    '<tr><td><span class="swatch" style="background:' + COLORS[i % COLORS.length] + '"></span>' +
    (i+1) + '</td><td>' + p.depth.map(Math.round).join(', ') + '</td>' +
    '<td>' + p.pi.map(Math.round).join(', ') + '</td>' +
    '<td>' + (residuals[i] !== undefined ? residuals[i].toFixed(2) + ' px' : '&mdash;') + '</td>' +
    '<td><button data-del="' + i + '">remove</button></td></tr>').join('');
  document.getElementById('table').querySelector('tbody').innerHTML =
    '<tr><th>#</th><th>Depth x, y</th><th>Pi x, y</th><th>Error</th><th></th></tr>' + rows;
}

function preview() {
  if (!viewPair) return;
  const box = document.getElementById('warpBox');
  const svg = document.getElementById('previewSvg');
  paint(svg, viewPair.depthSize, '');
  if (!H) { box.style.display = 'none'; return; }
  box.style.display = 'block';
  const host = document.getElementById('preview');
  const s = host.clientWidth / viewPair.depthSize[0];
  box.style.transform = 'scale(' + s + ') matrix3d(' +
    [H[0][0], H[1][0], 0, H[2][0],
     H[0][1], H[1][1], 0, H[2][1],
     0, 0, 1, 0,
     H[0][2], H[1][2], 0, H[2][2]].join(',') + ')';
}

function setPickPair(key) {
  pickPair = D.pairs.find(p => p.key === key) || D.pairs[0];
  document.getElementById('pickPair').value = pickPair.key;
  document.getElementById('depthImg').src = pickPair.depth;
  document.getElementById('piImg').src = pickPair.pi;
  document.getElementById('depthName').textContent = pickPair.depthFile;
  document.getElementById('piName').textContent = pickPair.piFile;
  pending = null;
  redrawPoints();
}
function setViewPair(key) {
  viewPair = D.pairs.find(p => p.key === key) || D.pairs[0];
  document.getElementById('viewPair').value = viewPair.key;
  document.getElementById('previewBase').src = viewPair.depth;
  const w = document.getElementById('warpImg');
  w.src = viewPair.pi;
  w.style.width = viewPair.piSize[0] + 'px';
  preview();
}

// ---------------------------------------------------------------- crop view
function cropSVG(which) {
  const cls = which === 'video' ? 'crop video' : 'crop';
  const dot = which === 'video' ? 'corner video' : 'corner';
  const size = which === 'video' ? cropPair.piSize : cropPair.depthSize;
  const r = Math.max(5, size[0] / 110);
  const pts = crops[which];
  let s = '<polygon class="' + cls + '" points="' + pts.map(p => p.join(',')).join(' ') + '"/>';
  pts.forEach((p, i) => {
    s += '<circle class="' + dot + '" data-i="' + i + '" cx="' + p[0] + '" cy="' + p[1] +
         '" r="' + r + '"/>';
  });
  if (cropMode[which] === 'place') {
    placing[which].forEach((p, i) => {
      s += '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="' + r + '" fill="none" stroke="#fff" ' +
           'stroke-width="' + (r/2.5) + '"/>';
    });
  }
  return s;
}
function redrawCrops() {
  if (!cropPair) return;
  paint(document.getElementById('cropDepthSvg'), cropPair.depthSize, cropSVG('depth'));
  paint(document.getElementById('cropVideoSvg'), cropPair.piSize, cropSVG('video'));
}
function setCropPair(key) {
  cropPair = D.pairs.find(p => p.key === key) || D.pairs[0];
  document.getElementById('cropPair').value = cropPair.key;
  document.getElementById('cropDepthImg').src = cropPair.depth;
  document.getElementById('cropVideoImg').src = cropPair.pi;
  redrawCrops();
}
function attachCropEditor(hostId, which, sizeOf) {
  const host = document.getElementById(hostId);
  let dragging = null;
  host.addEventListener('mousedown', ev => {
    const p = hostPoint(host, ev, sizeOf());
    if (cropMode[which] === 'place') return;
    let best = 0, bd = Infinity;
    crops[which].forEach((c, i) => {
      const d = Math.hypot(c[0]-p[0], c[1]-p[1]);
      if (d < bd) { bd = d; best = i; }
    });
    const grab = sizeOf()[0] / 12;
    if (bd < grab) { dragging = best; ev.preventDefault(); }
  });
  window.addEventListener('mousemove', ev => {
    if (dragging === null) return;
    const p = hostPoint(host, ev, sizeOf());
    crops[which][dragging] = [Math.round(p[0]), Math.round(p[1])];
    redrawCrops();
  });
  window.addEventListener('mouseup', () => { dragging = null; });
  host.addEventListener('click', ev => {
    if (cropMode[which] !== 'place') return;
    const p = hostPoint(host, ev, sizeOf());
    placing[which].push([Math.round(p[0]), Math.round(p[1])]);
    if (placing[which].length === 4) {
      crops[which] = placing[which];
      placing[which] = [];
      setCropMode(which, 'drag');
    }
    redrawCrops();
  });
}
function setCropMode(which, mode) {
  cropMode[which] = mode;
  if (mode === 'place') placing[which] = [];
  document.querySelectorAll('[data-mode][data-which="' + which + '"]').forEach(b =>
    b.setAttribute('aria-pressed', b.dataset.mode === mode));
  redrawCrops();
}

// ------------------------------------------------------------------- tabs
function setTab(which) {
  const points = which === 'points';
  document.getElementById('tabPoints').setAttribute('aria-selected', points);
  document.getElementById('tabCrop').setAttribute('aria-selected', !points);
  document.getElementById('viewPoints').hidden = !points;
  document.getElementById('viewCrop').hidden = points;
  if (points) preview(); else redrawCrops();
}

function flash(text, kind) {
  const m = document.getElementById('msg');
  m.className = 'msg ' + (kind || 'info');
  m.textContent = text;
}

function save() {
  const out = {
    schema: 'cichlid-registration/1',
    projectID: D.projectID, analysisID: D.analysisID, tankID: D.tankID,
    pickPair: pickPair.key, viewPair: viewPair.key, cropPair: cropPair.key,
    depthFile: pickPair.depthFile, piFile: pickPair.piFile,
    depthSize: pickPair.depthSize, piSize: pickPair.piSize,
    points: pairs.map(p => ({ depth: p.depth, pi: p.pi })),
    depthPoints: crops.depth.map(p => [Math.round(p[0]), Math.round(p[1])]),
    videoPoints: crops.video.map(p => [Math.round(p[0]), Math.round(p[1])]),
    browserTransM: H, browserRMS: rms,
    initials: document.getElementById('initials').value.trim().toUpperCase(),
    note: document.getElementById('note').value.trim(),
    created: new Date().toISOString(),
  };
  const text = JSON.stringify(out, null, 1);
  // Opened straight from disk there is nothing to post to, so fall back to the
  // file the user then drops into the submissions folder.
  if (location.protocol === 'file:') return saveAsFile(out, text);
  postToServer(out, text);
}

function postToServer(out, text) {
  const btn = document.getElementById('download');
  btn.disabled = true;
  flash('Saving\u2026', 'info');
  fetch('api/register', { method: 'POST',
                          headers: { 'Content-Type': 'application/json' }, body: text })
    .then(r => r.json().then(j => ({ ok: r.ok, j })))
    .then(({ ok, j }) => {
      if (!ok) throw new Error(j.error || 'the server refused the change');
      poll(j.job, j.queued_behind);
    })
    .catch(e => {
      btn.disabled = false;
      flash('Could not reach the server (' + e.message + '). Saving a file instead.', 'info');
      saveAsFile(out, text);
    });
}

function poll(jobId, behind) {
  const started = Date.now();
  const tick = () => {
    fetch('api/job/' + jobId).then(r => r.json()).then(j => {
      const secs = Math.round((Date.now() - started) / 1000);
      if (j.status === 'done') {
        let msg = 'Saved. ' + (j.record ? j.record.nPairs + ' pairs, ' + j.record.nInliers +
                  ' inliers, fit ' + j.record.rms_px + ' px. ' : '');
        const m = document.getElementById('msg');
        m.className = 'msg done';
        m.innerHTML = msg +
          '<a href="Prep.html" style="margin-left:8px">See it on the Prep page</a>' +
          (j.next ? ' &nbsp;<a href="../' + j.next + '/Prep.html">Next project &rarr;</a>' : '') +
          '<div class="stat" style="margin-top:6px">Reloading this page in a moment\u2026</div>';
        setTimeout(() => location.reload(), 4000);
        return;
      }
      if (j.status === 'failed') {
        document.getElementById('download').disabled = false;
        flash('The save failed: ' + (j.error || 'unknown') +
              '. Nothing was changed; the previous files are still in place.', 'info');
        return;
      }
      flash((j.step || j.status) + '\u2026 ' + secs + ' s' +
            (behind ? ' (' + behind + ' ahead in the queue)' : ''), 'info');
      setTimeout(tick, 1200);
    }).catch(() => setTimeout(tick, 2500));
  };
  tick();
}

function saveAsFile(out, text) {
  // A timestamp keeps every save a distinct file. Without it a second save of
  // the same project lands as "… (1).json", which no longer matches the glob
  // used to apply them — and since each save carries the crops as well as the
  // points, the file that gets skipped is the more complete one.
  const t = out.created.replace(/[-:]/g, '').replace('T', '_').slice(0, 15);
  const name = D.projectID + '__' + out.initials + '__' + t + '__registration.json';
  let ok = false;
  try {
    const blob = new Blob([text], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name; a.style.display = 'none';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 4000);
    ok = true;
  } catch (e) { ok = false; }

  const m = document.getElementById('msg');
  m.className = 'msg done';
  m.innerHTML = (ok
      ? 'Saved <b>' + name + '</b> with ' + pairs.length + ' pairs at ' + rms.toFixed(2) + ' px. ' +
        'It is in your Downloads folder \u2014 move it to WebServer/_submissions in Dropbox. '
      : 'The browser blocked the download. ') +
    '<button id="copyJson" style="margin-left:8px">Copy the file contents instead</button>' +
    '<div id="copyNote" class="stat" style="margin-top:8px"></div>';
  document.getElementById('copyJson').addEventListener('click', () => {
    const note = document.getElementById('copyNote');
    const done = () => { note.textContent = 'Copied. Paste into a new file named ' + name; };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, () => fallbackCopy(text, done));
    } else fallbackCopy(text, done);
  });
}

function fallbackCopy(text, done) {
  const ta = document.createElement('textarea');
  ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand('copy'); done(); }
  catch (e) { document.getElementById('copyNote').textContent = 'Could not copy automatically.'; }
  document.body.removeChild(ta);
}

function build() {
  document.getElementById('title').textContent = D.projectID + ' \u2014 registration';
  document.getElementById('sub').textContent =
    'Click the same speck of sand in both images. Six or more well spread pairs, kept away ' +
    'from any deep pit or tall castle, give the steadiest fit. Points are kept when you ' +
    'switch images, so you can add pairs from whichever trial has the clearest grain.';

  const opts = D.pairs.map(option).join('');
  ['pickPair', 'viewPair', 'cropPair'].forEach(id => {
    document.getElementById(id).innerHTML = opts;
  });

  const wanted = new URLSearchParams(location.search).get('pair');
  const start = D.pairs.find(p => p.key === wanted) || D.pairs[0];

  document.getElementById('pickPair').addEventListener('change', e => setPickPair(e.target.value));
  document.getElementById('viewPair').addEventListener('change', e => setViewPair(e.target.value));
  document.getElementById('cropPair').addEventListener('change', e => setCropPair(e.target.value));

  document.getElementById('depthHost').addEventListener('click', ev => {
    pending = hostPoint(document.getElementById('depthHost'), ev, pickPair.depthSize);
    redrawPoints();
  });
  document.getElementById('piHost').addEventListener('click', ev => {
    if (!pending) { flash('Click the point on the depth image first, then its match here.'); return; }
    pairs.push({ depth: pending,
                 pi: hostPoint(document.getElementById('piHost'), ev, pickPair.piSize) });
    pending = null;
    redrawPoints();
  });
  document.getElementById('table').addEventListener('click', ev => {
    const i = ev.target.dataset && ev.target.dataset.del;
    if (i !== undefined) { pairs.splice(+i, 1); redrawPoints(); }
  });
  document.getElementById('undo').addEventListener('click', () => {
    if (pending) pending = null; else pairs.pop();
    redrawPoints();
  });
  document.getElementById('clear').addEventListener('click', () => {
    pairs = []; pending = null; redrawPoints();
  });
  document.getElementById('initials').addEventListener('input', fit);

  attachLoupe(document.getElementById('depthHost'), document.getElementById('depthLoupe'),
              () => document.getElementById('depthImg').src);
  attachLoupe(document.getElementById('piHost'), document.getElementById('piLoupe'),
              () => document.getElementById('piImg').src);
  attachLoupe(document.getElementById('cropDepthHost'), document.getElementById('cropDepthLoupe'),
              () => document.getElementById('cropDepthImg').src);
  attachLoupe(document.getElementById('cropVideoHost'), document.getElementById('cropVideoLoupe'),
              () => document.getElementById('cropVideoImg').src);

  attachCropEditor('cropDepthHost', 'depth', () => cropPair.depthSize);
  attachCropEditor('cropVideoHost', 'video', () => cropPair.piSize);
  document.querySelectorAll('[data-mode]').forEach(b =>
    b.addEventListener('click', () => setCropMode(b.dataset.which, b.dataset.mode)));
  document.querySelectorAll('[data-reset]').forEach(b =>
    b.addEventListener('click', () => {
      const which = b.dataset.reset;
      crops[which] = ORIGINAL[which].map(p => p.slice());
      setCropMode(which, 'drag');
    }));

  document.getElementById('tabPoints').addEventListener('click', () => setTab('points'));
  document.getElementById('tabCrop').addEventListener('click', () => setTab('crop'));

  const host = document.getElementById('preview');
  host.addEventListener('mousemove', ev => {
    const r = host.getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
    document.getElementById('previewOver').style.width = (f * 100) + '%';
    document.getElementById('previewHandle').style.left = (f * 100) + '%';
  });
  window.addEventListener('resize', () => { if (viewPair) preview(); });
  document.getElementById('download').addEventListener('click', save);

  if (location.protocol !== 'file:') {
    fetch('api/whoami').then(r => r.json()).then(j => {
      if (!j.user || j.user === 'anonymous') return;
      const el = document.getElementById('initials');
      el.value = j.user.split('@')[0].slice(0, 4).toUpperCase();
      el.title = 'signed in as ' + j.user;
      fit();
    }).catch(() => {});
  }

  setPickPair(start.key);
  setViewPair(start.key);
  setCropPair(start.key);
  setTab('points');
}
build();
</script>
</body>
</html>
"""


def write_register(path, analysis_id, payload):
    html = REGISTER_PAGE.replace('__TITLE__', payload['projectID'] + ' · Registration')
    html = html.replace('__ANALYSIS__', analysis_id)
    html = html.replace('__PAYLOAD__', json.dumps(payload).replace('</', '<\\/'))
    with open(path, 'w') as f:
        f.write(html)
    return os.path.getsize(path)


def format_points(points):
    """Match the format PrepPreparer writes: ','.join(str(tuple))."""
    return ','.join([str((int(p[0]), int(p[1]))) for p in points])


def apply_submission(fm_obj, sub, s_dt, who=None):
    """Recompute the transform with OpenCV and write the three prep files."""
    projectID = sub['projectID']
    pts = sub['points']
    if len(pts) < 4:
        raise ValueError('only ' + str(len(pts)) + ' point pairs')

    pi = np.array([p['pi'] for p in pts], np.float32)
    depth = np.array([p['depth'] for p in pts], np.float32)
    transM, inliers = cv2.findHomography(pi, depth, cv2.RANSAC, 3.0)
    if transM is None:
        raise ValueError('findHomography failed')
    n_in = int(inliers.sum())

    projected = cv2.perspectiveTransform(pi.reshape(-1, 1, 2), transM).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected - depth) ** 2, axis=1))))

    depth_points = [(int(p[0]), int(p[1])) for p in sub['depthPoints']]
    if sub.get('videoPoints'):
        # The two crops are independent: the depth camera does not always see the
        # whole video field of view, so the video crop is placed by hand.
        video_points = [(int(p[0]), int(p[1])) for p in sub['videoPoints']]
    else:
        inv = np.linalg.inv(transM)
        video_points = cv2.perspectiveTransform(
            np.array(depth_points, np.float32).reshape(-1, 1, 2), inv).reshape(-1, 2)
        video_points = [(int(round(p[0])), int(round(p[1]))) for p in video_points]

    fm_obj.setProjectID(projectID)
    fm_obj.createDirectory(fm_obj.localAnalysisDir)
    fm_obj.createDirectory(fm_obj.localBackupDir)

    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for path in [fm_obj.localDepthCropFile, fm_obj.localVideoCropFile, fm_obj.localTransMFile]:
        fetch_optional(path.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir), path)
        if os.path.exists(path):
            shutil.copy2(path, fm_obj.localBackupDir + stamp + '_' + os.path.basename(path))

    with open(fm_obj.localDepthCropFile, 'w') as f:
        print(format_points(depth_points), file=f)
    with open(fm_obj.localVideoCropFile, 'w') as f:
        print(format_points(video_points), file=f)
    np.save(fm_obj.localTransMFile, transM)

    record = {'appliedAt': str(datetime.datetime.now().replace(microsecond=0)),
              'nPairs': len(pts), 'nInliers': n_in, 'rms_px': round(rms, 3),
              'initials': sub.get('initials', ''), 'note': sub.get('note', ''),
              'who': who or sub.get('initials', '') or 'unknown',
              'pairKey': sub.get('pickPair', sub.get('pairKey', '')),
              'backupStamp': stamp,
              'depthPoints': depth_points, 'videoPoints': video_points}

    # a single current-state file beside the crops, so the Prep page can say who
    # last changed the registration without trawling the backups
    with open(registration_info_path(fm_obj), 'w') as f:
        json.dump(record, f, indent=1)
    with open(fm_obj.localBackupDir + stamp + '_registration.json', 'w') as f:
        json.dump({'submission': sub, 'applied': record}, f, indent=1)

    for path in [fm_obj.localDepthCropFile, fm_obj.localVideoCropFile,
                 fm_obj.localTransMFile, registration_info_path(fm_obj)]:
        upload(fm_obj, path)
    upload(fm_obj, fm_obj.localBackupDir + stamp + '_registration.json')

    stale = []
    row = s_dt.loc[projectID]
    if 'Depth' in s_dt and bool(row.Depth):
        stale.append('Depth')
    if 'Cluster' in s_dt and str(row.Cluster).strip() not in ('VideoIndices:', 'VideoIndices: ', 'nan'):
        stale.append('Cluster')
    record['stale'] = stale
    return record


def build_one(fm_obj, projectID, out_root, category='', delete=False,
              page_type='Prep', neighbours=None):
    """Build the Prep and Register pages for a single project.

    Never raises: a project that cannot be built is recorded so the sweep
    continues and the index still shows it with a reason."""
    entry = {'id': projectID, 'tank': '', 'category': category, 'trials': 0, 'start': '',
             'thumb': None, 'missing': 0, 'status': 'ok', 'pages': {}, 'files': [],
             'error': ''}

    fm_obj.setProjectID(projectID)
    lp = getattr(fm_obj, 'lp', None)
    if lp is None:
        entry['status'] = 'failed'
        entry['error'] = 'logfile could not be parsed'
        return entry

    entry['tank'] = lp.tankID
    entry['trials'] = len(lp.trials)
    entry['start'] = str(lp.master_start)

    fm_obj.createDirectory(fm_obj.localAnalysisDir)
    fm_obj.createDirectory(fm_obj.localSummaryDir)
    fm_obj.createDirectory(fm_obj.localLogfileDir)
    fm_obj.downloadData(fm_obj.localPrepDir)
    prep2 = fm_obj.localProjectDir + 'PrepFiles2/'
    fm_obj.createDirectory(prep2)
    fetch_optional(prep2.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir),
                   prep2, directory=True)
    for f in [fm_obj.localDepthCropFile, fm_obj.localVideoCropFile, fm_obj.localTransMFile,
              fm_obj.localPrepLogfile]:
        fetch_optional(f.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir), f)

    if page_type == 'Cluster':
        for f in [fm_obj.localAllLabeledClustersFile]:
            fetch_optional(f.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir), f)
        try:
            clusters = load_clusters(fm_obj)
        except ClusterFileMissing as e:
            entry['status'] = 'needs_clusters'
            entry['error'] = str(e) + ' — run the Cluster stage of runAnalysis.py'
            return entry
        payload = build_cluster_payload(fm_obj, lp, clusters,
                                        parse_points(fm_obj.localVideoCropFile),
                                        parse_points(fm_obj.localDepthCropFile),
                                        np.load(fm_obj.localTransMFile))
        project_dir = out_root + projectID + '/'
        fm_obj.createDirectory(project_dir)
        entry['size'] = write_cluster(project_dir + 'Cluster.html', fm_obj.analysisID, payload)
        entry['pages']['Cluster'] = 'Cluster.html'
        entry['files'].append('Cluster.html')
        entry['trials'] = len(payload['trials'])
        first = load_npy(fm_obj.localFirstFrame)
        last = load_npy(fm_obj.localLastFrame)
        change, _ = filtered_change(first, last)
        entry['thumb'] = thumbnail(change)
        return entry

    if page_type == 'Depth':
        depth_dir = fm_obj.localProjectDir + 'DepthFiles/'
        fm_obj.createDirectory(depth_dir)
        fetch_optional(depth_dir.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir),
                       depth_dir, directory=True)
        try:
            raw, smooth, dmeta = load_depth_endpoints(fm_obj)
        except DepthFilesMissing as e:
            entry['status'] = 'needs_depthfiles'
            entry['error'] = str(e) + ' — rerun the Depth stage of runAnalysis.py'
            return entry
        payload = build_depth_payload(fm_obj, lp, raw, smooth, dmeta)
        project_dir = out_root + projectID + '/'
        fm_obj.createDirectory(project_dir)
        entry['size'] = write_depth(project_dir + 'Depth.html', fm_obj.analysisID, payload)
        entry['pages']['Depth'] = 'Depth.html'
        entry['files'].append('Depth.html')
        entry['trials'] = len(payload['trials'])
        first = load_npy(fm_obj.localFirstFrame)
        last = load_npy(fm_obj.localLastFrame)
        change, _ = filtered_change(first, last)
        entry['thumb'] = thumbnail(change)
        if delete:
            shutil.rmtree(fm_obj.localPrepDir, ignore_errors=True)
        return entry

    depth_dir = fm_obj.localProjectDir + 'DepthFiles/'
    fm_obj.createDirectory(depth_dir)
    fetch_optional(depth_dir.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir),
                   depth_dir, directory=True)

    problems, trial_status = check_prep_files(fm_obj, lp)
    blocking = [p for p in problems if p.startswith('missing ')]
    if blocking:
        entry['status'] = 'failed'
        entry['error'] = '; '.join(blocking)
        return entry

    try:
        pairs, missing = load_prepfiles2(fm_obj, lp)
    except PrepFiles2Missing as e:
        entry['status'] = 'needs_prepfiles2'
        entry['error'] = str(e) + ' — run: python createPrepFiles2.py ' + \
                         fm_obj.analysisID + ' --ProjectIDs ' + projectID
        return entry
    entry['missing'] = len(lp.trials) * 2 - len(pairs)

    payload = build_prep_payload(fm_obj, lp, trial_status, pairs, neighbours=neighbours)

    project_dir = out_root + projectID + '/'
    fm_obj.createDirectory(project_dir)
    size = write_page(project_dir + 'Prep.html', projectID + ' \u00b7 Prep', payload)
    entry['pages']['Prep'] = 'Prep.html'
    entry['files'].append('Prep.html')

    # Register is reached from the Prep page, not the index, so it is uploaded
    # without being listed in entry['pages'].
    reg_payload = build_register_payload(fm_obj, lp, pairs)
    size += write_register(project_dir + 'Register.html', fm_obj.analysisID, reg_payload)
    entry['files'].append('Register.html')
    entry['size'] = size

    first = load_npy(fm_obj.localFirstFrame)
    last = load_npy(fm_obj.localLastFrame)
    change, _ = filtered_change(first, last)
    entry['thumb'] = thumbnail(change)

    if delete:
        shutil.rmtree(fm_obj.localPrepDir, ignore_errors=True)

    return entry


def category_of(s_dt, projectID):
    if 'Category' not in s_dt.columns or projectID not in s_dt.index:
        return ''
    raw = s_dt.loc[projectID, 'Category']
    return '' if raw is None or str(raw).strip().lower() in ('nan', '') else str(raw).strip()


def remove_source(path, cloud_path=None):
    """Drop a submission file once it has been applied.

    Nothing is lost: apply_submission writes the whole submission, and what came
    of it, into the project's Backups directory before this runs.
    """
    try:
        os.remove(path)
    except OSError:
        return False
    if cloud_path:
        subprocess.run(['rclone', 'deletefile', cloud_path],
                       capture_output=True, encoding='utf-8')
    return True


def apply_registrations(args):
    fm_obj = FM(args.AnalysisID)
    s_dt = fm_obj.s_dt
    sub_dir = fm_obj.localAnalysisStatesDir + 'WebServer/_submissions/'
    ledger_path = sub_dir + 'applied.json'
    ledger = {}

    if args.Files:
        paths = []
        unmatched = []
        for pattern in args.Files:
            expanded = sorted(glob.glob(os.path.expanduser(pattern)))
            if expanded:
                paths.extend(expanded)
            else:
                unmatched.append(pattern)
        if unmatched:
            print('Cannot find: ' + ', '.join(unmatched))
            for pattern in unmatched:
                d = os.path.dirname(os.path.expanduser(pattern)) or '.'
                if os.path.isdir(d):
                    near = sorted([f for f in os.listdir(d) if 'registration' in f.lower()])
                    if near:
                        print('  registration files in ' + d + ': ' + ', '.join(near))
                    else:
                        print('  no registration files in ' + d)
            if not paths:
                return 1
    else:
        fm_obj.createDirectory(sub_dir)
        fetch_optional(sub_dir.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir),
                       sub_dir, directory=True)
        if os.path.exists(ledger_path):
            try:
                with open(ledger_path) as f:
                    ledger = json.load(f)
            except Exception:
                ledger = {}
        paths = sorted([sub_dir + f for f in os.listdir(sub_dir)
                        if f.endswith('.json') and f != 'applied.json'])
        if not args.Force:
            already = [p for p in paths if os.path.basename(p) in ledger]
            paths = [p for p in paths if os.path.basename(p) not in ledger]
            if already:
                print('Already applied, skipping ' + str(len(already)) +
                      ' file(s). Use --Force to reapply.')

    if not paths:
        cloud = sub_dir.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir)
        print('No new registration files to apply.')
        print('')
        print('Registration files are made in the browser: open index.html, pick a')
        print('project, click Register, match points and save. The file that downloads')
        print('goes in ' + cloud)
        print('')
        print('To apply one straight from your computer instead:')
        print('  python createServer.py ApplyRegistration ' + args.AnalysisID +
              ' --Files ~/Downloads/*__registration.json')
        return 1

    print('Found ' + str(len(paths)) + ' registration file(s)')

    # Several files can name the same project — a second pass over the crops, or
    # a browser-renamed duplicate. Each save carries the whole state, so the
    # newest one supersedes the rest rather than being merged with them.
    loaded = []
    for path in paths:
        try:
            with open(path) as f:
                loaded.append((path, json.load(f)))
        except Exception as e:
            print(os.path.basename(path) + ': not readable (' + repr(e) + ')')
    newest = {}
    for path, sub in loaded:
        pid = sub.get('projectID')
        stamp = sub.get('created', '')
        if pid not in newest or stamp > newest[pid][1].get('created', ''):
            newest[pid] = (path, sub)
    superseded_by = {}
    for path, sub in loaded:
        pid = sub.get('projectID')
        winner = newest.get(pid, (None,))[0]
        if winner != path:
            superseded_by.setdefault(pid, []).append(path)
    superseded = [os.path.basename(p) for ps in superseded_by.values() for p in ps]
    if superseded:
        print('Superseded by a newer save for the same project, skipping ' +
              str(len(superseded)) + ': ' + ', '.join(superseded))
    paths = [p for p, _ in sorted(newest.values(), key=lambda x: x[1].get('projectID', ''))]

    applied, skipped, stale_any = [], [], {}
    applied_paths = {}
    for path in paths:
        name = os.path.basename(path)
        try:
            with open(path) as f:
                sub = json.load(f)
        except Exception as e:
            print(name + ': not readable (' + repr(e) + ')')
            skipped.append(name)
            continue

        if sub.get('schema') != 'cichlid-registration/1':
            print(name + ': unrecognised schema, skipping')
            skipped.append(name)
            continue
        projectID = sub.get('projectID')
        if projectID not in s_dt.index:
            print(name + ': ' + str(projectID) + ' is not in ' + args.AnalysisID)
            skipped.append(name)
            continue
        if sub.get('analysisID') != args.AnalysisID:
            print(name + ': built for analysis ' + str(sub.get('analysisID')) + ', skipping')
            skipped.append(name)
            continue

        n = len(sub.get('points', []))
        who = sub.get('initials', '?')
        if args.DryRun:
            print(projectID + ': ' + str(n) + ' pairs from ' + who +
                  ', browser fit ' + str(round(sub.get('browserRMS') or 0, 2)) + ' px (dry run)')
            continue
        try:
            record = apply_submission(fm_obj, sub, s_dt, who=sub.get('initials'))
        except Exception as e:
            print(projectID + ': failed (' + repr(e) + ')')
            skipped.append(name)
            continue

        print(projectID + ': ' + str(record['nPairs']) + ' pairs from ' + who + ', ' +
              str(record['nInliers']) + ' inliers, fit ' + str(record['rms_px']) + ' px' +
              (' — backup ' + record['backupStamp']))
        applied.append(projectID)
        applied_paths[projectID] = [path] + superseded_by.get(projectID, [])
        ledger[os.path.basename(path)] = {
            'projectID': projectID, 'appliedAt': record['appliedAt'],
            'initials': who, 'rms_px': record['rms_px'], 'backupStamp': record['backupStamp']}
        if record['stale']:
            stale_any[projectID] = record['stale']

    if args.DryRun:
        return 0

    # rebuild the pages for whatever changed, so the loop is one command
    out_root = fm_obj.localAnalysisStatesDir + 'WebServer/'
    if applied and not args.NoRebuild and os.path.exists(out_root):
        order = project_order(s_dt)
        print('')
        for projectID in dict.fromkeys(applied):
            pos = order.index(projectID) if projectID in order else None
            nb = {} if pos is None else {
                'prev': order[pos - 1] if pos > 0 else None,
                'next': order[pos + 1] if pos + 1 < len(order) else None,
                'position': pos + 1, 'total': len(order)}
            print('Rebuilding ' + projectID)
            try:
                entry = build_one(fm_obj, projectID, out_root,
                                  category=category_of(s_dt, projectID),
                                  page_type='Prep', neighbours=nb)
            except Exception as e:
                print('  rebuild failed: ' + repr(e))
                continue
            if entry['status'] != 'ok':
                print('  ' + entry['status'] + ': ' + entry.get('error', ''))
                continue
            for fname in entry.get('files', []):
                upload(fm_obj, out_root + projectID + '/' + fname)
            refresh_index(fm_obj, out_root, entry, fm_obj.branch_name)
        upload(fm_obj, out_root + 'index.html')
        upload(fm_obj, out_root + INDEX_CACHE, quiet=True)

    if applied and not args.Files:
        fm_obj.createDirectory(sub_dir)
        with open(ledger_path, 'w') as f:
            json.dump(ledger, f, indent=1)
        upload(fm_obj, ledger_path)

    if applied and not args.Keep:
        removed = 0
        for projectID in dict.fromkeys(applied):
            for path in applied_paths.get(projectID, []):
                cloud = None
                if not args.Files and os.path.dirname(path).rstrip('/') == sub_dir.rstrip('/'):
                    cloud = (sub_dir + os.path.basename(path)).replace(
                        fm_obj.localMasterDir, fm_obj.cloudMasterDir)
                if remove_source(path, cloud):
                    removed += 1
        if removed:
            print('Removed ' + str(removed) + ' processed submission file(s). The full '
                  'submissions are kept in each project\'s Backups directory.')

    print('\nApplied ' + str(len(applied)) + ', skipped ' + str(len(skipped)) + '.')
    if stale_any:
        print('\nThese projects have Depth or Cluster results from before the change:')
        for projectID, what in stale_any.items():
            print('  ' + projectID + ': ' + ', '.join(what))
    if args.NoRebuild and applied:
        print('\nPages not rebuilt. To update them:')
        print('  python createServer.py Prep ' + args.AnalysisID +
              ' --ProjectIDs ' + ' '.join(dict.fromkeys(applied)))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Create the static HTML pages that back the bower dashboard')
    sub = parser.add_subparsers(required=True, title='Pages', dest='PageType')
    for name in ['Prep', 'Depth', 'Cluster', 'IntegratedData']:
        p = sub.add_parser(name, description='Build the ' + name + ' pages for an analysisID')
        p.add_argument('AnalysisID', type=str, help='The AnalysisID to build pages for')
        p.add_argument('--ProjectIDs', type=str, nargs='+',
                       help='Optional projectIDs to restrict the build to')
        p.add_argument('--NoUpload', action='store_true',
                       help='Build locally without uploading to the cloud')
        p.add_argument('--Delete', action='store_true',
                       help='Remove each downloaded PrepFiles directory after building')
        p.add_argument('--SkipExisting', action='store_true',
                       help='Leave projects whose page has already been built')

    ar = sub.add_parser('ApplyRegistration',
                        description='Apply signed-off registration files from the submissions folder')
    ar.add_argument('AnalysisID', type=str, help='The AnalysisID to apply submissions for')
    ar.add_argument('--Files', type=str, nargs='+',
                    help='Local .json files to apply instead of the cloud submissions folder')
    ar.add_argument('--DryRun', action='store_true',
                    help='Report what would change without writing or uploading anything')
    ar.add_argument('--Force', action='store_true',
                    help='Reapply submissions that have already been applied')
    ar.add_argument('--NoRebuild', action='store_true',
                    help='Apply the files without rebuilding the affected pages')
    ar.add_argument('--Keep', action='store_true',
                    help='Leave the submission files in place after applying them')
    args = parser.parse_args()

    if args.PageType == 'ApplyRegistration':
        return apply_registrations(args)

    if args.PageType not in ('Prep', 'Depth', 'Cluster'):
        print(args.PageType + ' pages are not implemented yet.')
        return 1

    fm_obj = FM(args.AnalysisID)
    s_dt = fm_obj.s_dt

    projectIDs = project_order(s_dt)
    full_order = list(projectIDs)
    if args.ProjectIDs is not None:
        unknown = [p for p in args.ProjectIDs if p not in s_dt.index]
        if unknown:
            print('Unknown ProjectIDs: ' + ', '.join(unknown))
            return 1
        not_ready = [p for p in args.ProjectIDs if p not in projectIDs]
        if not_ready:
            print('Skipping, Prep not complete: ' + ', '.join(not_ready))
        projectIDs = [p for p in args.ProjectIDs if p in projectIDs]

    if not projectIDs:
        print('No projects in ' + args.AnalysisID + ' have completed Prep.')
        return 1

    out_root = fm_obj.localAnalysisStatesDir + 'WebServer/'
    fm_obj.createDirectory(out_root)

    # Make the drop folder exist in the cloud, otherwise there is nowhere for
    # anyone to put a registration file and ApplyRegistration has nothing to read.
    sub_dir = out_root + '_submissions/'
    fm_obj.createDirectory(sub_dir)
    readme = sub_dir + 'README.txt'
    if not os.path.exists(readme):
        with open(readme, 'w') as f:
            f.write('Drop registration files here.\n\n'
                    'Open a project from index.html, click Register, match at least six\n'
                    'points between the two images, enter your initials and save. The file\n'
                    'that downloads (<projectID>__registration.json) goes in this folder.\n\n'
                    'They are applied with:\n'
                    '  python createServer.py ApplyRegistration ' + args.AnalysisID + '\n')
        if not args.NoUpload:
            upload(fm_obj, readme)

    print('Building ' + args.PageType + ' pages for ' + str(len(projectIDs)) +
          ' project(s) in ' + args.AnalysisID)
    entries, failed = [], []
    for i, projectID in enumerate(projectIDs, 1):
        prefix = '[' + str(i) + '/' + str(len(projectIDs)) + '] ' + projectID
        if args.SkipExisting and os.path.exists(
                out_root + projectID + '/' + args.PageType + '.html'):
            print(prefix + ': already built, skipping')
            continue
        print(prefix + ' ' + str(datetime.datetime.now()), flush=True)
        category = ''
        if 'Category' in s_dt.columns:
            raw = s_dt.loc[projectID, 'Category']
            category = '' if raw is None or str(raw).strip().lower() in ('nan', '') else str(raw).strip()
        try:
            pos = full_order.index(projectID) if projectID in full_order else None
            nb = {} if pos is None else {
                'prev': full_order[pos - 1] if pos > 0 else None,
                'next': full_order[pos + 1] if pos + 1 < len(full_order) else None,
                'position': pos + 1, 'total': len(full_order)}
            entry = build_one(fm_obj, projectID, out_root, category=category,
                              delete=args.Delete, page_type=args.PageType, neighbours=nb)
        except Exception as e:
            entry = {'id': projectID, 'tank': '', 'category': category, 'trials': 0, 'start': '',
                     'thumb': None, 'missing': 0, 'status': 'failed', 'pages': {}, 'files': [],
                     'error': repr(e)}
        entries.append(entry)

        if entry['status'] != 'ok':
            failed.append(projectID)
            print('    ' + entry['status'] + ': ' + entry['error'])
            continue
        print('    wrote ' + entry['id'] + '/' + args.PageType + '.html (' +
              str(round(entry['size'] / 1e6, 2)) + ' MB, ' + str(entry['trials']) + ' trials)')
        if not args.NoUpload:
            for name in entry.get('files', []):
                upload(fm_obj, out_root + projectID + '/' + name)

    # a Depth sweep must not blank the Prep buttons written by an earlier run
    for e in entries:
        d = out_root + e['id'] + '/'
        for name in ('Prep', 'Depth', 'Cluster', 'IntegratedData'):
            if name not in e['pages'] and os.path.exists(d + name + '.html'):
                e['pages'][name] = name + '.html'
    # merge into whatever the last sweep left, so a partial run does not drop
    # projects from the index
    merged = {e['id']: e for e in load_index_entries(out_root)}
    for e in entries:
        merged[e['id']] = e
    ordered = [merged[k] for k in sorted(merged)]
    save_index_entries(out_root, ordered)
    index_size = write_index(out_root + 'index.html', args.AnalysisID, ordered,
                             fm_obj.branch_name)
    print('Wrote ' + out_root + 'index.html (' + str(round(index_size / 1e6, 2)) + ' MB)')

    built = len([e for e in entries if e['status'] == 'ok'])
    incomplete = [e['id'] for e in entries if e['status'] == 'ok' and e['missing']]
    needs = [e['id'] for e in entries if e['status'] == 'needs_prepfiles2']
    needs_depth = [e['id'] for e in entries if e['status'] == 'needs_depthfiles']
    needs_cl = [e['id'] for e in entries if e['status'] == 'needs_clusters']
    if needs_cl:
        print('Missing cluster files (' + str(len(needs_cl)) + '): ' + ', '.join(needs_cl))
    if needs_depth:
        print('Missing DepthFiles (' + str(len(needs_depth)) + '): ' + ', '.join(needs_depth))
    print('Built ' + str(built) + ' of ' + str(len(entries)) + ' pages.')
    if needs:
        print('Missing PrepFiles2 (' + str(len(needs)) + '). Build them with:')
        print('  python createPrepFiles2.py ' + args.AnalysisID +
              ' --ProjectIDs ' + ' '.join(needs))
    if incomplete:
        print('Incomplete trial data in: ' + ', '.join(incomplete))
    if failed:
        print('Failed: ' + ', '.join(failed))

    if args.NoUpload:
        print('Skipping upload. Open ' + out_root + 'index.html')
        return 0

    upload(fm_obj, out_root + 'index.html')
    upload(fm_obj, out_root + INDEX_CACHE, quiet=True)
    print('Open ' + out_root.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir) + 'index.html')
    return 0


if __name__ == '__main__':
    sys.exit(main())