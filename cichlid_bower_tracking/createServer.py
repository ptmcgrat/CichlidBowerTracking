import argparse, base64, datetime, glob, json, os, re, shutil, subprocess, sys, warnings

import cv2
import numpy as np
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


def build_prep_payload(fm, lp, trial_status, pairs):
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

    # the depth crop inverse-warped into Pi coordinates, so the video crop can be
    # checked against what the depth camera can actually see
    depth_in_video = None
    try:
        inv = np.linalg.inv(transM)
        pts = cv2.perspectiveTransform(
            np.array(depth_points, np.float32).reshape(-1, 1, 2), inv).reshape(-1, 2)
        depth_in_video = [[int(round(x)), int(round(y))] for x, y in pts]
    except np.linalg.LinAlgError:
        pass

    prep_log = ''
    if os.path.exists(fm.localPrepLogfile):
        with open(fm.localPrepLogfile) as f:
            prep_log = ''.join([line for line in f if 'DateAnalyzed' in line
                                or 'Username' in line or 'Nodename' in line]).strip()

    return {
        'projectID': lp.projectID, 'tankID': lp.tankID, 'analysisID': fm.analysisID,
        'device': getattr(lp, 'device', 'unknown'),
        'masterStart': str(lp.master_start), 'masterStop': str(getattr(lp, 'master_stop', '')),
        'nFrames': len(lp.frames), 'nMovies': len(lp.movies),
        'depthPoints': depth_points, 'videoPoints': video_points,
        'depthInVideo': depth_in_video,
        'frameSize': [lp.width, lp.height], 'piSize': pi_size or [1296, 972],
        'trials': trials,
        'logIssues': lp.malformed_file, 'prepLog': prep_log,
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
  .wrap { max-width: 1240px; margin: 0 auto; padding: 28px 24px 72px; }
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
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 20px; }
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
  .stage polygon.derived { stroke: rgba(242,163,60,.45); stroke-width: 2; stroke-dasharray: 8 6; }
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
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<div class="wrap">
  <a class="back" href="../index.html">All projects in __ANALYSIS__</a>
  <h1 id="title"></h1>
  <p class="sub" id="subtitle"></p>
  <div class="meta" id="meta"></div>
  <div class="bar"><span class="stat">Crops and registration are reviewed per trial.</span>
    <a class="btn" id="fixLink" href="Register.html">Fix registration or crops</a></div>
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
const VIDEO_POLY = [{ points: D.videoPoints },
                    { points: D.depthInVideo, cls: 'derived' }];

function trialRow(t, build) {
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
  grid.className = 'grid';
  build(grid, t);
  sec.appendChild(grid);
  return sec;
}

function viewDepthCrop() {
  const box = document.createElement('div');
  box.innerHTML = '<p class="sub">The polygon should follow the tray walls. Change that ' +
    'reaches the boundary means the crop is clipping part of the bower.</p>';
  D.trials.forEach(t => box.appendChild(trialRow(t, (grid, t) => {
    grid.appendChild(imagePanel(t.depthFirst, '<b>Start</b> \u2014 depth camera', DEPTH_POLY));
    grid.appendChild(imagePanel(t.depthLast, '<b>Stop</b> \u2014 depth camera', DEPTH_POLY));
    grid.appendChild(depthPanel(t.change, '<b>Total change over the trial</b>', DEPTH_POLY));
  })));
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

  if (D.logIssues && D.logIssues.length) {
    const n = document.createElement('div');
    n.className = 'note';
    n.innerHTML = 'The log parser flagged this project:<ul>' +
      D.logIssues.map(x => '<li>' + x + '</li>').join('') + '</ul>';
    document.getElementById('tabs').before(n);
  }

  const views = [['Depth Crop', viewDepthCrop], ['Video Crop', viewVideoCrop],
                 ['Registration', viewRegistration]];
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


def half(frame):
    """Downsample by 2 for display, ignoring NaN."""
    h, w = frame.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    blocks = frame[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
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
        return {'rate': dict(zip(('src', 'meta'), encode_depth(half(rate)))),
                'excess': dict(zip(('src', 'meta'), encode_depth(half(excess)))),
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
        for key, arr in (('travel', extras.get('travel')),
                         ('stdMean', extras.get('stdMean')),
                         ('stdMax', extras.get('stdMax'))):
            if arr is not None:
                s, m = encode_depth(half(arr[i]))
                entry[key] = {'src': s, 'meta': m}
        # the depth camera stills copied out by depth_endpoints
        for key, name in (('jpgFirst', days[i].get('first_jpg')),
                          ('jpgLast', days[i].get('last_jpg'))):
            if name and os.path.exists(depth_dir + name):
                entry[key] = encode_image(cv2.imread(depth_dir + name), max_width=lp.width // 2)
        frames.append(entry)

    return {
        'projectID': lp.projectID, 'tankID': lp.tankID, 'analysisID': fm.analysisID,
        'frameSize': [lp.width // 2, lp.height // 2],
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
  .wrap { max-width:1400px; margin:0 auto; padding:24px 22px 80px; }
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
  for (let i = 0, p = 0; i < values.length; i++, p += 4) {
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
function renderTrial(t) {
  const box = document.createElement('div');
  const days = D.days.filter(d => d.trial === t.trial);
  let selected = days[0].day;
  let threshold = D.defaultThreshold;

  const bar = document.createElement('div');
  bar.className = 'bar';
  bar.innerHTML = '<label>Threshold <b id="thv">' + threshold.toFixed(2) + '</b> cm</label>' +
    '<input type="range" id="thr" min="0" max="3" step="0.05" value="' + threshold + '">' +
    '<span class="spacer"></span><span class="stat" id="dayInfo"></span>';
  box.appendChild(bar);

  const strip = document.createElement('div');
  strip.className = 'days';
  days.forEach(d => {
    const b = document.createElement('button');
    b.textContent = d.date.slice(5);
    if (d.partial) b.className = 'partial';
    b.title = d.nFrames + ' frames, ' + d.firstTime + ' to ' + d.lastTime +
              (d.partial ? ' (partial day)' : '');
    b.addEventListener('click', () => { selected = d.day; draw(); });
    b.dataset.day = d.day;
    strip.appendChild(b);
  });
  box.appendChild(strip);

  const noteSlot = document.createElement('div');
  box.appendChild(noteSlot);
  const rowA = document.createElement('div'); rowA.className = 'grid';
  const hB = document.createElement('h2'); hB.textContent = 'Change, and how variable the sensor was';
  const rowB = document.createElement('div'); rowB.className = 'grid';
  const hC = document.createElement('h2'); hC.textContent = 'Raw against interpolated, on a shared scale';
  const rowC = document.createElement('div'); rowC.className = 'grid';
  const hD = document.createElement('h2'); hD.textContent = 'Further diagnostics';
  const rowD = document.createElement('div'); rowD.className = 'grid';
  box.appendChild(rowA); box.appendChild(hB); box.appendChild(rowB);
  box.appendChild(hC); box.appendChild(rowC);
  box.appendChild(hD); box.appendChild(rowD);
  const tableSlot = document.createElement('div');
  box.appendChild(tableSlot);

  function draw() {
    const d = D.days[selected];
    Array.from(strip.children).forEach(b =>
      b.setAttribute('aria-pressed', String(+b.dataset.day === selected)));
    bar.querySelector('#dayInfo').innerHTML =
      'Day ' + (d.day - t.firstDay + 1) + ' of ' + t.nDays + ' · <b>' + d.nFrames +
      '</b> frames · raw valid <b>' + (d.rawValid[0]*100).toFixed(1) + '%</b>';

    noteSlot.innerHTML = '';
    const notes = [];
    if (d.partial) notes.push('This day is only ' + d.nFrames +
      ' frames (' + d.firstTime + ' to ' + d.lastTime + '), so its daily total is not comparable to a full day.');
    if (d.overnightNote) notes.push('No overnight value after this day: ' + d.overnightNote + '.');
    if (notes.length) noteSlot.innerHTML = '<div class="note">' + notes.join(' ') + '</div>';

    const f = D.frames[selected];
    const need = [f.smoothFirst, f.smoothLast, f.rawFirst, f.rawLast];
    ['stdMean', 'stdMax', 'travel'].forEach(k => { if (f[k]) need.push(f[k]); });
    const baseline = D.frames[+D.baselines[t.trial]];
    need.push(baseline.smoothFirst);
    const nextF = d.overnightOK ? D.frames[selected + 1] : null;
    if (nextF) need.push(nextF.smoothFirst);

    loadAll(need).then(() => {
      const sFirst = decode(f.smoothFirst), sLast = decode(f.smoothLast);
      const rFirst = decode(f.rawFirst), rLast = decode(f.rawLast);

      // row one: what the depth camera saw
      rowA.textContent = '';
      rowA.appendChild(photoPanel(f.jpgFirst, '<b>Depth camera</b> — morning, ' + d.firstTime));
      rowA.appendChild(photoPanel(f.jpgLast, '<b>Depth camera</b> — evening, ' + d.lastTime));

      // row two: change, always from the interpolated array
      rowB.textContent = '';
      rowB.appendChild(mapPanel(diff(decode(baseline.smoothFirst), sLast),
        '<b>Cumulative</b> — trial start to the end of this day',
        { threshold: threshold, range: 4 }));
      rowB.appendChild(mapPanel(diff(sFirst, sLast),
        '<b>Daily change</b> — ' + d.firstTime + ' to ' + d.lastTime,
        { threshold: threshold }));
      if (nextF) {
        rowB.appendChild(mapPanel(diff(sLast, decode(nextF.smoothFirst)),
          '<b>Overnight</b> — ' + d.lastTime + ' to ' + D.days[selected+1].firstTime,
          { threshold: threshold }));
      } else {
        rowB.appendChild(photoPanel(null, '<b>Overnight</b> — ' +
          (d.overnightNote || 'no following day')));
      }
      if (f.stdMean) {
        rowB.appendChild(mapPanel(decode(f.stdMean),
          '<b>Capture variability</b> — mean standard deviation across the ~30 captures ' +
          'behind each frame, averaged over the day', { positive: 1.0 }));
      } else {
        rowB.appendChild(photoPanel(null, '<b>Capture variability</b>'));
      }

      // row three: raw against interpolated, all four on one scale so the
      // difference between them is the only thing that changes
      const fin = [];
      for (let i = 0; i < sFirst.length; i++) {
        if (!Number.isNaN(sFirst[i])) fin.push(sFirst[i]);
        if (!Number.isNaN(sLast[i])) fin.push(sLast[i]);
      }
      fin.sort((a,b) => a-b);
      const shared = fin.length
        ? [fin[Math.floor(fin.length*0.02)], fin[Math.floor(fin.length*0.98)]]
        : [0, 1];
      rowC.textContent = '';
      rowC.appendChild(mapPanel(rFirst, '<b>Raw</b> — morning frame', { fixed: shared }));
      rowC.appendChild(mapPanel(sFirst, '<b>Interpolated</b> — morning frame', { fixed: shared }));
      rowC.appendChild(mapPanel(rLast, '<b>Raw</b> — evening frame', { fixed: shared }));
      rowC.appendChild(mapPanel(sLast, '<b>Interpolated</b> — evening frame', { fixed: shared }));

      // row four: the rest
      rowD.textContent = '';
      if (f.stdMax) rowD.appendChild(mapPanel(decode(f.stdMax),
        '<b>Worst capture variability</b> — the highest any frame reached today',
        { positive: 2.0 }));
      if (f.travel) rowD.appendChild(mapPanel(decode(f.travel),
        '<b>Total travel</b> — how far each pixel moved over the day, summed. ' +
        'Steady building gives a small number; churn gives a large one.',
        { positive: 6.0 }));
      hD.style.display = rowD.children.length ? '' : 'none';

      tableSlot.textContent = '';
      const h = document.createElement('h2'); h.textContent = 'Volumes';
      tableSlot.appendChild(h);
      tableSlot.appendChild(volTable([
        ['This day', d.daily],
        ['Overnight after', d.overnight],
        ['Cumulative to date', d.cumulative],
        ['Whole trial', t.total],
      ]));
      const p = document.createElement('p');
      p.className = 'stat';
      p.style.marginTop = '8px';
      p.textContent = 'Volumes are computed at full resolution on the server, at the ' +
        'pipeline thresholds (' + D.dailyThreshold + ' cm daily, ' + D.defaultThreshold +
        ' cm total). The threshold slider changes only what is coloured in the maps.';
      tableSlot.appendChild(p);
      tableSlot.appendChild(sweepChart(t));
      tableSlot.appendChild(travelSection(t));
    });
  }

  bar.querySelector('#thr').addEventListener('input', e => {
    threshold = parseFloat(e.target.value);
    bar.querySelector('#thv').textContent = threshold.toFixed(2);
    draw();
  });
  draw();
  return box;
}

function travelSection(t) {
  const box = document.createElement('div');
  if (!t.travel || !t.travel.stats.hist) return box;
  const s = t.travel.stats;
  const h = document.createElement('h2');
  h.textContent = 'Travel rate over the whole trial';
  box.appendChild(h);

  const grid = document.createElement('div');
  grid.className = 'grid';
  const hi = Math.max(s.max, s.median * 4);
  loadAll([t.travel.rate, t.travel.excess]).then(() => {
    grid.appendChild(mapPanel(decode(t.travel.rate),
      '<b>Travel rate</b> — total movement per lights-on hour, over ' + s.nDays +
      ' days and ' + s.hours + ' h. Log scale.', { log: [s.median / 2, hi] }));
    grid.appendChild(mapPanel(decode(t.travel.excess),
      '<b>Excess travel rate</b> — the same with |net change| subtracted, so a ' +
      'pixel that genuinely built a lot cannot look like churn. Log scale.',
      { log: [s.median / 2, hi] }));
  });
  box.appendChild(grid);

  const chart = document.createElement('div');
  chart.className = 'chart';
  chart.style.marginTop = '16px';
  const c = s.hist.counts, e = s.hist.edges;
  const w = 1000, hgt = 230, pad = 36;
  const maxc = Math.max(...c);
  const lx = v => pad + (Math.log(v) - Math.log(e[0])) /
                  (Math.log(e[e.length-1]) - Math.log(e[0])) * (w - 2*pad);
  let g = '';
  c.forEach((n, i) => {
    const x0 = lx(e[i]), x1 = lx(e[i+1]);
    const bh = (n / maxc) * (hgt - 2*pad);
    g += '<rect x="' + x0 + '" y="' + (hgt-pad-bh) + '" width="' + Math.max(1, x1-x0-1) +
         '" height="' + bh + '" fill="var(--tray)" opacity="0.75"><title>' +
         e[i].toFixed(3) + ' to ' + e[i+1].toFixed(3) + ' cm/h: ' + n + ' pixels</title></rect>';
  });
  s.cuts.forEach(cut => {
    const x = lx(cut.value);
    if (x < pad || x > w-pad) return;
    g += '<line x1="' + x + '" y1="' + (pad-4) + '" x2="' + x + '" y2="' + (hgt-pad) +
         '" stroke="#6fb2e8" stroke-width="1" stroke-dasharray="3 3"/>' +
         '<text x="' + x + '" y="' + (pad-8) + '" fill="#6fb2e8" font-size="11" ' +
         'text-anchor="middle">k=' + cut.k + '</text>';
  });
  g += '<line x1="' + pad + '" y1="' + (hgt-pad) + '" x2="' + (w-pad) + '" y2="' + (hgt-pad) +
       '" stroke="#262d38"/>';
  [e[0], s.median, s.p99, e[e.length-1]].forEach(v => {
    const x = lx(v);
    g += '<text x="' + x + '" y="' + (hgt-pad+15) + '" fill="#93a0b0" font-size="11" ' +
         'text-anchor="middle">' + v.toFixed(2) + '</text>';
  });
  chart.innerHTML = '<svg viewBox="0 0 ' + w + ' ' + hgt + '" preserveAspectRatio="none">' + g +
    '</svg><p class="stat" style="margin:6px 0 0">Excess travel rate per pixel, cm/h, log axis. ' +
    'Dashed lines are cuts at median &times; MAD<sup>k</sup>.</p>';
  box.appendChild(chart);

  const tbl = document.createElement('table');
  tbl.className = 'vol';
  tbl.style.marginTop = '14px';
  tbl.innerHTML = '<tr><th>cut</th><th>rate cm/h</th><th>tray masked</th></tr>' +
    s.cuts.map(cut => '<tr><td>k = ' + cut.k + '</td><td>' + cut.value + '</td><td>' +
      cut.masked + ' %</td></tr>').join('');
  box.appendChild(tbl);

  const p = document.createElement('p');
  p.className = 'stat';
  p.style.marginTop = '8px';
  p.innerHTML = 'median <b>' + s.median + '</b> cm/h &middot; MAD factor <b>&times;' +
    s.madFactor + '</b> &middot; 90th <b>' + s.p90 + '</b> &middot; 99th <b>' + s.p99 +
    '</b> &middot; max <b>' + s.max + '</b>';
  box.appendChild(p);
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
  .wrap { max-width:1400px; margin:0 auto; padding:24px 22px 80px; }
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
  const name = D.projectID + '__' + out.initials + '__registration.json';
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
        'It is in your Downloads folder &mdash; move it to WebServer/_submissions in Dropbox. '
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


def apply_submission(fm_obj, sub, s_dt):
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
              'pairKey': sub.get('pairKey', ''), 'backupStamp': stamp,
              'depthPoints': depth_points, 'videoPoints': video_points}
    with open(fm_obj.localBackupDir + stamp + '_registration.json', 'w') as f:
        json.dump({'submission': sub, 'applied': record}, f, indent=1)

    for path in [fm_obj.localDepthCropFile, fm_obj.localVideoCropFile, fm_obj.localTransMFile]:
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


def build_one(fm_obj, projectID, out_root, category='', delete=False, page_type='Prep'):
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

    payload = build_prep_payload(fm_obj, lp, trial_status, pairs)

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
    applied, skipped, stale_any = [], [], {}
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
            record = apply_submission(fm_obj, sub, s_dt)
        except Exception as e:
            print(projectID + ': failed (' + repr(e) + ')')
            skipped.append(name)
            continue

        print(projectID + ': ' + str(record['nPairs']) + ' pairs from ' + who + ', ' +
              str(record['nInliers']) + ' inliers, fit ' + str(record['rms_px']) + ' px' +
              (' — backup ' + record['backupStamp']))
        applied.append(projectID)
        ledger[os.path.basename(path)] = {
            'projectID': projectID, 'appliedAt': record['appliedAt'],
            'initials': who, 'rms_px': record['rms_px'], 'backupStamp': record['backupStamp']}
        if record['stale']:
            stale_any[projectID] = record['stale']

    if args.DryRun:
        return 0

    if applied and not args.Files:
        fm_obj.createDirectory(sub_dir)
        with open(ledger_path, 'w') as f:
            json.dump(ledger, f, indent=1)
        upload(fm_obj, ledger_path)

    print('\nApplied ' + str(len(applied)) + ', skipped ' + str(len(skipped)) + '.')
    if stale_any:
        print('\nThese projects now have results computed against the old transform:')
        for projectID, what in stale_any.items():
            print('  ' + projectID + ': ' + ', '.join(what))
        print('Rerun those stages, then rebuild the pages with:')
        print('  python createServer.py Prep ' + args.AnalysisID +
              ' --ProjectIDs ' + ' '.join(stale_any.keys()))
    elif applied:
        print('Rebuild the pages with:')
        print('  python createServer.py Prep ' + args.AnalysisID +
              ' --ProjectIDs ' + ' '.join(applied))
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
    args = parser.parse_args()

    if args.PageType == 'ApplyRegistration':
        return apply_registrations(args)

    if args.PageType not in ('Prep', 'Depth'):
        print(args.PageType + ' pages are not implemented yet.')
        return 1

    fm_obj = FM(args.AnalysisID)
    s_dt = fm_obj.s_dt

    projectIDs = s_dt[(s_dt.Prep == True) & (s_dt.RunAnalysis == True)].index.sort_values().to_list()
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
            entry = build_one(fm_obj, projectID, out_root, category=category,
                              delete=args.Delete, page_type=args.PageType)
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
    index_size = write_index(out_root + 'index.html', args.AnalysisID, entries, fm_obj.branch_name)
    print('Wrote ' + out_root + 'index.html (' + str(round(index_size / 1e6, 2)) + ' MB)')

    built = len([e for e in entries if e['status'] == 'ok'])
    incomplete = [e['id'] for e in entries if e['status'] == 'ok' and e['missing']]
    needs = [e['id'] for e in entries if e['status'] == 'needs_prepfiles2']
    needs_depth = [e['id'] for e in entries if e['status'] == 'needs_depthfiles']
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
    print('Open ' + out_root.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir) + 'index.html')
    return 0


if __name__ == '__main__':
    sys.exit(main())