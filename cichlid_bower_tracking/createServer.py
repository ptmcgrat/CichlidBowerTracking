import argparse, base64, datetime, glob, json, os, re, shutil, subprocess, sys, warnings

import cv2
import numpy as np

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


def encode_depth(arr, scale=DEPTH_SCALE):
    """Float array in cm -> base64 PNG carrying exact values.

    Value is quantized to int16 counts and split across the red (high byte) and
    green (low byte) channels; blue is 255 where the pixel is valid and 0 where
    it was NaN. Alpha is left fully opaque throughout so the browser never
    premultiplies away the payload."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None, None
    offset = float(np.floor(finite.min() / scale) - 1) * scale
    counts = np.round((arr - offset) / scale)
    counts = np.where(np.isfinite(counts), counts, 0)
    if counts.max() > 65535:
        raise ValueError('Depth range too wide for 16 bits at scale ' + str(scale))
    counts = counts.astype(np.uint16)

    valid = np.isfinite(arr)
    h, w = arr.shape
    bgra = np.zeros((h, w, 4), np.uint8)
    bgra[:, :, 0] = np.where(valid, 255, 0)          # blue: validity flag
    bgra[:, :, 1] = (counts & 0xFF).astype(np.uint8)  # green: low byte
    bgra[:, :, 2] = (counts >> 8).astype(np.uint8)    # red: high byte
    bgra[:, :, 3] = 255
    ok, buf = cv2.imencode('.png', bgra)
    if not ok:
        raise RuntimeError('PNG encoding failed')
    meta = {'scale': scale, 'offset': offset, 'width': w, 'height': h}
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
    """Depth change with out-of-tray pixels removed, matching the filter that
    PrepPreparer._cropDepth applies before showing the crop image."""
    difference = last - first
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
    depth_points = parse_points(fm.localDepthCropFile)
    video_points = parse_points(fm.localVideoCropFile)
    transM = np.load(fm.localTransMFile)
    crop_mask = polygon_mask((lp.height, lp.width), depth_points)

    # Overview: one depth still and one warped Pi still per pair, so the
    # registration can be checked against any trial rather than just setup.
    overview_pairs = []
    for p in pairs:
        depth_img = cv2.imread(p['depth_jpg'])
        pi_img = cv2.imread(p['pi_jpg'])
        if depth_img is None or pi_img is None:
            continue
        warped = cv2.warpPerspective(pi_img, transM, (lp.width, lp.height))
        overview_pairs.append({
            'key': p['key'], 'label': p['label'], 'gapMinutes': p['gapMinutes'],
            'depth': encode_image(depth_img, max_width=lp.width),
            'piWarped': encode_image(warped, max_width=lp.width),
        })

    first = pairs[0]
    pi_first = cv2.imread(first['pi_jpg'])
    overview = {
        'pairs': overview_pairs,
        'depthCropImage': encode_image(cv2.imread(first['depth_jpg']), max_width=lp.width),
        'videoCropImage': encode_image(pi_first, max_width=min(pi_first.shape[1], 900)),
        'piSize': [int(pi_first.shape[1]), int(pi_first.shape[0])],
    }

    trials = []
    by_trial = {}
    for p in pairs:
        by_trial.setdefault(p['trial'], {})[p['side']] = p

    for i, trial in enumerate(lp.trials, 1):
        sides = by_trial.get(i, {})
        entry = {
            'number': i,
            'start': str(trial.startTime), 'stop': str(trial.stopTime),
            'reset': str(trial.resetTime) if trial.resetTime is not None else None,
            'numDays': int(trial.num_days),
            'nFrames': len(trial.frames), 'nDaylightFrames': len(trial.daylight_frames),
            'movies': [int(m.index) for m in trial.movies],
            'missing': [] if len(sides) == 2 else ['PrepFiles2 pair for this trial'],
            'panels': {},
        }
        if len(sides) != 2:
            trials.append(entry)
            continue

        t_first = load_npy(sides['First']['depth_npy'])
        t_last = load_npy(sides['Last']['depth_npy'])
        t_change, _ = filtered_change(t_first, t_last)
        t_src, t_meta = encode_depth(t_change)

        panels = {
            'firstDepthRGB': encode_image(cv2.imread(sides['First']['depth_jpg'])),
            'lastDepthRGB': encode_image(cv2.imread(sides['Last']['depth_jpg'])),
            'firstPi': encode_image(cv2.warpPerspective(
                cv2.imread(sides['First']['pi_jpg']), transM, (lp.width, lp.height))),
            'lastPi': encode_image(cv2.warpPerspective(
                cv2.imread(sides['Last']['pi_jpg']), transM, (lp.width, lp.height))),
            'change': {'src': t_src, 'meta': t_meta,
                       'stats': depth_stats(t_change, crop_mask),
                       'label': 'Trial ' + str(i) + ', last frame minus first'},
        }

        # The reset frame only exists in the original PrepFiles.
        reset_path = fm.localPrepDir + 'Trial_' + str(i) + 'ResetDepth.npy'
        if os.path.exists(reset_path):
            r_change, _ = filtered_change(load_npy(reset_path), t_last)
            r_src, r_meta = encode_depth(r_change)
            panels['resetChange'] = {'src': r_src, 'meta': r_meta,
                                     'stats': depth_stats(r_change, crop_mask),
                                     'label': 'Trial ' + str(i) + ', last frame minus reset'}
        entry['panels'] = panels
        entry['gaps'] = {side: sides[side]['gapMinutes'] for side, _ in TRIAL_SIDES}
        trials.append(entry)

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
        'frameSize': [lp.width, lp.height],
        'overview': overview, 'trials': trials,
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
  .stage polygon { fill: none; stroke: var(--tray); stroke-width: 3; vector-effect: non-scaling-stroke; }
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
  <div class="tabs" id="tabs" role="tablist"></div>
  <div id="body"></div>
  <p class="foot" id="foot"></p>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const D = JSON.parse(document.getElementById('payload').textContent);

// jet, matching the colormap the pipeline figures use
function jet(t) {
  t = Math.min(1, Math.max(0, t));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 1)));
  return [r*255, g*255, b*255];
}

// Pull exact centimetre values back out of the packed PNG.
function decodeDepth(img, meta) {
  const c = document.createElement('canvas');
  c.width = meta.width; c.height = meta.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const px = ctx.getImageData(0, 0, meta.width, meta.height).data;
  const out = new Float32Array(meta.width * meta.height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    out[i] = px[p+2] === 0 ? NaN : ((px[p] << 8) | px[p+1]) * meta.scale + meta.offset;
  }
  return out;
}

function polygonSVG(points, w, h) {
  const pts = points.map(p => p.join(',')).join(' ');
  return '<svg viewBox="0 0 ' + w + ' ' + h + '" preserveAspectRatio="none" aria-hidden="true">' +
         '<polygon points="' + pts + '"/></svg>';
}

function statLine(s) {
  if (!s || s.median === undefined) return 'No valid pixels inside the tray.';
  return 'Median ' + s.median.toFixed(2) + ' cm · 1–99% ' + s.p1.toFixed(2) + ' to ' +
         s.p99.toFixed(2) + ' cm · ' + (100*s.valid_fraction).toFixed(1) + '% of tray pixels valid';
}

// A depth panel: colours itself from the decoded array, range adjustable, and
// reports the value under the cursor.
function depthPanel(layer, initialRange) {
  const fig = document.createElement('figure');
  fig.innerHTML =
    '<div class="stage"><canvas></canvas>' + polygonSVG(D.depthPoints, D.frameSize[0], D.frameSize[1]) +
    '<span class="readout">—</span></div>' +
    '<div class="scalebar"></div>' +
    '<div class="controls"><label>Range ±<span class="rangeval"></span> cm</label>' +
    '<input type="range" min="0.5" max="10" step="0.5"></div>' +
    '<figcaption><b>' + layer.label + '</b><br>' + statLine(layer.stats) + '</figcaption>';

  const canvas = fig.querySelector('canvas');
  const readout = fig.querySelector('.readout');
  const slider = fig.querySelector('input');
  const rangeval = fig.querySelector('.rangeval');
  const scalebar = fig.querySelector('.scalebar');
  slider.value = initialRange;

  const meta = layer.meta;
  canvas.width = meta.width; canvas.height = meta.height;
  const ctx = canvas.getContext('2d');
  const img = new Image();
  let values = null;

  function paint() {
    if (!values) return;
    const range = parseFloat(slider.value);
    rangeval.textContent = range.toFixed(1);
    const out = ctx.createImageData(meta.width, meta.height);
    for (let i = 0, p = 0; i < values.length; i++, p += 4) {
      const v = values[i];
      if (Number.isNaN(v)) { out.data[p] = out.data[p+1] = out.data[p+2] = 0; out.data[p+3] = 255; continue; }
      const [r, g, b] = jet((v + range) / (2 * range));
      out.data[p] = r; out.data[p+1] = g; out.data[p+2] = b; out.data[p+3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    let stops = [];
    for (let i = 0; i <= 10; i++) { const [r,g,b] = jet(i/10); stops.push('rgb('+[r|0,g|0,b|0]+') '+(i*10)+'%'); }
    scalebar.style.background = 'linear-gradient(to right,' + stops.join(',') + ')';
  }

  img.onload = () => { values = decodeDepth(img, meta); paint(); };
  img.src = layer.src;
  slider.addEventListener('input', paint);

  canvas.parentElement.addEventListener('mousemove', ev => {
    if (!values) return;
    const r = canvas.getBoundingClientRect();
    const col = Math.floor((ev.clientX - r.left) / r.width * meta.width);
    const row = Math.floor((ev.clientY - r.top) / r.height * meta.height);
    if (col < 0 || row < 0 || col >= meta.width || row >= meta.height) return;
    const v = values[row * meta.width + col];
    readout.textContent = (Number.isNaN(v) ? 'masked' : v.toFixed(2) + ' cm') +
                          '  ·  row ' + row + ', col ' + col;
  });
  return fig;
}

function imagePanel(src, caption, points, size) {
  const fig = document.createElement('figure');
  const w = size ? size[0] : D.frameSize[0], h = size ? size[1] : D.frameSize[1];
  fig.innerHTML = '<div class="stage"><img src="' + src + '" alt="">' +
                  (points ? polygonSVG(points, w, h) : '') + '</div>' +
                  '<figcaption>' + caption + '</figcaption>';
  return fig;
}

// The registration check: the divider follows the cursor across the panel, so the
// warped Pi frame and the depth frame can be compared edge to edge.
function swipePanel(base, overSrc, caption) {
  const fig = document.createElement('figure');
  fig.innerHTML =
    '<div class="stage"><img class="base" src="' + base + '" alt="">' +
    '<div class="swipe"><div class="over"><img src="' + overSrc + '" alt=""></div>' +
    '<div class="handle"></div></div>' +
    polygonSVG(D.depthPoints, D.frameSize[0], D.frameSize[1]) + '</div>' +
    '<div class="controls"><label>Divider</label>' +
    '<input type="range" min="0" max="100" step="1" value="50" aria-label="Divider position"></div>' +
    '<figcaption>' + caption + '</figcaption>';

  const stage = fig.querySelector('.stage');
  const baseImg = fig.querySelector('img.base');
  const swipe = fig.querySelector('.swipe');
  const over = fig.querySelector('.over');
  const overImg = over.querySelector('img');
  const handle = fig.querySelector('.handle');
  const slider = fig.querySelector('input');
  let frac = 0.5;

  // The overlay is clipped by its parent's width, so its own image must stay
  // pinned at the full panel width or it would squeeze as the divider moves.
  function sync() {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w) return;
    overImg.style.width = w + 'px';
    overImg.style.height = h + 'px';
    over.style.width = (frac * 100) + '%';
    handle.style.left = (frac * 100) + '%';
  }
  function set(f) { frac = Math.min(1, Math.max(0, f)); slider.value = frac * 100; sync(); }

  swipe.addEventListener('mousemove', ev => {
    const r = swipe.getBoundingClientRect();
    set((ev.clientX - r.left) / r.width);
  });
  swipe.addEventListener('touchmove', ev => {
    const r = swipe.getBoundingClientRect();
    set((ev.touches[0].clientX - r.left) / r.width);
  }, { passive: true });
  slider.addEventListener('input', () => set(slider.value / 100));

  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(sync).observe(stage);
  if (baseImg.complete) sync(); else baseImg.addEventListener('load', sync);
  return fig;
}

function renderOverview() {
  const o = D.overview, box = document.createElement('div');

  const bar = document.createElement('div');
  bar.className = 'bar';
  bar.innerHTML =
    '<label for="pairPick">Registration checked against</label>' +
    '<select id="pairPick">' + o.pairs.map(p =>
      '<option value="' + p.key + '">' + p.label +
      (p.gapMinutes === null || p.gapMinutes === undefined ? '' :
        ' (' + p.gapMinutes + ' min apart)') + '</option>').join('') + '</select>' +
    '<a class="btn" id="fixLink" href="Register.html">Fix registration or crops</a>';
  box.appendChild(bar);

  const grid = document.createElement('div');
  grid.className = 'grid';
  grid.appendChild(imagePanel(o.depthCropImage,
    'Depth camera with the tray crop.', D.depthPoints));
  grid.appendChild(imagePanel(o.videoCropImage,
    'Pi camera with the video crop.', D.videoPoints, o.piSize));

  const slot = document.createElement('div');
  grid.appendChild(slot);
  box.appendChild(grid);

  function showPair(key) {
    const p = o.pairs.find(x => x.key === key) || o.pairs[0];
    slot.textContent = '';
    slot.appendChild(swipePanel(p.depth, p.piWarped,
      'Registration on ' + p.label + '. Move the pointer across to wipe the warped Pi ' +
      'frame over the depth frame — the tray edges should stay continuous.'));
    document.getElementById('fixLink').href = 'Register.html?pair=' + encodeURIComponent(p.key);
  }
  bar.querySelector('#pairPick').addEventListener('change', e => showPair(e.target.value));
  if (o.pairs.length) showPair(o.pairs[0].key);
  else slot.innerHTML = '<div class="note">No image pairs are available.</div>';
  return box;
}

function renderTrial(t) {
  const box = document.createElement('div');
  if (t.missing.length) {
    box.innerHTML = '<div class="note">This trial is missing prep files, so its panels ' +
      'cannot be drawn:<ul>' + t.missing.map(m => '<li>' + m + '</li>').join('') + '</ul></div>';
    return box;
  }
  const p = t.panels, grid = document.createElement('div');
  grid.className = 'grid';
  grid.appendChild(imagePanel(p.firstDepthRGB, 'Depth view, first daylight frame.', D.depthPoints));
  grid.appendChild(imagePanel(p.lastDepthRGB, 'Depth view, last daylight frame.', D.depthPoints));
  grid.appendChild(imagePanel(p.firstPi, 'Pi view at the start, warped into depth space.', D.depthPoints));
  grid.appendChild(imagePanel(p.lastPi, 'Pi view at the end, warped into depth space.', D.depthPoints));
  grid.appendChild(depthPanel(p.change, 2));
  if (p.resetChange) grid.appendChild(depthPanel(p.resetChange, 2));
  box.appendChild(grid);
  return box;
}

function meta(pairs) {
  document.getElementById('meta').innerHTML =
    pairs.map(([k, v]) => '<div>' + k + '<b>' + v + '</b></div>').join('');
}

function build() {
  document.getElementById('title').textContent = D.projectID;
  document.getElementById('subtitle').textContent =
    'Prep — tray crop and camera registration, reviewed per trial.';
  meta([
    ['Tank', D.tankID], ['Analysis', D.analysisID], ['Sensor', D.device],
    ['Recording', D.masterStart.slice(0, 16) + ' to ' + D.masterStop.slice(0, 16)],
    ['Depth frames', D.nFrames], ['Videos', D.nMovies], ['Trials', D.trials.length],
  ]);

  const body = document.getElementById('body');
  const tabs = document.getElementById('tabs');
  const views = [{ name: 'Overview', render: renderOverview }].concat(
    D.trials.map(t => ({
      name: 'Trial ' + t.number + (t.missing.length ? ' ⚠' : ''),
      render: () => {
        const wrap = document.createElement('div');
        const bits = ['Ran ' + t.start.slice(0, 16) + ' to ' + t.stop.slice(0, 16),
                      t.numDays + ' days', t.nDaylightFrames + ' daylight frames of ' + t.nFrames,
                      'videos ' + (t.movies.length ? t.movies.join(', ') : 'none')];
        if (t.gaps) bits.push('pairs matched to ' + t.gaps.First + ' and ' +
                              t.gaps.Last + ' min');
        if (t.reset) bits.push('tank reset ' + t.reset.slice(0, 16));
        const p = document.createElement('p');
        p.className = 'sub'; p.textContent = bits.join(' · ');
        wrap.appendChild(p);
        wrap.appendChild(renderTrial(t));
        return wrap;
      }
    })));

  if (D.logIssues && D.logIssues.length) {
    const n = document.createElement('div');
    n.className = 'note';
    n.innerHTML = 'The log parser flagged this project:<ul>' +
                  D.logIssues.map(x => '<li>' + x + '</li>').join('') + '</ul>';
    body.parentElement.insertBefore(n, tabs);
  }

  const buttons = views.map((v, i) => {
    const b = document.createElement('button');
    b.textContent = v.name; b.setAttribute('role', 'tab');
    b.addEventListener('click', () => show(i));
    tabs.appendChild(b);
    return b;
  });
  const cache = [];
  function show(i) {
    buttons.forEach((b, j) => b.setAttribute('aria-selected', j === i));
    body.textContent = '';
    if (!cache[i]) cache[i] = views[i].render();
    body.appendChild(cache[i]);
  }
  show(0);

  document.getElementById('foot').textContent =
    'Built ' + D.built + ' from branch ' + D.branch + '. ' + D.prepLog.replace(/\n/g, ' · ');
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
  .wrap { max-width:1240px; margin:0 auto; padding:28px 24px 72px; }
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
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(196px,1fr)); gap:14px; }
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
  .card .body { padding:10px 11px; flex:1; }
  .card h2 { font-size:13px; font-weight:600; margin:0 0 3px; word-break:break-word; }
  .card p { margin:0; font-size:12px; color:var(--ink-dim); font-variant-numeric:tabular-nums; }
  .pages { display:flex; gap:4px; flex-wrap:wrap; padding:0 11px 11px; }
  .pages a, .pages em { font-size:11px; padding:3px 7px; border-radius:999px;
                        border:1px solid var(--line); font-style:normal; }
  .pages a { color:var(--ink); border-color:#3a4553; }
  .pages a:hover { background:var(--ink); color:var(--bg); text-decoration:none; }
  .pages em { color:#5c6675; }
  .flag { display:inline-block; margin-top:7px; font-size:12px; padding:2px 8px;
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
  <div class="grid" id="grid"></div>
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

function rank(x) { return x.status === 'failed' ? 0 : (x.status === 'needs_prepfiles2' ? 1 : (x.missing ? 2 : 3)); }

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
  const R = 66, OFFSET = 74;
  host.addEventListener('mousemove', ev => {
    const img = host.querySelector('img');
    const r = img.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    const zoom = 5;
    loupe.style.display = 'block';
    const above = y > 2 * R + OFFSET;
    loupe.style.left = Math.min(Math.max(x - R, 4), r.width - 2 * R - 4) + 'px';
    loupe.style.top = (above ? y - 2 * R - OFFSET : y + OFFSET) + 'px';
    loupe.style.backgroundImage = 'url(' + getSrc() + ')';
    loupe.style.backgroundSize = (r.width * zoom) + 'px ' + (r.height * zoom) + 'px';
    const lx = parseFloat(loupe.style.left), ly = parseFloat(loupe.style.top);
    loupe.style.backgroundPosition =
      (-x * zoom + (x - lx)) + 'px ' + (-y * zoom + (y - ly)) + 'px';
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
  paint(svg, viewPair.depthSize,
        '<polygon class="crop" points="' + crops.depth.map(p => p.join(',')).join(' ') + '"/>');
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


def build_one(fm_obj, projectID, out_root, category='', delete=False):
    """Build the Prep and Register pages for a single project.

    Never raises: a project that cannot be built is recorded so the sweep
    continues and the index still shows it with a reason."""
    entry = {'id': projectID, 'tank': '', 'category': category, 'trials': 0, 'start': '',
             'thumb': None, 'missing': 0, 'status': 'ok', 'pages': {}, 'error': ''}

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

    reg_payload = build_register_payload(fm_obj, lp, pairs)
    size += write_register(project_dir + 'Register.html', fm_obj.analysisID, reg_payload)
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

    if args.PageType != 'Prep':
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

    print('Building Prep pages for ' + str(len(projectIDs)) + ' project(s) in ' + args.AnalysisID)
    entries, failed = [], []
    for i, projectID in enumerate(projectIDs, 1):
        prefix = '[' + str(i) + '/' + str(len(projectIDs)) + '] ' + projectID
        if args.SkipExisting and os.path.exists(out_root + projectID + '/Prep.html'):
            print(prefix + ': already built, skipping')
            continue
        print(prefix + ' ' + str(datetime.datetime.now()), flush=True)
        category = ''
        if 'Category' in s_dt.columns:
            raw = s_dt.loc[projectID, 'Category']
            category = '' if raw is None or str(raw).strip().lower() in ('nan', '') else str(raw).strip()
        try:
            entry = build_one(fm_obj, projectID, out_root, category=category, delete=args.Delete)
        except Exception as e:
            entry = {'id': projectID, 'tank': '', 'category': category, 'trials': 0, 'start': '',
                     'thumb': None, 'missing': 0, 'status': 'failed', 'pages': {}, 'error': repr(e)}
        entries.append(entry)

        if entry['status'] != 'ok':
            failed.append(projectID)
            print('    ' + entry['status'] + ': ' + entry['error'])
            continue
        print('    wrote ' + entry['id'] + '/Prep.html (' +
              str(round(entry['size'] / 1e6, 2)) + ' MB, ' + str(entry['trials']) + ' trials)')
        if not args.NoUpload:
            for name in entry['pages'].values():
                upload(fm_obj, out_root + projectID + '/' + name)

    index_size = write_index(out_root + 'index.html', args.AnalysisID, entries, fm_obj.branch_name)
    print('Wrote ' + out_root + 'index.html (' + str(round(index_size / 1e6, 2)) + ' MB)')

    built = len([e for e in entries if e['status'] == 'ok'])
    incomplete = [e['id'] for e in entries if e['status'] == 'ok' and e['missing']]
    needs = [e['id'] for e in entries if e['status'] == 'needs_prepfiles2']
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