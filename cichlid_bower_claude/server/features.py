"""Project metrics, computed once and cached.

The per-project features page computes everything in the browser, which works
because it holds one project. Comparing thirty-one projects cannot: that is a
gigabyte of depth maps and several hundred megabytes of events, and no browser
should be asked to carry it.

So the same quantities are computed here, per trial, and written to
``Pages/features.json``. The cross-project page then reads thirty-one small
JSON files instead of thirty-one bundles.

A trial is the unit, not a project. The tank is reset between trials, so each
is an independent observation of a fish building from flat sand; averaging them
into one number per project throws away most of the replication.
"""

from __future__ import annotations

import base64
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..collect import bundle as B
from . import corrections as C

SCHEMA = 'cichlid-features/1'
THRESHOLD_CM = 0.6        # sand must move this far to count as moved
COVERAGE = 0.68           # the share of events a dispersion ellipse holds
PIXEL_CM = 0.1030168618

# Spawn depths are kept as a histogram rather than a list: a project can have
# thousands of spawns, and pooling a category means pooling every one of them.
# A quarter-centimetre bin is finer than the sensor's own noise floor, so the
# shape survives and the page carries sixty-four numbers per trial.
DEPTH_SPAN = 8.0          # centimetres either side of flat sand
DEPTH_BINS = 64


def _inside(points, shape) -> np.ndarray:
    """Which pixels fall inside the crop polygon, by ray casting.

    The same test the pages run in the browser, so a volume computed here and
    a map drawn there agree about where the tray is.
    """
    height, width = shape
    ys, xs = np.mgrid[0:height, 0:width]
    ys = ys + 0.5
    xs = xs + 0.5
    inside = np.zeros(shape, bool)
    n = len(points)
    for i in range(n):
        xi, yi = float(points[i][0]), float(points[i][1])
        xj, yj = float(points[i - 1][0]), float(points[i - 1][1])
        if yi == yj:
            continue                    # a horizontal edge crosses no ray
        crosses = (yi > ys) != (yj > ys)
        at = (xj - xi) * (ys - yi) / (yj - yi) + xi
        inside ^= crosses & (xs < at)
    return inside


def _volumes(change: np.ndarray, threshold: float) -> dict:
    """Sand added, sand removed, and the index that compares them.

    Castle is sand that ended higher, pit is sand that ended lower, and a pixel
    that moved less than the threshold is noise and counts as neither. The
    index runs from -1 for a pure pit to +1 for a pure castle.
    """
    area = PIXEL_CM * PIXEL_CM
    finite = np.isfinite(change)
    up = finite & (change >= threshold)
    down = finite & (change <= -threshold)
    castle = float(np.sum(change[up]) * area) if up.any() else 0.0
    pit = float(-np.sum(change[down]) * area) if down.any() else 0.0
    total = castle + pit
    return {
        'castleVolume': round(castle, 2),
        'pitVolume': round(pit, 2),
        'totalVolume': round(total, 2),
        'castleArea': round(float(up.sum()) * area, 2),
        'pitArea': round(float(down.sum()) * area, 2),
        'trayArea': round(float(finite.sum()) * area, 2),
        'bowerIndex': round((castle - pit) / total, 4) if total > 0 else 0.0,
    }


def _dispersion(points: np.ndarray, coverage: float = COVERAGE) -> Optional[dict]:
    """The ellipse holding a given share of the points, from the covariance."""
    if len(points) < 10:
        return None
    centre = points.mean(axis=0)
    centred = points - centre
    cov = (centred.T @ centred) / len(points)
    determinant = max(0.0, float(np.linalg.det(cov)))
    k = -2.0 * np.log(1.0 - coverage)
    return {'n': int(len(points)),
            'cx': float(centre[0]), 'cy': float(centre[1]),
            'area': round(float(np.pi * k * np.sqrt(determinant))
                          * PIXEL_CM * PIXEL_CM, 2)}


def _unpack(packed: dict, key: str, dtype) -> np.ndarray:
    return np.frombuffer(base64.b64decode(packed[key]), dtype=dtype)


def _apply_h(H, xs, ys):
    H = np.asarray(H, float)
    w = H[2, 0] * xs + H[2, 1] * ys + H[2, 2]
    w = np.where(w == 0, np.nan, w)
    return ((H[0, 0] * xs + H[0, 1] * ys + H[0, 2]) / w,
            (H[1, 0] * xs + H[1, 1] * ys + H[1, 2]) / w)


def compute(project_paths, threshold: float = THRESHOLD_CM) -> dict:
    """Every metric for one project, trial by trial."""
    manifest = B.read_manifest(project_paths.manifest)
    if manifest is None:
        raise FileNotFoundError('not collected: ' + project_paths.project_id)
    prep = C.load(project_paths)
    data = B.load_bundle(project_paths.bundle)
    first, last = data['first'], data['last']
    shape = first.shape[1:]

    days = manifest['days']
    by_trial: Dict[int, List[dict]] = {}
    for day in days:
        by_trial.setdefault(int(day['trial']), []).append(day)

    packed = None
    if project_paths.clusters_packed.exists():
        try:
            with open(project_paths.clusters_packed) as handle:
                packed = json.load(handle)
        except Exception:
            packed = None

    events = None
    if packed:
        code = {bid: i for i, bid in enumerate(packed['bids'])}
        events = {
            'code': code,
            'x': _unpack(packed, 'x', np.uint16).astype(float),
            'y': _unpack(packed, 'y', np.uint16).astype(float),
            'bid': _unpack(packed, 'bid', np.uint8),
            'prob': _unpack(packed, 'prob', np.uint8),
            'flags': _unpack(packed, 'flags', np.uint8),
            'day': _unpack(packed, 'day', np.uint16),
            'trial': _unpack(packed, 'trial', np.uint8),
            'hour': _unpack(packed, 'hour', np.uint8),
        }

    trials = []
    for number in sorted(by_trial):
        entry = _trial_metrics(project_paths, prep, number, by_trial[number],
                               first, last, shape, events, threshold)
        trials.append(entry)

    return {
        'schema': SCHEMA,
        'depthSpan': DEPTH_SPAN,
        'depthBins': DEPTH_BINS,
        'projectID': manifest['projectID'],
        'analysisID': manifest['analysisID'],
        'tankID': manifest.get('tankID', ''),
        'built': str(dt.datetime.now().replace(microsecond=0)),
        'threshold': threshold,
        'coverage': COVERAGE,
        'registered': prep.is_registered,
        'cropped': prep.is_cropped,
        'marks': {'newBower': prep.marked_days('new_bower'),
                  'wallBuilding': prep.marked_days('wall_building')},
        'trials': trials,
    }


def _trial_metrics(project_paths, prep, number, days, first, last, shape,
                   events, threshold) -> dict:
    """One trial: what the sand did, and what the fish did to it."""
    out = {'trial': number, 'days': len(days),
           'from': days[0]['date'], 'to': days[-1]['date'],
           'excluded': prep.is_excluded(number),
           'reason': prep.override(number).reason}
    if out['excluded']:
        return out

    crop = prep.depth_crop_for(number)
    outside = ~_inside(crop, shape) if crop and len(crop) >= 3 else None

    start = first[days[0]['index']]
    end = last[days[-1]['index']]
    change = end - start
    if outside is not None:
        change = np.where(outside, np.nan, change)
    out.update(_volumes(change, threshold))

    # how the build accumulated, so a fish that finished early is
    # distinguishable from one that was still going
    curve = []
    for day in days:
        daily = last[day['index']] - start
        if outside is not None:
            daily = np.where(outside, np.nan, daily)
        measured = _volumes(daily, threshold)
        curve.append({'date': day['date'],
                      'total': measured['totalVolume'],
                      'index': measured['bowerIndex']})
    out['curve'] = curve
    if out['totalVolume'] > 0:
        reached = [i for i, c in enumerate(curve)
                   if c['total'] >= 0.9 * out['totalVolume']]
        out['daysTo90'] = (reached[0] + 1) if reached else len(days)
    out['volumePerDay'] = round(out['totalVolume'] / max(1, len(days)), 2)

    if events is None:
        return out
    out.update(_event_metrics(prep, number, days, events, first, last, shape,
                             outside, threshold))
    return out


def _event_metrics(prep, number, days, events, first, last, shape, outside,
                   threshold) -> dict:
    """Behaviour, in depth coordinates, for one trial."""
    code = events['code']
    cut = int(0.67 * 255)
    usable = ((events['flags'] & 1) != 0) & ((events['flags'] & 2) != 0) \
        & (events['bid'] != 255) & (events['prob'] >= cut) \
        & (events['trial'] == number)

    out = {'events': int(usable.sum())}
    for name, bid in (('scoops', 'c'), ('spits', 'p'), ('multiples', 'b'),
                      ('spawns', 's'), ('feedScoops', 'f'), ('feedSpits', 't')):
        out[name] = int((usable & (events['bid'] == code[bid])).sum())
    out['eventsPerDay'] = round(out['events'] / max(1, len(days)), 1)

    H = prep.transform_for(number)
    if H is None:
        out['placed'] = False
        return out
    out['placed'] = True

    dx, dy = _apply_h(H, events['x'], events['y'])
    scoops = usable & (events['bid'] == code['c'])
    spits = usable & (events['bid'] == code['p'])
    scoop_shape = _dispersion(np.column_stack([dx[scoops], dy[scoops]]))
    spit_shape = _dispersion(np.column_stack([dx[spits], dy[spits]]))
    out['scoopEllipse'] = scoop_shape['area'] if scoop_shape else None
    out['spitEllipse'] = spit_shape['area'] if spit_shape else None
    if scoop_shape and spit_shape and scoop_shape['area'] > 0:
        out['spreadRatio'] = round(spit_shape['area'] / scoop_shape['area'], 3)
        out['centroidGap'] = round(float(np.hypot(
            scoop_shape['cx'] - spit_shape['cx'],
            scoop_shape['cy'] - spit_shape['cy'])) * PIXEL_CM, 2)

    # spawn depth against the sand as it was on the spawn's own day, which is
    # the comparison the summary page's histogram makes
    spawns = usable & (events['bid'] == code['s'])
    if spawns.any():
        index = {int(d['index']): d for d in days}
        start = first[days[0]['index']]
        depths = []
        for day_index in sorted(set(events['day'][spawns].tolist())):
            day = index.get(int(day_index))
            if day is None:
                continue
            cumulative = last[day['index']] - start
            if outside is not None:
                cumulative = np.where(outside, np.nan, cumulative)
            here = spawns & (events['day'] == day_index)
            xs = np.clip(np.nan_to_num(dx[here]), 0, shape[1] - 1).astype(int)
            ys = np.clip(np.nan_to_num(dy[here]), 0, shape[0] - 1).astype(int)
            values = cumulative[ys, xs]
            depths.extend(values[np.isfinite(values)].tolist())
        if depths:
            depths = np.asarray(depths)
            out['spawnDepthMedian'] = round(float(np.median(depths)), 3)
            out['spawnDepthMean'] = round(float(depths.mean()), 3)
            out['spawnDepthQuartiles'] = [
                round(float(np.percentile(depths, q)), 3) for q in (25, 50, 75)]
            out['spawnOverCastle'] = round(
                100.0 * float((depths > 0).sum()) / len(depths), 1)
            out['spawnsMeasured'] = int(len(depths))
            edges = np.linspace(-DEPTH_SPAN, DEPTH_SPAN, DEPTH_BINS + 1)
            counts, _ = np.histogram(np.clip(depths, -DEPTH_SPAN, DEPTH_SPAN),
                                     bins=edges)
            out['spawnDepthHistogram'] = counts.astype(int).tolist()
    return out


def path_for(project_paths) -> Path:
    return project_paths.pages_dir / 'features.json'


def is_stale(project_paths) -> bool:
    """Whether the cached metrics predate the data or the corrections.

    Both matter: a recollection changes the depth maps, and a new crop or
    registration changes what those maps mean. Checking only one would leave
    numbers that look current and are not.
    """
    cached = path_for(project_paths)
    if not cached.exists():
        return True
    when = cached.stat().st_mtime
    for source in (project_paths.manifest, project_paths.prep_json):
        if source.exists() and source.stat().st_mtime > when:
            return True
    return False


def load(project_paths, rebuild: bool = True) -> Optional[dict]:
    """The cached metrics, recomputing if the inputs have moved on."""
    cached = path_for(project_paths)
    if rebuild and is_stale(project_paths):
        try:
            result = compute(project_paths)
        except FileNotFoundError:
            return None
        cached.parent.mkdir(parents=True, exist_ok=True)
        with open(cached, 'w') as handle:
            json.dump(result, handle)
        return result
    if not cached.exists():
        return None
    try:
        with open(cached) as handle:
            return json.load(handle)
    except Exception:
        return None
