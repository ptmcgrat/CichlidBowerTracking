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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..collect import bundle as B
from ..collect import residual as R
from . import corrections as C

SCHEMA = 'cichlid-features/3'

# The ten codes the classifier assigns, with the name each gets in a column
# heading. 'b' is both a scoop and a spit, so it is counted on its own and
# left out of the scoop-to-spit ratio rather than counted twice.
BEHAVIOUR = (('scoops', 'c'), ('spits', 'p'), ('multiples', 'b'),
             ('feedScoops', 'f'), ('feedSpits', 't'), ('feedMultiples', 'm'),
             ('spawns', 's'), ('dropSand', 'd'), ('fishOther', 'o'),
             ('noFishOther', 'x'))


@dataclass(frozen=True)
class Params:
    """Every setting that changes a number, in one object.

    These were constants, and two of them were written twice -- the confidence
    cut was 0.67 here and a slider there, so a count in a spreadsheet need not
    have matched the count on screen. Passing them explicitly makes a
    disagreement impossible to have by accident, and lets the workbook record
    exactly what produced it.

    ``spread`` changes only a drawing, and is carried so provenance can say so
    rather than leaving the reader to wonder whether it moved a volume.
    """

    confidence: float = 0.67      # minimum classifier probability
    hour_from: int = 8            # events outside these hours are not counted
    hour_to: int = 18
    threshold: float = 0.6        # cm before sand counts as moved
    coverage: float = 0.68        # share of events a dispersion ellipse holds
    split_at: float = 0.5         # share of gross volume that ends half one
    min_for_shape: int = 25       # events before an aspect ratio is reported
    spread: int = 30              # depth px an event is drawn over (display)

    @property
    def cut(self) -> int:
        """The confidence as the packed byte the event columns hold."""
        return int(round(self.confidence * 255))


DEFAULTS = Params()

# The median aspect ratio a perfectly circular cloud of n points produces, from
# 20,000 simulated isotropic draws per n. An ellipse fitted to a handful of
# points is elongated whatever the cloud is, and the bias runs the wrong way:
# the categories with more events a day get the smaller bias. The same table
# the pages carry, so the two cannot disagree.
NULL_ASPECT = ((10, 1.528), (15, 1.392), (20, 1.325), (25, 1.278), (30, 1.251),
               (40, 1.212), (50, 1.185), (75, 1.148), (100, 1.126),
               (150, 1.101), (200, 1.088), (300, 1.070), (500, 1.054),
               (1000, 1.038), (2000, 1.027))


def null_aspect(n: int) -> float:
    """Interpolated in 1/sqrt(n), which is where the curve is near straight."""
    if not n or n <= NULL_ASPECT[0][0]:
        return NULL_ASPECT[0][1]
    if n >= NULL_ASPECT[-1][0]:
        last_n, last_a = NULL_ASPECT[-1]
        return 1 + (last_a - 1) * (last_n / n) ** 0.5
    for i in range(1, len(NULL_ASPECT)):
        n1, a1 = NULL_ASPECT[i]
        if n > n1:
            continue
        n0, a0 = NULL_ASPECT[i - 1]
        x, x0, x1 = n ** -0.5, n0 ** -0.5, n1 ** -0.5
        return a1 + (a0 - a1) * (x - x1) / (x0 - x1)
    return 1.0
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


def _shape_of(points: np.ndarray, params: 'Params') -> Optional[dict]:
    """The ellipse's size and its elongation, from one covariance.

    Area is sqrt(det) and aspect is sqrt(l1/l2): the same matrix answers both,
    and they are independent -- a cloud can be large and round or small and
    stretched. Aspect is withheld below min_for_shape because a fitted ellipse
    is elongated whatever the cloud is when the points are few.
    """
    if len(points) < 10:
        return None
    centre = points.mean(axis=0)
    centred = points - centre
    cov = (centred.T @ centred) / len(points)
    determinant = max(0.0, float(np.linalg.det(cov)))
    k = -2.0 * np.log(1.0 - params.coverage)
    trace = float(cov[0, 0] + cov[1, 1])
    root = max(0.0, trace * trace / 4 - determinant) ** 0.5
    major = (k * max(0.0, trace / 2 + root)) ** 0.5
    minor = (k * max(0.0, trace / 2 - root)) ** 0.5
    out = {'n': int(len(points)),
           'cx': round(float(centre[0]), 2), 'cy': round(float(centre[1]), 2),
           'areaCm2': round(float(np.pi * k * determinant ** 0.5)
                            * PIXEL_CM * PIXEL_CM, 2)}
    if len(points) >= params.min_for_shape and minor > 0:
        # float(), not the numpy scalar: np.float64 happens to subclass float
        # so json.dump survives it, but np.float32 does not and the cache is
        # written without a guard
        out['aspect'] = round(float(major / minor), 3)
        out['nullAspect'] = round(null_aspect(len(points)), 3)
        out['angleDeg'] = round(float(np.degrees(
            0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1]))), 1)
    return out


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


def compute(project_paths, params: Optional['Params'] = None,
            threshold: Optional[float] = None) -> dict:
    """Every metric for one project, trial by trial.

    ``threshold`` is kept for callers that passed it positionally before
    Params existed; it overrides the one in params.
    """
    params = params or DEFAULTS
    if threshold is not None:
        params = Params(**dict(asdict(params), threshold=threshold))
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
                               first, last, shape, events, params,
                               residual_maps=data.get('residual'))
        trials.append(entry)

    return {
        'schema': SCHEMA,
        'depthSpan': DEPTH_SPAN,
        'depthBins': DEPTH_BINS,
        'projectID': manifest['projectID'],
        'analysisID': manifest['analysisID'],
        'tankID': manifest.get('tankID', ''),
        'built': str(dt.datetime.now().replace(microsecond=0)),
        'threshold': params.threshold,
        'coverage': params.coverage,
        'params': asdict(params),
        'category': '',          # filled by the caller, which has the table
        'registered': prep.is_registered,
        'cropped': prep.is_cropped,
        'marks': {'newBower': prep.marked_days('new_bower'),
                  'wallBuilding': prep.marked_days('wall_building')},
        'trials': trials,
    }


def truncate_at_new_bower(prep, days) -> tuple:
    """The days to analyse, and where the trial was cut short.

    A day marked 'new bower' means the fish abandoned what it had and started
    again somewhere else, so the cumulative map from there on is two
    overlapping structures and its volume and index describe neither. For
    analysis the trial ends at the previous day's close.

    The FIRST such mark in the trial wins: a second new bower is already past
    the end. A mark on the trial's own first day leaves nothing to analyse, so
    it is reported and the trial is left whole -- dropping the trial silently
    would be a worse answer than an obviously odd one.
    """
    marks = prep.day_marks or {}
    for position, day in enumerate(days):
        if not getattr(marks.get(str(day['index'])), 'new_bower', False):
            continue
        if position == 0:
            return days, {'at': None, 'why': 'new bower on the first day'}
        return days[:position], {'at': int(day['index']),
                                 'date': day['date'],
                                 'droppedDays': len(days) - position}
    return days, None


def _trial_metrics(project_paths, prep, number, days, first, last, shape,
                   events, params, residual_maps=None) -> dict:
    """One trial: what the sand did, and what the fish did to it."""
    logged = len(days)
    days, truncation = truncate_at_new_bower(prep, days)
    out = {'trial': number, 'days': len(days), 'daysLogged': logged,
           'from': days[0]['date'], 'to': days[-1]['date'],
           'excluded': prep.is_excluded(number),
           'reason': prep.override(number).reason}
    if truncation:
        out['truncatedAtNewBower'] = truncation
    if out['excluded']:
        return out

    crop = prep.depth_crop_for(number)
    inside = _inside(crop, shape) if crop and len(crop) >= 3 else None
    outside = ~inside if inside is not None else None

    # The residual mask, from this trial's own days, on top of the crop. It
    # was applied on the depth tab and nowhere else, so every volume, index
    # and spawn depth here was measured partly on pixels the depth tab had
    # already called unreliable.
    unreliable = trial_mask(residual_maps, days, prep.residual_k, inside)
    if unreliable is not None:
        outside = unreliable if outside is None else (outside | unreliable)
        out['maskedPixels'] = int(unreliable.sum())
    out['residualK'] = prep.residual_k

    # Depth is distance from the sensor down to the sand, so sand piling up
    # makes the number SMALLER. Castle-positive change is therefore
    # start - end, not end - start. It was the other way round here, and
    # everything this module reported had castle and pit swapped: a castle
    # builder came out with a bower index of -0.85, 'spawnOverCastle' counted
    # spawns over pits, and the cross-project page's spawn-depth axis was
    # mirrored. The browser had it right, which is why the features page and
    # this file disagreed.
    start = first[days[0]['index']]
    end = last[days[-1]['index']]
    change = start - end
    if outside is not None:
        change = np.where(outside, np.nan, change)
    out.update(_volumes(change, params.threshold))

    # how the build accumulated, so a fish that finished early is
    # distinguishable from one that was still going
    curve = []
    for day in days:
        daily = start - last[day['index']]
        if outside is not None:
            daily = np.where(outside, np.nan, daily)
        measured = _volumes(daily, params.threshold)
        curve.append({'date': day['date'],
                      'total': measured['totalVolume'],
                      'index': measured['bowerIndex']})
    out['curve'] = curve
    if out['totalVolume'] > 0:
        reached = [i for i, c in enumerate(curve)
                   if c['total'] >= 0.9 * out['totalVolume']]
        out['daysTo90'] = (reached[0] + 1) if reached else len(days)
    out['volumePerDay'] = round(out['totalVolume'] / max(1, len(days)), 2)

    # --- per day, and the halves the gross volume implies -----------------
    rows = day_volumes(first, last, days, outside, params)
    out['dayVolumes'] = rows
    dark = sum(r['dark']['totalVolume'] for r in rows if r['dark'])
    light = sum(r['light']['totalVolume'] for r in rows)
    dark_hours = [r.get('darkHours') for r in rows if r.get('darkHours')]
    light_hours = [r.get('lightHours') for r in rows if r.get('lightHours')]
    out['grossVolume'] = round(dark + light, 2)
    out['nightVolume'] = round(dark, 2)
    out['dayVolume'] = round(light, 2)
    out['nightShare'] = round(dark / (dark + light), 4) if dark + light else None
    out['darkHoursMean'] = round(sum(dark_hours) / len(dark_hours), 2) \
        if dark_hours else None
    out['lightHoursMean'] = round(sum(light_hours) / len(light_hours), 2) \
        if light_hours else None
    # a 14 hour night against a 10 hour day makes a half share mean the night
    # rate is lower, so the rate ratio goes beside the raw share
    if dark_hours and light_hours and light > 0:
        per_dark = dark / sum(dark_hours)
        per_light = light / sum(light_hours)
        out['nightRateRatio'] = round(per_dark / per_light, 3) if per_light else None

    split = split_point(rows, params)
    out['split'] = split
    if split is not None:
        middle = last[days[split['at']]['index']]
        for name, a, b, from_day, to_day in (
                ('firstHalf', start, middle, 0, split['at']),
                ('secondHalf', middle, end, split['at'], len(days) - 1)):
            half = _volumes(_mask(a - b, outside), params.threshold)
            half['fromDate'] = days[from_day]['date']
            half['toDate'] = days[to_day]['date']
            half['days'] = to_day - from_day + (1 if name == 'firstHalf' else 0)
            half['grossShare'] = round(
                split['fraction'] if name == 'firstHalf'
                else 1 - split['fraction'], 4)
            out[name] = half

    marks = prep.day_marks or {}
    indices = {str(d['index']) for d in days}
    out['newBowerDays'] = sum(
        1 for k, v in marks.items()
        if k in indices and getattr(v, 'new_bower', False))
    out['wallBuildingDays'] = sum(
        1 for k, v in marks.items()
        if k in indices and getattr(v, 'wall_building', False))

    if events is None:
        return out
    out.update(_event_metrics(prep, number, days, events, first, last, shape,
                             outside, params))
    return out


def _event_metrics(prep, number, days, events, first, last, shape, outside,
                   params) -> dict:
    """Behaviour, in depth coordinates, for one trial."""
    code = events['code']
    # Every filter here is a parameter, and all of them have to match what a
    # page applies or a count in a spreadsheet will not be the count on
    # screen. The hour window was not applied at all before.
    usable = ((events['flags'] & 1) != 0) & ((events['flags'] & 2) != 0) \
        & (events['bid'] != 255) & (events['prob'] >= params.cut) \
        & (events['hour'] >= params.hour_from) \
        & (events['hour'] < params.hour_to) \
        & (events['trial'] == number)

    out = {'events': int(usable.sum())}
    # all ten codes, so the workbook has a column for each rather than a
    # chosen six and an unexplained remainder
    for name, bid in BEHAVIOUR:
        out[name] = int((usable & (events['bid'] == code[bid])).sum()) \
            if bid in code else 0
    if out['spits'] > 0:
        out['scoopSpitRatio'] = round(out['scoops'] / out['spits'], 3)
    out['eventsPerDay'] = round(out['events'] / max(1, len(days)), 1)

    # the three families, because 'events per day' mixes building with
    # feeding and spawning and so moves for reasons that are not building
    out['buildEvents'] = out['scoops'] + out['spits'] + out['multiples']
    out['feedEvents'] = (out['feedScoops'] + out['feedSpits']
                         + out['feedMultiples'])
    for name, total in (('buildEventsPerDay', out['buildEvents']),
                        ('feedEventsPerDay', out['feedEvents']),
                        ('spawnEventsPerDay', out['spawns'])):
        out[name] = round(total / max(1, len(days)), 2)

    # --- day by day, for the day-level sheet ------------------------------
    # Counts first, because they need no registration: a project not yet
    # registered still has events, and building these after the early return
    # below meant its day rows came back empty.
    per_day = {}
    for day in days:
        index = int(day['index'])
        here = usable & (events['day'] == index)
        row = {'events': int(here.sum())}
        for name, bid in BEHAVIOUR:
            row[name] = int((here & (events['bid'] == code[bid])).sum()) \
                if bid in code else 0
        if row['spits'] > 0:
            row['scoopSpitRatio'] = round(row['scoops'] / row['spits'], 3)
        per_day[str(index)] = row
    out['dayEvents'] = per_day

    H = prep.transform_for(number)
    if H is None:
        out['placed'] = False
        return out
    out['placed'] = True
    dx, dy = _apply_h(H, events['x'], events['y'])

    scoops = usable & (events['bid'] == code['c'])
    spits = usable & (events['bid'] == code['p'])
    scoop_shape = _shape_of(np.column_stack([dx[scoops], dy[scoops]]), params)
    spit_shape = _shape_of(np.column_stack([dx[spits], dy[spits]]), params)
    out['scoopShape'] = scoop_shape
    out['spitShape'] = spit_shape
    out['scoopEllipse'] = scoop_shape['areaCm2'] if scoop_shape else None
    out['spitEllipse'] = spit_shape['areaCm2'] if spit_shape else None
    if scoop_shape and spit_shape and scoop_shape['areaCm2'] > 0:
        out['spreadRatio'] = round(
            spit_shape['areaCm2'] / scoop_shape['areaCm2'], 3)
        out['centroidGap'] = round(float(np.hypot(
            scoop_shape['cx'] - spit_shape['cx'],
            scoop_shape['cy'] - spit_shape['cy'])) * PIXEL_CM, 2)

    # now the shapes per day, which do need it
    start_map = first[days[0]['index']]
    for day in days:
        index = int(day['index'])
        here = usable & (events['day'] == index)
        row = per_day[str(index)]
        for label, bid in (('scoopShape', 'c'), ('spitShape', 'p')):
            picked = here & (events['bid'] == code[bid])
            row[label] = _shape_of(
                np.column_stack([dx[picked], dy[picked]]), params)
        if row['scoopShape'] and row['spitShape']:
            row['centroidGap'] = round(float(np.hypot(
                row['scoopShape']['cx'] - row['spitShape']['cx'],
                row['scoopShape']['cy'] - row['spitShape']['cy'])) * PIXEL_CM, 2)
        # spawn depth for this day, against this day's own sand
        spawn_here = here & (events['bid'] == code['s'])
        if spawn_here.any():
            cumulative = _mask(start_map - last[index], outside)
            xs = np.clip(np.nan_to_num(dx[spawn_here]), 0, shape[1] - 1).astype(int)
            ys = np.clip(np.nan_to_num(dy[spawn_here]), 0, shape[0] - 1).astype(int)
            values = cumulative[ys, xs]
            values = values[np.isfinite(values)]
            if len(values):
                row['spawnDepthMean'] = round(float(values.mean()), 3)
                row['spawnDepthMin'] = round(float(values.min()), 3)
                row['spawnDepthMax'] = round(float(values.max()), 3)
                row['spawnsMeasured'] = int(len(values))

    # --- clouds averaged over days, which is not the pooled cloud ----------
    # Pooling every event in a trial adds the day-to-day wandering of the
    # centre, so a male who works one spot a day but moves about over the
    # trial pools much wider than he scatters. Both are wanted, and they are
    # different measurements rather than two estimates of one.
    out.update(_daily_means(per_day))

    # --- the halves, pooled, for the clouds -------------------------------
    rows = day_volumes(first, last, days, outside, params)
    split = split_point(rows, params)
    if split is not None:
        for name, lo, hi in (('firstHalfShapes', 0, split['at']),
                             ('secondHalfShapes', split['at'], len(days) - 1)):
            window = {int(days[i]['index']) for i in range(lo, hi + 1)}
            inside = usable & np.isin(events['day'], list(window))
            out[name] = {
                'scoop': _shape_of(np.column_stack([
                    dx[inside & (events['bid'] == code['c'])],
                    dy[inside & (events['bid'] == code['c'])]]), params),
                'spit': _shape_of(np.column_stack([
                    dx[inside & (events['bid'] == code['p'])],
                    dy[inside & (events['bid'] == code['p'])]]), params)}

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
            cumulative = start - last[day['index']]
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


def _daily_means(per_day: dict) -> dict:
    """Cloud size and shape averaged across days, with the day counts.

    A day whose cloud was not reported -- too few events for an ellipse, or
    too few for its shape -- contributes nothing rather than a zero, so the
    counts are given too: a mean over three days and a mean over twenty are
    not the same number however alike they look.
    """
    out: Dict[str, object] = {}
    for label, key in (('scoop', 'scoopShape'), ('spit', 'spitShape')):
        areas, aspects = [], []
        for row in per_day.values():
            shape = row.get(key)
            if not shape:
                continue
            areas.append(shape['areaCm2'])
            if shape.get('aspect') is not None:
                aspects.append(shape['aspect'])
        name = label.capitalize()
        out[label + 'AreaDaily'] = round(sum(areas) / len(areas), 2) \
            if areas else None
        out[label + 'AreaDailyDays'] = len(areas)
        out[label + 'AspectDaily'] = round(sum(aspects) / len(aspects), 3) \
            if aspects else None
        out[label + 'AspectDailyDays'] = len(aspects)

    # scoop over spit, of the daily averages. NOT spreadRatio, which is spit
    # over scoop of the pooled clouds -- the other direction and the other
    # pooling, so the two must never be read as the same quantity.
    scoop, spit = out.get('scoopAreaDaily'), out.get('spitAreaDaily')
    if scoop and spit:
        out['scoopSpitAreaIndex'] = round(scoop / spit, 3)
    return out


def score_mask(score: np.ndarray, k: float) -> Optional[np.ndarray]:
    """Which pixels are too unreliable to measure sand with.

    The residual is a pixel's RMS departure from its own fitted line, so a
    pixel that builds steadily scores low and one that thrashes scores high.
    The cut is k median-absolute-deviations above the median *of the logs*,
    because the scores are heavy-tailed and a cut on the raw values is set by
    the worst pixel.

    This is a second implementation of common.js's scoreMask, and a test pins
    the two together: a mask that differs between the page and the server
    means a volume on screen and a volume in the spreadsheet disagree for a
    reason nobody can see.
    """
    finite = score[np.isfinite(score) & (score > 0)]
    if not finite.size:
        return None
    logs = np.log(np.sort(finite))
    median = float(logs[len(logs) >> 1])
    deviations = np.sort(np.abs(logs - median))
    mad = float(deviations[len(deviations) >> 1]) * 1.4826
    cut = np.exp(median + k * mad)
    # NaN counts as unreliable, not as fine
    return ~np.isfinite(score) | (score > cut)


def trial_mask(residual_maps, days, k: float,
               inside: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """The mask for one trial, from that trial's own days.

    Per trial rather than per project: the tank is reset between trials, and a
    camera knocked during one of them should not mask the others. A pixel that
    thrashed in trial two was fine in trial one and its sand should still be
    counted there.
    """
    if residual_maps is None:
        return None
    picked = [residual_maps[int(d['index'])] for d in days
              if int(d['index']) < len(residual_maps)]
    if not picked:
        return None
    return score_mask(R.project_score(picked, inside=inside), k)


def _mask(values, outside):
    return np.where(outside, np.nan, values) if outside is not None else values


def day_volumes(first, last, days, outside, params: 'Params') -> List[dict]:
    """Light, dark and cumulative volume for each day of one trial.

    The night comes free: a day runs from its first lights-on frame to its
    last, so the gap between one day's last and the next day's first IS the
    dark period, and both endpoints are already in the bundle.
    """
    rows = []
    start = first[days[0]['index']]
    previous = None
    for day in days:
        index = int(day['index'])
        row = {'index': index, 'trial': int(day['trial']), 'date': day['date'],
               'from': day.get('first'), 'to': day.get('last'),
               'light': _volumes(_mask(first[index] - last[index], outside),
                                 params.threshold),
               'dark': None,
               'cumulative': _volumes(_mask(start - last[index], outside),
                                      params.threshold)}
        if previous is not None:
            row['dark'] = _volumes(_mask(previous - first[index], outside),
                                   params.threshold)
        row['lightHours'] = _hours(day.get('first'), day.get('last'))
        rows.append(row)
        previous = last[index]
    for i in range(1, len(rows)):
        rows[i]['darkHours'] = _hours(days[i - 1].get('last'),
                                      days[i].get('first'))
    return rows


def _hours(a, b) -> Optional[float]:
    """A span in hours, or None when it is not one.

    A negative span means the days are out of order or a clock moved, and
    anything past two days is not a night. Summing those blindly once printed
    a dark period of minus forty hours.
    """
    if not a or not b:
        return None
    try:
        hours = (dt.datetime.fromisoformat(str(b))
                 - dt.datetime.fromisoformat(str(a))).total_seconds() / 3600
    except ValueError:
        return None
    return round(hours, 2) if 0 < hours < 48 else None


def split_point(rows: List[dict], params: 'Params') -> Optional[dict]:
    """The day by which half the trial's gross sand has moved.

    Gross, because it accumulates monotonically; the net change across a
    window does not, so a 50% crossing of it can happen twice. Snapped to a
    day boundary, and the fraction actually reached is reported rather than
    assumed to be a half.
    """
    moved = [r['light']['totalVolume'] +
             (r['dark']['totalVolume'] if r['dark'] else 0.0) for r in rows]
    gross = sum(moved)
    if gross <= 0:
        return None
    running = 0.0
    for i, amount in enumerate(moved):
        running += amount
        if running >= params.split_at * gross:
            return {'at': i, 'dayIndex': rows[i]['index'],
                    'date': rows[i]['date'],
                    'fraction': round(running / gross, 4),
                    'grossVolume': round(gross, 2)}
    return {'at': len(rows) - 1, 'dayIndex': rows[-1]['index'],
            'date': rows[-1]['date'], 'fraction': 1.0,
            'grossVolume': round(gross, 2)}


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

    # This module's own mtime, before anything else. A change here changes the
    # output without touching a single data file, so file timestamps alone
    # call a cache current when the code that wrote it has moved on.
    #
    # The schema check below was meant to cover that, and did not: bumping
    # SCHEMA is a thing someone has to remember, and the very next change --
    # adding the daily-mean fields -- did not get one. Every cached file was
    # served as current while missing the new keys, so nine panels on the
    # phenotypes page drew nothing. Comparing against the module is the
    # version of this that needs no discipline.
    try:
        if Path(__file__).stat().st_mtime > when:
            return True
    except OSError:
        pass

    try:
        with open(cached) as handle:
            if json.load(handle).get('schema') != SCHEMA:
                return True
    except Exception:
        return True            # unreadable is stale, not current

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
