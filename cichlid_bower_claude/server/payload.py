"""Turning a collected bundle into what a page needs.

Writes a directory of PNGs and one small JSON. The page fetches the images by
URL rather than carrying them inline, so the HTML stays a few kilobytes, the
browser caches what it has already seen, and a project with thirty-four days
does not become a fifteen-megabyte document.

Nothing here applies a crop, a mask or a threshold. Those are live controls in
the page, so baking any of them in would mean a rebuild every time somebody
moved a slider.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..collect import bundle as B
from ..collect import residual as R
from . import corrections as C
from .encode import write_depth

SCHEMA = 'cichlid-page/1'
WINDOW_CM = 30.0


def _physical_window(first: np.ndarray) -> tuple:
    """The range real sand occupies, from the endpoint frames.

    Raw frames hold no-return pixels thousands of centimetres away. Clamping to
    the window the sand actually occupies keeps the quantisation fine; without
    it a single bad pixel coarsens the whole frame.
    """
    finite = first[np.isfinite(first)]
    median = float(np.median(finite)) if finite.size else 60.0
    return (median - WINDOW_CM, median + WINDOW_CM)


def is_stale(project_paths, out_dir: Optional[Path] = None) -> bool:
    """Whether the built assets predate the collection they came from.

    A recollection writes a new bundle and new stills; without this the pages
    keep serving the assets built from the previous one, which looks like
    nothing happened. Comparing against the manifest works because the manifest
    is written last, after every artefact it describes.
    """
    out_dir = Path(out_dir or project_paths.pages_dir)
    payload = out_dir / 'page.json'
    if not payload.exists():
        return True
    if not project_paths.manifest.exists():
        return False
    return project_paths.manifest.stat().st_mtime > payload.stat().st_mtime


def build_prep_payload(project_paths, out_dir: Optional[Path] = None) -> dict:
    """Assets and payload for the prep page. Returns the payload.

    Raises if the project has not been collected: a page built from a missing
    bundle would look like a project with no data rather than one that has not
    been through the collector.
    """
    manifest = B.read_manifest(project_paths.manifest)
    if manifest is None:
        raise FileNotFoundError('no manifest for ' + project_paths.project_id +
                                ' — collect it first')
    out_dir = Path(out_dir or project_paths.pages_dir)
    assets = out_dir / 'assets'
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)

    data = B.load_bundle(project_paths.bundle)
    first, last = data['first'], data['last']
    window = _physical_window(first[0])

    days = []
    for entry in manifest['days']:
        index = entry['index']
        day = dict(entry)
        for key, array in (('first', first[index]), ('last', last[index])):
            name = 'day_%02d_%s.png' % (index, key)
            day[key + 'Png'] = write_depth(assets / name, array, clip=window)
            day[key + 'Png']['url'] = 'assets/' + name
        if 'residual' in data:
            name = 'day_%02d_residual.png' % index
            day['residualPng'] = write_depth(assets / name, data['residual'][index])
            day['residualPng']['url'] = 'assets/' + name
            day['residualStats'] = _residual_stats(data['residual'][index])
        for which in ('first', 'last'):
            still = project_paths.collected_dir / ('Day_%02d_%s.jpg' % (index, which))
            if still.exists():
                shutil.copy2(still, assets / still.name)
                day[which + 'Jpg'] = 'assets/' + still.name
        video_still = project_paths.collected_dir / ('Video_%02d.jpg' % index)
        if video_still.exists():
            shutil.copy2(video_still, assets / video_still.name)
            day['videoJpg'] = 'assets/' + video_still.name
        days.append(day)

    # the project-wide residual score, which is what the mask is built from
    score_url = None
    if 'residual' in data:
        maps = [data['residual'][i] for i in range(data['residual'].shape[0])]
        score = R.project_score(maps)
        meta = write_depth(assets / 'residual_score.png', score)
        meta['url'] = 'assets/residual_score.png'
        score_url = meta

    candidates = []
    for entry in manifest.get('candidates', []):
        item = dict(entry)
        stem = entry['stem']
        npy = project_paths.collected_dir / (stem + '.npy')
        jpg = project_paths.collected_dir / (stem + '.jpg')
        if npy.exists():
            array = np.load(npy).astype(np.float64)
            meta = write_depth(assets / (stem + '.png'), array, clip=window)
            meta['url'] = 'assets/' + stem + '.png'
            item['depthPng'] = meta
        if jpg.exists():
            shutil.copy2(jpg, assets / jpg.name)
            item['jpg'] = 'assets/' + jpg.name
        candidates.append(item)

    pairs = []
    for entry in manifest.get('pairs', []):
        item = dict(entry)
        stem = entry['stem']
        for suffix, key in (('_depth.npy', 'depthPng'), ('_pi.jpg', 'piJpg'),
                            ('_depth.jpg', 'depthJpg')):
            source = project_paths.collected_dir / (stem + suffix)
            if not source.exists():
                continue
            if suffix.endswith('.npy'):
                array = np.load(source).astype(np.float64)
                meta = write_depth(assets / (stem + '_depth.png'), array, clip=window)
                meta['url'] = 'assets/' + stem + '_depth.png'
                item[key] = meta
            else:
                shutil.copy2(source, assets / source.name)
                item[key] = 'assets/' + source.name
        pairs.append(item)

    # the corrections are embedded for a first paint, but the page fetches the
    # live copy: this snapshot goes stale the moment anybody saves
    prep = C.load(project_paths)
    payload = {
        'schema': SCHEMA,
        'projectID': manifest['projectID'],
        'analysisID': manifest['analysisID'],
        'tankID': manifest.get('tankID', ''),
        'depthSize': manifest.get('depthSize'),
        'videoSize': manifest.get('videoSize'),
        'nFrames': manifest.get('nFrames'),
        'collected': manifest.get('built'),
        'built': str(dt.datetime.now().replace(microsecond=0)),
        'trials': manifest['trials'],
        'days': days,
        'candidates': candidates,
        'pairs': pairs,
        'offsets': sorted({c['offset'] for c in manifest.get('candidates', [])}),
        'residualScore': score_url,
        # the packed events are served separately: they are larger than every
        # image put together, and only one page needs them
        'hasClusters': project_paths.clusters_packed.exists(),
        'clusterSummary': manifest.get('clusters'),
        'logIssues': manifest.get('logIssues', []),
        'missing': manifest.get('missing', []),
        'prep': prep.to_dict(),
    }
    with open(out_dir / 'page.json', 'w') as handle:
        json.dump(payload, handle)
    return payload


def _residual_stats(residual: np.ndarray) -> dict:
    """The threshold statistics for one day, at full resolution.

    Computed here rather than in the page because the page only ever holds the
    half-resolution image: a threshold read off a downsampled map disagrees
    with one applied to the data.
    """
    values = residual[np.isfinite(residual) & (residual > 0)]
    if values.size < 100:
        return {}
    logs = np.log(values)
    log_median = float(np.median(logs))
    log_mad = float(np.median(np.abs(logs - log_median))) * 1.4826
    return {'median': round(float(np.exp(log_median)), 5),
            'madFactor': round(float(np.exp(log_mad)), 4),
            'p99': round(float(np.percentile(values, 99)), 5),
            'max': round(float(values.max()), 4),
            'n': int(values.size)}


def payload_size(out_dir: Path) -> Dict[str, int]:
    """What the build weighs, so a page that has grown is visible."""
    out_dir = Path(out_dir)
    assets = out_dir / 'assets'
    files = list(assets.glob('*')) if assets.is_dir() else []
    return {'files': len(files),
            'assetBytes': sum(f.stat().st_size for f in files),
            'payloadBytes': (out_dir / 'page.json').stat().st_size
                            if (out_dir / 'page.json').exists() else 0}