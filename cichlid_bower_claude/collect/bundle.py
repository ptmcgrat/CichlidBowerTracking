"""The collected bundle: what the server reads instead of the archive.

Two files. ``bundle.npz`` holds the arrays; ``manifest.json`` describes them
and is **written last**, after every artefact it names has been uploaded.

That ordering is the whole design. Presence of files is not a completion
marker — a run that failed halfway leaves files that look finished. The
manifest is the marker, and it carries a schema version so a later change to
the format forces a rebuild rather than silently mixing generations.

Arrays are float16. About thirteen times smaller than float64, NaN preserved
natively, and 0.03 cm of quantisation against a sensor whose own noise floor
is 0.05 cm. Reductions over them must accumulate in float64: ``np.nansum`` on
float16 returns infinity, which is a quiet way to produce a wrong volume.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCHEMA = 'cichlid-bundle/1'

ARRAY_KEYS = ('first', 'last', 'residual', 'trend', 'travel', 'valid',
              'std_mean', 'std_max')


def write_bundle(path: Path, arrays: Dict[str, np.ndarray]) -> int:
    """Store the per-day arrays. Returns the file size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, value in arrays.items():
        if value is None:
            continue
        payload[key] = np.asarray(value, dtype=np.float16)
    np.savez_compressed(path, **payload)
    return path.stat().st_size


def load_bundle(path: Path) -> Dict[str, np.ndarray]:
    """Read the arrays back, widened to float64.

    Widening here rather than at each call site is deliberate: every reduction
    downstream would otherwise have to remember to do it, and forgetting shows
    up as an infinity rather than an error.
    """
    with np.load(path) as data:
        return {key: data[key].astype(np.float64) for key in data.files}


def build_manifest(log, plan, *, extracted: int, missing: List[str],
                   bundle_bytes: int, source_bytes: Optional[int] = None,
                   settings: Optional[dict] = None, branch: str = '',
                   depth_size: Optional[List[int]] = None) -> dict:
    """Describe what was collected, and under what assumptions.

    The trial settings are recorded because the day endpoints depend on them:
    a bundle collected under one set of offsets and read under another would
    silently disagree with the interface about where a day starts.
    """
    return {
        'schema': SCHEMA,
        'projectID': log.project_id,
        'tankID': log.tank_id,
        'analysisID': log.analysis_id,
        'built': str(dt.datetime.now().replace(microsecond=0)),
        'branch': branch,
        # the two cameras differ: depth is typically 640x480 and the Pi
        # 1296x972. Recording one 'frameSize' invited reading the Pi's
        # resolution as the shape of the depth arrays, which it is not.
        'depthSize': depth_size,
        'videoSize': [log.width, log.height],
        'nFrames': len(log.frames),
        'nMovies': len(log.movies),
        'sourceBytes': source_bytes,
        'bundleBytes': bundle_bytes,
        'extracted': extracted,
        'missing': missing,
        'logIssues': log.issues,
        'trialSettings': settings,
        'trials': [{'number': t.number, 'start': str(t.start_time),
                    'stop': str(t.stop_time),
                    'reset': str(t.reset_time) if t.reset_time else None}
                   for t in log.trials],
        'days': [{'index': d.index, 'trial': d.trial, 'date': str(d.date),
                  'first': str(d.first.time), 'last': str(d.last.time),
                  'firstIndex': d.first.index, 'lastIndex': d.last.index,
                  'hours': round(d.hours, 3)}
                 for d in plan.days],
        'candidates': [c.as_dict() for c in plan.candidates],
        'pairs': [p.as_dict() for p in plan.pairs],
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(manifest, handle, indent=1)


def read_manifest(path: Path) -> Optional[dict]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path) as handle:
            return json.load(handle)
    except Exception:
        return None


def is_current(manifest: Optional[dict], settings: Optional[dict] = None) -> bool:
    """Whether a collected bundle can be reused.

    Stale for three reasons, each of which would otherwise produce a bundle
    that looks fine and is not: the schema changed, the trial offsets changed
    so the day endpoints no longer match, or there is no manifest at all
    because the run that should have written it did not finish.
    """
    if not manifest:
        return False
    if manifest.get('schema') != SCHEMA:
        return False
    stored = (manifest.get('trialSettings') or {}).get('trials') or {}
    current = (settings or {}).get('trials') or {}
    return stored == current