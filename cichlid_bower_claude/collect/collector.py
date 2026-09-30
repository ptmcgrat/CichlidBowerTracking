"""Collecting one project.

Orchestration only: every decision lives in the module that owns it. This
reads the log, plans what it needs, streams the archive once, and writes the
bundle — then uploads, and writes the manifest last.

The ordering is the contract. A run that dies partway leaves artefacts but no
manifest, so the next run sees an incomplete collection rather than a
convincing one.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..logfile.parse import parse_logfile
from ..paths import Layout, ProjectPaths
from . import bundle as B
from . import residual as R
from .archive import Archive, ArchiveError, ensure_archive
from .frames import plan_for


@dataclass
class Result:
    """What happened to one project."""

    project_id: str
    status: str = 'ok'          # ok | skipped | failed
    reason: str = ''
    days: int = 0
    candidates: int = 0
    extracted: int = 0
    missing: List[str] = field(default_factory=list)
    bundle_bytes: int = 0
    source_bytes: Optional[int] = None
    seconds: float = 0.0

    def line(self) -> str:
        if self.status == 'skipped':
            return '%s: already collected' % self.project_id
        if self.status == 'failed':
            return '%s: failed — %s' % (self.project_id, self.reason)
        text = ('%s: %d days, %d candidates, %.1f MB bundle in %.0fs'
                % (self.project_id, self.days, self.candidates,
                   self.bundle_bytes / 1e6, self.seconds))
        if self.missing:
            text += ' (%d frames missing from the archive)' % len(self.missing)
        return text


def read_trial_settings(paths: ProjectPaths) -> Optional[dict]:
    if not paths.trial_settings.exists():
        return None
    try:
        with open(paths.trial_settings) as handle:
            return json.load(handle)
    except Exception:
        return None


def collect_project(layout: Layout, cloud, project_id: str, *,
                    force: bool = False, keep_archive: bool = False,
                    upload: bool = True, branch: str = '') -> Result:
    """Collect one project. Never raises — the sweep keeps going."""
    started = dt.datetime.now()
    paths = layout.project(project_id)
    result = Result(project_id=project_id)

    try:
        # the corrections first: the day endpoints depend on the trial offsets
        cloud.download_optional(paths.trial_settings)
        settings = read_trial_settings(paths)

        existing = B.read_manifest(paths.manifest)
        if existing is None and cloud.download_optional(paths.manifest):
            existing = B.read_manifest(paths.manifest)
        if not force and B.is_current(existing, settings):
            result.status = 'skipped'
            return result

        if not paths.logfile.exists():
            cloud.download(paths.logfile)
        log = parse_logfile(paths.logfile)
        if not log.trials:
            result.status = 'failed'
            result.reason = 'no trials in the logfile'
            return result

        plan = plan_for(log, settings)
        if not plan.days:
            result.status = 'failed'
            result.reason = 'no days survived the trial offsets'
            return result
        result.days = len(plan.days)
        result.candidates = len(plan.candidates)

        archive_path, source_bytes = ensure_archive(cloud, paths, force=force)
        result.source_bytes = source_bytes

        arrays: Dict[str, List[np.ndarray]] = {}
        missing: List[str] = []
        with Archive(archive_path) as archive:
            wanted = plan.members()
            missing.extend(archive.extract(wanted, paths.collected_dir))
            result.extracted = len(wanted) - len(missing)

            std_by_day = {}
            for day, std_stack in archive.iter_std_days(log, plan.days):
                std_by_day[day.index] = std_stack

            for day, stacked, day_missing in archive.iter_days(log, plan.days):
                missing.extend(day_missing)
                if not stacked.size:
                    continue
                summary = R.summarise_day(stacked, std_stack=std_by_day.get(day.index))
                summary['first'] = stacked[0].astype(np.float32)
                summary['last'] = stacked[-1].astype(np.float32)
                for key, value in summary.items():
                    arrays.setdefault(key, []).append(value)

        stacked_arrays = {key: np.stack(value) for key, value in arrays.items() if value}
        result.bundle_bytes = B.write_bundle(paths.bundle, stacked_arrays)
        result.missing = sorted(set(missing))

        if upload:
            cloud.upload(paths.collected_dir)

        # last, and only now: the collection is complete
        manifest = B.build_manifest(log, plan, extracted=result.extracted,
                                    missing=result.missing,
                                    bundle_bytes=result.bundle_bytes,
                                    source_bytes=source_bytes,
                                    settings=settings, branch=branch)
        B.write_manifest(paths.manifest, manifest)
        if upload:
            cloud.upload(paths.manifest)

        if not keep_archive and archive_path.exists():
            archive_path.unlink()

    except (ArchiveError, Exception) as error:   # a bad project must not stop a sweep
        result.status = 'failed'
        result.reason = repr(error)
    finally:
        result.seconds = (dt.datetime.now() - started).total_seconds()
    return result


def collect_many(layout: Layout, cloud, project_ids: List[str], **kwargs) -> List[Result]:
    """Collect several projects, reporting each as it finishes."""
    results = []
    for number, project_id in enumerate(project_ids, 1):
        print('[%d/%d] %s' % (number, len(project_ids), project_id))
        result = collect_project(layout, cloud, project_id, **kwargs)
        print('    ' + result.line())
        results.append(result)
    return results
