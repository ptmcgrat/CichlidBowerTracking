"""What a person decided, and when.

One file per project, ``Corrections/prep.json``. It holds the registration, both
crops, the per-trial time offsets and the residual k, together with who set
them and when.

One file rather than five. The old layout spread the same decision across
``DepthCrop.txt``, ``VideoCrop.txt``, ``TransMFile.npy``, ``TrialSettings.json``
and ``RegistrationInfo.json``: five writes, five chances to half-succeed, and
provenance kept apart from the thing it described. Here a save either lands or
does not.

Every previous version is kept in ``Corrections/history/``, so a change can be
read back and undone. The history sits beside the file rather than in a
separate backups directory, because an audit trail parted from its data stops
being consulted.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

SCHEMA = 'cichlid-prep/1'
DEFAULT_K = 4.0


class CorrectionsError(Exception):
    """The corrections file is unusable."""


@dataclass
class TrialTimes:
    """Minutes to move one trial's boundaries inward.

    Always inward. The risk being managed is a boundary that takes in disturbed
    sand, not one that leaves out good data.
    """

    start: int = 0
    stop: int = 0
    reset: int = 0


@dataclass
class Prep:
    """Everything a person decides about a project before it is analysed."""

    project_id: str = ''
    analysis_id: str = ''
    schema: str = SCHEMA
    transform: Optional[List[List[float]]] = None      # 3x3 depth-to-Pi homography
    depth_crop: Optional[List[List[int]]] = None       # four points, depth coordinates
    video_crop: Optional[List[List[int]]] = None       # four points, Pi coordinates
    trials: Dict[str, TrialTimes] = field(default_factory=dict)
    residual_k: float = DEFAULT_K
    points: List[dict] = field(default_factory=list)   # the pairs the fit came from
    fit_rms_px: Optional[float] = None
    who: str = ''
    note: str = ''
    updated: str = ''

    @property
    def is_registered(self) -> bool:
        return self.transform is not None

    @property
    def is_cropped(self) -> bool:
        return bool(self.depth_crop) and bool(self.video_crop)

    @property
    def is_complete(self) -> bool:
        """Whether the depth tab should open for this project."""
        return self.is_registered and self.is_cropped

    def times_for(self, trial_number: int) -> TrialTimes:
        """One trial's offsets, falling back to trial 1's.

        Most projects have one camera position and one settling time, so the
        common case should be one decision rather than one per trial. Trial 1
        is the default and any trial can override it.
        """
        entry = self.trials.get(str(trial_number)) or self.trials.get('1')
        return entry or TrialTimes()

    def to_dict(self) -> dict:
        data = asdict(self)
        data['trials'] = {k: asdict(v) if not isinstance(v, dict) else v
                          for k, v in self.trials.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Prep':
        trials = {}
        for key, value in (data.get('trials') or {}).items():
            trials[str(key)] = TrialTimes(**value) if isinstance(value, dict) else value
        known = {f for f in cls.__dataclass_fields__}
        kept = {k: v for k, v in data.items() if k in known and k != 'trials'}
        kept['trials'] = trials
        return cls(**kept)


def path_for(project_paths) -> Path:
    return project_paths.corrections_dir / 'prep.json'


def load(project_paths) -> Prep:
    """The current corrections, or an empty set if there are none yet."""
    path = path_for(project_paths)
    if not path.exists():
        return Prep(project_id=project_paths.project_id,
                    analysis_id=project_paths.analysis_id)
    try:
        with open(path) as handle:
            data = json.load(handle)
    except Exception as error:
        raise CorrectionsError('could not read ' + str(path) + ': ' + repr(error))
    if data.get('schema') != SCHEMA:
        raise CorrectionsError(str(path) + ' is schema ' + str(data.get('schema')) +
                               ', this code writes ' + SCHEMA)
    return Prep.from_dict(data)


def save(project_paths, prep: Prep, who: str = '', note: str = '') -> Path:
    """Write the corrections, keeping the version they replace.

    Written to a temporary file and moved into place, so a save interrupted
    halfway leaves the previous version intact rather than a truncated one.
    """
    path = path_for(project_paths)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        history = project_paths.corrections_dir / 'history'
        history.mkdir(parents=True, exist_ok=True)
        (history / (stamp + '_prep.json')).write_bytes(path.read_bytes())

    prep.project_id = prep.project_id or project_paths.project_id
    prep.analysis_id = prep.analysis_id or project_paths.analysis_id
    prep.schema = SCHEMA
    prep.updated = str(dt.datetime.now().replace(microsecond=0))
    if who:
        prep.who = who
    if note:
        prep.note = note

    handle = tempfile.NamedTemporaryFile('w', delete=False, dir=str(path.parent),
                                         prefix='.prep-', suffix='.json')
    try:
        json.dump(prep.to_dict(), handle, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)
    return path


def history(project_paths) -> List[Path]:
    """Previous versions, newest first."""
    directory = project_paths.corrections_dir / 'history'
    if not directory.is_dir():
        return []
    return sorted(directory.glob('*_prep.json'), reverse=True)


def restore(project_paths, stamp: str) -> Prep:
    """Bring back a previous version, keeping the current one in history."""
    candidates = [p for p in history(project_paths) if p.name.startswith(stamp)]
    if not candidates:
        raise CorrectionsError('no history entry starting ' + stamp)
    with open(candidates[0]) as handle:
        prep = Prep.from_dict(json.load(handle))
    save(project_paths, prep, note='restored from ' + candidates[0].name)
    return prep