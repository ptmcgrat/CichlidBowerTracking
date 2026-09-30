"""Choosing which frames to keep out of the archive.

Pure functions of a parsed log. Nothing here reads the archive; it decides what
to ask for, and ``archive.py`` fetches it.

Two sets come out, and they overlap only by accident:

- **Endpoints** — the first and last lights-on frame of each day. Daylight
  change is a day's own pair, night change is one day's last against the next
  day's first, and 24-hour change is one day's first against the next day's
  first. All three from the same two frames per day.
- **Boundary candidates** — the frames a trial start or stop could be moved
  to. These are what make the trial-times tab work, and they are why the
  server can honour an offset without anything being recollected.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional

from ..logfile.days import Day, Offsets, days_for
from ..logfile.model import Frame, ProjectLog, Trial

OFFSET_MINUTES = (0, 15, 30, 45, 60)


@dataclass(frozen=True)
class Candidate:
    """One frame a trial boundary could be moved to."""

    kind: str          # 'start', 'stop' or 'reset'
    trial: int
    offset: int        # minutes inward from the logged boundary
    frame: Frame

    @property
    def stem(self) -> str:
        letter = 'm' if self.kind == 'stop' else 'p'
        return 'Bound_T%d_%s_%s%02d' % (self.trial, self.kind, letter, self.offset)

    @property
    def label(self) -> str:
        sign = '-' if self.kind == 'stop' else '+'
        return '%s %s%d min' % (self.kind, sign, self.offset)

    def as_dict(self) -> dict:
        return {'kind': self.kind, 'trial': self.trial, 'offset': self.offset,
                'label': self.label, 'stem': self.stem,
                'index': self.frame.index, 'time': str(self.frame.time)}


def frame_near(log: ProjectLog, when: dt.datetime, forward: bool = True,
               lights_on: bool = True) -> Optional[Frame]:
    """The nearest lights-on frame at or after ``when``, or at or before it."""
    pool = [f for f in log.frames if f.lights_on or not lights_on]
    if forward:
        later = [f for f in pool if f.time >= when]
        return later[0] if later else None
    earlier = [f for f in pool if f.time <= when]
    return earlier[-1] if earlier else None


def candidates_for(log: ProjectLog, trial: Trial, is_last: bool,
                   offsets=OFFSET_MINUTES) -> List[Candidate]:
    """Boundary frames for one trial.

    Start candidates run forward from the logged start, stop candidates back
    from the logged stop — always inward, because the risk being managed is a
    boundary that includes disturbed sand, not one that excludes good data.

    The reset following a trial is the next trial's start, so it only needs its
    own control on the last trial, where nothing follows it and those frames
    are the final picture of the bower.
    """
    out: List[Candidate] = []
    for minutes in offsets:
        frame = frame_near(log, trial.start_time + dt.timedelta(minutes=minutes), True)
        if frame is not None:
            out.append(Candidate('start', trial.number, minutes, frame))
    for minutes in offsets:
        frame = frame_near(log, trial.stop_time - dt.timedelta(minutes=minutes), False)
        if frame is not None:
            out.append(Candidate('stop', trial.number, minutes, frame))
    if is_last and trial.reset_time is not None:
        for minutes in offsets:
            frame = frame_near(log, trial.reset_time + dt.timedelta(minutes=minutes), True)
            if frame is not None:
                out.append(Candidate('reset', trial.number, minutes, frame))
    return out


def all_candidates(log: ProjectLog, offsets=OFFSET_MINUTES) -> List[Candidate]:
    last = len(log.trials)
    out: List[Candidate] = []
    for trial in log.trials:
        out.extend(candidates_for(log, trial, is_last=(trial.number == last),
                                  offsets=offsets))
    return out


@dataclass
class Plan:
    """Everything one project needs out of its archive."""

    days: List[Day] = field(default_factory=list)
    candidates: List[Candidate] = field(default_factory=list)

    def day_frames(self) -> List[Frame]:
        """Endpoint frames, in order, one pair per day."""
        out = []
        for day in self.days:
            out.extend([day.first, day.last])
        return out

    def members(self) -> dict:
        """Archive member name to output filename, for everything to extract.

        Candidate frames are named by their stem so the interface can find
        them; day endpoints keep their original names so a frame appearing in
        both sets is extracted once.
        """
        wanted = {}
        for day in self.days:
            for which, frame in (('first', day.first), ('last', day.last)):
                wanted[frame.npy_file] = frame.npy_file.split('/')[-1]
                wanted[frame.pic_file] = 'Day_%02d_%s.jpg' % (day.index, which)
        for candidate in self.candidates:
            wanted[candidate.frame.npy_file] = candidate.stem + '.npy'
            wanted[candidate.frame.pic_file] = candidate.stem + '.jpg'
        return wanted

    def residual_members(self) -> dict:
        """Every lights-on frame of every day.

        The residual is fitted through all of them, which is the reason the
        whole archive is downloaded rather than seeked into: by the time you
        need this many members there is nothing left to save.
        """
        wanted = {}
        for day in self.days:
            wanted[day.index] = (day.first.index, day.last.index)
        return wanted


def plan_for(log: ProjectLog, settings: Optional[dict] = None,
             offsets=OFFSET_MINUTES) -> Plan:
    """What to collect for a project, given whatever trial settings exist.

    Days are computed under the *current* offsets so the bundle's endpoints
    match what the interface shows. The candidates are collected regardless,
    so a later change of mind costs a rebuild rather than a recollection.
    """
    days: List[Day] = []
    for trial in log.trials:
        trial_offsets = Offsets.for_trial(settings, trial.number)
        for day in days_for(log, trial, trial_offsets):
            days.append(Day(index=len(days), trial=trial.number,
                            first=day.first, last=day.last))
    return Plan(days=days, candidates=all_candidates(log, offsets=offsets))
