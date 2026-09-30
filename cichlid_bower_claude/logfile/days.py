"""Where a day starts and stops.

Separate from the model because the answer depends on the trial offsets, which
are a human choice made after parsing and changed whenever someone decides the
sand had not settled. The old code fixed a trial's days in ``Trial.__init__``,
so changing an offset meant reparsing.

A day is a run of lights-on frames on one calendar date, clipped to the trial's
adjusted window. Only the first and last day of a trial can move: intermediate
days begin and end at lights-on boundaries, which no offset touches.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional

from .model import Frame, ProjectLog, Trial

MIN_FRAMES = 6


@dataclass(frozen=True)
class Offsets:
    """Minutes to move a trial's boundaries inward."""

    start: int = 0
    stop: int = 0
    reset: int = 0

    @classmethod
    def for_trial(cls, settings: Optional[dict], number: int) -> 'Offsets':
        """Read one trial's offsets out of TrialSettings.json.

        Falls back to trial 1's choice, which is the interface's default: most
        projects have one settling time, so the common case is one decision
        rather than one per trial.
        """
        trials = (settings or {}).get('trials') or {}
        entry = trials.get(str(number)) or trials.get('1') or {}
        return cls(start=int(entry.get('startOffset') or 0),
                   stop=int(entry.get('stopOffset') or 0),
                   reset=int(entry.get('resetOffset') or 0))


@dataclass(frozen=True)
class Day:
    """One day of one trial, after the offsets have been applied."""

    index: int
    trial: int
    first: Frame
    last: Frame

    @property
    def date(self) -> dt.date:
        return self.first.time.date()

    @property
    def hours(self) -> float:
        return (self.last.time - self.first.time).total_seconds() / 3600.0

    @property
    def span(self) -> slice:
        return slice(self.first.index, self.last.index + 1)


def trial_window(trial: Trial, offsets: Offsets):
    """A trial's adjusted start and stop."""
    return (trial.start_time + dt.timedelta(minutes=offsets.start),
            trial.stop_time - dt.timedelta(minutes=offsets.stop))


def days_for(log: ProjectLog, trial: Trial, offsets: Offsets = Offsets(),
             min_frames: int = MIN_FRAMES) -> List[Day]:
    """The days of one trial, with offsets applied.

    A day left with fewer than ``min_frames`` lights-on frames is dropped
    rather than carried as a stub: a two-frame day produces a residual fit
    through two points, which is not a fit.
    """
    start, stop = trial_window(trial, offsets)
    by_date: Dict[dt.date, List[Frame]] = {}
    for frame in log.frames:
        if frame.lights_on and start <= frame.time <= stop:
            by_date.setdefault(frame.time.date(), []).append(frame)

    days = []
    for date in sorted(by_date):
        frames = by_date[date]
        if len(frames) < min_frames:
            continue
        days.append(Day(index=len(days), trial=trial.number,
                        first=frames[0], last=frames[-1]))
    return days


def all_days(log: ProjectLog, settings: Optional[dict] = None) -> List[Day]:
    """Every day of every trial, numbered across the project."""
    out: List[Day] = []
    for trial in log.trials:
        offsets = Offsets.for_trial(settings, trial.number)
        for day in days_for(log, trial, offsets):
            out.append(Day(index=len(out), trial=trial.number,
                           first=day.first, last=day.last))
    return out


def nights_for(days: List[Day]) -> List[tuple]:
    """Consecutive day pairs within a trial.

    Night change is one day's last frame against the next day's first, and
    24-hour change is one day's first against the next day's first — both come
    from the same pairs, so neither needs its own frames collected.
    """
    return [(a, b) for a, b in zip(days, days[1:]) if a.trial == b.trial]
