"""Reading a logfile.

Parsing only. A malformed line is recorded and skipped, never raised — losing
a whole project because one line is short has cost real time. Validation lives
in ``validate.py`` and runs over the result.

The file is one record per line, a type before the first colon and then
``Key: value`` pairs separated by ``,,``:

    FrameCaptured: NpyFile: Frames/Frame_000001.npy,,PicFile: ...,,Time: ...
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Optional

from .model import Frame, Movie, ProjectLog, Trial

_TIME_FORMATS = ('%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S',
                 '%Y-%m-%d_%H:%M:%S', '%Y-%m-%d')


def parse_time(text: str) -> Optional[dt.datetime]:
    text = (text or '').strip()
    for fmt in _TIME_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _fields(line: str) -> Dict[str, str]:
    """Key-value pairs from one record.

    The writer has used ``Key: value``, ``Key:value`` and ``Key=value`` over
    the years, so all three are accepted. The old parser tried each in turn
    inside nested try blocks; one regex is easier to reason about.
    """
    body = line.split(':', 1)[1] if ':' in line else ''
    out = {}
    for chunk in body.split(',,'):
        match = re.match(r'\s*([A-Za-z_][\w ]*?)\s*[:=]\s*(.*)$', chunk)
        if match:
            out[match.group(1).strip()] = match.group(2).strip()
    return out


def _as_float(text: Optional[str]) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _as_bool(text: Optional[str]) -> bool:
    return str(text).strip().lower() in ('true', '1', 'yes')


def _resolution(text: Optional[str]):
    """'1296x972' or '(1296, 972)' to a pair, or (None, None)."""
    if not text:
        return None, None
    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    return None, None


def parse_logfile(path: Path, running: bool = False) -> ProjectLog:
    """Read a logfile into a ProjectLog. Never raises on bad content."""
    log = ProjectLog(running=running)
    seen_master_start = False
    open_movies: Dict[str, Movie] = {}

    with open(path, errors='replace') as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.rstrip('\n')
            if ':' not in line:
                continue
            kind = line.split(':', 1)[0].strip()
            data = _fields(line)

            if kind == 'MasterStart':
                if seen_master_start:
                    log.issues.append('MasterStart appears more than once')
                    continue
                seen_master_start = True
                log.system = data.get('System', '')
                log.device = data.get('Device', '')
                log.camera = data.get('Camera', '')
                log.uname = data.get('Uname', '')
                log.tank_id = data.get('TankID', '')
                log.project_id = data.get('ProjectID', '')
                log.analysis_id = data.get('AnalysisID', '')
                log.sample_id = data.get('SampleID')

            elif kind == 'MasterRecordInitialStart':
                log.master_start = parse_time(data.get('Time', ''))

            elif kind == 'MasterRecordStop':
                log.master_stop = parse_time(data.get('Time', ''))

            elif kind == 'MasterRecordRestart':
                when = parse_time(data.get('Time', ''))
                if when:
                    log.restarts.append(when)

            elif kind == 'FrameCaptured':
                when = parse_time(data.get('Time', ''))
                npy = data.get('NpyFile', '')
                if when is None or not npy:
                    log.issues.append('line %d: frame with no time or file' % number)
                    continue
                log.frames.append(Frame(
                    npy_file=npy, pic_file=data.get('PicFile', ''), time=when,
                    median=_as_float(data.get('AvgMed')),
                    std=_as_float(data.get('AvgStd')),
                    good_pixels=_as_float(data.get('GP')),
                    lights_on=_as_bool(data.get('LOF'))))

            elif kind == 'PiCameraStarted':
                when = parse_time(data.get('Time', ''))
                video = data.get('VideoFile', '')
                if when is None or not video:
                    log.issues.append('line %d: video start with no time or file' % number)
                    continue
                width, height = _resolution(data.get('Resolution'))
                movie = Movie(start_time=when, h264_file=video,
                              pic_file=data.get('PicFile', ''),
                              framerate=_as_float(data.get('FrameRate')),
                              width=width, height=height)
                log.movies.append(movie)
                open_movies[video] = movie

            elif kind == 'PiCameraStopped':
                video = data.get('File', '')
                movie = open_movies.get(video)
                if movie is None:
                    log.issues.append('no PiCameraStarted for ' + (video or '?'))
                    continue
                movie.stop_time = parse_time(data.get('Time', ''))

            elif kind == 'TankResetStart':
                when = parse_time(data.get('Time', ''))
                if when:
                    log.reset_starts.append(when)

            elif kind == 'TankResetStop':
                when = parse_time(data.get('Time', ''))
                if when:
                    log.reset_stops.append(when)

    log.frames.sort(key=lambda f: f.time)
    log.movies.sort(key=lambda m: m.start_time)
    if log.running and log.movies and log.movies[-1].stop_time is None and log.frames:
        log.movies[-1].stop_time = log.frames[-1].time
    log.trials = build_trials(log)
    return log


def build_trials(log: ProjectLog) -> List[Trial]:
    """Split the recording into trials at tank resets.

    Each trial runs from one reset's end to the next reset's start, the first
    starting at the master start. ``reset_time`` on a trial is the reset that
    follows it.

    When the project is still running, everything after the last reset is a
    trial in its own right. When it is finished, it is not: those frames are
    the final picture of the bower rather than an active trial.
    """
    if log.master_start is None:
        return []

    starts = sorted(log.reset_starts)
    stops = sorted(log.reset_stops)
    trials: List[Trial] = []
    boundary_start = log.master_start

    # never reset: the whole recording is one trial. Splitting on resets and
    # then requiring one produced no trials at all for a project that simply
    # ran once and stopped.
    if not starts:
        end = log.master_stop or (log.frames[-1].time if log.frames else boundary_start)
        if end > boundary_start:
            return [Trial(number=1, start_time=boundary_start, stop_time=end,
                          reset_time=None)]
        return []

    for i, reset_start in enumerate(starts):
        reset_stop = stops[i] if i < len(stops) else None
        trials.append(Trial(number=len(trials) + 1, start_time=boundary_start,
                            stop_time=reset_start, reset_time=reset_stop))
        if reset_stop is None:
            return trials
        boundary_start = reset_stop

    if log.running:
        end = log.master_stop or (log.frames[-1].time if log.frames else boundary_start)
        trials.append(Trial(number=len(trials) + 1, start_time=boundary_start,
                            stop_time=end, reset_time=None))
    return trials
