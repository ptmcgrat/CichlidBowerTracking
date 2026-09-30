"""What a logfile describes, as data.

Dataclasses with no behaviour beyond what is derivable from their own fields.
The old FrameObj took seven positional arguments and MovieObj five, so a
transposed pair was a silent error; these are keyword-constructed and typed.

Trials deliberately do not hold their own frames. In the old code
``Trial.__init__`` filtered the global frame list, which fixed a trial's days
at construction — but trial start and stop offsets are now a human choice that
can change after parsing. Day boundaries are computed by ``logfile.days``
instead, as a function of the offsets.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Frame:
    """One depth capture: a NumPy array and a JPEG, five minutes apart."""

    npy_file: str
    pic_file: str
    time: dt.datetime
    median: Optional[float]
    std: Optional[float]
    good_pixels: Optional[float]
    lights_on: bool

    @property
    def index(self) -> int:
        """Zero-based, from the filename — Frame_000001.npy is index 0."""
        stem = self.npy_file.split('/')[-1]
        return int(stem.split('_')[1].split('.')[0]) - 1

    @property
    def std_file(self) -> str:
        """The per-capture spread written alongside each frame."""
        return self.npy_file.replace('Frame_', 'Frame_std_')

    @property
    def directory(self) -> str:
        return self.npy_file.rsplit('/', 1)[0] + '/' if '/' in self.npy_file else ''


@dataclass
class Movie:
    """One video, with the still the camera wrote when it started."""

    start_time: dt.datetime
    h264_file: str
    pic_file: str
    framerate: Optional[float]
    width: Optional[int]
    height: Optional[int]
    stop_time: Optional[dt.datetime] = None

    @property
    def mp4_file(self) -> str:
        return self.h264_file.replace('.h264', '') + '.mp4'

    @property
    def base_name(self) -> str:
        return self.mp4_file.split('/')[-1].replace('.mp4', '')

    @property
    def index(self) -> int:
        stem = self.h264_file.split('/')[-1]
        return int(stem.split('_')[0]) - 1


@dataclass(frozen=True)
class Trial:
    """One recording period between tank resets.

    ``reset_time`` is the reset that *follows* this trial, which is the next
    trial's start — so it is redundant except on the last trial, where no trial
    follows it and the frames after it are the final picture of the bower.
    """

    number: int
    start_time: dt.datetime
    stop_time: dt.datetime
    reset_time: Optional[dt.datetime] = None

    @property
    def is_open_ended(self) -> bool:
        return self.reset_time is None


@dataclass
class ProjectLog:
    """A parsed logfile. Frames are sorted by time."""

    project_id: str = ''
    tank_id: str = ''
    analysis_id: str = ''
    sample_id: Optional[str] = None
    device: str = ''
    camera: str = ''
    system: str = ''
    uname: str = ''
    master_start: Optional[dt.datetime] = None
    master_stop: Optional[dt.datetime] = None
    running: bool = False
    frames: List[Frame] = field(default_factory=list)
    movies: List[Movie] = field(default_factory=list)
    trials: List[Trial] = field(default_factory=list)
    reset_starts: List[dt.datetime] = field(default_factory=list)
    reset_stops: List[dt.datetime] = field(default_factory=list)
    restarts: List[dt.datetime] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        return self.movies[0].width if self.movies and self.movies[0].width else 1296

    @property
    def height(self) -> int:
        return self.movies[0].height if self.movies and self.movies[0].height else 972

    def frames_between(self, start: dt.datetime, stop: dt.datetime,
                       lights_on: bool = True) -> List[Frame]:
        return [f for f in self.frames
                if start <= f.time <= stop and (f.lights_on or not lights_on)]

    def movies_between(self, start: dt.datetime, stop: dt.datetime) -> List[Movie]:
        """Movies that *started* inside the window.

        A movie that began before the trial is not the trial's, even though it
        overlaps it: its opening frames are of the previous trial's sand.
        """
        return [m for m in self.movies if start <= m.start_time <= stop]
