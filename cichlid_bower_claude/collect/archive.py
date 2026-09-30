"""Getting frames out of Frames.tar.

The whole archive is downloaded. The residual is fitted through every lights-on
frame of every day, so by the time you need that many members there is nothing
left for seeking to save — and dropping it removes the ordering probe, the
cached member index, and the fallback for archives written in readdir order.
That machinery produced real failures: a four-sample ordering test passes by
luck one time in twenty-four, which is roughly one project in a sweep of
thirty, and the failure surfaced much later as a member that could not be
found.

Reading is streamed one member at a time. A month of depth frames does not fit
in memory, and the arrays are reduced per day and discarded.
"""

from __future__ import annotations

import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


class ArchiveError(Exception):
    """The archive is missing or unreadable."""


@dataclass
class Archive:
    """An open Frames.tar, indexed by member name."""

    path: Path
    _index: Dict[str, tarfile.TarInfo] = field(default_factory=dict, repr=False)
    _tar: Optional[tarfile.TarFile] = field(default=None, repr=False)

    def __enter__(self) -> 'Archive':
        if not Path(self.path).exists():
            raise ArchiveError('no archive at ' + str(self.path))
        self._tar = tarfile.open(self.path, 'r')
        for member in self._tar.getmembers():
            if not member.isfile():
                continue
            self._index[member.name] = member
            # also by bare filename: the logfile records Frames/Frame_000001.npy
            # while some archives were written with a different prefix
            self._index.setdefault(member.name.split('/')[-1], member)
        return self

    def __exit__(self, *exc) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None

    def __contains__(self, name: str) -> bool:
        return name in self._index or name.split('/')[-1] in self._index

    @property
    def member_count(self) -> int:
        return len({m.name for m in self._index.values()})

    def _member(self, name: str) -> Optional[tarfile.TarInfo]:
        return self._index.get(name) or self._index.get(name.split('/')[-1])

    def read_bytes(self, name: str) -> Optional[bytes]:
        member = self._member(name)
        if member is None or self._tar is None:
            return None
        source = self._tar.extractfile(member)
        return source.read() if source is not None else None

    def read_array(self, name: str) -> Optional[np.ndarray]:
        """One depth frame as float64, or None if it is not in the archive."""
        raw = self.read_bytes(name)
        if raw is None:
            return None
        import io
        try:
            return np.load(io.BytesIO(raw), allow_pickle=False).astype(np.float64)
        except Exception:
            return None

    def extract(self, members: Dict[str, str], into: Path) -> List[str]:
        """Write named members out under new filenames.

        Returns the names that were not found, so a caller can report them
        rather than discovering the gap much later.
        """
        into = Path(into)
        into.mkdir(parents=True, exist_ok=True)
        missing = []
        for name, out_name in members.items():
            raw = self.read_bytes(name)
            if raw is None:
                missing.append(name)
                continue
            (into / out_name).write_bytes(raw)
        return missing

    def stack(self, names: List[str], shape: Optional[Tuple[int, int]] = None
              ) -> Tuple[np.ndarray, List[str]]:
        """Read a day's frames into one array.

        A frame that is absent or the wrong shape becomes a plane of NaN
        rather than shifting everything after it. Losing one frame of a day
        should cost that frame, not the day.
        """
        arrays, missing = [], []
        for name in names:
            array = self.read_array(name)
            if array is None:
                missing.append(name)
                arrays.append(None)
                continue
            if shape is None:
                shape = array.shape
            if array.shape != shape:
                missing.append(name)
                arrays.append(None)
                continue
            arrays.append(array)
        if shape is None:
            return np.empty((0, 0, 0)), names
        blank = np.full(shape, np.nan)
        return np.stack([a if a is not None else blank for a in arrays]), missing

    def iter_days(self, log, days) -> Iterator[Tuple[object, np.ndarray, List[str]]]:
        """Yield each day's lights-on frames as one array, then discard it.

        Streaming rather than accumulating: a month of frames at 640x480 is
        tens of gigabytes in float64, and only the reductions are kept.
        """
        for day in days:
            names = [f.npy_file for f in log.frames
                     if day.first.index <= f.index <= day.last.index and f.lights_on]
            stacked, missing = self.stack(names)
            yield day, stacked, missing

    def iter_std_days(self, log, days) -> Iterator[Tuple[object, Optional[np.ndarray]]]:
        """The per-capture spread alongside each frame, where it exists.

        Not every project has these. Absent is normal, not an error: the
        residual works without them and they only add a second view of noise.
        """
        for day in days:
            names = [f.std_file for f in log.frames
                     if day.first.index <= f.index <= day.last.index and f.lights_on]
            present = [n for n in names if n in self]
            if not present:
                yield day, None
                continue
            stacked, _ = self.stack(present)
            yield day, stacked


def ensure_archive(cloud, project_paths, force: bool = False) -> Tuple[Path, Optional[int]]:
    """Make sure Frames.tar is on disk. Returns its path and cloud size.

    The size is reported before the transfer starts so a multi-gigabyte pull is
    a stated cost rather than a surprise.
    """
    local = project_paths.frames_tar
    size = cloud.size(local)
    if local.exists() and not force:
        return local, size
    if size is None:
        raise ArchiveError('no Frames.tar in the cloud for ' + project_paths.project_id)
    cloud.download(local)
    if not local.exists():
        raise ArchiveError('download of Frames.tar produced nothing')
    return local, size
