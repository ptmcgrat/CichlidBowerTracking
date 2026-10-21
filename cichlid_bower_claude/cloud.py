"""Talking to Dropbox, through rclone.

One class, one job, a small surface. The old downloadData carried four boolean
flags — tarred, tarred_subdirs, allow_errors, quiet — which is usually several
functions wearing one name. Here each operation is its own method and the
caller decides what to do about failure.

Nothing else in the package shells out. That makes this the only module a test
has to replace, and FakeCloud below is that replacement.
"""

from __future__ import annotations

import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .paths import Layout


class CloudError(Exception):
    """An rclone command failed."""


# rclone has no timeout of its own, so a stalled transfer waits forever. That
# is not merely slow: the server holds a project's lock across its upload, so
# one hung rclone made that project permanently unsaveable until a restart,
# reported to whoever tried next as 'someone else is saving this project'.
METADATA_TIMEOUT = 120      # lsf, size: a listing that takes two minutes is broken
TRANSFER_TIMEOUT = 3600     # a Frames.tar is gigabytes and legitimately slow
QUICK_TIMEOUT = 120         # a small file the server is waiting on

_METADATA = ('lsf', 'lsd', 'size', 'lsjson')


@dataclass
class Cloud:
    layout: Layout
    dry_run: bool = False
    verbose: bool = False
    transfers: int = 4

    def _run(self, args: List[str], check: bool = True,
             timeout: Optional[float] = None) -> subprocess.CompletedProcess:
        if self.verbose:
            print('    rclone ' + ' '.join(args))
        if self.dry_run and args[0] in ('copy', 'copyto', 'delete', 'deletefile', 'moveto'):
            return subprocess.CompletedProcess(args, 0, '', '')
        if timeout is None:
            timeout = (METADATA_TIMEOUT if args[0] in _METADATA
                       else TRANSFER_TIMEOUT)
        try:
            result = subprocess.run(['rclone'] + args, capture_output=True,
                                    encoding='utf-8', timeout=timeout)
        except FileNotFoundError:
            raise CloudError('rclone is not installed, or not on PATH')
        except subprocess.TimeoutExpired:
            # killed, so nothing is left holding whatever waits on this
            raise CloudError('rclone %s gave up after %gs: %s'
                             % (args[0], timeout, ' '.join(args[1:3])))
        if check and result.returncode != 0:
            raise CloudError('rclone ' + args[0] + ' failed: ' + (result.stderr or '').strip())
        return result

    # -- reading -----------------------------------------------------------
    def exists(self, local: Path) -> bool:
        """Whether the cloud copy of a local path is there."""
        remote = self.layout.cloud(local)
        result = self._run(['lsf', remote], check=False)
        return result.returncode == 0 and bool(result.stdout.strip())

    def listdir(self, local_dir: Path, dirs_only: bool = False,
                files_only: bool = False, max_depth: Optional[int] = None) -> List[str]:
        """Names in a cloud directory.

        ``dirs_only`` matters where the caller means subdirectories: lsf lists
        files too, so a loose CSV sitting beside the real directories came back
        looking like one of them.

        ``max_depth`` above 1 recurses, and names then carry their subdirectory
        ('YH_MC/YH_MC.csv'), which lets one round trip answer a question that
        would otherwise cost one call per name.

        Raises on failure rather than returning nothing. Swallowing the error
        made a misconfigured remote indistinguishable from an empty directory,
        which is the difference between a five-second fix and an afternoon.
        """
        remote = self.layout.cloud(local_dir)
        args = ['lsf', remote]
        if dirs_only:
            args.append('--dirs-only')
        if files_only:
            args.append('--files-only')
        if max_depth is not None and max_depth > 1:
            args += ['-R', '--max-depth', str(max_depth)]
        result = self._run(args, check=False)
        if result.returncode != 0:
            raise CloudError('rclone lsf ' + remote + ' failed: ' +
                             ((result.stderr or '').strip() or
                              'exit %d with no message' % result.returncode))
        return [n.strip().rstrip('/') for n in result.stdout.splitlines() if n.strip()]

    def size(self, local: Path) -> Optional[int]:
        """Bytes in the cloud copy, or None if it is not there.

        Worth knowing before pulling an archive that may be several gigabytes.
        """
        result = self._run(['size', '--json', self.layout.cloud(local)], check=False)
        if result.returncode != 0:
            return None
        import json
        try:
            return int(json.loads(result.stdout)['bytes'])
        except Exception:
            return None

    def download(self, local: Path) -> None:
        """Fetch one file. Raises if it is not there — the caller decides."""
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        self._run(['copyto', self.layout.cloud(local), str(local),
                   '--transfers', str(self.transfers)])

    def download_optional(self, local: Path) -> bool:
        """Fetch one file if it exists. False if it does not."""
        try:
            self.download(local)
        except CloudError:
            return False
        return local.exists()

    def download_dir(self, local_dir: Path) -> bool:
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        result = self._run(['copy', self.layout.cloud(local_dir), str(local_dir),
                            '--transfers', str(self.transfers)], check=False)
        return result.returncode == 0

    # -- writing -----------------------------------------------------------
    def upload(self, local: Path, timeout: Optional[float] = None) -> None:
        """Send one file or directory up.

        ``timeout`` is for callers who are holding something while they wait —
        the server passes a short one, because a save that hangs takes the
        project's lock with it.
        """
        local = Path(local)
        if not local.exists():
            raise CloudError('nothing to upload at ' + str(local))
        if local.is_dir():
            self._run(['copy', str(local), self.layout.cloud(local),
                       '--transfers', str(self.transfers)], timeout=timeout)
        else:
            self._run(['copyto', str(local), self.layout.cloud(local)],
                      timeout=timeout)

    def delete(self, local: Path) -> None:
        self._run(['deletefile', self.layout.cloud(local)], check=False)


def extract_tar(archive: Path, members: dict, into: Path) -> List[str]:
    """Pull named members out of a downloaded archive.

    members maps the name inside the archive to the filename to write. Returns
    the names that were not found, so a caller can report them rather than
    discovering a missing file much later.
    """
    into = Path(into)
    into.mkdir(parents=True, exist_ok=True)
    missing = []
    with tarfile.open(archive, 'r') as tar:
        index = {}
        for member in tar.getmembers():
            if member.isfile():
                index[member.name] = member
                index[member.name.split('/')[-1]] = member
        for name, out_name in members.items():
            member = index.get(name) or index.get(name.split('/')[-1])
            if member is None:
                missing.append(name)
                continue
            source = tar.extractfile(member)
            if source is None:
                missing.append(name)
                continue
            with open(into / out_name, 'wb') as f:
                f.write(source.read())
    return missing


@dataclass
class FakeCloud:
    """A Cloud that never leaves the machine, for tests.

    Backed by a directory standing in for the remote, so upload and download
    really move bytes and a test can assert on what ended up where.
    """

    layout: Layout
    remote_dir: Path
    uploaded: List[str] = field(default_factory=list)
    downloaded: List[str] = field(default_factory=list)

    def _remote_path(self, local: Path) -> Path:
        rel = Path(local).resolve().relative_to(self.layout.local_root.resolve())
        return Path(self.remote_dir) / rel

    def exists(self, local: Path) -> bool:
        return self._remote_path(local).exists()

    def listdir(self, local_dir: Path, dirs_only: bool = False,
                files_only: bool = False, max_depth: Optional[int] = None):
        remote = self._remote_path(local_dir)
        if not remote.is_dir():
            return []
        if max_depth is not None and max_depth > 1:
            names = []
            for item in sorted(remote.rglob('*')):
                depth = len(item.relative_to(remote).parts)
                if depth > max_depth:
                    continue
                if dirs_only and not item.is_dir():
                    continue
                if files_only and not item.is_file():
                    continue
                names.append(str(item.relative_to(remote)))
            return names
        out = []
        for item in sorted(remote.iterdir()):
            if dirs_only and not item.is_dir():
                continue
            if files_only and not item.is_file():
                continue
            out.append(item.name)
        return out

    def size(self, local: Path):
        remote = self._remote_path(local)
        return remote.stat().st_size if remote.exists() else None

    def download(self, local: Path) -> None:
        remote = self._remote_path(local)
        if not remote.exists():
            raise CloudError('nothing at ' + str(remote))
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        Path(local).write_bytes(remote.read_bytes())
        self.downloaded.append(str(local))

    def download_optional(self, local: Path) -> bool:
        try:
            self.download(local)
        except CloudError:
            return False
        return True

    def download_dir(self, local_dir: Path) -> bool:
        remote = self._remote_path(local_dir)
        if not remote.is_dir():
            return False
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for item in remote.iterdir():
            if item.is_file():
                (Path(local_dir) / item.name).write_bytes(item.read_bytes())
        return True

    def upload(self, local: Path, timeout: Optional[float] = None) -> None:
        local = Path(local)
        if not local.exists():
            raise CloudError('nothing to upload at ' + str(local))
        remote = self._remote_path(local)
        remote.parent.mkdir(parents=True, exist_ok=True)
        if local.is_dir():
            remote.mkdir(exist_ok=True)
            for item in local.iterdir():
                if item.is_file():
                    (remote / item.name).write_bytes(item.read_bytes())
        else:
            remote.write_bytes(local.read_bytes())
        self.uploaded.append(str(local))

    def delete(self, local: Path) -> None:
        remote = self._remote_path(local)
        if remote.exists():
            remote.unlink()
