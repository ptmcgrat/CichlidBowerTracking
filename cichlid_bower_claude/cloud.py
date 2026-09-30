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


@dataclass
class Cloud:
    layout: Layout
    dry_run: bool = False
    verbose: bool = False
    transfers: int = 4

    def _run(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        if self.verbose:
            print('    rclone ' + ' '.join(args))
        if self.dry_run and args[0] in ('copy', 'copyto', 'delete', 'deletefile', 'moveto'):
            return subprocess.CompletedProcess(args, 0, '', '')
        try:
            result = subprocess.run(['rclone'] + args, capture_output=True,
                                    encoding='utf-8')
        except FileNotFoundError:
            raise CloudError('rclone is not installed, or not on PATH')
        if check and result.returncode != 0:
            raise CloudError('rclone ' + args[0] + ' failed: ' + (result.stderr or '').strip())
        return result

    # -- reading -----------------------------------------------------------
    def exists(self, local: Path) -> bool:
        """Whether the cloud copy of a local path is there."""
        remote = self.layout.cloud(local)
        result = self._run(['lsf', remote], check=False)
        return result.returncode == 0 and bool(result.stdout.strip())

    def listdir(self, local_dir: Path) -> List[str]:
        """Names in a cloud directory.

        Raises on failure rather than returning nothing. Swallowing the error
        made a misconfigured remote indistinguishable from an empty directory,
        which is the difference between a five-second fix and an afternoon.
        """
        remote = self.layout.cloud(local_dir)
        result = self._run(['lsf', remote], check=False)
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
    def upload(self, local: Path) -> None:
        local = Path(local)
        if not local.exists():
            raise CloudError('nothing to upload at ' + str(local))
        if local.is_dir():
            self._run(['copy', str(local), self.layout.cloud(local),
                       '--transfers', str(self.transfers)])
        else:
            self._run(['copyto', str(local), self.layout.cloud(local)])

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

    def listdir(self, local_dir: Path):
        remote = self._remote_path(local_dir)
        return sorted(p.name for p in remote.iterdir()) if remote.is_dir() else []

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

    def upload(self, local: Path) -> None:
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