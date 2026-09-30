"""Where everything lives.

Pure computation. Nothing here touches the disk or the network, so a wrong
path is visible in a test rather than three minutes into a download.

The old FileManager set 146 ``self.localX`` attributes in its constructor.
Adding a file meant editing the constructor, a typo failed at runtime, and
nothing was discoverable. Here a path is an attribute of the thing it belongs
to: ``project.frames_tar``, ``analysis.states_csv``.

Local and cloud layouts are identical below the root, so one method converts
between them and there is no second set of names to keep in step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

DEFAULT_REMOTE = 'ptm_dropbox:'
DEFAULT_CLOUD_ROOT = '/CoS/BioSci/BioSci-McGrath/Apps/CichlidPiData/'
LOCAL_ROOT_ENV = 'CICHLID_LOCAL_ROOT'
REMOTE_ENV = 'CICHLID_REMOTE'


class LayoutError(Exception):
    """The local root is not configured or does not exist."""


@dataclass(frozen=True)
class Layout:
    """The two roots every other path hangs off.

    The old code probed mounted directories to guess where it was running,
    which meant the answer depended on the machine and failed obscurely when
    it guessed wrong. This is explicit: the environment says, or the caller
    says, or it raises.
    """

    local_root: Path
    remote: str = DEFAULT_REMOTE
    cloud_root: PurePosixPath = PurePosixPath(DEFAULT_CLOUD_ROOT)

    @classmethod
    def from_env(cls) -> 'Layout':
        root = os.environ.get(LOCAL_ROOT_ENV)
        if not root:
            raise LayoutError(
                LOCAL_ROOT_ENV + ' is not set. Point it at the local working '
                'directory, e.g. export ' + LOCAL_ROOT_ENV + '=~/Temp/CichlidAnalyzer')
        return cls(local_root=Path(root).expanduser(),
                   remote=os.environ.get(REMOTE_ENV, DEFAULT_REMOTE))

    def cloud(self, local: Path) -> str:
        """The rclone address of a local path.

        Raises rather than guessing if the path is outside the root: a silent
        wrong answer here uploads a file to the wrong place.
        """
        try:
            rel = Path(local).resolve().relative_to(self.local_root.resolve())
        except ValueError:
            raise LayoutError(str(local) + ' is not inside ' + str(self.local_root))
        # the leading slash matters: this remote resolves ptm_dropbox:CoS/...
        # and ptm_dropbox:/CoS/... to different places, and only the second is
        # where the data is
        return self.remote + str(self.cloud_root / PurePosixPath(rel))

    def analysis(self, analysis_id: str) -> 'AnalysisPaths':
        return AnalysisPaths(self, analysis_id)

    def project(self, project_id: str, analysis_id: str) -> 'ProjectPaths':
        """A project's paths.

        The analysisID is required: project data lives under
        ``__ProjectData/<analysisID>/<projectID>/``, so the same projectID
        under two analyses is two directories.
        """
        return ProjectPaths(self, project_id, analysis_id)


@dataclass(frozen=True)
class AnalysisPaths:
    """One analysisID: the set of projects reviewed together."""

    layout: Layout
    analysis_id: str

    @property
    def root(self) -> Path:
        return self.layout.local_root / '__AnalysisStates' / self.analysis_id

    @property
    def states_csv(self) -> Path:
        """Which projects belong to this analysis, and which stages have run."""
        return self.root / (self.analysis_id + '.csv')

    @property
    def server_dir(self) -> Path:
        return self.root / 'WebServer'

    @property
    def index_html(self) -> Path:
        return self.server_dir / 'index.html'

    @property
    def index_cache(self) -> Path:
        """Per-project index entries, so one project can be rebuilt alone."""
        return self.server_dir / '_index.json'

    @property
    def submissions_dir(self) -> Path:
        return self.server_dir / '_submissions'

    def project_dir(self, project_id: str) -> Path:
        """Where this project's generated pages go."""
        return self.server_dir / project_id

    def project(self, project_id: str) -> 'ProjectPaths':
        return ProjectPaths(self.layout, project_id, self.analysis_id)


@dataclass(frozen=True)
class ProjectPaths:
    """One projectID: one tank recorded over days to weeks.

    Project data is stored per analysis, not at the top level: the same
    projectID can appear in more than one analysis and each has its own copy.
    """

    layout: Layout
    project_id: str
    analysis_id: str

    @property
    def root(self) -> Path:
        return (self.layout.local_root / '__ProjectData' / self.analysis_id
                / self.project_id)

    # -- source data, written by the tank ---------------------------------
    @property
    def logfile(self) -> Path:
        return self.root / 'Logfile.txt'

    @property
    def frames_tar(self) -> Path:
        return self.root / 'Frames.tar'

    @property
    def frames_dir(self) -> Path:
        return self.root / 'Frames'

    @property
    def videos_dir(self) -> Path:
        return self.root / 'Videos'

    # -- human corrections, written by the new server only ------------------
    @property
    def corrections_dir(self) -> Path:
        """Everything this server writes, and nothing the old pipeline did.

        A directory of its own so provenance is unambiguous: a file here was
        made by this code, and anything in MasterAnalysisFiles was not.
        """
        return self.root / 'Corrections'

    @property
    def prep_json(self) -> Path:
        return self.corrections_dir / 'prep.json'

    @property
    def corrections_history(self) -> Path:
        return self.corrections_dir / 'history'

    @property
    def analysis_dir(self) -> Path:
        """The old pipeline's outputs. Read for cluster data; never written."""
        return self.root / 'MasterAnalysisFiles'

    # -- collected data, written by the collector ---------------------------
    @property
    def collected_dir(self) -> Path:
        return self.root / 'Collected'

    @property
    def manifest(self) -> Path:
        """Written last, after every artefact it names has uploaded."""
        return self.collected_dir / 'manifest.json'

    @property
    def bundle(self) -> Path:
        """Endpoint frames, residual maps and diagnostics, one file."""
        return self.collected_dir / 'bundle.npz'

    def candidate_jpg(self, stem: str) -> Path:
        return self.collected_dir / (stem + '.jpg')

    def candidate_npy(self, stem: str) -> Path:
        return self.collected_dir / (stem + '.npy')

    def day_still(self, day: int, which: str) -> Path:
        return self.collected_dir / ('Day_%02d_%s.jpg' % (day, which))

    # -- analysis outputs ---------------------------------------------------
    @property
    def pages_dir(self) -> Path:
        """Generated page assets: small HTML, a payload, and PNGs served
        separately so the browser can cache them."""
        return self.root / 'Pages'

    @property
    def clusters_csv(self) -> Path:
        return self.analysis_dir / 'AllLabeledClusters.csv'

    def pose_csv(self, video_index: int) -> Path:
        return self.analysis_dir / ('%04d_pose.csv' % (video_index + 1))
