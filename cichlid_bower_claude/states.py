"""Which projects belong to an analysis, and what has been run on them.

The one place pandas appears. The old FileManager mixed this with path layout
and cloud sync, so reading the states table meant constructing an object that
also wanted a network.

The table is the lab's record, not this package's: columns come and go, and
older analyses lack ones newer analyses have. So every accessor tolerates a
missing column rather than raising, and ``missing_columns`` reports what was
absent instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .paths import AnalysisPaths, Layout

PROJECT_COLUMN = 'projectID'
EXPECTED = ('Prep', 'Depth', 'Cluster', 'RunAnalysis', 'Category', 'tankID')


class StatesError(Exception):
    """The states table is missing or unusable."""


def _truthy(value) -> bool:
    """Whether a cell means yes.

    The table has held True, 'TRUE', 1 and 'Yes' for the same thing over the
    years, and a cluster column holds 'VideoIndices: 0,1,2' when it has run.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip()
    if not text or text.lower() in ('nan', 'none', 'false', '0', 'no'):
        return False
    return True


@dataclass
class AnalysisStates:
    """One analysis's project table."""

    analysis_id: str
    table: pd.DataFrame

    @classmethod
    def load(cls, paths: AnalysisPaths, cloud=None,
             refresh: bool = False) -> 'AnalysisStates':
        """Read the states table, fetching it if it is not here yet.

        A fresh machine, or a new local root, has nothing downloaded. Without
        the fetch the first command on a new machine fails with a path the user
        has no way to create, which is a poor welcome.
        """
        csv = paths.states_csv
        if cloud is not None and (refresh or not csv.exists()):
            cloud.download_optional(csv)
        if not csv.exists():
            message = 'no states file at ' + str(csv)
            if cloud is not None:
                known = _cloud_analyses(paths, cloud)
                if known:
                    message += '\nAnalysisIDs in the cloud: ' + ', '.join(known)
                else:
                    message += ('\nNothing found in the cloud either — check the '
                                'rclone remote and that the analysisID is spelt right.')
            raise StatesError(message)
        table = pd.read_csv(csv)
        if PROJECT_COLUMN not in table.columns:
            raise StatesError(str(csv) + ' has no ' + PROJECT_COLUMN + ' column')
        table = table.set_index(PROJECT_COLUMN)
        table.index = table.index.astype(str)
        return cls(analysis_id=paths.analysis_id, table=table)

    @property
    def missing_columns(self) -> List[str]:
        return [c for c in EXPECTED if c not in self.table.columns]

    def project_ids(self, require_prep: bool = True,
                    require_analysis: bool = True) -> List[str]:
        """Projects in this analysis, in the order pages link them.

        Sorted, because the previous and next links on a page and the order of
        a sweep have to agree; if they disagree, stepping through an analysis
        skips projects.
        """
        keep = pd.Series(True, index=self.table.index)
        if require_prep and 'Prep' in self.table.columns:
            keep &= self.table['Prep'].map(_truthy)
        if require_analysis and 'RunAnalysis' in self.table.columns:
            keep &= self.table['RunAnalysis'].map(_truthy)
        return sorted(self.table.index[keep].tolist())

    def category(self, project_id: str) -> str:
        """The group a project belongs to on the landing page."""
        if 'Category' not in self.table.columns or project_id not in self.table.index:
            return ''
        value = self.table.loc[project_id, 'Category']
        text = str(value).strip()
        return '' if text.lower() in ('nan', 'none', '') else text

    def broods(self, project_id: str) -> Optional[int]:
        """How many broods this female produced, if the table says.

        Optional, and deliberately absent from EXPECTED: an analysis without
        the column is not missing anything, and reporting it as missing would
        make every older analysis look incomplete.
        """
        if 'Broods' not in self.table.columns or project_id not in self.table.index:
            return None
        value = self.table.loc[project_id, 'Broods']
        try:
            number = int(float(str(value).strip()))
        except (TypeError, ValueError):
            return None
        return number if 0 <= number <= 10 else None

    def tank(self, project_id: str) -> str:
        if 'tankID' not in self.table.columns or project_id not in self.table.index:
            return ''
        value = str(self.table.loc[project_id, 'tankID']).strip()
        return '' if value.lower() in ('nan', 'none') else value

    def has_run(self, project_id: str, stage: str) -> bool:
        """Whether a stage has completed for a project."""
        if stage not in self.table.columns or project_id not in self.table.index:
            return False
        return _truthy(self.table.loc[project_id, stage])

    def neighbours(self, project_id: str) -> dict:
        """Previous and next project, for stepping through without the index."""
        order = self.project_ids()
        if project_id not in order:
            return {}
        position = order.index(project_id)
        return {'prev': order[position - 1] if position > 0 else None,
                'next': order[position + 1] if position + 1 < len(order) else None,
                'position': position + 1, 'total': len(order)}


def _cloud_analyses(paths: AnalysisPaths, cloud) -> List[str]:
    """Analysis directories in the cloud, for a useful error message."""
    try:
        return sorted(cloud.listdir(paths.root.parent))
    except Exception:
        return []


def local_analyses(layout: Layout) -> List[str]:
    """AnalysisIDs this machine can actually open, by reading no further than
    its own disk.

    The one the server wants. Discovery is a different question from service:
    asking the cloud what exists costs a round trip per name and then lists
    analyses whose states file is not here, which the master page can only
    render as a row saying it could not read them. Collecting an analysis
    downloads its states file first, so anything worth serving is already here.
    """
    local = layout.local_root / '__AnalysisStates'
    if not local.is_dir():
        return []
    names = []
    for child in sorted(local.iterdir()):
        if not child.is_dir() or child.name.startswith('__'):
            continue
        if (child / (child.name + '.csv')).is_file():
            names.append(child.name)
    return names


def list_analyses(layout: Layout, cloud=None) -> List[str]:
    """Every analysisID, including ones not yet on this machine.

    For discovery — the ``analyses`` command — not for serving; see
    ``local_analyses``.

    An analysis is a directory holding a states file named after it. Listing
    directory names alone invented analyses out of whatever else sits under
    __AnalysisStates: a loose CSV came back as a name, because lsf lists files
    too, and a stray directory came back as an empty analysis. One recursive
    listing answers it exactly, and still in a single round trip.
    """
    found = set(local_analyses(layout))
    local = layout.local_root / '__AnalysisStates'
    if cloud is not None:
        # raises if the remote is unreachable, rather than reporting nothing
        for entry in cloud.listdir(local, files_only=True, max_depth=2):
            parts = entry.replace('\\', '/').strip('/').split('/')
            if len(parts) == 2 and parts[1] == parts[0] + '.csv':
                found.add(parts[0])
    # __DeletedData and its like live alongside the analyses and are not ones
    return sorted(name for name in found if name and not name.startswith('__'))
