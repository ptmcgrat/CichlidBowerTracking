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


def list_analyses(layout: Layout, cloud=None) -> List[str]:
    """Every analysisID, for the master page.

    Local first, then the cloud if one is given — a machine that has never run
    a sweep has nothing locally and would otherwise show an empty master page.
    """
    found = set()
    local = layout.local_root / '__AnalysisStates'
    if local.is_dir():
        found.update(p.name for p in local.iterdir() if p.is_dir())
    if cloud is not None:
        found.update(cloud.listdir(local))
    return sorted(found)
