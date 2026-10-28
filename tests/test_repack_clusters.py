"""Repacking clusters.json in place, without re-collecting.

The point of the tool is that it is cheap: adding the Sex column to an
already-collected project must not require the archive, which collection
re-downloads. So the test collects once, deletes nothing, writes a cluster
CSV that has Sex where the first one did not, and checks the packed file
gains the column while the bundle is left alone.
"""

import datetime as dt
import json

import pandas as pd
import pytest

from cichlid_bower_claude.cloud import FakeCloud
from cichlid_bower_claude.collect import bundle as B
from cichlid_bower_claude.collect.collector import collect_project
from cichlid_bower_claude.paths import Layout
from tests.make_project import build
from tools import repack_clusters as R

ANALYSIS = 'YHMC_BC_Data'
PROJECT = 'MC_920_t001_tr1'


def _clusters(path, days, sex=None):
    """One event a day, optionally carrying a Sex column."""
    rows = []
    for entry in days:
        first = dt.datetime.fromisoformat(entry['first'])
        for i in range(3):
            row = {'TimeStamp': first + dt.timedelta(minutes=5 * i + 5),
                   'X': 400, 'Y': 600, 'Prediction': 'p', 'Probability': 0.9,
                   'LID': 1, 'ClipCreated': 'Yes', 'TrackID': 20 + i}
            if sex is not None:
                row['Sex'] = sex[i % len(sex)]
            rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).set_index('TimeStamp').to_csv(path)


@pytest.fixture
def project(tmp_path):
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir(); remote.mkdir()
    layout = Layout(local_root=local)
    cloud = FakeCloud(layout=layout, remote_dir=remote)
    build(remote, project_id=PROJECT, days=3, resets=(), H=32, W=40,
          analysis_id=ANALYSIS)
    assert collect_project(layout, cloud, PROJECT, ANALYSIS).status == 'ok'
    return layout, cloud, layout.project(PROJECT, ANALYSIS)


def _packed(paths):
    with open(paths.clusters_packed) as handle:
        return json.load(handle)


def test_a_repack_adds_sex_without_rebuilding_the_bundle(project):
    layout, cloud, paths = project
    manifest = B.read_manifest(paths.manifest)
    _clusters(paths.clusters_csv, manifest['days'])
    R.repack(paths, go=True, cloud=cloud)
    assert 'sex' not in _packed(paths)

    bundle_before = paths.bundle.stat().st_mtime_ns
    _clusters(paths.clusters_csv, manifest['days'],
              sex=['Male', 'Female', 'Unknown'])
    change = R.repack(paths, go=True, cloud=cloud)

    packed = _packed(paths)
    assert 'sex' in packed
    assert packed['sexes'] == ['Unknown', 'Male', 'Female']
    assert change['has_sex'] and not change['had_sex']
    # one of each per day, so the counts follow the fixture's day count
    per_day = len(manifest['days'])
    assert change['by_sex'] == {'Male': per_day, 'Female': per_day,
                                'Unknown': per_day}
    # the expensive artefact was never touched, which is the whole point
    assert paths.bundle.stat().st_mtime_ns == bundle_before


def test_the_schema_is_not_bumped_so_the_bundle_stays_current(project):
    """Bumping it would force the full re-collection this tool avoids. The
    cost is that an unrepacked project keeps its old packed file, which the
    page has to tolerate -- and does, by treating a missing column as
    'not collected'."""
    layout, cloud, paths = project
    manifest = B.read_manifest(paths.manifest)
    _clusters(paths.clusters_csv, manifest['days'], sex=['Male'])
    R.repack(paths, go=True, cloud=cloud)
    assert B.is_current(B.read_manifest(paths.manifest))


def test_a_dry_run_changes_nothing(project):
    layout, cloud, paths = project
    manifest = B.read_manifest(paths.manifest)
    _clusters(paths.clusters_csv, manifest['days'], sex=['Female'])
    before = paths.clusters_packed.read_bytes()
    change = R.repack(paths, go=False, cloud=cloud)
    assert change['has_sex']          # it says what it would do
    assert paths.clusters_packed.read_bytes() == before


def test_an_uncollected_project_is_skipped_with_a_reason(project):
    layout, cloud, paths = project
    paths.manifest.unlink()
    with pytest.raises(R.Skip) as error:
        R.repack(paths, go=True, cloud=cloud)
    assert 'not collected' in str(error.value)


def test_a_project_without_a_cluster_csv_is_skipped(project):
    layout, cloud, paths = project
    if paths.clusters_csv.exists():
        paths.clusters_csv.unlink()
    with pytest.raises(R.Skip):
        R.repack(paths, go=True, cloud=cloud)


def test_the_manifest_summary_is_refreshed_too(project):
    """The page reads its header counts from manifest['clusters'], so a
    repack that updated only the packed file would show the old totals."""
    layout, cloud, paths = project
    manifest = B.read_manifest(paths.manifest)
    _clusters(paths.clusters_csv, manifest['days'],
              sex=['Male', 'Male', 'Female'])
    R.repack(paths, go=True, cloud=cloud)
    after = B.read_manifest(paths.manifest)
    days = len(manifest['days'])
    assert after['clusters']['bySex'] == {'Male': 2 * days, 'Female': days}
    assert after['clusters']['n'] == 3 * days
