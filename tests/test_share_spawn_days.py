"""The sharing script, against collected fixtures.

Worth testing because its mistakes are published outside the lab: a wrong
day, a day counted from unclustered transitions, or two days that are really
the same tank twice.
"""

import datetime as dt

import pandas as pd
import pytest

from cichlid_bower_claude.cloud import FakeCloud
from cichlid_bower_claude.collect.collector import collect_project
from cichlid_bower_claude.paths import Layout
from cichlid_bower_claude.states import AnalysisStates
from tests.make_project import build
from tools import share_spawn_days as S

ANALYSIS = 'YH_MC_Parentals'

# two projects per category, so 'one day per project' has something to choose
PROJECTS = (('MC_920_t001_tr1', 'MC'), ('MC_921_t002_tr1', 'MC'),
            ('TI_404_t077_tr1', 'TI'), ('TI_405_t078_tr1', 'TI'))


def _clusters(path, days, spawns_on_day, clustered=True, probability=0.9):
    """A cluster file with a known number of spawns on a chosen day."""
    rows = []
    for day_index, entry in enumerate(days):
        first = dt.datetime.fromisoformat(entry['first'])
        count = spawns_on_day.get(day_index, 0)
        for i in range(count):
            rows.append({'TimeStamp': first + dt.timedelta(minutes=5 * i + 5),
                         'X': 400, 'Y': 600, 'Prediction': 's',
                         'Probability': probability,
                         'LID': 1 if clustered else -1, 'ClipCreated': 'Yes'})
        # a feed event every day, so 's' is really being selected for
        rows.append({'TimeStamp': first + dt.timedelta(minutes=2), 'X': 100,
                     'Y': 100, 'Prediction': 'f', 'Probability': 0.95,
                     'LID': 2, 'ClipCreated': 'Yes'})
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).set_index('TimeStamp').to_csv(path)


@pytest.fixture
def analysis(tmp_path):
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir(); remote.mkdir()
    layout = Layout(local_root=local)
    cloud = FakeCloud(layout=layout, remote_dir=remote)

    states_dir = local / '__AnalysisStates' / ANALYSIS
    states_dir.mkdir(parents=True)
    pd.DataFrame({'projectID': [p for p, _ in PROJECTS],
                  'tankID': ['t'] * len(PROJECTS),
                  'Prep': [True] * len(PROJECTS),
                  'RunAnalysis': [True] * len(PROJECTS),
                  'Category': [c for _, c in PROJECTS]}
                 ).to_csv(states_dir / (ANALYSIS + '.csv'), index=False)

    for project_id, _ in PROJECTS:
        build(remote, project_id=project_id, days=4, resets=(), H=32, W=40,
              analysis_id=ANALYSIS)
        assert collect_project(layout, cloud, project_id, ANALYSIS).status == 'ok'
    return layout, cloud


def _write_clusters(layout, cloud, spawns_by_project, **kwargs):
    """Replace the cluster file of *every* project, not only the named ones.

    make_project.build writes its own realistic AllLabeledClusters.csv with
    spawns scattered through every day. Writing only the projects a test cares
    about leaves that fixture data in the others, and the test then measures
    the fixture instead of the thing it is checking.
    """
    from cichlid_bower_claude.collect import bundle as B
    for project_id, _ in PROJECTS:
        paths = layout.project(project_id, ANALYSIS)
        manifest = B.read_manifest(paths.manifest)
        _clusters(paths.clusters_csv, manifest['days'],
                  spawns_by_project.get(project_id, {}), **kwargs)


def _days(layout, cloud, confidence=0.5):
    states = AnalysisStates.load(layout.analysis(ANALYSIS))
    return S._spawn_days(layout, cloud, states, ANALYSIS, confidence, False)


def test_spawns_are_counted_on_the_day_they_happened(analysis):
    layout, cloud = analysis
    _write_clusters(layout, cloud, {'MC_920_t001_tr1': {1: 7, 2: 3}})
    counts = {(d.project_id, d.index): d.spawns for d in _days(layout, cloud)}
    assert counts[('MC_920_t001_tr1', 1)] == 7
    assert counts[('MC_920_t001_tr1', 2)] == 3
    assert counts[('MC_920_t001_tr1', 0)] == 0


def test_an_unclustered_transition_is_not_a_spawn(analysis):
    """LID -1 is a transition that never became an event. Counting those
    would publish a day chosen by a clustering failure."""
    layout, cloud = analysis
    _write_clusters(layout, cloud, {'MC_920_t001_tr1': {1: 9}},
                    clustered=False)
    assert all(day.spawns == 0 for day in _days(layout, cloud))


def test_the_confidence_threshold_is_applied(analysis):
    layout, cloud = analysis
    _write_clusters(layout, cloud, {'MC_920_t001_tr1': {1: 9}},
                    probability=0.3)
    assert all(d.spawns == 0 for d in _days(layout, cloud))
    found = [d for d in _days(layout, cloud, confidence=0.2) if d.spawns]
    assert found and found[0].spawns == 9


def test_two_days_per_category_come_from_different_projects(analysis):
    """The busiest two days may both be one tank; two looks at the same fish
    is rarely what a category example should be."""
    layout, cloud = analysis
    _write_clusters(layout, cloud, {
        'MC_920_t001_tr1': {1: 20, 2: 18},    # the top two days overall
        'MC_921_t002_tr1': {3: 11},
        'TI_404_t077_tr1': {0: 6},
        'TI_405_t078_tr1': {2: 4},
    })
    chosen = S._choose(_days(layout, cloud), 2, True)
    assert sorted(chosen) == ['MC', 'TI']
    mc = chosen['MC']
    assert [d.spawns for d in mc] == [20, 11]
    assert len({d.project_id for d in mc}) == 2
    assert [d.spawns for d in chosen['TI']] == [6, 4]


def test_both_days_may_share_a_project_when_asked(analysis):
    layout, cloud = analysis
    _write_clusters(layout, cloud, {'MC_920_t001_tr1': {1: 20, 2: 18}})
    chosen = S._choose(_days(layout, cloud), 2, False)
    assert [d.spawns for d in chosen['MC']] == [20, 18]


def test_one_project_in_a_category_still_yields_two_days(analysis):
    """Preferring different projects must not silently return a short list."""
    layout, cloud = analysis
    _write_clusters(layout, cloud, {'MC_920_t001_tr1': {1: 20, 2: 18}})
    chosen = S._choose(_days(layout, cloud), 2, True)
    assert len(chosen['MC']) == 2


def test_a_video_overlapping_the_day_is_included(analysis):
    """Not just one that started inside it: a recording that began before the
    window still holds the first part of the day."""
    layout, cloud = analysis
    from cichlid_bower_claude.collect import bundle as B
    from cichlid_bower_claude.logfile.parse import parse_logfile
    paths = layout.project('MC_920_t001_tr1', ANALYSIS)
    cloud.download_optional(paths.logfile)
    log = parse_logfile(paths.logfile)
    manifest = B.read_manifest(paths.manifest)
    entry = manifest['days'][1]
    day = S.Day(project_id='MC_920_t001_tr1', category='MC', index=1, trial=1,
                first=dt.datetime.fromisoformat(entry['first']),
                last=dt.datetime.fromisoformat(entry['last']))
    videos = S._videos_for(log, day)
    assert videos and all(v.endswith('.mp4') for v in videos)
    assert all(v.startswith('Videos/') for v in videos)


def _pose(remote, project_id, names, flat=False, analysis=ANALYSIS):
    """Put parquet in the cloud, in one of the two layouts seen in the wild."""
    if flat:
        directory = (remote / '__ProjectData'
                     / ('%s_%s' % (analysis, project_id)) / 'PoseData' / 'days')
    else:
        directory = (remote / '__ProjectData' / analysis / project_id
                     / 'PoseData' / 'days')
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b'parquet')
    return directory


def _find(layout, cloud, project_id, stems, extra=()):
    paths = layout.project(project_id, ANALYSIS)
    return S._parquet_for(cloud,
                          S._pose_dirs(layout, paths, ANALYSIS, list(extra)),
                          stems)


def test_parquet_is_found_under_the_project(analysis):
    layout, cloud = analysis
    remote = cloud.remote_dir
    _pose(remote, 'MC_920_t001_tr1', ['0001_vid.parquet', '0002_vid.parquet'])
    found, searched, _ = _find(layout, cloud, 'MC_920_t001_tr1',
                               ['0001_vid', '0002_vid'])
    assert sorted(p.name for p in found) == ['0001_vid.parquet',
                                             '0002_vid.parquet']
    assert searched


def test_parquet_is_also_found_under_the_flat_analysis_project_name(analysis):
    """PoseData has been seen under YH_MC_Parentals_MC_959_t011_tr1/ rather
    than under the project, so both shapes are tried."""
    layout, cloud = analysis
    _pose(cloud.remote_dir, 'MC_920_t001_tr1', ['0001_vid.parquet'], flat=True)
    found, searched, _ = _find(layout, cloud, 'MC_920_t001_tr1', ['0001_vid'])
    assert [p.name for p in found] == ['0001_vid.parquet']
    assert any('YH_MC_Parentals_MC_920_t001_tr1' in s for s in searched)


def test_only_the_exact_video_stem_matches(analysis):
    """A substring match would publish a stray '_old' copy alongside it."""
    layout, cloud = analysis
    _pose(cloud.remote_dir, 'MC_920_t001_tr1',
          ['0001_vid.parquet', '0001_vid_old.parquet', '0009_vid.parquet'])
    found, _, ignored = _find(layout, cloud, 'MC_920_t001_tr1', ['0001_vid'])
    assert [p.name for p in found] == ['0001_vid.parquet']
    assert '0001_vid_old.parquet' in ignored


def test_missing_parquet_says_where_it_looked(analysis):
    layout, cloud = analysis
    found, searched, ignored = _find(layout, cloud, 'MC_920_t001_tr1',
                                     ['0001_vid'])
    assert found == []
    # nothing was readable, so nothing is claimed to have been searched
    assert all('PoseData' in s for s in searched)


def test_an_extra_parquet_directory_can_be_given(analysis):
    layout, cloud = analysis
    directory = (cloud.remote_dir / '__ProjectData' / ANALYSIS
                 / 'MC_920_t001_tr1' / 'Elsewhere')
    directory.mkdir(parents=True)
    (directory / '0001_vid.parquet').write_bytes(b'parquet')
    found, _, _ = _find(layout, cloud, 'MC_920_t001_tr1', ['0001_vid'],
                        extra=['Elsewhere'])
    assert [p.name for p in found] == ['0001_vid.parquet']


def test_a_project_with_no_cluster_file_is_skipped_not_fatal(analysis):
    """A project can be collected long before it is classified, so a missing
    cluster file must not stop the sweep."""
    layout, cloud = analysis
    _write_clusters(layout, cloud, {'MC_920_t001_tr1': {1: 5}})
    gone = layout.project('TI_404_t077_tr1', ANALYSIS)
    gone.clusters_csv.unlink()
    cloud._remote_path(gone.clusters_csv).unlink()     # and not re-downloadable

    days = _days(layout, cloud)
    assert 'TI_404_t077_tr1' not in {d.project_id for d in days}
    assert 'MC_920_t001_tr1' in {d.project_id for d in days}
    # the rest of the analysis still came through
    assert len({d.project_id for d in days}) == len(PROJECTS) - 1


def test_a_dry_run_goes_all_the_way_through_and_copies_nothing(analysis,
                                                               monkeypatch,
                                                               tmp_path,
                                                               capsys):
    """The pieces all pass their tests; this is the wiring between them.

    A blank page once survived every unit test because the failure was in the
    assembly, so main() is exercised rather than only its parts.
    """
    layout, cloud = analysis
    _write_clusters(layout, cloud, {
        'MC_920_t001_tr1': {1: 20}, 'MC_921_t002_tr1': {2: 9},
        'TI_404_t077_tr1': {1: 7}, 'TI_405_t078_tr1': {3: 4}})
    for project_id, _ in PROJECTS:
        _pose(cloud.remote_dir, project_id,
              ['%04d_vid.parquet' % n for n in range(1, 12)])

    monkeypatch.setattr(S, 'Cloud', lambda **kw: cloud)
    # any rclone call at all would be a dry run that is not dry
    monkeypatch.setattr(S.subprocess, 'run',
                        lambda *a, **k: pytest.fail('ran %r' % (a,)))
    monkeypatch.chdir(tmp_path)

    code = S.main([ANALYSIS, '--root', str(layout.local_root),
                   '--dest', 'ptm_dropbox:/CoS-EXT/x/MorePoseData'])
    out = capsys.readouterr().out
    assert code == 0
    assert 'dry run' in out and 'nothing was copied' in out
    # one day from each of the four projects: two categories, two each
    for project_id, _ in PROJECTS:
        assert project_id in out
    assert '.parquet' in out and '.mp4' in out
    # the index is written on a dry run too: it is the list to check before --go
    assert (tmp_path / 'shared_spawn_days.csv').exists()


def test_the_index_records_the_spawn_times(analysis, monkeypatch, tmp_path):
    """Recipients get a day of video; without the clock times they have no way
    to find the spawns in it."""
    import csv as _csv
    layout, cloud = analysis
    _write_clusters(layout, cloud, {'MC_920_t001_tr1': {1: 3}})
    monkeypatch.setattr(S, 'Cloud', lambda **kw: cloud)
    monkeypatch.chdir(tmp_path)
    S.main([ANALYSIS, '--root', str(layout.local_root), '--dest', 'x:/y'])

    with open(tmp_path / 'shared_spawn_days.csv') as handle:
        rows = list(_csv.DictReader(handle))
    mine = [r for r in rows if r['projectID'] == 'MC_920_t001_tr1']
    assert mine and mine[0]['spawns'] == '3'
    assert len(mine[0]['spawnTimes'].split()) == 3


def _log_with(movies, frames_end):
    """A stand-in log: only what _videos_for reads."""
    from cichlid_bower_claude.logfile.model import Frame, Movie, ProjectLog
    log = ProjectLog()
    for index, (start, stop) in enumerate(movies, 1):
        log.movies.append(Movie(start_time=start, stop_time=stop,
                                h264_file='Videos/%04d_vid.h264' % index,
                                pic_file='', framerate=30,
                                width=1296, height=972))
    log.frames.append(Frame(npy_file='Frames/Frame_000001.npy',
                            pic_file='Frames/Frame_000001.jpg',
                            time=frames_end, median=60.0, std=0.1,
                            good_pixels=1.0, lights_on=True))
    return log


def _day(first, last):
    return S.Day(project_id='p', category='c', index=38, trial=1,
                 first=first, last=last)


def test_a_recording_with_no_logged_end_does_not_claim_every_later_day():
    """The real failure: a video truncated in week one was offered as material
    for day 38, because an unknown end was treated as 'still recording'.

    Videos 1 and 2 died early and logged no stop; video 3 is the day's own.
    """
    base = dt.datetime(2025, 1, 1, 8)
    movies = [
        (base, None),                                        # died, no stop
        (base + dt.timedelta(days=7), None),                 # died, no stop
        (base + dt.timedelta(days=37), base + dt.timedelta(days=37, hours=10)),
    ]
    log = _log_with(movies, base + dt.timedelta(days=40))
    day = _day(base + dt.timedelta(days=37, hours=1),
               base + dt.timedelta(days=37, hours=9))

    assert S._videos_for(log, day) == ['Videos/0003_vid.mp4']


def test_an_unlogged_end_is_taken_as_the_next_recording_start():
    """Not dropped: the recording is real and does cover its own window."""
    base = dt.datetime(2025, 1, 1, 8)
    movies = [(base, None), (base + dt.timedelta(hours=6), None)]
    log = _log_with(movies, base + dt.timedelta(hours=12))

    early = _day(base + dt.timedelta(hours=1), base + dt.timedelta(hours=2))
    assert S._videos_for(log, early) == ['Videos/0001_vid.mp4']

    later = _day(base + dt.timedelta(hours=7), base + dt.timedelta(hours=8))
    assert S._videos_for(log, later) == ['Videos/0002_vid.mp4']


def test_the_last_recording_without_a_stop_ends_with_the_project():
    base = dt.datetime(2025, 1, 1, 8)
    log = _log_with([(base, None)], base + dt.timedelta(hours=10))
    day = _day(base + dt.timedelta(hours=8), base + dt.timedelta(hours=9))
    assert S._videos_for(log, day) == ['Videos/0001_vid.mp4']
    # but not a day after the project ended
    after = _day(base + dt.timedelta(days=5), base + dt.timedelta(days=5, hours=1))
    assert S._videos_for(log, after) == []


def test_a_recording_straddling_midnight_still_counts_for_the_day():
    """Overlap, not 'started inside' -- the point of the original choice."""
    base = dt.datetime(2025, 1, 1, 20)
    log = _log_with([(base, base + dt.timedelta(hours=14))],
                    base + dt.timedelta(hours=20))
    day = _day(dt.datetime(2025, 1, 2, 8), dt.datetime(2025, 1, 2, 9))
    assert S._videos_for(log, day) == ['Videos/0001_vid.mp4']


def test_inferred_ends_are_reported():
    base = dt.datetime(2025, 1, 1, 8)
    log = _log_with([(base, None), (base + dt.timedelta(hours=6),
                                    base + dt.timedelta(hours=9))],
                    base + dt.timedelta(hours=12))
    assert S._inferred(log) == ['Videos/0001_vid.mp4']
