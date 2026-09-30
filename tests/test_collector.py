import json
from pathlib import Path

import numpy as np
import pytest

from cichlid_bower_claude.cloud import FakeCloud
from cichlid_bower_claude.collect import bundle as B
from cichlid_bower_claude.collect.collector import collect_project
from cichlid_bower_claude.paths import Layout, LayoutError
from tests.make_project import build

ANALYSIS = 'YH_MC_Parentals'


@pytest.fixture
def project(tmp_path):
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir(); remote.mkdir()
    project_id, _ = build(remote, days=5, resets=(2,), H=32, W=40)
    layout = Layout(local_root=local)
    return layout, FakeCloud(layout=layout, remote_dir=remote), project_id


def test_collects_and_writes_the_manifest_last(project):
    layout, cloud, project_id = project
    result = collect_project(layout, cloud, project_id, ANALYSIS)
    assert result.status == 'ok' and result.days > 0
    paths = layout.project(project_id, ANALYSIS)
    assert paths.bundle.exists() and paths.manifest.exists()
    assert Path(cloud.uploaded[-1]).name == 'manifest.json'


def test_a_second_run_is_skipped(project):
    layout, cloud, project_id = project
    collect_project(layout, cloud, project_id, ANALYSIS)
    assert collect_project(layout, cloud, project_id, ANALYSIS).status == 'skipped'


def test_changing_an_offset_forces_a_rebuild(project):
    layout, cloud, project_id = project
    collect_project(layout, cloud, project_id, ANALYSIS)
    settings = layout.project(project_id, ANALYSIS).prep_json
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(json.dumps({'schema': 'cichlid-prep/1',
                                    'trials': {'1': {'start': 30, 'stop': 0, 'reset': 0}}}))
    cloud.upload(settings)
    assert collect_project(layout, cloud, project_id, ANALYSIS).status == 'ok'


def test_nan_survives_the_bundle_round_trip(project):
    layout, cloud, project_id = project
    collect_project(layout, cloud, project_id, ANALYSIS)
    data = B.load_bundle(layout.project(project_id, ANALYSIS).bundle)
    assert set(data) >= {'first', 'last', 'residual', 'trend', 'travel', 'valid'}
    assert data['first'].dtype == np.float64        # widened on read
    assert np.isnan(data['residual']).any()         # the dead strip


def test_missing_frames_are_reported_not_absorbed(tmp_path):
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir(); remote.mkdir()
    project_id, log = build(remote, days=4, resets=(), H=32, W=40)
    lights_on = [f for f in log.frames if f.lights_on]
    dropped = {f.index for f in lights_on[40:43]}
    import shutil
    shutil.rmtree(remote / '__ProjectData' / ANALYSIS / project_id)
    build(remote, project_id=project_id, days=4, resets=(), H=32, W=40,
          drop_frames=dropped)
    layout = Layout(local_root=local)
    cloud = FakeCloud(layout=layout, remote_dir=remote)
    result = collect_project(layout, cloud, project_id, ANALYSIS)
    assert result.status == 'ok' and result.missing


def test_a_path_outside_the_root_is_refused(tmp_path):
    layout = Layout(local_root=tmp_path / 'local')
    with pytest.raises(LayoutError):
        layout.cloud(Path('/etc/passwd'))


def test_states_file_is_fetched_when_absent(tmp_path):
    """A fresh local root has nothing downloaded; the first command must work."""
    import pandas as pd
    from cichlid_bower_claude.states import AnalysisStates, StatesError
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir(); remote.mkdir()
    csv = remote / '__AnalysisStates' / 'YH_MC_Parentals' / 'YH_MC_Parentals.csv'
    csv.parent.mkdir(parents=True)
    pd.DataFrame({'projectID': ['MC_920_t001_tr1'], 'Prep': [True],
                  'RunAnalysis': [True], 'Category': ['MC']}).to_csv(csv, index=False)
    layout = Layout(local_root=local)
    cloud = FakeCloud(layout=layout, remote_dir=remote)
    paths = layout.analysis('YH_MC_Parentals')
    assert not paths.states_csv.exists()
    states = AnalysisStates.load(paths, cloud)
    assert states.project_ids() == ['MC_920_t001_tr1']
    assert paths.states_csv.exists()


def test_a_missing_analysis_lists_what_is_there(tmp_path):
    import pandas as pd
    from cichlid_bower_claude.states import AnalysisStates, StatesError
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir(); remote.mkdir()
    for name in ('YH_MC_Parentals', 'MC_singles'):
        d = remote / '__AnalysisStates' / name
        d.mkdir(parents=True)
        pd.DataFrame({'projectID': ['p']}).to_csv(d / (name + '.csv'), index=False)
    layout = Layout(local_root=local)
    cloud = FakeCloud(layout=layout, remote_dir=remote)
    with pytest.raises(StatesError) as caught:
        AnalysisStates.load(layout.analysis('Typo_Parentals'), cloud)
    assert 'MC_singles' in str(caught.value)


def test_a_failed_listing_is_not_mistaken_for_an_empty_one(tmp_path):
    """An unreachable remote must not read as 'nothing there'."""
    from cichlid_bower_claude.cloud import Cloud, CloudError
    layout = Layout(local_root=tmp_path)
    cloud = Cloud(layout=layout)

    class Failing(Cloud):
        def _run(self, args, check=True):
            import subprocess
            return subprocess.CompletedProcess(args, 1, '', 'remote not found')

    with pytest.raises(CloudError) as caught:
        Failing(layout=layout).listdir(tmp_path / '__AnalysisStates')
    assert 'remote not found' in str(caught.value)


def test_cloud_paths_keep_the_leading_slash(tmp_path):
    """ptm_dropbox:CoS/... and ptm_dropbox:/CoS/... are different places."""
    layout = Layout(local_root=tmp_path)
    remote = layout.cloud(tmp_path / 'MC_920_t001_tr1' / 'Frames.tar')
    assert remote.startswith('ptm_dropbox:/CoS/')
    assert remote.endswith('/MC_920_t001_tr1/Frames.tar')