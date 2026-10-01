import json

import pandas as pd
import pytest

flask = pytest.importorskip('flask')

from cichlid_bower_claude.cloud import FakeCloud
from cichlid_bower_claude.collect.collector import collect_project
from cichlid_bower_claude.paths import Layout
from cichlid_bower_claude.server import app as A
from cichlid_bower_claude.server import corrections as C
from tests.make_project import build

ANALYSIS = 'YH_MC_Parentals'
PROJECT = 'MC_920_t001_tr1'


@pytest.fixture
def client(tmp_path):
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir(); remote.mkdir()
    build(remote, project_id=PROJECT, days=4, resets=(), H=32, W=40)
    states_dir = local / '__AnalysisStates' / ANALYSIS
    states_dir.mkdir(parents=True)
    pd.DataFrame({'projectID': [PROJECT], 'tankID': ['t001'], 'Prep': [True],
                  'RunAnalysis': [True], 'Category': ['MC']}
                 ).to_csv(states_dir / (ANALYSIS + '.csv'), index=False)
    layout = Layout(local_root=local)
    cloud = FakeCloud(layout=layout, remote_dir=remote)
    assert collect_project(layout, cloud, PROJECT, ANALYSIS).status == 'ok'
    A.configure(layout, ANALYSIS, cloud=cloud, upload=False)
    A.app.config['TESTING'] = True
    return A.app.test_client(), layout


def test_the_index_lists_the_project(client):
    response = client[0].get('/')
    assert response.status_code == 200
    assert PROJECT in response.get_data(as_text=True)


def test_the_project_page_builds_on_first_visit(client):
    http, layout = client
    paths = layout.project(PROJECT, ANALYSIS)
    assert not (paths.pages_dir / 'page.json').exists()
    assert http.get('/' + PROJECT + '/').status_code == 200
    assert (paths.pages_dir / 'page.json').exists()


def test_the_payload_and_assets_are_served(client):
    http, _ = client
    http.get('/' + PROJECT + '/')
    payload = http.get('/' + PROJECT + '/page.json')
    assert payload.status_code == 200
    data = json.loads(payload.get_data(as_text=True))
    url = data['days'][0]['firstPng']['url']
    image = http.get('/' + PROJECT + '/' + url)
    assert image.status_code == 200
    assert image.get_data()[:8] == b'\x89PNG\r\n\x1a\n'


def test_a_save_lands_and_reads_back(client):
    http, layout = client
    body = {'residual_k': 4.5, 'trials': {'1': {'start': 30, 'stop': 0, 'reset': 0}},
            'depth_crop': [[5, 5], [30, 5], [30, 25], [5, 25]]}
    response = http.post('/' + PROJECT + '/save', json=body)
    assert response.status_code == 200
    prep = C.load(layout.project(PROJECT, ANALYSIS))
    assert prep.residual_k == 4.5
    assert prep.times_for(1).start == 30


def test_a_save_keeps_the_previous_version(client):
    http, layout = client
    http.post('/' + PROJECT + '/save', json={'residual_k': 3.0})
    http.post('/' + PROJECT + '/save', json={'residual_k': 5.0})
    assert len(C.history(layout.project(PROJECT, ANALYSIS))) == 1


def test_an_unknown_project_is_refused(client):
    assert client[0].get('/NOT_A_PROJECT/').status_code == 404


def test_paths_cannot_escape_the_pages_directory(client):
    http, _ = client
    http.get('/' + PROJECT + '/')
    assert http.get('/' + PROJECT + '/../../../etc/passwd').status_code in (403, 404)


def test_a_save_writes_nowhere_but_corrections(client):
    """Collecting reads the old pipeline's files; saving must not touch them."""
    http, layout = client
    http.get('/' + PROJECT + '/')
    paths = layout.project(PROJECT, ANALYSIS)
    before = {p: p.stat().st_mtime_ns for p in paths.root.rglob('*') if p.is_file()}
    http.post('/' + PROJECT + '/save', json={'residual_k': 4.0})
    after = {p: p.stat().st_mtime_ns for p in paths.root.rglob('*') if p.is_file()}
    changed = {p for p in after if before.get(p) != after[p]}
    assert changed, 'the save wrote nothing at all'
    for path in changed:
        assert path.relative_to(paths.root).parts[0] == 'Corrections', str(path)


def test_a_save_survives_a_reload(client):
    """The payload is cached, so the corrections inside it must be read fresh."""
    http, layout = client
    http.get('/' + PROJECT + '/')                      # builds and caches the payload
    http.post('/' + PROJECT + '/save',
              json={'residual_k': 6.0,
                    'depth_crop': [[5, 5], [30, 5], [30, 25], [5, 25]],
                    'transform': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    'points': [{'pi': [1, 2], 'depth': [3, 4]}],
                    'trials': {'1': {'start': 45, 'stop': 0, 'reset': 0}}})
    payload = json.loads(http.get('/' + PROJECT + '/page.json').get_data(as_text=True))
    assert payload['prep']['residual_k'] == 6.0
    assert payload['prep']['depth_crop'][0] == [5, 5]
    assert payload['prep']['transform'][0][0] == 1
    assert payload['prep']['points']
    assert payload['prep']['trials']['1']['start'] == 45
    assert payload['prep']['updated']


def test_the_cached_payload_is_not_rebuilt_on_every_request(client):
    http, layout = client
    http.get('/' + PROJECT + '/')
    built = (layout.project(PROJECT, ANALYSIS).pages_dir / 'page.json').stat().st_mtime
    http.get('/' + PROJECT + '/page.json')
    after = (layout.project(PROJECT, ANALYSIS).pages_dir / 'page.json').stat().st_mtime
    assert built == after


def test_the_depth_page_is_served(client):
    http, _ = client
    response = http.get('/' + PROJECT + '/depth')
    assert response.status_code == 200
    assert 'depth.js' in response.get_data(as_text=True)


def test_the_shared_and_page_scripts_are_served(client):
    http, _ = client
    for name in ('common.js', 'prep.js', 'depth.js'):
        assert http.get('/' + PROJECT + '/' + name).status_code == 200


def test_prep_and_depth_link_to_each_other(client):
    http, _ = client
    assert 'depth' in http.get('/' + PROJECT + '/prep').get_data(as_text=True)
    assert 'prep' in http.get('/' + PROJECT + '/depth').get_data(as_text=True)


def test_cluster_events_are_served_separately(client):
    """They are larger than every image together; only one page needs them."""
    http, layout = client
    http.get('/' + PROJECT + '/')
    payload = json.loads(http.get('/' + PROJECT + '/page.json').get_data(as_text=True))
    assert 'clusters' not in payload                 # not embedded
    assert payload['hasClusters'] is True
    packed = http.get('/' + PROJECT + '/clusters.json')
    assert packed.status_code == 200
    assert json.loads(packed.get_data(as_text=True))['n'] > 0


def test_the_clusters_page_is_served(client):
    http, _ = client
    response = http.get('/' + PROJECT + '/clusters')
    assert response.status_code == 200
    assert 'clusters.js' in response.get_data(as_text=True)
    assert http.get('/' + PROJECT + '/clusters.js').status_code == 200


def test_a_save_records_the_name_the_page_sent(client):
    """Behind a VPN there is no authenticated identity, so the page supplies one."""
    http, layout = client
    http.post('/' + PROJECT + '/save', json={'residual_k': 4.0, 'who': 'Emily Keaton'})
    assert C.load(layout.project(PROJECT, ANALYSIS)).who == 'Emily Keaton'


def test_an_authenticated_identity_wins_over_a_claimed_one(client):
    http, layout = client
    http.post('/' + PROJECT + '/save', json={'residual_k': 4.0, 'who': 'Someone Else'},
              headers={'Cf-Access-Authenticated-User-Email': 'ek@gatech.edu'})
    assert C.load(layout.project(PROJECT, ANALYSIS)).who == 'ek@gatech.edu'


def test_the_statistics_page_is_served(client):
    http, _ = client
    assert http.get('/' + PROJECT + '/stats').status_code == 200
    assert http.get('/' + PROJECT + '/stats.js').status_code == 200
