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
ROOT = '/' + ANALYSIS + '/' + PROJECT


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
    A.configure(layout, cloud=cloud, upload=False)
    A.app.config['TESTING'] = True
    return A.app.test_client(), layout


def test_the_master_page_lists_the_analysis(client):
    """The root is now every analysis on the machine, not one analysis."""
    response = client[0].get('/')
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert ANALYSIS in page
    assert 'href="/%s/"' % ANALYSIS in page
    assert PROJECT not in page          # projects live a level down


def test_the_analysis_page_lists_its_projects(client):
    response = client[0].get('/' + ANALYSIS + '/')
    assert response.status_code == 200
    assert PROJECT in response.get_data(as_text=True)


def test_an_unknown_analysis_is_refused(client):
    assert client[0].get('/NOT_AN_ANALYSIS/').status_code == 404


def test_the_project_page_builds_on_first_visit(client):
    http, layout = client
    paths = layout.project(PROJECT, ANALYSIS)
    assert not (paths.pages_dir / 'page.json').exists()
    assert http.get(ROOT + '/').status_code == 200
    assert (paths.pages_dir / 'page.json').exists()


def test_the_payload_and_assets_are_served(client):
    http, _ = client
    http.get(ROOT + '/')
    payload = http.get(ROOT + '/page.json')
    assert payload.status_code == 200
    data = json.loads(payload.get_data(as_text=True))
    url = data['days'][0]['firstPng']['url']
    image = http.get(ROOT + '/' + url)
    assert image.status_code == 200
    assert image.get_data()[:8] == b'\x89PNG\r\n\x1a\n'


def test_a_save_lands_and_reads_back(client):
    http, layout = client
    body = {'residual_k': 4.5, 'trials': {'1': {'start': 30, 'stop': 0, 'reset': 0}},
            'depth_crop': [[5, 5], [30, 5], [30, 25], [5, 25]]}
    response = http.post(ROOT + '/save', json=body)
    assert response.status_code == 200
    prep = C.load(layout.project(PROJECT, ANALYSIS))
    assert prep.residual_k == 4.5
    assert prep.times_for(1).start == 30


def test_a_save_keeps_the_previous_version(client):
    http, layout = client
    http.post(ROOT + '/save', json={'residual_k': 3.0})
    http.post(ROOT + '/save', json={'residual_k': 5.0})
    assert len(C.history(layout.project(PROJECT, ANALYSIS))) == 1


def test_an_unknown_project_is_refused(client):
    assert client[0].get('/NOT_A_PROJECT/').status_code == 404


def test_paths_cannot_escape_the_pages_directory(client):
    http, _ = client
    http.get(ROOT + '/')
    assert http.get(ROOT + '/../../../etc/passwd').status_code in (403, 404)


def test_a_save_writes_nowhere_but_corrections(client):
    """Collecting reads the old pipeline's files; saving must not touch them."""
    http, layout = client
    http.get(ROOT + '/')
    paths = layout.project(PROJECT, ANALYSIS)
    before = {p: p.stat().st_mtime_ns for p in paths.root.rglob('*') if p.is_file()}
    http.post(ROOT + '/save', json={'residual_k': 4.0})
    after = {p: p.stat().st_mtime_ns for p in paths.root.rglob('*') if p.is_file()}
    changed = {p for p in after if before.get(p) != after[p]}
    assert changed, 'the save wrote nothing at all'
    for path in changed:
        assert path.relative_to(paths.root).parts[0] == 'Corrections', str(path)


def test_a_save_survives_a_reload(client):
    """The payload is cached, so the corrections inside it must be read fresh."""
    http, layout = client
    http.get(ROOT + '/')                      # builds and caches the payload
    http.post(ROOT + '/save',
              json={'residual_k': 6.0,
                    'depth_crop': [[5, 5], [30, 5], [30, 25], [5, 25]],
                    'transform': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    'points': [{'pi': [1, 2], 'depth': [3, 4]}],
                    'trials': {'1': {'start': 45, 'stop': 0, 'reset': 0}}})
    payload = json.loads(http.get(ROOT + '/page.json').get_data(as_text=True))
    assert payload['prep']['residual_k'] == 6.0
    assert payload['prep']['depth_crop'][0] == [5, 5]
    assert payload['prep']['transform'][0][0] == 1
    assert payload['prep']['points']
    assert payload['prep']['trials']['1']['start'] == 45
    assert payload['prep']['updated']


def test_the_cached_payload_is_not_rebuilt_on_every_request(client):
    http, layout = client
    http.get(ROOT + '/')
    built = (layout.project(PROJECT, ANALYSIS).pages_dir / 'page.json').stat().st_mtime
    http.get(ROOT + '/page.json')
    after = (layout.project(PROJECT, ANALYSIS).pages_dir / 'page.json').stat().st_mtime
    assert built == after


def test_the_depth_page_is_served(client):
    http, _ = client
    response = http.get(ROOT + '/depth')
    assert response.status_code == 200
    assert 'depth.js' in response.get_data(as_text=True)


def test_the_shared_and_page_scripts_are_served(client):
    http, _ = client
    for name in ('common.js', 'prep.js', 'depth.js'):
        assert http.get(ROOT + '/' + name).status_code == 200


def test_prep_and_depth_link_to_each_other(client):
    http, _ = client
    assert 'depth' in http.get(ROOT + '/prep').get_data(as_text=True)
    assert 'prep' in http.get(ROOT + '/depth').get_data(as_text=True)


def test_cluster_events_are_served_separately(client):
    """They are larger than every image together; only one page needs them."""
    http, layout = client
    http.get(ROOT + '/')
    payload = json.loads(http.get(ROOT + '/page.json').get_data(as_text=True))
    assert 'clusters' not in payload                 # not embedded
    assert payload['hasClusters'] is True
    packed = http.get(ROOT + '/clusters.json')
    assert packed.status_code == 200
    assert json.loads(packed.get_data(as_text=True))['n'] > 0


def test_the_clusters_page_is_served(client):
    http, _ = client
    response = http.get(ROOT + '/clusters')
    assert response.status_code == 200
    assert 'clusters.js' in response.get_data(as_text=True)
    assert http.get(ROOT + '/clusters.js').status_code == 200


def test_a_save_records_the_name_the_page_sent(client):
    """Behind a VPN there is no authenticated identity, so the page supplies one."""
    http, layout = client
    http.post(ROOT + '/save', json={'residual_k': 4.0, 'who': 'Emily Keaton'})
    assert C.load(layout.project(PROJECT, ANALYSIS)).who == 'Emily Keaton'


def test_an_authenticated_identity_wins_over_a_claimed_one(client):
    http, layout = client
    http.post(ROOT + '/save', json={'residual_k': 4.0, 'who': 'Someone Else'},
              headers={'Cf-Access-Authenticated-User-Email': 'ek@gatech.edu'})
    assert C.load(layout.project(PROJECT, ANALYSIS)).who == 'ek@gatech.edu'


def test_statistics_is_a_view_inside_the_cluster_page(client):
    """Not its own page: the two views share the same loaded events."""
    http, _ = client
    assert http.get(ROOT + '/stats').status_code == 404
    page = http.get(ROOT + '/clusters').get_data(as_text=True)
    assert 'stats.js' in page and 'clusters.js' in page
    assert http.get(ROOT + '/stats.js').status_code == 200


def test_the_summary_page_is_served(client):
    http, _ = client
    assert http.get(ROOT + '/summary').status_code == 200
    assert http.get(ROOT + '/summary.js').status_code == 200


def test_the_landing_page_reflects_a_saved_registration(client):
    """It is read fresh on every request, so a save shows up on reload."""
    http, _ = client
    before = http.get('/' + ANALYSIS + '/').get_data(as_text=True)
    assert '0 of 1' in before                 # no trial registered or cropped
    http.post(ROOT + '/save',
              json={'who': 'Emily Keaton',
                    'transform': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    'depth_crop': [[5, 5], [30, 5], [30, 25], [5, 25]],
                    'video_crop': [[5, 5], [30, 5], [30, 25], [5, 25]]})
    after = http.get('/' + ANALYSIS + '/').get_data(as_text=True)
    assert 'Emily' in after
    assert '1 of 1' in after


def test_every_page_links_to_the_other_three(client):
    http, _ = client
    pages = ['prep', 'depth', 'clusters', 'summary', 'features']
    for page in pages:
        text = http.get(ROOT + '/' + page).get_data(as_text=True)
        for other in pages:
            if other == page:
                assert 'href="%s"' % other not in text, page + ' links to itself'
            else:
                assert 'href="%s"' % other in text, page + ' is missing ' + other


def test_a_project_registered_only_per_trial_counts_as_registered(client):
    """Each trial carrying its own registration is a registered project.

    Checking the project-level field alone reported such a project as having
    no registration at all, which is what it looked like on the landing page.
    """
    http, layout = client
    http.post(ROOT + '/save', json={
        'who': 'Emily Keaton',
        'overrides': {'1': {'transform': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                            'depth_crop': [[0, 0], [9, 0], [9, 9], [0, 9]],
                            'video_crop': [[0, 0], [9, 0], [9, 9], [0, 9]]}}})
    prep = C.load(layout.project(PROJECT, ANALYSIS))
    assert prep.is_registered and prep.is_cropped
    assert prep.transform is None              # nothing at the project level
    assert '1 of 1' in http.get('/' + ANALYSIS + '/').get_data(as_text=True)


def test_an_empty_override_is_reported_as_a_gap(client):
    """A trial with an override but nothing in it looks configured and is not:
    its events have no transform and cannot be placed."""
    http, layout = client
    http.post(ROOT + '/save', json={
        'transform': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'depth_crop': [[0, 0], [9, 0], [9, 9], [0, 9]],
        'video_crop': [[0, 0], [9, 0], [9, 9], [0, 9]],
        'overrides': {'2': {'transform': None, 'depth_crop': None,
                            'video_crop': None}}})
    prep = C.load(layout.project(PROJECT, ANALYSIS))
    # trial 2 falls back to the project, so it is covered
    assert prep.coverage([1, 2])['ready']
    # but a trial with nothing anywhere is not
    bare = C.Prep(overrides={'2': C.TrialOverride()})
    assert bare.coverage([1, 2])['missingRegistration'] == [1, 2]


def test_day_marks_save_without_touching_the_rest(client):
    """The summary page owns only the marks: posting from a copy loaded before
    someone changed the crops must not undo that change."""
    http, layout = client
    http.post(ROOT + '/save', json={
        'who': 'Emily Keaton',
        'transform': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'depth_crop': [[5, 5], [30, 5], [30, 25], [5, 25]],
        'video_crop': [[5, 5], [30, 5], [30, 25], [5, 25]], 'residual_k': 6.0})
    response = http.post(ROOT + '/marks', json={
        'who': 'Patrick', 'day_marks': {'2': {'new_bower': True},
                                        '4': {'wall_building': True}}})
    assert response.status_code == 200
    prep = C.load(layout.project(PROJECT, ANALYSIS))
    assert prep.mark(2).new_bower and prep.mark(4).wall_building
    assert prep.transform is not None            # the registration survived
    assert prep.depth_crop[0] == [5, 5]
    assert prep.residual_k == 6.0
    assert prep.who == 'Patrick'


def test_unknown_mark_fields_are_ignored(client):
    http, layout = client
    http.post(ROOT + '/marks',
              json={'who': 'pm', 'day_marks': {'1': {'new_bower': True,
                                                     'nonsense': 'x'}}})
    assert C.load(layout.project(PROJECT, ANALYSIS)).mark(1).new_bower


def test_the_features_page_is_served(client):
    http, _ = client
    assert http.get(ROOT + '/features').status_code == 200
    assert http.get(ROOT + '/features.js').status_code == 200


def test_the_landing_page_groups_by_category_and_links_to_each_tab(client):
    http, _ = client
    page = http.get('/' + ANALYSIS + '/').get_data(as_text=True)
    for slug in ('prep', 'depth', 'clusters', 'summary', 'features'):
        assert 'href="/%s/%s/%s"' % (ANALYSIS, PROJECT, slug) in page
    assert 'band0' in page                      # category banding applied


def test_the_category_comparison_is_served(client):
    http, _ = client
    assert http.get('/' + ANALYSIS + '/features').status_code == 200
    assert http.get('/' + ANALYSIS + '/analysis.js').status_code == 200


def test_the_comparison_data_covers_every_collected_project(client):
    http, _ = client
    payload = json.loads(http.get('/' + ANALYSIS + '/features.json').get_data(as_text=True))
    assert payload['projects']
    names = [p['projectID'] for p in payload['projects']]
    assert PROJECT in names
    project = [p for p in payload['projects'] if p['projectID'] == PROJECT][0]
    assert project['category'] == 'MC'        # taken from the states file
    assert project['trials']


def test_the_analysis_page_does_not_shadow_a_project_page(client):
    """/features is the comparison; /<project>/features is one project."""
    http, _ = client
    assert 'analysis.js' in http.get('/' + ANALYSIS + '/features').get_data(as_text=True)
    assert 'features.js' in http.get(ROOT + '/features').get_data(as_text=True)


def test_page_scripts_are_never_cached(client):
    """A cached script means a browser keeps running the previous version of a
    page after an update, which looks exactly like the update not working."""
    http, _ = client
    for name in ('common.js', 'summary.js', 'features.js'):
        response = http.get(ROOT + '/' + name)
        assert response.status_code == 200
        assert 'no-store' in response.headers.get('Cache-Control', '')
    assert 'no-store' in http.get('/' + ANALYSIS + '/analysis.js').headers.get('Cache-Control', '')


def test_a_page_that_fails_to_start_says_so(client):
    """A script error renders nothing; without this the page is blank and the
    reason is only in the console."""
    http, _ = client
    for page in ('prep', 'depth', 'clusters', 'summary', 'features'):
        text = http.get(ROOT + '/' + page).get_data(as_text=True)
        assert 'could not start' in text, page


def test_two_analyses_are_served_side_by_side(tmp_path):
    """The point of the master page: more than one analysis at a time."""
    import pandas as pd
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir(); remote.mkdir()
    layout = Layout(local_root=local)
    cloud = FakeCloud(layout=layout, remote_dir=remote)
    for analysis, project, category in (('YH_MC_Parentals', 'MC_920_t001_tr1', 'MC'),
                                        ('TI_singles', 'TI_404_t077_tr1', 'TI')):
        build(remote, project_id=project, days=4, resets=(), H=32, W=40,
              analysis_id=analysis)
        states_dir = local / '__AnalysisStates' / analysis
        states_dir.mkdir(parents=True)
        pd.DataFrame({'projectID': [project], 'tankID': ['t'], 'Prep': [True],
                      'RunAnalysis': [True], 'Category': [category]}
                     ).to_csv(states_dir / (analysis + '.csv'), index=False)
        assert collect_project(layout, cloud, project, analysis).status == 'ok'

    A.configure(layout, cloud=cloud, upload=False)
    A.app.config['TESTING'] = True
    http = A.app.test_client()

    master = http.get('/').get_data(as_text=True)
    assert 'YH_MC_Parentals' in master and 'TI_singles' in master
    assert 'MC 1' in master and 'TI 1' in master       # categories present

    for analysis, project in (('YH_MC_Parentals', 'MC_920_t001_tr1'),
                              ('TI_singles', 'TI_404_t077_tr1')):
        page = http.get('/' + analysis + '/').get_data(as_text=True)
        assert project in page
        assert http.get('/%s/%s/summary' % (analysis, project)).status_code == 200

    # and one analysis cannot reach another's project
    assert http.get('/TI_singles/MC_920_t001_tr1/prep').status_code == 404


def _states_dir(local, name, with_csv=True):
    directory = local / '__AnalysisStates' / name
    directory.mkdir(parents=True)
    if with_csv:
        (directory / (name + '.csv')).write_text('projectID\n')
    return directory


def test_an_analysis_is_a_directory_with_a_states_file_in_it(tmp_path):
    """Everything under __AnalysisStates was taken for an analysis, so loose
    files and leftover directories showed up on the master page as analyses
    that do not exist."""
    from cichlid_bower_claude.states import local_analyses
    local = tmp_path / 'local'
    _states_dir(local, 'YH_MC_Parentals')
    _states_dir(local, 'TI_singles')
    _states_dir(local, '__DeletedData')             # bookkeeping
    _states_dir(local, 'LeftOver', with_csv=False)  # directory, no states file
    (local / '__AnalysisStates' / 'notes.csv').write_text('x\n')   # loose file
    assert local_analyses(Layout(local_root=local)) == ['TI_singles',
                                                        'YH_MC_Parentals']


def test_cloud_discovery_counts_only_names_with_their_own_states_file(tmp_path):
    """``analyses`` still reaches the cloud, and applies the same rule there:
    lsf lists files as well as directories, so a CSV lying beside the analyses
    came back looking like one."""
    from cichlid_bower_claude.states import list_analyses
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir()
    cloud_states = remote / '__AnalysisStates'
    for name in ('YH_MC_Parentals', 'TI_singles'):
        (cloud_states / name).mkdir(parents=True)
        (cloud_states / name / (name + '.csv')).write_text('projectID\n')
    (cloud_states / 'Scratch').mkdir()                       # no states file
    (cloud_states / 'Scratch' / 'something.csv').write_text('x\n')  # wrong name
    cloud_states.joinpath('notes.csv').write_text('x\n')     # loose file
    layout = Layout(local_root=local)
    cloud = FakeCloud(layout=layout, remote_dir=remote)
    assert list_analyses(layout, cloud) == ['TI_singles', 'YH_MC_Parentals']


def test_the_master_page_asks_the_cloud_nothing(client):
    """It used to cost an rclone listing plus a failed fetch per absent
    analysis, which is tens of seconds to render rows nobody can open."""
    http, layout = client
    cloud = A.STATE['cloud']
    before = len(cloud.downloaded)
    calls = []
    cloud.listdir = lambda *a, **k: calls.append(a) or []
    assert http.get('/').status_code == 200
    assert calls == []
    assert len(cloud.downloaded) == before


def test_a_landing_page_reparses_only_what_changed(client):
    """Both landing pages read every project in the analysis, so each one is
    cached against the mtimes of the files it reads."""
    http, layout = client
    assert http.get('/').status_code == 200
    assert http.get('/' + ANALYSIS + '/').status_code == 200
    paths = layout.project(PROJECT, ANALYSIS)
    reads = []
    real = C.load
    C.load = lambda p: reads.append(str(p.root)) or real(p)
    try:
        http.get('/')
        http.get('/' + ANALYSIS + '/')
        assert reads == []                       # nothing re-read
        prep = real(paths)
        C.save(paths, prep, who='tester')        # a save moves prep.json
        http.get('/')
        http.get('/' + ANALYSIS + '/')
        assert len(reads) == 2                   # both pages notice, once each
    finally:
        C.load = real


def test_a_save_waits_briefly_for_the_one_ahead_of_it(client, monkeypatch):
    """Two people saving within a second of each other is ordinary, and used
    to fail outright with 'someone else is saving this project'."""
    import threading
    import time as _time
    http, layout = client
    monkeypatch.setattr(A, 'LOCK_WAIT', 5.0)

    holder = A._lock_for(ANALYSIS, PROJECT)
    holder.acquire()
    A.HELD[ANALYSIS + '/' + PROJECT] = _time.time()

    def let_go():
        _time.sleep(0.4)
        A.HELD.pop(ANALYSIS + '/' + PROJECT, None)
        holder.release()

    threading.Thread(target=let_go).start()
    response = http.post(ROOT + '/save', json={'who': 'tester'})
    assert response.status_code == 200, response.get_json()


def test_a_lock_held_too_long_says_the_server_is_stuck(client, monkeypatch):
    """A wedged lock and a busy colleague are different problems, and the
    message has to tell them apart or the advice is useless."""
    import time as _time
    http, layout = client
    monkeypatch.setattr(A, 'LOCK_WAIT', 0.1)

    stuck = A._lock_for(ANALYSIS, PROJECT)
    stuck.acquire()
    A.HELD[ANALYSIS + '/' + PROJECT] = _time.time() - 600
    try:
        response = http.post(ROOT + '/save', json={'who': 'tester'})
        assert response.status_code == 409
        error = response.get_json()['error']
        assert 'stuck' in error and 'restart' in error
        assert 'do not reload' in error    # their edits are still on screen
    finally:
        A.HELD.pop(ANALYSIS + '/' + PROJECT, None)
        stuck.release()


def test_the_lock_is_given_back_when_a_save_fails(client):
    """Whatever goes wrong inside, the next save must still be possible."""
    http, layout = client
    assert http.post(ROOT + '/save', data='not json').status_code == 400
    assert http.post(ROOT + '/save', json={'trial_times': 'nonsense'}
                     ).status_code in (200, 500)
    # and the project is not left locked by either
    assert http.post(ROOT + '/save', json={'who': 'tester'}).status_code == 200
    assert A.HELD.get(ANALYSIS + '/' + PROJECT) is None


def test_the_workbook_downloads_with_three_sheets(client):
    http, layout = client
    response = http.get(ROOT + '/export.xlsx')
    assert response.status_code == 200, response.get_data(as_text=True)[:300]
    assert 'spreadsheet' in response.headers['Content-Type']

    import io
    import openpyxl
    book = openpyxl.load_workbook(io.BytesIO(response.data))
    assert book.sheetnames == ['Settings', 'Trials', 'Days']
    assert book['Trials'].max_row >= 2          # a header and at least one trial
    assert book['Days'].max_row >= 2


def test_the_workbook_uses_the_settings_it_was_given(client):
    """The point of passing the sliders through: the numbers in the file are
    the numbers that were on screen, and the first sheet says which."""
    import io
    import openpyxl
    http, layout = client

    def settings_of(query):
        response = http.get(ROOT + '/export.xlsx' + query)
        assert response.status_code == 200
        book = openpyxl.load_workbook(io.BytesIO(response.data))
        return {row[0]: row[1] for row in
                book['Settings'].iter_rows(values_only=True) if row[0]}

    loose = settings_of('?confidence=0.2&threshold=0.3')
    assert loose['Confidence threshold'] == 0.2
    assert loose['Sand moved threshold'] == 0.3

    strict = settings_of('?confidence=0.9&threshold=0.3')
    assert strict['Confidence threshold'] == 0.9

    # and the counts really followed the threshold, rather than the sheet
    # merely recording a number nothing used
    def events_of(query):
        response = http.get(ROOT + '/export.xlsx' + query)
        book = openpyxl.load_workbook(io.BytesIO(response.data))
        head = [c.value for c in book['Trials'][1]]
        column = head.index('events')
        return [row[column] for row in
                book['Trials'].iter_rows(min_row=2, values_only=True)]

    assert sum(events_of('?confidence=0.2')) >= sum(events_of('?confidence=0.95'))


def test_a_bad_setting_falls_back_rather_than_failing(client):
    """A typo in a query parameter should not cost someone their download."""
    import io
    import openpyxl
    http, layout = client
    response = http.get(ROOT + '/export.xlsx?confidence=banana')
    assert response.status_code == 200
    book = openpyxl.load_workbook(io.BytesIO(response.data))
    settings = {row[0]: row[1] for row in
                book['Settings'].iter_rows(values_only=True) if row[0]}
    assert settings['Confidence threshold'] == A.F.DEFAULTS.confidence


def test_the_trials_page_and_its_rows_are_served(client):
    http, layout = client
    page = http.get('/' + ANALYSIS + '/trials')
    assert page.status_code == 200
    assert 'trials.js' in page.get_data(as_text=True)
    assert http.get('/' + ANALYSIS + '/trials.js').status_code == 200

    rows = http.get('/' + ANALYSIS + '/trials.json').get_json()
    assert rows['analysisID'] == ANALYSIS
    assert rows['params']['confidence'] == A.F.DEFAULTS.confidence
    assert rows['rows'], rows
    first = rows['rows'][0]
    for key in ('projectID', 'category', 'trial', 'totalVolume', 'bowerIndex',
                'grossVolume', 'nightShare', 'events'):
        assert key in first, key
    assert first['projectID'] == PROJECT
    assert first['category'] == 'MC'


def test_an_excluded_trial_is_left_out_of_the_comparison(client):
    """Asked for explicitly: the comparison is of trials worth comparing."""
    http, layout = client
    paths = layout.project(PROJECT, ANALYSIS)
    before = len(http.get('/' + ANALYSIS + '/trials.json').get_json()['rows'])
    assert before >= 1

    prep = C.load(paths)
    number = prep and 1
    override = prep.overrides.setdefault('1', C.TrialOverride())
    override.excluded = True
    override.reason = 'test'
    C.save(paths, prep, who='tester')

    after = http.get('/' + ANALYSIS + '/trials.json').get_json()['rows']
    assert all(r['trial'] != 1 for r in after), after
    assert len(after) == before - 1


def test_a_project_that_cannot_be_read_is_named_not_dropped(client):
    """A silently shorter comparison is worse than a stated omission."""
    http, layout = client
    import pandas as pd
    states_csv = (layout.local_root / '__AnalysisStates' / ANALYSIS /
                  (ANALYSIS + '.csv'))
    table = pd.read_csv(states_csv)
    table.loc[len(table)] = {'projectID': 'MC_999_t099_tr1', 'tankID': 't099',
                             'Prep': True, 'RunAnalysis': True, 'Category': 'MC'}
    table.to_csv(states_csv, index=False)
    A.TABLES.clear()

    payload = http.get('/' + ANALYSIS + '/trials.json').get_json()
    named = [s['projectID'] for s in payload['skipped']]
    assert 'MC_999_t099_tr1' in named, payload['skipped']


def test_editing_a_category_shows_up_without_a_restart(client):
    """The row caches are keyed on what the rows read, and the rows carry the
    category -- which comes from the states file and nowhere else. Watching
    only the manifests and the corrections left an edited Category column
    invisible until an unrelated save moved a prep.json."""
    import time
    import pandas as pd
    http, layout = client

    assert 'MC' in http.get('/' + ANALYSIS + '/').get_data(as_text=True)
    assert 'MC' in http.get('/').get_data(as_text=True)

    states_csv = (layout.local_root / '__AnalysisStates' / ANALYSIS /
                  (ANALYSIS + '.csv'))
    table = pd.read_csv(states_csv)
    table.loc[table['projectID'] == PROJECT, 'Category'] = 'CVxMC_F1'
    time.sleep(0.01)                     # so the mtime really differs
    table.to_csv(states_csv, index=False)

    index = http.get('/' + ANALYSIS + '/').get_data(as_text=True)
    assert 'CVxMC_F1' in index, 'the per-analysis page did not notice'
    master = http.get('/').get_data(as_text=True)
    assert 'CVxMC_F1' in master, 'the master page did not notice'
