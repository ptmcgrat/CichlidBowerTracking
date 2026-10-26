import json

import numpy as np
import pytest

from cichlid_bower_claude.cloud import FakeCloud
from cichlid_bower_claude.collect.collector import collect_project
from cichlid_bower_claude.paths import Layout
from cichlid_bower_claude.server import corrections as C
from cichlid_bower_claude.server import features as F
from tests.make_project import build

ANALYSIS = 'YH_MC_Parentals'


@pytest.fixture
def collected(tmp_path):
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir(); remote.mkdir()
    project_id, _ = build(remote, days=6, resets=(2,), H=40, W=52)
    layout = Layout(local_root=local)
    cloud = FakeCloud(layout=layout, remote_dir=remote)
    assert collect_project(layout, cloud, project_id, ANALYSIS).status == 'ok'
    return layout.project(project_id, ANALYSIS)


def test_the_polygon_test_matches_known_areas():
    """A volume computed here and a map drawn in the browser have to agree
    about where the tray is."""
    rectangle = F._inside([[10, 10], [40, 10], [40, 30], [10, 30]], (60, 80))
    assert rectangle.sum() == 600
    triangle = F._inside([[10, 10], [50, 10], [30, 40]], (60, 80))
    assert abs(int(triangle.sum()) - 600) <= 20


def test_volumes_and_the_bower_index():
    change = np.zeros((240, 320))
    change[50:70, 50:70] = 3.0          # 400 px raised 3 cm
    change[150:160, 150:170] = -2.0     # 200 px lowered 2 cm
    out = F._volumes(change, 0.6)
    area = F.PIXEL_CM ** 2
    # the stored values are rounded to two decimals for the wire
    assert out['castleVolume'] == pytest.approx(400 * 3.0 * area, abs=0.01)
    assert out['pitVolume'] == pytest.approx(200 * 2.0 * area, abs=0.01)
    assert out['bowerIndex'] == pytest.approx(0.5, abs=1e-4)


def test_sub_threshold_movement_counts_as_neither():
    out = F._volumes(np.full((50, 50), 0.3), 0.6)
    assert out['totalVolume'] == 0 and out['bowerIndex'] == 0


def test_a_pure_pit_and_a_pure_castle():
    pit = np.zeros((60, 60)); pit[10:30, 10:30] = -3.0
    castle = np.zeros((60, 60)); castle[10:30, 10:30] = 3.0
    assert F._volumes(pit, 0.6)['bowerIndex'] == pytest.approx(-1.0)
    assert F._volumes(castle, 0.6)['bowerIndex'] == pytest.approx(1.0)


def test_the_ellipse_area_scales_with_the_spread():
    rng = np.random.default_rng(0)
    tight = rng.normal(0, 5, (3000, 2))
    wide = rng.normal(0, 10, (3000, 2))
    a = F._dispersion(tight)['area']
    b = F._dispersion(wide)['area']
    assert b / a == pytest.approx(4.0, rel=0.08)


def test_compute_produces_a_trial_per_trial(collected):
    metrics = F.compute(collected)
    assert metrics['schema'] == F.SCHEMA
    assert metrics['trials']
    for trial in metrics['trials']:
        assert 'trial' in trial and 'days' in trial


def test_an_excluded_trial_carries_no_metrics(collected):
    prep = C.load(collected)
    prep.overrides['1'] = C.TrialOverride(excluded=True, reason='no building')
    C.save(collected, prep, who='pm')
    metrics = F.compute(collected)
    first = [t for t in metrics['trials'] if t['trial'] == 1][0]
    assert first['excluded'] and first['reason'] == 'no building'
    assert 'bowerIndex' not in first        # nothing computed for it


def test_the_cache_goes_stale_when_the_corrections_change(collected):
    F.load(collected)
    assert not F.is_stale(collected)
    prep = C.load(collected)
    prep.residual_k = 4.5
    C.save(collected, prep, who='pm')       # a correction, not a recollection
    assert F.is_stale(collected)


def test_the_spawn_histogram_keeps_the_median():
    """The page pools these across a category, so the shape has to survive."""
    rng = np.random.default_rng(1)
    depths = rng.normal(1.6, 0.9, 5000)
    edges = np.linspace(-F.DEPTH_SPAN, F.DEPTH_SPAN, F.DEPTH_BINS + 1)
    counts, _ = np.histogram(np.clip(depths, -F.DEPTH_SPAN, F.DEPTH_SPAN), bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    recovered = centres[np.searchsorted(np.cumsum(counts), counts.sum() / 2)]
    assert abs(recovered - np.median(depths)) < (2 * F.DEPTH_SPAN / F.DEPTH_BINS)


def test_a_castle_is_reported_as_a_castle(collected):
    """Depth is distance down to the sand, so building makes the number
    smaller and castle-positive change is start minus end.

    This module had it the other way round, so every index it reported was
    sign-flipped: the fixture below builds a castle and it came back as a pit
    at -0.85, 'spawnOverCastle' counted spawns over pits, and the
    cross-project spawn-depth axis was mirrored. The browser was right, which
    is exactly why nobody noticed from looking at a page.
    """
    from cichlid_bower_claude.collect import bundle as B

    paths = collected

    # make_project writes surface = 60 - 3*frac*castle: the depth value falls
    # where sand accumulates, so this fixture builds a castle
    data = B.load_bundle(paths.bundle)
    centre = tuple(d // 2 for d in data['first'][0].shape)
    fell = float(data['first'][0][centre]) - float(data['last'][-1][centre])
    assert fell > 0, 'fixture no longer builds up at the centre'

    trial = F.compute(paths)['trials'][0]
    assert trial['bowerIndex'] > 0, trial
    assert trial['castleVolume'] > trial['pitVolume'], trial


def test_the_null_aspect_table_matches_the_browser_copy():
    """The same table is carried in common.js. Two copies of a lookup are two
    chances to be different, so they are checked against each other."""
    import json
    import re
    import subprocess
    from pathlib import Path

    common = (Path(__file__).resolve().parents[1] / 'cichlid_bower_claude' /
              'server' / 'templates' / 'common.js').read_text()
    rows = re.findall(r'\[(\d+), (1\.\d+)\]', common)
    assert rows, 'no NULL_ASPECT table found in common.js'
    from_js = [(int(n), float(a)) for n, a in rows]
    assert from_js == [(n, a) for n, a in F.NULL_ASPECT], (from_js, F.NULL_ASPECT)


def test_a_cache_from_an_older_schema_is_rebuilt(collected):
    """A fix to this module changes the numbers without touching any data
    file, so mtimes alone would call the old cache current.

    That is not hypothetical: when castle and pit were swapped here, every
    cached features.json held flipped values and every mtime said they were
    fine. Bumping SCHEMA is how a correction gets announced, so staleness has
    to read it.
    """
    import json

    metrics = F.load(collected)
    assert metrics and metrics['schema'] == F.SCHEMA
    assert not F.is_stale(collected)

    cached = F.path_for(collected)
    stale = dict(metrics, schema='cichlid-features/1')
    stale['trials'] = [dict(t, bowerIndex=-999) for t in stale['trials']]
    with open(cached, 'w') as handle:
        json.dump(stale, handle)

    assert F.is_stale(collected), 'an older schema was taken for current'
    fresh = F.load(collected)
    assert fresh['schema'] == F.SCHEMA
    assert all(t.get('bowerIndex') != -999 for t in fresh['trials'])


def test_an_unreadable_cache_is_stale_not_current(collected):
    F.load(collected)
    F.path_for(collected).write_text('{ truncated')
    assert F.is_stale(collected)
    assert F.load(collected)['schema'] == F.SCHEMA


def test_a_cache_older_than_this_module_is_rebuilt(collected):
    """The failure the schema check was supposed to prevent and did not.

    Bumping SCHEMA is something a person has to remember, and the change
    right after it -- adding the daily-mean fields -- did not get one. Every
    cache was served as current while missing the new keys, and nine panels
    on the phenotypes page drew nothing. Comparing against this module's own
    mtime needs nobody to remember anything.
    """
    import os
    import time
    from pathlib import Path

    metrics = F.load(collected)
    assert metrics and not F.is_stale(collected)

    # the module edited after the cache was written, schema unchanged
    module = Path(F.__file__)
    later = time.time() + 5
    os.utime(module, (later, later))
    try:
        assert F.is_stale(collected), 'a cache older than the code was current'
    finally:
        os.utime(module, None)


def test_every_metric_the_phenotypes_page_plots_is_computed(collected):
    """A field the page asks for and compute() never produces draws an empty
    panel, which looks like no data rather than like a missing key."""
    from cichlid_bower_claude.server import app as A

    metrics = F.compute(collected)
    trial = next(t for t in metrics['trials'] if not t.get('excluded'))
    row = A._trial_row(metrics, trial, 'MC', 't001', 'MC_920_t001_tr1')

    for _, key in A.CSV_COLUMNS:
        if key == 'sampleID':
            continue          # read from the logfile by the CSV route itself
        assert key in row, key

    # These come from the trial's endpoints, so they need no registration, no
    # events and no day-to-day movement.
    for key in ('totalVolume', 'volumePerDay', 'pitVolume', 'castleVolume',
                'bowerIndex'):
        assert row[key] is not None, key

    # The rest legitimately can be None, and on this fixture are: night share
    # and the half indices all come from per-DAY volumes, and no single day of
    # a gradual six-day build moves 0.6 cm of sand anywhere once the residual
    # mask has excluded the churning patch -- which was the only thing moving
    # that far in a day. Requiring them here would be requiring the mask not
    # to work.
    for key in ('nightShare', 'firstHalfIndex', 'secondHalfIndex'):
        assert key in row, key


def _js_score_mask(score, k):
    """common.js's scoreMask, on the same array, through node."""
    import json
    import shutil
    import subprocess
    from pathlib import Path

    import numpy as np
    import pytest as _pytest

    if not shutil.which('node'):
        _pytest.skip('needs node')
    root = Path(__file__).resolve().parents[1]
    flat = np.asarray(score, dtype=np.float64).ravel()
    payload = {'values': [0.0 if not np.isfinite(v) else float(v) for v in flat],
               'nan': [i for i, v in enumerate(flat) if not np.isfinite(v)]}
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as handle:
        json.dump(payload, handle)
        where = handle.name
    result = subprocess.run(
        ['node', str(root / 'tests' / 'score_mask.js'), where, str(k)],
        cwd=str(root), capture_output=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_the_residual_mask_matches_the_browser_copy():
    """The mask decides which pixels every volume is measured on, and it is
    implemented twice. A difference between the two means a volume on screen
    and the same volume in the spreadsheet disagree invisibly -- which is how
    castle and pit stayed swapped on the server for weeks.
    """
    import numpy as np

    rng = np.random.default_rng(7)
    score = np.abs(rng.lognormal(0.0, 0.45, (40, 50))).astype(np.float64)
    score[6:9, 10:14] = 55.0          # a thrashing patch
    score[0, 0] = np.nan
    score[1, 1] = 0.0                 # zero is not a score, and is skipped

    for k in (2.0, 5.0, 8.0):
        mine = F.score_mask(score, k)
        theirs = _js_score_mask(score, k)
        assert theirs['marked'] is not None
        assert mine.ravel().astype(int).tolist() == theirs['marked'], k


def test_a_new_bower_mark_ends_the_trial_for_analysis(collected):
    """Asked for: from a new bower the cumulative map is two overlapping
    structures, so its volume and index describe neither."""
    paths = collected
    whole = F.compute(paths)['trials'][0]
    assert whole['days'] == whole['daysLogged']
    assert 'truncatedAtNewBower' not in whole

    prep = C.load(paths)
    days = F.B.read_manifest(paths.manifest)['days']
    in_trial = [d for d in days if d['trial'] == whole['trial']]
    assert len(in_trial) >= 3
    cut_at = in_trial[2]['index']
    prep.day_marks[str(cut_at)] = C.DayMark(new_bower=True)
    C.save(paths, prep, who='tester')

    after = F.compute(paths)['trials'][0]
    assert after['days'] == 2, after
    assert after['daysLogged'] == whole['daysLogged']
    assert after['truncatedAtNewBower']['at'] == cut_at
    assert after['truncatedAtNewBower']['droppedDays'] == len(in_trial) - 2
    assert after['to'] == in_trial[1]['date']


def test_a_new_bower_on_the_first_day_leaves_the_trial_whole(collected):
    """There would be nothing left to analyse. Reported rather than silently
    dropping the trial, which would be a worse answer than an odd one."""
    paths = collected
    whole = F.compute(paths)['trials'][0]
    days = F.B.read_manifest(paths.manifest)['days']
    first = [d for d in days if d['trial'] == whole['trial']][0]

    prep = C.load(paths)
    prep.day_marks[str(first['index'])] = C.DayMark(new_bower=True)
    C.save(paths, prep, who='tester')

    after = F.compute(paths)['trials'][0]
    assert after['days'] == whole['days']
    assert after['truncatedAtNewBower']['at'] is None
    assert 'first day' in after['truncatedAtNewBower']['why']


def test_the_residual_mask_reaches_the_volumes(collected):
    """It was applied on the depth tab and nowhere else, so every volume here
    was measured partly on pixels that tab had already called unreliable."""
    paths = collected

    prep = C.load(paths)
    prep.residual_k = 50.0           # so loose nothing is masked
    C.save(paths, prep, who='tester')
    loose = F.compute(paths)['trials'][0]

    prep = C.load(paths)
    prep.residual_k = 0.5            # so tight much is masked
    C.save(paths, prep, who='tester')
    tight = F.compute(paths)['trials'][0]

    assert loose['maskedPixels'] < tight['maskedPixels'], (loose, tight)
    assert tight['totalVolume'] < loose['totalVolume'], (loose, tight)
    assert tight['residualK'] == 0.5 and loose['residualK'] == 50.0
