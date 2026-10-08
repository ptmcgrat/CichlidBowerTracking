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
