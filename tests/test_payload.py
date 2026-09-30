import json

import numpy as np
import pytest

from cichlid_bower_claude.cloud import FakeCloud
from cichlid_bower_claude.collect.collector import collect_project
from cichlid_bower_claude.paths import Layout
from cichlid_bower_claude.server import payload as P
from cichlid_bower_claude.server.encode import decode_depth, read_png
from tests.make_project import build

ANALYSIS = 'YH_MC_Parentals'


@pytest.fixture
def collected(tmp_path):
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    local.mkdir(); remote.mkdir()
    project_id, _ = build(remote, days=5, resets=(2,), H=32, W=40)
    layout = Layout(local_root=local)
    cloud = FakeCloud(layout=layout, remote_dir=remote)
    result = collect_project(layout, cloud, project_id, ANALYSIS)
    assert result.status == 'ok'
    return layout.project(project_id, ANALYSIS)


def test_builds_assets_and_a_payload(collected):
    payload = P.build_prep_payload(collected)
    assert payload['schema'] == P.SCHEMA
    assert payload['days'] and payload['trials']
    assert (collected.pages_dir / 'page.json').exists()
    assert (collected.pages_dir / 'assets').is_dir()


def test_the_payload_is_small_and_the_images_are_separate(collected):
    P.build_prep_payload(collected)
    sizes = P.payload_size(collected.pages_dir)
    assert sizes['files'] > 0
    assert sizes['payloadBytes'] < sizes['assetBytes']


def test_depth_values_survive_into_the_assets(collected):
    payload = P.build_prep_payload(collected)
    day = payload['days'][0]
    meta = day['firstPng']
    values = decode_depth(read_png(collected.pages_dir / meta['url']), meta)
    from cichlid_bower_claude.collect import bundle as B
    original = B.load_bundle(collected.bundle)['first'][0]
    inside = np.isfinite(original) & np.isfinite(values)
    assert inside.any()
    assert np.max(np.abs(values[inside] - original[inside])) < 0.05


def test_residual_statistics_are_full_resolution(collected):
    payload = P.build_prep_payload(collected)
    stats = payload['days'][0]['residualStats']
    assert stats['n'] > 0 and stats['median'] > 0
    assert stats['madFactor'] >= 1.0


def test_an_uncollected_project_is_refused(tmp_path):
    layout = Layout(local_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        P.build_prep_payload(layout.project('nope', ANALYSIS))


def test_rebuilding_does_not_accumulate_stale_assets(collected):
    P.build_prep_payload(collected)
    (collected.pages_dir / 'assets' / 'leftover.png').write_bytes(b'x')
    P.build_prep_payload(collected)
    assert not (collected.pages_dir / 'assets' / 'leftover.png').exists()


def test_current_corrections_ride_along(collected):
    from cichlid_bower_claude.server import corrections as C
    prep = C.load(collected)
    prep.residual_k = 4.5
    prep.depth_crop = [[5, 5], [30, 5], [30, 25], [5, 25]]
    C.save(collected, prep, who='pm@gatech.edu')
    payload = P.build_prep_payload(collected)
    assert payload['prep']['residual_k'] == 4.5
    assert payload['prep']['who'] == 'pm@gatech.edu'
