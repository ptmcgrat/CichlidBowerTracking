import json

import pytest

from cichlid_bower_claude.paths import Layout
from cichlid_bower_claude.server import corrections as C


@pytest.fixture
def paths(tmp_path):
    return Layout(local_root=tmp_path).project('MC_920_t001_tr1', 'YH_MC_Parentals')


def test_absent_file_gives_an_empty_set(paths):
    prep = C.load(paths)
    assert not prep.is_registered and not prep.is_complete
    assert prep.residual_k == C.DEFAULT_K


def test_round_trip(paths):
    prep = C.load(paths)
    prep.transform = [[1.9, .1, -140], [.05, 1.85, -95], [.0002, .0001, 1]]
    prep.depth_crop = [[95, 75], [555, 78], [558, 425], [92, 422]]
    prep.video_crop = [[150, 120], [1180, 130], [1190, 900], [140, 890]]
    prep.trials = {'1': C.TrialTimes(start=30, stop=15)}
    prep.residual_k = 4.5
    C.save(paths, prep, who='pmcgrath@gatech.edu', note='tray corners')

    back = C.load(paths)
    assert back.is_complete
    assert back.transform[0][0] == 1.9
    assert back.residual_k == 4.5
    assert back.who == 'pmcgrath@gatech.edu'
    assert back.updated


def test_trial_two_falls_back_to_trial_one(paths):
    prep = C.load(paths)
    prep.trials = {'1': C.TrialTimes(start=30)}
    C.save(paths, prep)
    back = C.load(paths)
    assert back.times_for(2).start == 30          # inherited
    back.trials['2'] = C.TrialTimes(start=60)
    assert back.times_for(2).start == 60          # overridden
    assert back.times_for(1).start == 30          # trial one untouched


def test_saving_keeps_the_version_it_replaces(paths):
    prep = C.load(paths)
    prep.residual_k = 3.0
    C.save(paths, prep)
    prep.residual_k = 5.0
    C.save(paths, prep)
    assert len(C.history(paths)) == 1
    with open(C.history(paths)[0]) as handle:
        assert json.load(handle)['residual_k'] == 3.0


def test_restore_brings_back_an_earlier_version(paths):
    prep = C.load(paths)
    prep.residual_k = 3.0
    C.save(paths, prep)
    prep.residual_k = 5.0
    C.save(paths, prep)
    stamp = C.history(paths)[0].name.split('_prep')[0]
    restored = C.restore(paths, stamp)
    assert restored.residual_k == 3.0
    assert C.load(paths).residual_k == 3.0


def test_a_foreign_schema_is_refused(paths):
    paths.corrections_dir.mkdir(parents=True)
    paths.prep_json.write_text(json.dumps({'schema': 'cichlid-prep/0'}))
    with pytest.raises(C.CorrectionsError):
        C.load(paths)


def test_nothing_is_written_outside_corrections(paths):
    prep = C.load(paths)
    prep.residual_k = 4.0
    C.save(paths, prep)
    written = {p.relative_to(paths.root).parts[0] for p in paths.root.rglob('*') if p.is_file()}
    assert written == {'Corrections'}
