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


def test_a_trial_falls_back_to_the_project_registration(paths):
    """One registration should serve a project; an override is the exception."""
    prep = C.load(paths)
    prep.transform = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    prep.depth_crop = [[0, 0], [10, 0], [10, 10], [0, 10]]
    C.save(paths, prep, who='pm')
    back = C.load(paths)
    assert back.transform_for(1) == back.transform
    assert back.transform_for(3) == back.transform


def test_a_trial_can_override_the_registration_and_crops(paths):
    prep = C.load(paths)
    prep.transform = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    prep.depth_crop = [[0, 0], [10, 0], [10, 10], [0, 10]]
    prep.overrides['2'] = C.TrialOverride(
        transform=[[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        depth_crop=[[5, 5], [20, 5], [20, 20], [5, 20]])
    C.save(paths, prep, who='pm')
    back = C.load(paths)
    assert back.transform_for(1)[0][0] == 1
    assert back.transform_for(2)[0][0] == 2
    assert back.depth_crop_for(2)[0] == [5, 5]
    assert back.depth_crop_for(3) == back.depth_crop


def test_a_trial_can_be_excluded(paths):
    prep = C.load(paths)
    prep.overrides['3'] = C.TrialOverride(excluded=True, reason='no building')
    C.save(paths, prep, who='pm')
    back = C.load(paths)
    assert back.is_excluded(3)
    assert not back.is_excluded(1)
    assert back.override(3).reason == 'no building'


def test_unknown_keys_in_an_override_are_ignored(paths):
    paths.corrections_dir.mkdir(parents=True, exist_ok=True)
    paths.prep_json.write_text(json.dumps(
        {'schema': C.SCHEMA, 'overrides': {'2': {'excluded': True, 'nonsense': 1}}}))
    assert C.load(paths).is_excluded(2)


def test_day_marks_round_trip(paths):
    """What a person noticed about a day, which the analysis cannot see."""
    prep = C.load(paths)
    prep.day_marks['7'] = C.DayMark(new_bower=True, note='moved to the left wall')
    prep.day_marks['9'] = C.DayMark(wall_building=True)
    C.save(paths, prep, who='pm')
    back = C.load(paths)
    assert back.mark(7).new_bower and back.mark(7).note
    assert back.mark(9).wall_building and not back.mark(9).new_bower
    assert not back.mark(3).new_bower        # an unmarked day is simply clear
    assert back.marked_days('new_bower') == [7]
    assert back.marked_days('wall_building') == [9]
