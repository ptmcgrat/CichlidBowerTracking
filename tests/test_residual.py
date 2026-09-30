import numpy as np

from cichlid_bower_claude.collect import residual as R


def _line(frames=40, shape=(12, 14), slope=-0.05):
    t = np.arange(frames, dtype=float)[:, None, None]
    return 60 + slope * t + np.zeros((frames,) + shape)


def test_an_exact_line_has_no_residual():
    assert np.nanmax(R.residual(_line())) < 1e-9


def test_residual_recovers_the_scatter():
    rng = np.random.default_rng(0)
    day = _line() + rng.normal(0, 0.10, (40, 12, 14))
    assert 0.08 < float(np.nanmean(R.residual(day))) < 0.12


def test_a_builder_scores_far_below_a_churner():
    rng = np.random.default_rng(1)
    build = _line(slope=-0.075) + rng.normal(0, 0.01, (40, 12, 14))
    churn = 60 + rng.normal(0, 0.8, (40, 12, 14))
    assert np.nanmean(R.residual(build)) * 10 < np.nanmean(R.residual(churn))


def test_a_sparse_pixel_is_refused_rather_than_fitted():
    rng = np.random.default_rng(2)
    day = _line() + rng.normal(0, 0.05, (40, 12, 14))
    day[:30, 0, 0] = np.nan          # 25 per cent valid
    day[:10, 1, 1] = np.nan          # 75 per cent valid
    out = R.residual(day)
    assert np.isnan(out[0, 0]) and np.isfinite(out[1, 1])


def test_residual_is_stable_with_length_where_travel_is_not():
    rng = np.random.default_rng(3)
    day = _line(frames=40) + rng.normal(0, 0.1, (40, 12, 14))
    short, long = R.residual(day[:20]), R.residual(day)
    assert abs(np.nanmean(short) - np.nanmean(long)) < 0.02
    assert np.nanmean(R.travel(day)) > 1.7 * np.nanmean(R.travel(day[:20]))


def test_theil_sen_recovers_a_known_slope():
    assert abs(float(R.theil_sen(_line(slope=-0.05))[0, 0]) + 1.95) < 0.01


def test_project_score_keeps_a_pixel_bad_on_a_minority_of_days():
    rng = np.random.default_rng(4)
    maps = []
    for d in range(24):
        m = np.abs(rng.normal(0, 0.010 + 0.006 * d / 24, (60, 80)))
        m[20:24, 30:34] += 0.8                       # every day
        if d in (5, 11):
            m[40:44, 10:14] += 0.8                   # two days of twenty-four
        maps.append(m)
    score = R.project_score(maps)
    cut, info = R.threshold(score, k=4)
    assert score[21, 31] > cut
    assert score[41, 11] < cut
    assert info['n'] == 60 * 80
