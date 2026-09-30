import numpy as np
import pytest

from cichlid_bower_claude.server import encode as E


def test_values_survive_the_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    array = 60 + rng.normal(0, 0.5, (40, 52))
    meta = E.write_depth(tmp_path / 'd.png', array)
    back = E.decode_depth(E.read_png(tmp_path / 'd.png'), meta)
    assert np.nanmax(np.abs(back - array)) <= meta['scale'] / 2 + 1e-9


def test_missing_pixels_stay_missing(tmp_path):
    array = np.full((20, 24), 60.0)
    array[:3] = np.nan
    array[10, 10] = np.nan
    meta = E.write_depth(tmp_path / 'd.png', array)
    back = E.decode_depth(E.read_png(tmp_path / 'd.png'), meta)
    assert np.array_equal(np.isnan(back), np.isnan(array))


def test_a_no_return_pixel_does_not_ruin_the_rest(tmp_path):
    """One pixel thousands of cm away must not coarsen the whole frame."""
    rng = np.random.default_rng(1)
    array = 60 + rng.normal(0, 0.05, (60, 60))
    array[5, 5] = 4000.0
    array[7, 7] = -1500.0
    meta = E.write_depth(tmp_path / 'd.png', array, clip=(30.0, 90.0))
    back = E.decode_depth(E.read_png(tmp_path / 'd.png'), meta)
    inside = (array > 30) & (array < 90)
    assert meta['scale'] == E.DEFAULT_SCALE
    assert np.nanmax(np.abs(back[inside] - array[inside])) < 0.01
    assert meta['clipped'] == 2


def test_a_wide_span_coarsens_rather_than_failing(tmp_path):
    array = np.linspace(0, 5000, 40 * 40).reshape(40, 40)
    meta = E.write_depth(tmp_path / 'd.png', array, clip=(0.0, 5000.0))
    assert meta['scale'] > E.DEFAULT_SCALE
    back = E.decode_depth(E.read_png(tmp_path / 'd.png'), meta)
    assert np.nanmax(np.abs(back - array)) <= meta['scale']


def test_an_all_missing_frame_is_not_an_error(tmp_path):
    array = np.full((8, 8), np.nan)
    meta = E.write_depth(tmp_path / 'd.png', array)
    assert meta.get('empty')


def test_the_png_is_a_real_png(tmp_path):
    E.write_depth(tmp_path / 'd.png', np.full((4, 6), 60.0))
    assert (tmp_path / 'd.png').read_bytes()[:8] == b'\x89PNG\r\n\x1a\n'