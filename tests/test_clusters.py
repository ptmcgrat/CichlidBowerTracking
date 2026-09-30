import base64
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cichlid_bower_claude.collect import clusters as C


def make_table(n=200, start=dt.datetime(2025, 8, 8, 9)):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append({'TimeStamp': start + dt.timedelta(minutes=2 * i),
                     'X': rng.uniform(0, 972), 'Y': rng.uniform(0, 1296),
                     'Prediction': C.BIDS[i % len(C.BIDS)] if i % 13 else None,
                     'Probability': rng.uniform(0.2, 1.0),
                     'ClipCreated': 'No' if i % 13 == 0 else 'Yes'})
    return pd.DataFrame(rows).set_index('TimeStamp').sort_index()


def unpack(packed, key, dtype):
    return np.frombuffer(base64.b64decode(packed[key]), dtype=dtype)


def test_the_transposition_is_undone_once(tmp_path):
    """The file's X is the row index and Y the column; everything after this
    sees x across and y down."""
    table = make_table(10)
    days = [{'index': 0, 'trial': 1, 'first': str(table.index[0]),
             'last': str(table.index[-1])}]
    packed = C.pack(table, days)
    assert np.allclose(unpack(packed, 'x', np.uint16), table['Y'].to_numpy().astype(np.uint16))
    assert np.allclose(unpack(packed, 'y', np.uint16), table['X'].to_numpy().astype(np.uint16))


def test_events_are_assigned_to_days(tmp_path):
    table = make_table(60)
    middle = len(table) // 2
    days = [{'index': 0, 'trial': 1, 'first': str(table.index[0]),
             'last': str(table.index[middle - 1])},
            {'index': 1, 'trial': 1, 'first': str(table.index[middle]),
             'last': str(table.index[-1])}]
    packed = C.pack(table, days)
    day = unpack(packed, 'day', np.uint16)
    assert set(np.unique(day)) <= {0, 1}
    assert (day == 0).sum() == middle


def test_an_event_outside_every_day_is_kept_not_dropped(tmp_path):
    table = make_table(20)
    days = [{'index': 0, 'trial': 1, 'first': str(table.index[5]),
             'last': str(table.index[10])}]
    packed = C.pack(table, days)
    day = unpack(packed, 'day', np.uint16)
    assert (day == 65535).sum() == 14      # outside the one day, still counted
    assert packed['n'] == 20


def test_a_missing_prediction_is_its_own_code(tmp_path):
    table = make_table(30)
    days = [{'index': 0, 'trial': 1, 'first': str(table.index[0]),
             'last': str(table.index[-1])}]
    packed = C.pack(table, days)
    bid = unpack(packed, 'bid', np.uint8)
    assert (bid == C.NO_PREDICTION).sum() == table['Prediction'].isna().sum()


def test_clips_are_flagged(tmp_path):
    table = make_table(30)
    days = [{'index': 0, 'trial': 1, 'first': str(table.index[0]),
             'last': str(table.index[-1])}]
    packed = C.pack(table, days)
    flags = unpack(packed, 'flags', np.uint8)
    assert (flags & C.FLAG_CLIP).sum() // C.FLAG_CLIP == (table['ClipCreated'] == 'Yes').sum()


def test_packing_is_about_eleven_bytes_an_event(tmp_path):
    table = make_table(2000)
    days = [{'index': 0, 'trial': 1, 'first': str(table.index[0]),
             'last': str(table.index[-1])}]
    packed = C.pack(table, days)
    raw = sum(len(base64.b64decode(v)) for k, v in packed.items()
              if isinstance(v, str) and k not in ('n',))
    assert 10 <= raw / len(table) <= 13


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(C.ClustersMissing):
        C.read_clusters(tmp_path / 'nope.csv')


def test_summary_counts_without_unpacking(tmp_path):
    table = make_table(100)
    summary = C.summarise(table)
    assert summary['n'] == 100
    assert summary['noClip'] == (table['ClipCreated'] != 'Yes').sum()
    assert sum(summary['byBehaviour'].values()) + summary['noPrediction'] == 100
