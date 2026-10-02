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


def test_unclustered_transitions_are_flagged_not_dropped(tmp_path):
    """LID of -1 never joined a cluster: not an event, but worth counting."""
    table = make_table(40)
    table['LID'] = [-1 if i % 5 == 0 else i for i in range(len(table))]
    days = [{'index': 0, 'trial': 1, 'first': str(table.index[0]),
             'last': str(table.index[-1])}]
    packed = C.pack(table, days)
    flags = unpack(packed, 'flags', np.uint8)
    clustered = (flags & C.FLAG_CLUSTERED) != 0
    assert packed['n'] == 40                       # nothing dropped
    assert clustered.sum() == 32                   # 8 of 40 are unclustered


def test_a_file_without_LID_treats_everything_as_clustered(tmp_path):
    table = make_table(20)
    days = [{'index': 0, 'trial': 1, 'first': str(table.index[0]),
             'last': str(table.index[-1])}]
    flags = unpack(C.pack(table, days), 'flags', np.uint8)
    assert ((flags & C.FLAG_CLUSTERED) != 0).all()


def test_the_summary_counts_unclustered_transitions(tmp_path):
    table = make_table(50)
    table['LID'] = [-1 if i % 10 == 0 else i for i in range(len(table))]
    summary = C.summarise(table)
    assert summary['unclustered'] == 5
    assert summary['clustered'] == 45


def test_unclustered_transitions_are_flagged_not_dropped(tmp_path):
    """LID -1 never joined a cluster: not an event, but a measure of the
    clustering, so it is kept and marked."""
    table = make_table(60)
    table['LID'] = [-1 if i % 5 == 0 else i for i in range(len(table))]
    days = [{'index': 0, 'trial': 1, 'first': str(table.index[0]),
             'last': str(table.index[-1])}]
    packed = C.pack(table, days)
    flags = unpack(packed, 'flags', np.uint8)
    clustered = (flags & C.FLAG_CLUSTERED) != 0
    assert packed['n'] == 60
    assert (~clustered).sum() == 12


def test_the_clustered_fraction_is_reported(tmp_path):
    table = make_table(50)
    table['LID'] = [-1 if i % 10 == 0 else i for i in range(len(table))]
    summary = C.summarise(table)
    assert summary['unclustered'] == 5
    assert abs(summary['clusteredFraction'] - 0.9) < 1e-9


def test_a_file_with_no_LID_treats_everything_as_clustered(tmp_path):
    table = make_table(20)
    days = [{'index': 0, 'trial': 1, 'first': str(table.index[0]),
             'last': str(table.index[-1])}]
    packed = C.pack(table, days)
    flags = unpack(packed, 'flags', np.uint8)
    assert ((flags & C.FLAG_CLUSTERED) != 0).all()


def test_values_beyond_float16_are_clipped_not_turned_into_infinity(tmp_path):
    """float16 tops out at 65504 and silently overflows to inf, which then
    propagates through every reduction downstream."""
    import numpy as np
    from cichlid_bower_claude.collect import bundle as B
    arrays = {'first': np.full((2, 4, 4), 60.0)}
    arrays['first'][0, 0, 0] = 90000.0          # a no-return pixel
    arrays['first'][0, 1, 1] = np.nan
    size, clipped = B.write_bundle(tmp_path / 'b.npz', arrays)
    assert clipped == 1
    back = B.load_bundle(tmp_path / 'b.npz')['first']
    assert np.isfinite(back[0, 0, 0])
    assert back[0, 0, 0] == B.FLOAT16_MAX
    assert np.isnan(back[0, 1, 1])
    assert not np.isinf(back).any()


def test_an_ordinary_bundle_reports_nothing_clipped(tmp_path):
    import numpy as np
    from cichlid_bower_claude.collect import bundle as B
    size, clipped = B.write_bundle(tmp_path / 'b.npz',
                                   {'first': np.full((2, 4, 4), 60.0)})
    assert clipped == 0