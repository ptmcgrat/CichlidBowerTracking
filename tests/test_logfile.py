import datetime as dt
from pathlib import Path

from cichlid_bower_claude.logfile.days import Offsets, days_for, nights_for
from cichlid_bower_claude.logfile.parse import parse_logfile, parse_time
from tests.make_log import write_log


def _log(tmp_path, **kwargs):
    write_log(tmp_path / 'Logfile.txt', **kwargs)
    return parse_logfile(tmp_path / 'Logfile.txt')


def test_parses_header_and_counts(tmp_path):
    log = _log(tmp_path, days=6, resets=(2, 4))
    assert log.project_id == 'MC_920_t001_tr1'
    assert log.tank_id == 't001'
    assert (log.width, log.height) == (1296, 972)
    assert len(log.frames) == 1728
    assert len(log.movies) == 6
    assert not log.issues


def test_frame_index_comes_from_the_filename(tmp_path):
    log = _log(tmp_path, days=2, resets=())
    assert log.frames[0].index == 0
    assert log.frames[0].std_file.endswith('Frame_std_000001.npy')


def test_trials_split_at_resets(tmp_path):
    log = _log(tmp_path, days=6, resets=(2, 4))
    assert len(log.trials) == 2
    # a trial's reset_time is the reset that follows it, which is the next start
    assert log.trials[0].reset_time == log.trials[1].start_time


def test_offsets_move_only_the_outer_days(tmp_path):
    log = _log(tmp_path, days=6, resets=(2, 4))
    trial = log.trials[0]
    plain = days_for(log, trial, Offsets())
    shifted = days_for(log, trial, Offsets(start=30, stop=30))
    assert plain[0].first.time + dt.timedelta(minutes=30) == shifted[0].first.time
    assert plain[-1].last.time - dt.timedelta(minutes=30) == shifted[-1].last.time
    # an intermediate day begins at a lights-on boundary, which no offset touches
    assert plain[1].first.time == shifted[1].first.time


def test_a_day_too_short_after_clipping_is_dropped(tmp_path):
    log = _log(tmp_path, days=6, resets=(2, 4))
    trial = log.trials[0]
    assert len(days_for(log, trial, Offsets(start=60, stop=60))) < \
           len(days_for(log, trial, Offsets()))


def test_nights_do_not_span_trials(tmp_path):
    from cichlid_bower_claude.logfile.days import all_days
    log = _log(tmp_path, days=6, resets=(2, 4))
    days = all_days(log)
    assert all(a.trial == b.trial for a, b in nights_for(days))


def test_bad_lines_are_recorded_not_raised(tmp_path):
    path = tmp_path / 'Logfile.txt'
    write_log(path, days=2, resets=())
    text = path.read_text().splitlines()
    text.insert(5, 'FrameCaptured: NpyFile: ,,Time: not-a-time')
    path.write_text('\n'.join(text))
    log = parse_logfile(path)
    assert log.issues and len(log.frames) > 0


def test_parse_time_accepts_the_formats_in_use():
    assert parse_time('2025-01-31 14:25:00') is not None
    assert parse_time('2025-01-31 14:25:00.123456') is not None
    assert parse_time('nonsense') is None
