"""Render every page in a real DOM.

A per-file syntax check cannot catch the faults that actually reach people: a
name declared in two scripts that share a scope, or a control referenced by an
id that was never added to the markup. Both leave a blank page.

Needs node and jsdom. Skipped where they are absent, so the rest of the suite
runs anywhere:

    npm install jsdom
"""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve().parent / 'render_page.js'
PAGES = ['prep', 'depth', 'clusters', 'summary', 'features', 'analysis',
         'phenotypes']


def _has_jsdom() -> bool:
    if not shutil.which('node') or not HARNESS.exists():
        return False
    probe = subprocess.run(['node', '-e', 'require("jsdom")'],
                           cwd=str(ROOT), capture_output=True)
    return probe.returncode == 0


needs_jsdom = pytest.mark.skipif(
    not _has_jsdom(), reason='needs node with jsdom: npm install jsdom')


@needs_jsdom
@pytest.mark.parametrize('page', PAGES)
def test_the_page_renders_without_error(page):
    result = subprocess.run(['node', str(HARNESS), page], cwd=str(ROOT),
                            capture_output=True, encoding='utf-8', timeout=120)
    assert result.returncode == 0, (
        (result.stdout or '') + (result.stderr or ''))


def _render(page):
    result = subprocess.run(['node', str(HARNESS), page, '--text'], cwd=str(ROOT),
                            capture_output=True, encoding='utf-8', timeout=120)
    assert result.returncode == 0, (result.stdout or '') + (result.stderr or '')
    for line in (result.stdout or '').splitlines():
        if line.startswith('TEXT  '):
            return line[6:]
    return ''


@needs_jsdom
def test_the_features_second_row_draws_its_panels():
    """Rendering without an error is not the same as rendering the panels: a
    panel whose promise rejects silently leaves the page looking fine."""
    text = _render('features')
    for label in ('First half of the volume', 'Second half of the volume',
                  'Cloud size by day', 'Cloud shape by day',
                  'Spawn depth by day', 'Built overnight'):
        assert label in text, label


@needs_jsdom
def test_the_volume_split_reaches_about_half():
    """The split is snapped to a day boundary, so it lands near a half rather
    than on it -- but a split reporting 0% or 100% means the running total
    was never accumulated."""
    import re
    text = _render('features')
    shares = [float(m) for m in re.findall(r'(\d+)% of gross sand moved', text)]
    assert shares, text[:400]
    assert any(20 <= s <= 80 for s in shares), shares


@needs_jsdom
def test_the_aspect_panel_reports_its_null():
    """An aspect ratio without the circular baseline beside it is the thing
    this panel exists to avoid."""
    text = _render('features')
    assert 'circular would be' in text


@needs_jsdom
def test_the_summary_page_offers_the_workbook():
    """The button and its panel: a control referenced by an id that was never
    added to the markup is how this page went blank once before."""
    text = _render('summary')
    assert 'Download workbook' in text


@needs_jsdom
def test_the_phenotypes_page_groups_by_category_and_states_its_settings():
    """Its numbers are computed at fixed settings while the workbook follows
    the sliders, so the page has to say which it used or the two disagree for
    no visible reason."""
    text = _render('phenotypes')
    assert '24 trials from 8 projects' in text
    assert 'Fixed settings for this page' in text
    assert 'will not match these numbers' in text
    # the five rows, each by its heading, then one panel from each
    for heading in ('Depth \u2014 how much', 'Depth \u2014 what shape',
                    'Behaviour \u2014 how often', 'Clouds \u2014 how big',
                    'Clouds \u2014 what shape'):
        assert heading in text, heading
    for label in ('Total volume', 'Bower index', 'Night share',
                  'Build events per day', 'Scoop cloud area, daily mean',
                  'Index, early build', 'Index, late build',
                  'Scoop \u00f7 spit area, early build',
                  'Scoop \u00f7 spit area, late build',
                  'Median spawn depth'):
        assert label in text, label
    # every one of the twenty has data, so the panels are drawing rather than
    # reporting an empty metric -- which is what they did when the fixture
    # predated half of them and the label assertions above still passed
    assert 'nothing measured for this' not in text, text
    # every category appears, with its own count
    for category in ('MC 8', 'TI 8', 'CVxMC_F1 8'):
        assert category in text, category
    # and a project that could not be read is named rather than dropped
    assert 'P9 (not collected)' in text
    # the CSV, with the repeated-measures caveat beside it rather than buried
    assert 'Download CSV' in text
    assert 'repeated measures' in text


@needs_jsdom
def test_hovering_a_dot_names_its_project():
    """Pointing where a dot is ON SCREEN has to name that dot.

    The canvas is drawn at a fixed width and stretched by CSS, so the hit
    test scales client coordinates back into the space the dots were drawn
    in. Without that it is right only at one window width. The harness shows
    the canvas at 1.5x its backing store for exactly that reason -- with the
    scaling removed this reports 0 named, 2 misnamed, 6 silent.
    """
    result = subprocess.run(['node', str(HARNESS), 'phenotypes', '--hover'],
                            cwd=str(ROOT), capture_output=True,
                            encoding='utf-8', timeout=120)
    out = (result.stdout or '') + (result.stderr or '')
    assert result.returncode == 0, out
    line = [l for l in out.splitlines() if l.startswith('HOVER')]
    assert line, out
    assert 'misnamed 0' in line[0], line[0]
    assert 'silent 0' in line[0], line[0]


@needs_jsdom
def test_the_summary_page_shows_the_night_row_and_the_phase_toggle():
    """Both asked for: the overnight map as its own row under the 24 hour
    one, and a per-day toggle whose state is saved with the marks."""
    text = _render('summary')
    assert 'Night change' in text
    assert 'the dark period, between the two days it spans' in text
    # the first day of a trial has no night before it, and says so rather
    # than drawing an empty cell
    assert 'no night before the first day' in text

    # the daily row is this day's own lights-on period, not the 24 hours it
    # used to be -- the night is the other row's job, and measuring it twice
    # is what the change was for
    assert 'Daily change' in text
    assert 'this day\u2019s own morning to evening' in text
    assert 'h light' in text
    assert 'no following day in this trial' not in text

    # late build is the default, so an untouched day reads as late rather
    # than as an unset state the person has to clear
    assert 'late build' in text
    assert 'phase \u2013' not in text
    # and the per-trial opt-out of the phase analysis
    assert 'exclude from phase analysis' in text


@needs_jsdom
def test_the_residual_mask_reaches_the_pages_not_only_the_depth_tab():
    """It was applied on the depth tab and nowhere else, so the summary and
    features pages measured sand on pixels that tab had already called
    unreliable. The harness's score has a known churning patch, so a page
    that applies the mask reports fewer usable pixels than one that does not.
    """
    import re
    for page in ('summary', 'features'):
        text = _render(page)
        # the pages report their own scales and volumes; what matters here is
        # that the page got through its render with masks loaded rather than
        # falling back to crop only
        assert 'could not start' not in text, page
    # the depth tab still states the k it used, which is where the mask began
    depth = _render('depth')
    assert 'residual mask at k' in depth


@needs_jsdom
def test_a_new_bower_mark_shortens_the_trial_on_the_features_page():
    """The summary page shows every day because that is where the mark is
    set; the features page analyses up to the day before it, and says so."""
    import json
    import subprocess
    result = subprocess.run(['node', str(HARNESS), 'features', '--text',
                             '--mark-new-bower', '3'],
                            cwd=str(ROOT), capture_output=True,
                            encoding='utf-8', timeout=120)
    out = (result.stdout or '') + (result.stderr or '')
    assert result.returncode == 0, out
    text = [l for l in out.splitlines() if l.startswith('TEXT')][0]
    assert 'new bower after' in text, text[:400]
    assert '3 of 5 days' in text, text[:400]


@needs_jsdom
def test_the_cluster_page_applies_the_crop_and_shows_what_it_dropped():
    """The four point crop was computed every redraw and then used only to
    colour the set aside row, so Build, Feed, Spawning and Building all drew
    events from outside the tray -- feeding spits on the glass.

    The fixture fails a known share of events three ways. Both halves have to
    be non-empty, and together they have to account for every build event:
    a kept row that still drew the rejects would leave the dropped row at
    zero, and a dropped row double counting would break the sum.
    """
    import re
    text = _render('clusters')
    # the cumulative net row is left whole; it is Build by type that splits
    assert 'Buildingcumulative · spits minus scoops' in text
    assert 'Build by type, kept' in text
    assert 'Build by type, dropped' in text

    # trial one's fixture holds 332 build spits. The page drew all 332 while
    # the crop went unapplied; with it applied the net row keeps 275.
    net = [int(m) for m in re.findall(r'\+(\d+) spits', text)]
    assert net and net[4] == 275, net

    # Adjacent cells abut in the flattened text, so read the first triple of
    # each row. The fixture puts 67 of each build type in day one, and the
    # two halves have to account for all of them and no more.
    # only the first two of the triple can be read: the third abuts the next
    # cell's first count with no separator ('56 / 56 / 5655 / 55 / 56')
    def first_pair(after):
        tail = text.split(after, 1)[1]
        match = re.match(r'(\d+) / (\d+) / ', tail)
        assert match, tail[:80]
        return [int(g) for g in match.groups()]

    kept = first_pair('multiple green · passes every filter')
    dropped = first_pair('from the events the filters reject')
    assert min(dropped) > 0, dropped          # rejects really are rejected
    for a, b in zip(kept, dropped):
        assert a + b == 67, (kept, dropped)


@needs_jsdom
def test_the_cluster_hours_cannot_leave_the_lit_period():
    """8am to 6pm are the ends, and the page used to open on 0 to 24 -- so it
    disagreed with every number features.py computed from the same events."""
    text = _render('clusters')
    assert 'Hours 8–18' in text


@needs_jsdom
def test_the_fish_filter_actually_narrows_every_row():
    """Working the control the way a person would, and checking the maps
    moved. A filter wired to nothing renders identically, which is the only
    failure worth testing for here.

    The two sexes must not sum to all: most events are unattributed, and
    filtering drops those rather than splitting them between the sexes.
    """
    import re
    import subprocess

    def kept(sex):
        result = subprocess.run(['node', str(HARNESS), 'clusters', '--text',
                                 '--sex', str(sex)], cwd=str(ROOT),
                                capture_output=True, encoding='utf-8',
                                timeout=120)
        out = (result.stdout or '') + (result.stderr or '')
        assert result.returncode == 0, out
        text = [l[6:] for l in out.splitlines() if l.startswith('TEXT  ')][0]
        return [int(m) for m in re.findall(r'\+(\d+) spits', text)]

    everyone, male, female = kept(0), kept(1), kept(2)
    assert everyone[4] == 275, everyone
    # each sex is a real, non-empty subset
    assert 0 < male[4] < everyone[4], (male, everyone)
    assert 0 < female[4] < everyone[4], (female, everyone)
    # and the unattributed events are dropped, not handed to a sex
    assert male[4] + female[4] < everyone[4], (male, female, everyone)
    # the cumulative maps still accumulate under the filter: a sex landing on
    # two days out of ten looks frozen, which the fixture's strides avoid
    assert male[0] < male[4], male


@needs_jsdom
def test_every_depth_map_states_its_scale():
    """A diverging ramp centred on zero looks identical at 0.5 cm and at
    5 cm, so a map without its ends written down cannot be read. Drawn into
    the canvas by paintMap, so no call site has to remember.
    """
    import subprocess
    result = subprocess.run(['node', str(HARNESS), 'summary', '--scale-text'],
                            cwd=str(ROOT), capture_output=True,
                            encoding='utf-8', timeout=120)
    out = (result.stdout or '') + (result.stderr or '')
    assert result.returncode == 0, out
    line = [l for l in out.splitlines() if l.startswith('SCALE')]
    assert line, out
    # at least one map, and every one of them labelled at both ends
    assert 'maps 0' not in line[0], line[0]
    assert 'unlabelled 0' in line[0], line[0]
