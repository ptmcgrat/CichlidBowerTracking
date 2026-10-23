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
         'trials']


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
def test_the_trials_page_groups_by_category_and_states_its_settings():
    """Its numbers are computed at fixed settings while the workbook follows
    the sliders, so the page has to say which it used or the two disagree for
    no visible reason."""
    text = _render('trials')
    assert '24 trials from 8 projects' in text
    assert 'Fixed settings for this page' in text
    assert 'will not match these numbers' in text
    for label in ('Total volume', 'Bower index', 'Built overnight',
                  'Scoop cloud aspect', 'Median spawn depth'):
        assert label in text, label
    # every category appears, with its own count
    for category in ('MC 8', 'TI 8', 'CVxMC_F1 8'):
        assert category in text, category
    # and a project that could not be read is named rather than dropped
    assert 'P9 (not collected)' in text
    # the CSV, with the repeated-measures caveat beside it rather than buried
    assert 'Download CSV' in text
    assert 'repeated measures' in text
