"""Checks on the page scripts that a per-file syntax check cannot make.

Each page loads several scripts as separate tags into one shared scope, so a
name declared at the top level of two of them is a SyntaxError that stops the
whole page from running. Nothing renders, and the file that caused it parses
perfectly well on its own.
"""

import re
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / 'cichlid_bower_claude' \
    / 'server' / 'templates'

TOP_LEVEL = re.compile(r'^(?:let|const|var|function|class)\s+([A-Za-z_$][\w$]*)',
                       re.M)


def scripts_of(page: str):
    """The scripts a page loads, in order."""
    html = (TEMPLATES / page).read_text()
    return re.findall(r'<script src="([^"]+)"></script>', html)


def pages():
    return sorted(p.name for p in TEMPLATES.glob('*.html'))


@pytest.mark.parametrize('page', pages())
def test_no_two_scripts_on_a_page_declare_the_same_name(page):
    owner = {}
    clashes = []
    for script in scripts_of(page):
        source = TEMPLATES / script
        if not source.exists():
            continue
        for name in TOP_LEVEL.findall(source.read_text()):
            if name in owner and owner[name] != script:
                clashes.append('%s declared in both %s and %s'
                               % (name, owner[name], script))
            owner[name] = script
    assert not clashes, page + ': ' + '; '.join(sorted(set(clashes)))


@pytest.mark.parametrize('page', pages())
def test_every_script_a_page_loads_exists(page):
    for script in scripts_of(page):
        assert (TEMPLATES / script).exists(), page + ' loads missing ' + script


@pytest.mark.parametrize('page', pages())
def test_a_template_is_html(page):
    """A .html clobbered by a .js serves its own source as the page.

    Saving a file under the wrong name is easy and the result is unmistakable
    once you know it: the browser shows JavaScript where the page should be.
    """
    text = (TEMPLATES / page).read_text().lstrip()
    assert text.lower().startswith('<!doctype html'), (
        page + ' does not begin with a doctype; it starts: ' + text[:60])
    assert '</html>' in text, page + ' has no closing html tag'


@pytest.mark.parametrize('script', sorted(p.name for p in TEMPLATES.glob('*.js')))
def test_a_script_is_not_html(script):
    """And the reverse: a .js holding a page serves markup as a script."""
    text = (TEMPLATES / script).read_text().lstrip()
    assert not text.lower().startswith('<!doctype'), (
        script + ' contains HTML, not JavaScript')


def test_every_template_is_reachable():
    """A template with no route is dead weight that still gets shipped."""
    app = (TEMPLATES.parent / 'app.py').read_text()
    for page in pages():
        assert page in app, page + ' has no route in app.py'
