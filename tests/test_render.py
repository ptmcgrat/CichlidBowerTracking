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
PAGES = ['prep', 'depth', 'clusters', 'summary', 'features', 'analysis']


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
