"""Serving the pages, and taking corrections back.

Runs where the cloud credentials already are, so nothing else is trusted with
write access to the lab's data.

Saves are small — one JSON file — so unlike the old design there is no job
queue and no polling: a save writes, uploads, and returns. What made the old
one slow was rebuilding a page from the archive on every save, and nothing
here does that. The crop, the mask and the thresholds are applied in the
browser, so changing them costs a redraw.

One lock per project still, because two people saving the same project would
otherwise interleave and the later write would silently lose the earlier.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, abort, jsonify, request, send_from_directory

from ..cloud import Cloud, CloudError
from ..paths import Layout
from ..states import AnalysisStates
from ..collect import bundle as B
from . import corrections as C
from . import features as F
from . import payload as PL

app = Flask(__name__)

STATE: Dict = {'layout': None, 'cloud': None, 'analysis_id': None, 'states': None,
               'upload': True}
LOCKS: Dict[str, threading.Lock] = {}
LOCKS_GUARD = threading.Lock()


def _lock_for(project_id: str) -> threading.Lock:
    with LOCKS_GUARD:
        return LOCKS.setdefault(project_id, threading.Lock())


def _paths(project_id: str):
    states = STATE['states']
    if project_id not in states.table.index:
        abort(404)
    return STATE['layout'].project(project_id, STATE['analysis_id'])


def viewer(claimed: str = '') -> str:
    """Who is making the change.

    An authenticated address wins, because it cannot be mistyped. Behind a VPN
    there is none, so the page asks and sends what it was told — trusted here
    because everyone who can reach the server can already edit the files.
    """
    header = (request.headers.get('Cf-Access-Authenticated-User-Email')
              or request.headers.get('X-Forwarded-User'))
    if header:
        return header
    return (claimed or '').strip() or 'anonymous'


@app.after_request
def no_cache(response):
    # pages are rebuilt in place; a cached copy is a wrong copy. Assets are
    # content-addressed by rebuild, so they may be cached.
    if response.content_type and ('html' in response.content_type
                                  or 'json' in response.content_type):
        response.headers['Cache-Control'] = 'no-store, must-revalidate'
    return response


@app.route('/')
@app.route('/index.html')
def index():
    states = STATE['states']
    rows = []
    for project_id in states.project_ids():
        paths = STATE['layout'].project(project_id, STATE['analysis_id'])
        collected = paths.manifest.exists()
        try:
            prep = C.load(paths)
            ready = prep.is_complete
            updated = prep.updated
        except C.CorrectionsError:
            prep, ready, updated = C.Prep(), False, 'unreadable'
        # per trial, not per project: a project whose trials each carry their
        # own registration is registered, and one trial short is not the same
        # as none at all
        trials = []
        if collected:
            manifest = B.read_manifest(paths.manifest)
            trials = [t['number'] for t in (manifest or {}).get('trials', [])]
        coverage = prep.coverage(trials) if trials else None
        rows.append({'projectID': project_id,
                     'category': states.category(project_id),
                     'tank': states.tank(project_id),
                     'collected': collected, 'prepped': ready, 'updated': updated,
                     'coverage': coverage, 'who': prep.who})
    return _render_index(rows)


@app.route('/features')
@app.route('/features.html')
def analysis_features():
    """Every project's metrics side by side, grouped by category."""
    return _template('analysis.html').replace('__ANALYSIS__', STATE['analysis_id'])


@app.route('/features.json')
def analysis_features_data():
    """Metrics for every collected project in this analysis.

    Computed per project and cached, because the alternative is asking a
    browser to carry a gigabyte of depth maps. A project whose bundle or
    corrections have changed is recomputed; the rest are read from disk.
    """
    states = STATE['states']
    out = []
    for project_id in states.project_ids():
        paths = STATE['layout'].project(project_id, STATE['analysis_id'])
        if not paths.manifest.exists():
            continue
        try:
            metrics = F.load(paths)
        except Exception as error:
            out.append({'projectID': project_id, 'error': repr(error)})
            continue
        if not metrics:
            continue
        metrics['category'] = states.category(project_id)
        metrics['tankID'] = states.tank(project_id)
        out.append(metrics)
    return jsonify({'analysisID': STATE['analysis_id'],
                    'threshold': F.THRESHOLD_CM, 'coverage': F.COVERAGE,
                    'projects': out})


@app.route('/analysis.js')
def analysis_script():
    return send_from_directory(str(Path(__file__).parent / 'templates'),
                               'analysis.js')


@app.route('/<project_id>/')
@app.route('/<project_id>/index.html')
@app.route('/<project_id>/prep')
def project_page(project_id: str):
    paths = _paths(project_id)
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    return _render_prep(project_id)


@app.route('/<project_id>/depth')
def depth_page(project_id: str):
    paths = _paths(project_id)
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    return _template('depth.html').replace('__TITLE__', project_id + ' \u00b7 Depth')


@app.route('/<project_id>/clusters')
def clusters_page(project_id: str):
    paths = _paths(project_id)
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    return _template('clusters.html').replace('__TITLE__',
                                              project_id + ' \u00b7 Clusters')


@app.route('/<project_id>/summary')
def summary_page(project_id: str):
    paths = _paths(project_id)
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    return _template('summary.html').replace('__TITLE__',
                                             project_id + ' \u00b7 Summary')


@app.route('/<project_id>/features')
def features_page(project_id: str):
    paths = _paths(project_id)
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    return _template('features.html').replace('__TITLE__',
                                              project_id + ' \u00b7 Features')


@app.route('/<project_id>/clusters.json')
def project_clusters(project_id: str):
    """The packed events, served on their own.

    Larger than every image in the page put together, and only the cluster
    view needs them, so they are not embedded in the payload.
    """
    paths = _paths(project_id)
    if not paths.clusters_packed.exists():
        return jsonify({'error': 'no cluster data collected for this project'}), 404
    return send_from_directory(str(paths.collected_dir), 'clusters.json')


@app.route('/<project_id>/page.json')
def project_payload(project_id: str):
    """The built payload, with the corrections read fresh.

    The payload is generated once and cached, so the copy of the corrections
    inside it is a snapshot from build time. Serving that back after a save
    loses the save, which is exactly what happened: the file on disk was right
    and the page was reading a stale embedded copy.
    """
    paths = _paths(project_id)
    source = paths.pages_dir / 'page.json'
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    with open(source) as handle:
        payload = json.load(handle)
    payload['prep'] = C.load(paths).to_dict()
    return jsonify(payload)


@app.route('/<project_id>/<path:name>')
def project_asset(project_id: str, name: str):
    paths = _paths(project_id)
    if name in ('prep.js', 'depth.js', 'clusters.js', 'stats.js',
                'summary.js', 'features.js', 'common.js'):
        return send_from_directory(str(Path(__file__).parent / 'templates'), name)
    root = paths.pages_dir.resolve()
    target = (root / name).resolve()
    if not str(target).startswith(str(root)):
        abort(403)
    if not target.exists():
        abort(404)
    return send_from_directory(str(target.parent), target.name)


@app.route('/<project_id>/save', methods=['POST'])
def save(project_id: str):
    paths = _paths(project_id)
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return jsonify({'error': 'expected a JSON object'}), 400

    lock = _lock_for(project_id)
    if not lock.acquire(blocking=False):
        return jsonify({'error': 'someone else is saving this project. '
                                 'Reload and try again.'}), 409
    try:
        prep = C.Prep.from_dict(dict(body, project_id=project_id,
                                     analysis_id=STATE['analysis_id'],
                                     schema=C.SCHEMA))
        C.save(paths, prep, who=viewer(body.get('who', '')))
        if STATE['upload']:
            try:
                STATE['cloud'].upload(paths.prep_json)
            except CloudError as error:
                return jsonify({'error': 'saved locally but not uploaded: ' +
                                         str(error)}), 502
        saved = C.load(paths)
        return jsonify({'ok': True, 'updated': saved.updated, 'who': saved.who})
    except Exception as error:
        return jsonify({'error': repr(error)}), 500
    finally:
        lock.release()


@app.route('/<project_id>/marks', methods=['POST'])
def save_marks(project_id: str):
    """Day marks only, merged onto what is already saved.

    The summary page has no registration or crop controls, so posting the whole
    corrections object from there would let a copy loaded before someone
    changed the crops overwrite that change. Only the field this page owns is
    touched.
    """
    paths = _paths(project_id)
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return jsonify({'error': 'expected a JSON object'}), 400

    lock = _lock_for(project_id)
    if not lock.acquire(blocking=False):
        return jsonify({'error': 'someone else is saving this project. '
                                 'Reload and try again.'}), 409
    try:
        prep = C.load(paths)
        marks = {}
        for key, value in (body.get('day_marks') or {}).items():
            if not isinstance(value, dict):
                continue
            allowed = {f for f in C.DayMark.__dataclass_fields__}
            marks[str(key)] = C.DayMark(
                **{k: v for k, v in value.items() if k in allowed})
        prep.day_marks = marks
        C.save(paths, prep, who=viewer(body.get('who', '')))
        if STATE['upload']:
            try:
                STATE['cloud'].upload(paths.prep_json)
            except CloudError as error:
                return jsonify({'error': 'saved locally but not uploaded: ' +
                                         str(error)}), 502
        saved = C.load(paths)
        return jsonify({'ok': True, 'updated': saved.updated, 'who': saved.who})
    except Exception as error:
        return jsonify({'error': repr(error)}), 500
    finally:
        lock.release()


@app.route('/<project_id>/rebuild', methods=['POST'])
def rebuild(project_id: str):
    """Regenerate the assets, for when a project has been recollected."""
    paths = _paths(project_id)
    PL.build_prep_payload(paths)
    return jsonify({'ok': True})


def _template(name: str) -> str:
    return (Path(__file__).parent / 'templates' / name).read_text()


def _render_prep(project_id: str) -> str:
    return _template('prep.html').replace('__TITLE__', project_id + ' \u00b7 Prep')


def _tick(done: bool) -> str:
    return ('<span style="color:var(--ok)">yes</span>' if done
            else '<span style="color:#6c7684">no</span>')


def _of_trials(have, coverage, missing_key) -> str:
    """How many trials are covered, and which are not.

    "no" on a project where two trials of three are done hides the thing worth
    knowing, which is that one trial is left.
    """
    if not coverage:
        return '<span style="color:#6c7684">\u2014</span>'
    total = coverage['active']
    if len(have) == total and total:
        return '<span style="color:var(--ok)">%d of %d</span>' % (len(have), total)
    missing = coverage[missing_key]
    label = ', '.join(str(t) for t in missing[:4])
    return ('<span style="color:var(--warn)">%d of %d</span>'
            '<span style="color:#6c7684"> \u00b7 trial %s</span>'
            % (len(have), total, label))


def _render_index(rows) -> str:
    # grouped by category, because that is how projects are compared: two
    # banded colours so the eye finds a group's edge without a heading row
    rows = sorted(rows, key=lambda r: (r['category'] or '~', r['projectID']))
    body = []
    seen = []
    for row in rows:
        category = row['category'] or '-'
        if category not in seen:
            seen.append(category)
        band = 'band%d' % (seen.index(category) % 2)
        note = row['updated'][:16] if row['updated'] else ''
        if row['who']:
            note += ' \u00b7 ' + row['who'].split('@')[0]
        coverage = row['coverage']
        if coverage and coverage['excluded']:
            note += ' \u00b7 trial %s excluded' % ', '.join(
                str(t) for t in coverage['excluded'])
        project = row['projectID']
        links = ' '.join(
            '<a class="tab" href="/%s/%s">%s</a>' % (project, slug, label)
            for slug, label in (('prep', 'prep'), ('depth', 'depth'),
                                ('clusters', 'clusters'), ('summary', 'summary'),
                                ('features', 'features')))
        body.append(
            '<tr class="%s"><td><a class="name" href="/%s/prep">%s</a></td>'
            '<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
            '<td class="links">%s</td><td>%s</td></tr>'
            % (band, project, project, category, row['tank'] or '-',
               _tick(row['collected']),
               _of_trials((coverage or {}).get('registered', []), coverage,
                          'missingRegistration'),
               _of_trials((coverage or {}).get('cropped', []), coverage,
                          'missingCrop'),
               links, note or '-'))
    done = sum(1 for row in rows
               if row['coverage'] and row['coverage']['ready'])
    return _template('index.html') \
        .replace('__ANALYSIS__', STATE['analysis_id']) \
        .replace('__ROWS__', '\n'.join(body)) \
        .replace('__DONE__', str(done)) \
        .replace('__COUNT__', str(len(rows)))


def configure(layout: Layout, analysis_id: str, cloud: Optional[Cloud] = None,
              upload: bool = True) -> None:
    STATE['layout'] = layout
    STATE['analysis_id'] = analysis_id
    STATE['cloud'] = cloud or Cloud(layout=layout)
    STATE['upload'] = upload
    STATE['states'] = AnalysisStates.load(layout.analysis(analysis_id), STATE['cloud'])


def run(layout: Layout, analysis_id: str, host: str = '127.0.0.1', port: int = 8080,
        cloud: Optional[Cloud] = None, upload: bool = True) -> None:
    configure(layout, analysis_id, cloud=cloud, upload=upload)
    print('Serving %s on http://%s:%d' % (analysis_id, host, port))
    app.run(host=host, port=port, threaded=True)
