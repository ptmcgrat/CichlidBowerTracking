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
import time
from pathlib import Path
from typing import Dict, Optional

from flask import Flask, abort, jsonify, request, send_from_directory

from ..cloud import QUICK_TIMEOUT, Cloud, CloudError
from ..paths import Layout
from ..states import AnalysisStates, StatesError, local_analyses
from ..collect import bundle as B
from . import corrections as C
from . import features as F
from . import payload as PL

app = Flask(__name__)

STATE: Dict = {'layout': None, 'cloud': None, 'upload': True}
# One states table per analysis, loaded when first asked for. Loading all of
# them at startup would read a dozen files and several hundred manifests before
# the server could answer anything.
TABLES: Dict[str, tuple] = {}         # analysis -> (csv mtime, states table)
TABLES_GUARD = threading.Lock()
SUMMARIES: Dict[str, tuple] = {}      # analysis -> (fingerprint, summary)
INDEX_ROWS: Dict[str, tuple] = {}     # analysis -> (fingerprint, project rows)
LOCKS: Dict[str, threading.Lock] = {}
LOCKS_GUARD = threading.Lock()


LOCK_WAIT = 15.0          # seconds a save will wait for the one ahead of it
HELD: Dict[str, float] = {}


def _lock_for(analysis_id: str, project_id: str) -> threading.Lock:
    with LOCKS_GUARD:
        return LOCKS.setdefault(analysis_id + '/' + project_id, threading.Lock())


def _hold(analysis_id: str, project_id: str):
    """Take a project's lock, waiting a little rather than refusing at once.

    Two people saving the same project within a second of each other is
    ordinary and used to fail outright. A real wait also makes the failure
    meaningful: if the lock is still held after LOCK_WAIT seconds, something is
    stuck rather than busy, and the message says how long so that it reads as
    the fault it is instead of as a colleague typing.
    """
    key = analysis_id + '/' + project_id
    lock = _lock_for(analysis_id, project_id)
    if lock.acquire(timeout=LOCK_WAIT):
        HELD[key] = time.time()
        return lock, None
    waited = time.time() - HELD.get(key, time.time())
    if waited > LOCK_WAIT * 2:
        message = ('this project has been locked for %d seconds, which means a '
                   'previous save is stuck rather than in progress. The '
                   'server needs restarting; your edits are still on screen, '
                   'so do not reload.' % waited)
    else:
        message = 'someone else is saving this project. Try again in a moment.'
    return None, message


def _release(lock, analysis_id: str, project_id: str) -> None:
    HELD.pop(analysis_id + '/' + project_id, None)
    lock.release()


def _states_or_none(analysis_id: str) -> Optional[AnalysisStates]:
    """The states table, read once and kept until the file changes.

    Keyed on the file's mtime so adding a project to the CSV shows up without
    a restart, which it did not when the cache was keyed on the name alone.
    """
    paths = STATE['layout'].analysis(analysis_id)
    try:
        stamp = paths.states_csv.stat().st_mtime_ns
    except OSError:
        stamp = 0
    with TABLES_GUARD:
        cached = TABLES.get(analysis_id)
        if cached and cached[0] == stamp:
            return cached[1]
    try:
        table = AnalysisStates.load(paths, STATE['cloud'])
    except StatesError:
        return None
    with TABLES_GUARD:
        TABLES[analysis_id] = (stamp, table)
    return table


def _states(analysis_id: str) -> AnalysisStates:
    table = _states_or_none(analysis_id)
    if table is None:
        abort(404)
    return table


def _paths(analysis_id: str, project_id: str):
    states = _states(analysis_id)
    if project_id not in states.table.index:
        abort(404)
    return STATE['layout'].project(project_id, analysis_id)


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
    """Pages, payloads and page scripts are rebuilt in place.

    The scripts were missing from this list, so a browser kept running the
    previous version of a page's JavaScript after an update — which looks
    exactly like the update not working. Only the generated images may be
    cached, and those are rewritten wholesale when a project is rebuilt.
    """
    kind = response.content_type or ''
    if ('html' in kind or 'json' in kind or 'javascript' in kind):
        response.headers['Cache-Control'] = 'no-store, must-revalidate'
    return response


def _fingerprint(analysis_id: str, projects) -> tuple:
    """Cheap evidence that nothing a landing page reads has changed.

    Two stats per project, against two JSON parses per project. Stat is cheap;
    parsing is not, and a landing page reads every project in the analysis.

    The states file is in here too, and has to be: the rows carry each
    project's category, which comes from that file and from nothing else. With
    only the manifests and the corrections watched, editing the Category
    column changed nothing the cache could see, so both landing pages went on
    serving the old categories until some unrelated save happened to move a
    prep.json -- or until the server was restarted, which is not a thing
    anyone should have to know.
    """
    layout = STATE['layout']
    stamps = []
    try:
        stamps.append(layout.analysis(analysis_id).states_csv.stat().st_mtime_ns)
    except OSError:
        stamps.append(0)
    for project_id in projects:
        paths = layout.project(project_id, analysis_id)
        for source in (paths.manifest, paths.prep_json):
            try:
                stamps.append(source.stat().st_mtime_ns)
            except OSError:
                stamps.append(0)
    return (len(projects), tuple(stamps))


@app.route('/<analysis_id>/')
@app.route('/<analysis_id>/index.html')
def index(analysis_id: str):
    states = _states(analysis_id)
    projects = states.project_ids()
    fingerprint = _fingerprint(analysis_id, projects)
    cached = INDEX_ROWS.get(analysis_id)
    if cached and cached[0] == fingerprint:
        return _render_index(analysis_id, cached[1])
    rows = []
    for project_id in projects:
        paths = STATE['layout'].project(project_id, analysis_id)
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
    INDEX_ROWS[analysis_id] = (fingerprint, rows)
    return _render_index(analysis_id, rows)


@app.route('/<analysis_id>/features')
@app.route('/<analysis_id>/features.html')
def analysis_features(analysis_id: str):
    """Every project's metrics side by side, grouped by category."""
    _states(analysis_id)
    return _template('analysis.html').replace('__ANALYSIS__', analysis_id)


@app.route('/<analysis_id>/features.json')
def analysis_features_data(analysis_id: str):
    """Metrics for every collected project in this analysis.

    Computed per project and cached, because the alternative is asking a
    browser to carry a gigabyte of depth maps. A project whose bundle or
    corrections have changed is recomputed; the rest are read from disk.
    """
    states = _states(analysis_id)
    out = []
    for project_id in states.project_ids():
        paths = STATE['layout'].project(project_id, analysis_id)
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
    return jsonify({'analysisID': analysis_id,
                    'threshold': F.THRESHOLD_CM, 'coverage': F.COVERAGE,
                    'projects': out})


@app.route('/<analysis_id>/phenotypes')
@app.route('/<analysis_id>/phenotypes.html')
@app.route('/<analysis_id>/trials')              # the name it had first
@app.route('/<analysis_id>/trials.html')
def analysis_trials(analysis_id: str):
    """Trial-level metrics for every project, compared by category."""
    _states(analysis_id)
    return _template('phenotypes.html').replace('__ANALYSIS__', analysis_id)


@app.route('/<analysis_id>/phenotypes.json')
@app.route('/<analysis_id>/trials.json')
def analysis_trials_data(analysis_id: str):
    """One flat row per trial, across every collected project.

    Fixed settings, deliberately. Recomputing thirty projects on every slider
    move is not something to do on a page load, so this runs at the defaults
    and says so in its own header. A workbook downloaded with the sliders
    somewhere else will not match these numbers, which is why both carry the
    settings they used.
    """
    states = _states(analysis_id)
    params = F.DEFAULTS
    rows = []
    skipped = []
    for project_id in states.project_ids():
        paths = STATE['layout'].project(project_id, analysis_id)
        if not paths.manifest.exists():
            skipped.append({'projectID': project_id, 'why': 'not collected'})
            continue
        try:
            metrics = F.load(paths)
        except Exception as error:
            skipped.append({'projectID': project_id, 'why': repr(error)})
            continue
        if not metrics:
            skipped.append({'projectID': project_id, 'why': 'no metrics'})
            continue
        category = states.category(project_id) or 'uncategorised'
        for trial in metrics.get('trials', []):
            if trial.get('excluded'):
                continue
            rows.append(_trial_row(metrics, trial, category,
                                   states.tank(project_id), project_id))
        logged = metrics.get('projectID')
        if logged and logged != project_id:
            skipped.append({'projectID': project_id,
                            'why': 'its logfile calls it ' + str(logged)})
    from dataclasses import asdict
    return jsonify({'analysisID': analysis_id, 'params': asdict(params),
                    'rows': rows, 'skipped': skipped})


def _trial_row(metrics: dict, trial: dict, category: str, tank: str,
               project_id: Optional[str] = None) -> dict:
    """The flat shape the comparison page plots.

    Flat on purpose: the page has to sort and group by any of these, and
    reaching three levels into a nested metric at every dot is how the
    summary page became slow.
    """
    def shape(key, field):
        value = trial.get(key) or {}
        return value.get(field)

    # The directory, not the manifest. The manifest's projectID is whatever
    # the logfile's ProjectID field says, and a logfile copied from another
    # tank carries that tank's name -- which would attribute these rows to the
    # wrong fish without anything looking wrong. The directory is what the
    # states file lists and what the server looked the project up by, so it is
    # the identity that counts; a disagreement is reported separately.
    return {
        'projectID': project_id or metrics.get('projectID'),
        'loggedProjectID': metrics.get('projectID'),
        'tankID': tank, 'category': category,
        'trial': trial.get('trial'), 'days': trial.get('days'),

        'totalVolume': trial.get('totalVolume'),
        'volumePerDay': trial.get('volumePerDay'),
        'pitVolume': trial.get('pitVolume'),
        'castleVolume': trial.get('castleVolume'),

        'bowerIndex': trial.get('bowerIndex'),
        'firstHalfIndex': (trial.get('firstHalf') or {}).get('bowerIndex'),
        'secondHalfIndex': (trial.get('secondHalf') or {}).get('bowerIndex'),
        'nightShare': trial.get('nightShare'),

        'eventsPerDay': trial.get('eventsPerDay'),
        'buildEventsPerDay': trial.get('buildEventsPerDay'),
        'feedEventsPerDay': trial.get('feedEventsPerDay'),
        'spawnEventsPerDay': trial.get('spawnEventsPerDay'),

        'scoopArea': shape('scoopShape', 'areaCm2'),
        'scoopAreaDaily': trial.get('scoopAreaDaily'),
        'spitArea': shape('spitShape', 'areaCm2'),
        'spitAreaDaily': trial.get('spitAreaDaily'),

        'scoopSpitAreaIndex': trial.get('scoopSpitAreaIndex'),
        'scoopAspectDaily': trial.get('scoopAspectDaily'),
        'spitAspectDaily': trial.get('spitAspectDaily'),
        'spawnDepthMedian': trial.get('spawnDepthMedian'),
    }


# The CSV's column names, in order. Explicit rather than derived from the
# metric keys: these are what someone's R script will refer to by name, so
# they have to be stable, and a name generated by a rule changes whenever the
# rule does. Lower case with underscores and no units in the name, because
# make.names() mangles anything else and a mangled name is a broken script.
CSV_COLUMNS = (
    # identifiers, then the twenty measures the page plots, in page order.
    # Matching the page exactly was asked for: a file holding more than the
    # figure invites someone to compare a column nobody has looked at.
    ('project_id', 'projectID'), ('logged_project_id', 'loggedProjectID'),
    ('sample_id', 'sampleID'), ('tank_id', 'tankID'),
    ('category', 'category'), ('trial', 'trial'), ('days', 'days'),

    ('total_volume_cm3', 'totalVolume'),
    ('volume_per_day_cm3', 'volumePerDay'),
    ('pit_volume_cm3', 'pitVolume'),
    ('castle_volume_cm3', 'castleVolume'),

    ('bower_index', 'bowerIndex'),
    ('first_half_index', 'firstHalfIndex'),
    ('second_half_index', 'secondHalfIndex'),
    ('night_share', 'nightShare'),

    ('events_per_day', 'eventsPerDay'),
    ('build_events_per_day', 'buildEventsPerDay'),
    ('feed_events_per_day', 'feedEventsPerDay'),
    ('spawn_events_per_day', 'spawnEventsPerDay'),

    ('scoop_area_pooled_cm2', 'scoopArea'),
    ('scoop_area_daily_cm2', 'scoopAreaDaily'),
    ('spit_area_pooled_cm2', 'spitArea'),
    ('spit_area_daily_cm2', 'spitAreaDaily'),

    ('scoop_spit_area_index', 'scoopSpitAreaIndex'),
    ('scoop_aspect_daily', 'scoopAspectDaily'),
    ('spit_aspect_daily', 'spitAspectDaily'),
    ('spawn_depth_median_cm', 'spawnDepthMedian'),
)


def _sample_id(paths) -> str:
    """The SampleID the tank recorded, read from the logfile.

    It is parsed into the log model but never written to the manifest, so it
    cannot be had from the collected data -- and adding it there would mean
    recollecting every project to get an identifier that is already on disk.
    The logfile is local and small, so it is read here instead.
    """
    try:
        from ..logfile.parse import parse_logfile
        if paths.logfile.exists():
            return parse_logfile(paths.logfile).sample_id or ''
    except Exception:
        pass
    return ''


@app.route('/<analysis_id>/phenotypes.csv')
@app.route('/<analysis_id>/trials.csv')
def analysis_trials_csv(analysis_id: str):
    """The trial rows as one flat CSV, for mapping.

    One row per trial, not per fish. A trial is a repeated measure -- the same
    male builds again after the tank is reset -- so collapsing to one row per
    individual is a modelling choice, and this file leaves it to the person
    making it. Averaging here would throw away the replication that tells you
    whether a phenotype is repeatable at all.
    """
    import csv
    import io

    states = _states(analysis_id)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow([name for name, _ in CSV_COLUMNS])

    for project_id in states.project_ids():
        paths = STATE['layout'].project(project_id, analysis_id)
        if not paths.manifest.exists():
            continue
        try:
            metrics = F.load(paths)
        except Exception:
            continue
        if not metrics:
            continue
        sample = _sample_id(paths)
        category = states.category(project_id) or 'uncategorised'
        for trial in metrics.get('trials', []):
            if trial.get('excluded'):
                continue
            row = _trial_row(metrics, trial, category,
                             states.tank(project_id), project_id)
            row['sampleID'] = sample
            # NA rather than an empty cell: read.csv turns an empty numeric
            # into NA but an empty character into "", so a missing category
            # and a missing volume would not behave alike
            writer.writerow(['NA' if row.get(key) is None else row.get(key)
                             for _, key in CSV_COLUMNS])

    body = buffer.getvalue()
    return app.response_class(
        body, mimetype='text/csv',
        headers={'Content-Disposition':
                 'attachment; filename="%s_phenotypes.csv"' % analysis_id})


@app.route('/<analysis_id>/analysis.js')
@app.route('/<analysis_id>/phenotypes.js')
def analysis_script(analysis_id: str):
    """Scripts for the analysis-level pages.

    These sit one level above a project, so the project asset route cannot
    reach them: /YH_MC/trials.js has no project in it.
    """
    name = request.path.rsplit('/', 1)[-1]
    if name not in ('analysis.js', 'phenotypes.js'):
        abort(404)
    return send_from_directory(str(Path(__file__).parent / 'templates'), name)


@app.route('/<analysis_id>/<project_id>/')
@app.route('/<analysis_id>/<project_id>/index.html')
@app.route('/<analysis_id>/<project_id>/prep')
def project_page(analysis_id: str, project_id: str):
    paths = _paths(analysis_id, project_id)
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    return _render_prep(project_id)


@app.route('/<analysis_id>/<project_id>/depth')
def depth_page(analysis_id: str, project_id: str):
    paths = _paths(analysis_id, project_id)
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    return _template('depth.html').replace('__TITLE__', project_id + ' \u00b7 Depth')


@app.route('/<analysis_id>/<project_id>/clusters')
def clusters_page(analysis_id: str, project_id: str):
    paths = _paths(analysis_id, project_id)
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    return _template('clusters.html').replace('__TITLE__',
                                              project_id + ' \u00b7 Clusters')


@app.route('/<analysis_id>/<project_id>/summary')
def summary_page(analysis_id: str, project_id: str):
    paths = _paths(analysis_id, project_id)
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    return _template('summary.html').replace('__TITLE__',
                                             project_id + ' \u00b7 Summary')


@app.route('/<analysis_id>/<project_id>/features')
def features_page(analysis_id: str, project_id: str):
    paths = _paths(analysis_id, project_id)
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    return _template('features.html').replace('__TITLE__',
                                              project_id + ' \u00b7 Features')


@app.route('/<analysis_id>/<project_id>/clusters.json')
def project_clusters(analysis_id: str, project_id: str):
    """The packed events, served on their own.

    Larger than every image in the page put together, and only the cluster
    view needs them, so they are not embedded in the payload.
    """
    paths = _paths(analysis_id, project_id)
    if not paths.clusters_packed.exists():
        return jsonify({'error': 'no cluster data collected for this project'}), 404
    return send_from_directory(str(paths.collected_dir), 'clusters.json')


@app.route('/<analysis_id>/<project_id>/page.json')
def project_payload(analysis_id: str, project_id: str):
    """The built payload, with the corrections read fresh.

    The payload is generated once and cached, so the copy of the corrections
    inside it is a snapshot from build time. Serving that back after a save
    loses the save, which is exactly what happened: the file on disk was right
    and the page was reading a stale embedded copy.
    """
    paths = _paths(analysis_id, project_id)
    source = paths.pages_dir / 'page.json'
    if PL.is_stale(paths):
        PL.build_prep_payload(paths)
    with open(source) as handle:
        payload = json.load(handle)
    payload['prep'] = C.load(paths).to_dict()
    return jsonify(payload)


def _params_from_query(args) -> 'F.Params':
    """The slider values the page sent, falling back to the defaults.

    Every setting that changes a number is a query parameter, so the workbook
    can be computed at exactly what was on screen. A value that will not parse
    falls back rather than failing the download, and the sheet records what was
    actually used — so a typo shows up as a number you did not expect rather
    than as an error page.
    """
    from dataclasses import asdict, fields
    chosen = asdict(F.DEFAULTS)
    for field in fields(F.Params):
        raw = args.get(field.name)
        if raw is None or raw == '':
            continue
        try:
            chosen[field.name] = (int(raw) if field.type in ('int', int)
                                  else float(raw))
        except (TypeError, ValueError):
            continue
    return F.Params(**chosen)


@app.route('/<analysis_id>/<project_id>/export.xlsx')
def export_workbook(analysis_id: str, project_id: str):
    """One project as a three-tab workbook, at the chosen settings."""
    paths = _paths(analysis_id, project_id)
    params = _params_from_query(request.args)
    states = _states(analysis_id)
    from . import workbook as WB
    try:
        metrics = F.compute(paths, params)
    except FileNotFoundError:
        return jsonify({'error': 'this project has not been collected'}), 404
    target = paths.pages_dir / (project_id + '.xlsx')
    try:
        WB.write_workbook(target, paths, metrics, states.category(project_id))
    except WB.WorkbookError as error:
        return jsonify({'error': str(error)}), 501
    return send_from_directory(
        str(target.parent), target.name, as_attachment=True,
        download_name='%s_%s.xlsx' % (analysis_id, project_id))


@app.route('/<analysis_id>/<project_id>/<path:name>')
def project_asset(analysis_id: str, project_id: str, name: str):
    paths = _paths(analysis_id, project_id)
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


@app.route('/<analysis_id>/<project_id>/save', methods=['POST'])
def save(analysis_id: str, project_id: str):
    paths = _paths(analysis_id, project_id)
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return jsonify({'error': 'expected a JSON object'}), 400

    lock, busy = _hold(analysis_id, project_id)
    if lock is None:
        return jsonify({'error': busy}), 409
    try:
        prep = C.Prep.from_dict(dict(body, project_id=project_id,
                                     analysis_id=analysis_id,
                                     schema=C.SCHEMA))
        C.save(paths, prep, who=viewer(body.get('who', '')))
        if STATE['upload']:
            try:
                STATE['cloud'].upload(paths.prep_json, timeout=QUICK_TIMEOUT)
            except CloudError as error:
                return jsonify({'error': 'saved locally but not uploaded: ' +
                                         str(error)}), 502
        saved = C.load(paths)
        return jsonify({'ok': True, 'updated': saved.updated, 'who': saved.who})
    except Exception as error:
        return jsonify({'error': repr(error)}), 500
    finally:
        _release(lock, analysis_id, project_id)


@app.route('/<analysis_id>/<project_id>/marks', methods=['POST'])
def save_marks(analysis_id: str, project_id: str):
    """Day marks only, merged onto what is already saved.

    The summary page has no registration or crop controls, so posting the whole
    corrections object from there would let a copy loaded before someone
    changed the crops overwrite that change. Only the field this page owns is
    touched.
    """
    paths = _paths(analysis_id, project_id)
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return jsonify({'error': 'expected a JSON object'}), 400

    lock, busy = _hold(analysis_id, project_id)
    if lock is None:
        return jsonify({'error': busy}), 409
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
                STATE['cloud'].upload(paths.prep_json, timeout=QUICK_TIMEOUT)
            except CloudError as error:
                return jsonify({'error': 'saved locally but not uploaded: ' +
                                         str(error)}), 502
        saved = C.load(paths)
        return jsonify({'ok': True, 'updated': saved.updated, 'who': saved.who})
    except Exception as error:
        return jsonify({'error': repr(error)}), 500
    finally:
        _release(lock, analysis_id, project_id)


@app.route('/<analysis_id>/<project_id>/rebuild', methods=['POST'])
def rebuild(analysis_id: str, project_id: str):
    """Regenerate the assets, for when a project has been recollected."""
    paths = _paths(analysis_id, project_id)
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


def _analysis_summary(analysis_id: str) -> dict:
    """What the master page shows for one analysis.

    Reading every project's manifest and corrections is a few hundred small
    file operations, so the result is kept and only rebuilt when one of those
    files has actually moved. Stat is cheap; parsing is not.
    """
    layout = STATE['layout']
    # through the cache: loading the table again per request cost a CSV parse
    # per analysis, and for an analysis absent from this machine an rclone
    # fetch that failed slowly before the row could say so
    states = _states_or_none(analysis_id)
    if states is None:
        return {'analysisID': analysis_id, 'unreadable': True}

    projects = states.project_ids()
    fingerprint = _fingerprint(analysis_id, projects)
    cached = SUMMARIES.get(analysis_id)
    if cached and cached[0] == fingerprint:
        return cached[1]

    collected = ready = 0
    categories: Dict[str, int] = {}
    latest, latest_who = '', ''
    for project_id in projects:
        paths = layout.project(project_id, analysis_id)
        category = states.category(project_id) or 'uncategorised'
        categories[category] = categories.get(category, 0) + 1
        if not paths.manifest.exists():
            continue
        collected += 1
        try:
            prep = C.load(paths)
        except C.CorrectionsError:
            continue
        manifest = B.read_manifest(paths.manifest)
        trials = [t['number'] for t in (manifest or {}).get('trials', [])]
        if trials and prep.coverage(trials)['ready']:
            ready += 1
        if prep.updated and prep.updated > latest:
            latest, latest_who = prep.updated, prep.who

    summary = {'analysisID': analysis_id, 'projects': len(projects),
               'collected': collected, 'ready': ready,
               'categories': categories, 'updated': latest, 'who': latest_who}
    SUMMARIES[analysis_id] = (fingerprint, summary)
    return summary


@app.route('/')
@app.route('/index.html')
def master():
    """Every analysis on this machine.

    Local only, and deliberately: the page used to ask the cloud as well, so
    loading it cost one rclone listing plus a failed fetch for every analysis
    that is not here — tens of seconds, to show rows that cannot be opened.
    ``python -m cichlid_bower_claude analyses`` is where to look for what else
    exists in the cloud.
    """
    names = local_analyses(STATE['layout'])
    rows = [_analysis_summary(name) for name in names]
    return _render_master(rows)


def _render_master(rows) -> str:
    body = []
    for row in rows:
        if row.get('unreadable'):
            body.append(
                '<tr><td><a class="name" href="/%s/">%s</a></td>'
                '<td colspan="5">states file could not be read</td></tr>'
                % (row['analysisID'], row['analysisID']))
            continue
        categories = ' '.join(
            '<span class="chip">%s %d</span>' % (name, count)
            for name, count in sorted(row['categories'].items()))
        note = row['updated'][:16] if row['updated'] else ''
        if row['who']:
            note += ' \u00b7 ' + row['who'].split('@')[0]
        body.append(
            '<tr><td><a class="name" href="/%s/">%s</a></td><td>%d</td>'
            '<td>%s</td><td>%s</td><td class="cats">%s</td><td>%s</td></tr>'
            % (row['analysisID'], row['analysisID'], row['projects'],
               _count(row['collected'], row['projects']),
               _count(row['ready'], row['projects']),
               categories or '\u2014', note or '\u2014'))
    return _template('master.html') \
        .replace('__ROWS__', '\n'.join(body)) \
        .replace('__COUNT__', str(len(rows)))


def _count(done: int, total: int) -> str:
    if not total:
        return '<span style="color:#6c7684">\u2014</span>'
    colour = 'var(--ok)' if done == total else (
        '#6c7684' if done == 0 else 'var(--warn)')
    return '<span style="color:%s">%d of %d</span>' % (colour, done, total)


def _render_index(analysis_id, rows) -> str:
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
            '<a class="tab" href="/%s/%s/%s">%s</a>' % (analysis_id, project,
                                                        slug, label)
            for slug, label in (('prep', 'prep'), ('depth', 'depth'),
                                ('clusters', 'clusters'), ('summary', 'summary'),
                                ('features', 'features')))
        body.append(
            '<tr class="%s"><td><a class="name" href="/%s/%s/prep">%s</a></td>'
            '<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>'
            '<td class="links">%s</td><td>%s</td></tr>'
            % (band, analysis_id, project, project, category, row['tank'] or '-',
               _tick(row['collected']),
               _of_trials((coverage or {}).get('registered', []), coverage,
                          'missingRegistration'),
               _of_trials((coverage or {}).get('cropped', []), coverage,
                          'missingCrop'),
               links, note or '-'))
    done = sum(1 for row in rows
               if row['coverage'] and row['coverage']['ready'])
    return _template('index.html') \
        .replace('__ANALYSIS__', analysis_id) \
        .replace('__ROWS__', '\n'.join(body)) \
        .replace('__DONE__', str(done)) \
        .replace('__COUNT__', str(len(rows)))


def configure(layout: Layout, cloud: Optional[Cloud] = None,
              upload: bool = True) -> None:
    """Point the server at a working directory.

    Every analysis under it is served; none is loaded until something asks for
    it, so starting up costs nothing however many there are.
    """
    STATE['layout'] = layout
    STATE['cloud'] = cloud or Cloud(layout=layout)
    STATE['upload'] = upload
    TABLES.clear()
    SUMMARIES.clear()
    INDEX_ROWS.clear()
    # pointing at a different working directory means the old projects are
    # gone, so their locks are too; keeping them would carry a wedged lock
    # across a reconfigure, which is the one case where recovery is wanted
    LOCKS.clear()
    HELD.clear()


def run(layout: Layout, host: str = '127.0.0.1', port: int = 8080,
        cloud: Optional[Cloud] = None, upload: bool = True) -> None:
    configure(layout, cloud=cloud, upload=upload)
    print('Serving every analysis in %s on http://%s:%d'
          % (layout.local_root, host, port))
    app.run(host=host, port=port, threaded=True)
