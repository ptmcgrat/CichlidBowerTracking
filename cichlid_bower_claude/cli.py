"""Command line.

    python -m cichlid_bower_claude collect YH_MC_Parentals
    python -m cichlid_bower_claude collect YH_MC_Parentals --projects MC_920_t001_tr1
    python -m cichlid_bower_claude status YH_MC_Parentals
    python -m cichlid_bower_claude analyses

Thin: argument parsing and printing, with every decision in the module that
owns it. ``--raise`` exists because the collector swallows exceptions so one
bad project cannot stop a sweep, which is right for a sweep and useless when
the thing you are debugging is the exception.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .cloud import Cloud, CloudError
from .collect import bundle as B
from .collect.collector import collect_project
from .paths import Layout, LayoutError
from .states import AnalysisStates, StatesError, list_analyses


def _layout(args) -> Layout:
    """The working directory, and say so: a wrong root should be visible in
    the first line of output rather than inferred from odd results later."""
    if args.root:
        from pathlib import Path
        layout = Layout(local_root=Path(args.root).expanduser())
        source = '--root'
    else:
        layout = Layout.from_env()
        source = 'CICHLID_LOCAL_ROOT'
    if not getattr(args, 'quiet', False):
        print('Working directory: %s  (%s)' % (layout.local_root, source))
    return layout


def _projects(states: AnalysisStates, requested: Optional[List[str]]) -> List[str]:
    order = states.project_ids()
    if not requested:
        return order
    unknown = [p for p in requested if p not in states.table.index]
    if unknown:
        print('not in this analysis: ' + ', '.join(unknown))
    return [p for p in requested if p in states.table.index]


def cmd_analyses(args) -> int:
    layout = _layout(args)
    cloud = Cloud(layout=layout, verbose=args.verbose)
    try:
        names = list_analyses(layout, cloud)
    except CloudError as error:
        print('Could not reach the cloud.')
        print('  ' + str(error))
        print('\nCheck that rclone has the remote: rclone listremotes')
        return 2
    if not names:
        remote = layout.cloud(layout.local_root / '__AnalysisStates')
        print('No analyses found.')
        print('  locally: ' + str(layout.local_root / '__AnalysisStates'))
        print('  cloud:   ' + remote)
        print('\nThe cloud directory was readable but empty. If that is not what '
              'you expect,\ncheck the path above against where the data actually '
              'lives.')
        return 1
    for name in names:
        paths = layout.analysis(name)
        try:
            states = AnalysisStates.load(paths)
            count = len(states.project_ids())
            print('%-28s %3d projects' % (name, count))
        except StatesError:
            print('%-28s   (no states file locally)' % name)
    return 0


def cmd_status(args) -> int:
    layout = _layout(args)
    cloud = Cloud(layout=layout, verbose=args.verbose)
    try:
        states = AnalysisStates.load(layout.analysis(args.analysis_id), cloud)
    except StatesError as error:
        print(str(error))
        return 1
    if states.missing_columns:
        print('columns absent from the states file: ' +
              ', '.join(states.missing_columns))
    projects = _projects(states, args.projects)
    print('%-30s %-10s %-9s %s' % ('project', 'category', 'collected', 'days'))
    collected = 0
    for project_id in projects:
        paths = layout.project(project_id)
        manifest = B.read_manifest(paths.manifest)
        if manifest:
            collected += 1
            days = len(manifest.get('days', []))
            mark = 'yes'
        else:
            days, mark = 0, 'no'
        print('%-30s %-10s %-9s %s' % (project_id, states.category(project_id) or '-',
                                       mark, days or '-'))
    print('\n%d of %d collected' % (collected, len(projects)))
    return 0


def cmd_collect(args) -> int:
    layout = _layout(args)
    cloud = Cloud(layout=layout, dry_run=args.dry_run, verbose=args.verbose)
    try:
        states = AnalysisStates.load(layout.analysis(args.analysis_id), cloud,
                                     refresh=args.refresh_states)
    except StatesError as error:
        print(str(error))
        return 1

    projects = _projects(states, args.projects)
    if not projects:
        print('nothing to do')
        return 1
    print('Collecting %d project(s) in %s' % (len(projects), args.analysis_id))

    results = []
    for number, project_id in enumerate(projects, 1):
        print('[%d/%d] %s' % (number, len(projects), project_id))
        if args.raise_errors:
            # deliberately outside the collector's own guard
            from .collect.collector import collect_project as run
            result = run(layout, cloud, project_id, force=args.force,
                         keep_archive=args.keep_archive, upload=not args.no_upload,
                         branch=args.branch)
            if result.status == 'failed':
                raise RuntimeError(result.reason)
        else:
            result = collect_project(layout, cloud, project_id, force=args.force,
                                     keep_archive=args.keep_archive,
                                     upload=not args.no_upload, branch=args.branch)
        print('    ' + result.line())
        results.append(result)

    ok = [r for r in results if r.status == 'ok']
    skipped = [r for r in results if r.status == 'skipped']
    failed = [r for r in results if r.status == 'failed']
    print('\nCollected %d, skipped %d, failed %d.' % (len(ok), len(skipped), len(failed)))
    if failed:
        for result in failed:
            print('  %s: %s' % (result.project_id, result.reason))
    missing = [r for r in ok if r.missing]
    if missing:
        print('Frames absent from the archive:')
        for result in missing:
            print('  %s: %d' % (result.project_id, len(result.missing)))
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='cichlid_bower_claude',
                                     description='Collect and serve cichlid bower data')
    parser.add_argument('--root', help='local working directory; '
                                       'defaults to CICHLID_LOCAL_ROOT')
    parser.add_argument('--verbose', action='store_true', help='show every rclone call')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('analyses', help='list every analysisID, local and cloud')
    p.set_defaults(func=cmd_analyses)

    p = sub.add_parser('status', help='what has been collected')
    p.add_argument('analysis_id')
    p.add_argument('--projects', nargs='+')
    p.set_defaults(func=cmd_status)

    p = sub.add_parser('collect', help='gather what the server needs')
    p.add_argument('analysis_id')
    p.add_argument('--projects', nargs='+')
    p.add_argument('--force', action='store_true',
                   help='recollect even where the manifest is current')
    p.add_argument('--keep-archive', action='store_true',
                   help='leave Frames.tar on disk afterwards')
    p.add_argument('--no-upload', action='store_true',
                   help='collect locally without uploading')
    p.add_argument('--dry-run', action='store_true', help='make no cloud changes')
    p.add_argument('--raise', dest='raise_errors', action='store_true',
                   help='let an exception escape instead of recording it')
    p.add_argument('--branch', default='', help='recorded in the manifest')
    p.add_argument('--refresh-states', action='store_true',
                   help='refetch the states file even if it is already here')
    p.set_defaults(func=cmd_collect)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except LayoutError as error:
        print(str(error))
        return 2


if __name__ == '__main__':
    sys.exit(main())