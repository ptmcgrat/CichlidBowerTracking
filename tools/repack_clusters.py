#!/usr/bin/env python
"""Rebuild clusters.json for projects that are already collected.

    python tools/repack_clusters.py YHMC_BC_Data
    python tools/repack_clusters.py YHMC_BC_Data --go
    python tools/repack_clusters.py YHMC_BC_Data --project MCYHBC1-P-569_t035_tr1 --go

Why this exists: adding a column to the packed cluster file would otherwise
mean re-collecting every project, and collection re-downloads the archive. The
cluster CSV and the manifest are both already on disk once a project has been
collected, and they are the only two inputs ``pack`` needs -- so the packed
file can be rebuilt in seconds without touching the archive.

The bundle schema is deliberately NOT bumped for this. Bumping it would force
a full rebuild of exactly the thing this tool exists to avoid. What that costs
is that a project nobody repacks keeps its old clusters.json, which is why the
cluster page treats a missing column as 'not collected' rather than as an
empty one, and says so.

Dry run unless --go is passed. The manifest is rewritten last, after the
packed file it describes has uploaded, for the same reason the collector does
it in that order: presence of a file is not a completion marker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cichlid_bower_claude.cloud import Cloud, CloudError           # noqa: E402
from cichlid_bower_claude.collect import bundle as B               # noqa: E402
from cichlid_bower_claude.collect.clusters import (                # noqa: E402
    ClustersMissing, pack, read_clusters, summarise)
from cichlid_bower_claude.paths import Layout, LayoutError         # noqa: E402
from cichlid_bower_claude.states import AnalysisStates, local_analyses  # noqa: E402


class Skip(Exception):
    """This project cannot be repacked, with the reason."""


def repack(paths, go: bool, cloud: Optional[Cloud]) -> dict:
    """Rebuild one project's packed clusters. Returns what changed."""
    manifest = B.read_manifest(paths.manifest)
    if not manifest:
        raise Skip('not collected yet (no manifest)')
    if 'days' not in manifest:
        raise Skip('manifest has no days')

    try:
        table = read_clusters(paths.clusters_csv)
    except ClustersMissing as error:
        raise Skip(str(error))

    summary = summarise(table)
    packed = pack(table, manifest['days'])

    before = None
    if paths.clusters_packed.exists():
        try:
            with open(paths.clusters_packed) as handle:
                before = json.load(handle)
        except Exception:
            before = None

    change = {
        'events': packed['n'],
        'had_sex': bool(before and 'sex' in before),
        'has_sex': 'sex' in packed,
        'by_sex': summary.get('bySex'),
    }
    if not go:
        return change

    with open(paths.clusters_packed, 'w') as handle:
        json.dump(packed, handle)
    if cloud:
        cloud.upload(paths.clusters_packed)

    # the manifest last, and only after the file it names is up
    manifest['clusters'] = summary
    B.write_manifest(paths.manifest, manifest)
    if cloud:
        cloud.upload(paths.manifest)
    return change


def projects_in(layout: Layout, analysis_id: str, only: List[str],
                cloud: Optional[Cloud]) -> List[str]:
    if only:
        return only
    states = AnalysisStates.load(layout.analysis(analysis_id), cloud)
    return list(states.table.index)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='rebuild clusters.json from the cluster CSV already on disk')
    parser.add_argument('analysis_id', nargs='?',
                        help='the analysis to repack; omit to list what is here')
    parser.add_argument('--project', action='append', default=[],
                        help='one project, repeatable; default is all of them')
    parser.add_argument('--root', help='local working directory')
    parser.add_argument('--no-upload', action='store_true',
                        help='write locally and do not push to the cloud')
    parser.add_argument('--go', action='store_true',
                        help='actually write; without it nothing is changed')
    args = parser.parse_args()

    try:
        layout = (Layout(local_root=Path(args.root).expanduser()) if args.root
                  else Layout.from_env())
    except LayoutError as error:
        print(str(error))
        return 2

    if not args.analysis_id:
        here = local_analyses(layout)
        print('Analyses in %s:' % layout.local_root)
        for name in here:
            print('  ' + name)
        if not here:
            print('  (none)')
        return 0

    cloud = None if args.no_upload else Cloud(layout)
    print('Working directory: %s' % layout.local_root)
    print('Analysis: %s' % args.analysis_id)
    if not args.go:
        print('DRY RUN -- pass --go to write. Nothing below is being changed.')
    print('')

    done = skipped = failed = 0
    for project_id in projects_in(layout, args.analysis_id, args.project,
                                  cloud):
        paths = layout.project(project_id, args.analysis_id)
        try:
            change = repack(paths, args.go, cloud)
        except Skip as error:
            print('  %-42s skip: %s' % (project_id, error))
            skipped += 1
            continue
        except CloudError as error:
            print('  %-42s FAILED: %s' % (project_id, error))
            failed += 1
            continue

        note = '%d events' % change['events']
        if change['has_sex'] and not change['had_sex']:
            note += ', sex added'
        elif change['has_sex']:
            note += ', sex present'
        else:
            note += ', NO Sex column in the CSV'
        if change['by_sex']:
            note += ' (%s)' % ', '.join(
                '%s %d' % (k, v) for k, v in sorted(change['by_sex'].items()))
        print('  %-42s %s' % (project_id, note))
        done += 1

    print('')
    print('%d repacked, %d skipped, %d failed%s'
          % (done, skipped, failed, '' if args.go else ' (dry run)'))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
