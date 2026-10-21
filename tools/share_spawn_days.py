#!/usr/bin/env python
"""Pick the best spawning days in an analysis and copy that day's material to
a share anyone outside the lab can be given.

    python tools/share_spawn_days.py YH_MC_Parentals
    python tools/share_spawn_days.py YH_MC_Parentals --per-category 2 --go

Dry run unless --go is passed, because the destination is visible outside the
lab and a wrong path there is not a private mistake.

What counts as a spawning day: events the classifier called 's', that actually
clustered (LID is not -1, so a bare transition is never counted as a spawn),
and whose probability clears --confidence. Those are grouped into the day whose
lights-on window contains them, using the same windows the pages use, so a
number here and a number on the cluster page agree.

Copies are server-side where the source and destination are the same rclone
remote: nothing is downloaded to mzebra, so a day of video moves in seconds
rather than an hour.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cichlid_bower_claude.cloud import Cloud, CloudError          # noqa: E402
from cichlid_bower_claude.collect import bundle as B              # noqa: E402
from cichlid_bower_claude.collect.clusters import (                # noqa: E402
    ClustersMissing, read_clusters)
from cichlid_bower_claude.logfile.parse import parse_logfile      # noqa: E402
from cichlid_bower_claude.paths import Layout                     # noqa: E402
from cichlid_bower_claude.states import AnalysisStates            # noqa: E402

SPAWN = 's'

DEFAULT_DEST = 'ptm_dropbox:/CoS-EXT/BioSci-Ext/BioSci-McGrath-Ext/MorePoseData'

# One parquet per video, named by the video's stem: Videos/0001_vid.mp4 goes
# with PoseData/days/0001_vid.parquet.
POSE_SUBDIR = 'PoseData/days'


@dataclass
class Day:
    project_id: str
    category: str
    index: int
    trial: int
    first: dt.datetime
    last: dt.datetime
    spawns: int = 0
    times: List[dt.datetime] = field(default_factory=list)

    @property
    def label(self) -> str:
        return '%s_day%02d' % (self.project_id, self.index)

    @property
    def date(self) -> str:
        return self.first.date().isoformat()


def _spawn_days(layout: Layout, cloud: Cloud, states: AnalysisStates,
                analysis_id: str, confidence: float,
                verbose: bool) -> List[Day]:
    """Every day in the analysis, with its spawn count."""
    days: List[Day] = []
    for project_id in states.project_ids():
        paths = layout.project(project_id, analysis_id)
        manifest = B.read_manifest(paths.manifest)
        if not manifest:
            print('  %-32s not collected, skipped' % project_id)
            continue

        if not paths.clusters_csv.exists():
            cloud.download_optional(paths.clusters_csv)
        try:
            table = read_clusters(paths.clusters_csv)
        except ClustersMissing:
            print('  %-32s no cluster file, skipped' % project_id)
            continue

        keep = table['Prediction'] == SPAWN
        if 'LID' in table.columns:
            # a transition that never joined a cluster is not an event
            keep &= table['LID'].astype('Int64').fillna(-1) != -1
        if 'Probability' in table.columns:
            keep &= table['Probability'].fillna(0) >= confidence
        spawns = table[keep]

        category = states.category(project_id) or 'uncategorised'
        found = []
        for entry in manifest.get('days', []):
            day = Day(project_id=project_id, category=category,
                      index=int(entry['index']), trial=int(entry['trial']),
                      first=dt.datetime.fromisoformat(entry['first']),
                      last=dt.datetime.fromisoformat(entry['last']))
            inside = spawns.index[(spawns.index >= day.first)
                                 & (spawns.index <= day.last)]
            day.spawns = len(inside)
            day.times = [t.to_pydatetime() for t in inside]
            found.append(day)
        days.extend(found)
        if verbose:
            best = max((d.spawns for d in found), default=0)
            print('  %-32s %-10s %3d days, %4d spawns, best day %d'
                  % (project_id, category, len(found), len(spawns), best))
    return days


def _choose(days: List[Day], per_category: int,
            one_per_project: bool) -> Dict[str, List[Day]]:
    """The best days in each category, by default from different projects.

    Two days from one tank are two looks at the same fish, which is rarely
    what you want when the point is to show what a category looks like.
    """
    by_category: Dict[str, List[Day]] = defaultdict(list)
    for day in days:
        if day.spawns:
            by_category[day.category].append(day)

    chosen: Dict[str, List[Day]] = {}
    for category, candidates in sorted(by_category.items()):
        candidates.sort(key=lambda d: (-d.spawns, d.project_id, d.index))
        picked: List[Day] = []
        used = set()
        for day in candidates:
            if one_per_project and day.project_id in used:
                continue
            picked.append(day)
            used.add(day.project_id)
            if len(picked) == per_category:
                break
        # fewer projects than asked for: fall back to more days from the same
        # project rather than returning a short list silently
        if len(picked) < per_category and one_per_project:
            for day in candidates:
                if day not in picked:
                    picked.append(day)
                if len(picked) == per_category:
                    break
        chosen[category] = picked
    return chosen


def _stops(log) -> Dict[str, dt.datetime]:
    """When each recording actually ended.

    The log has no PiCameraStopped line for a recording that died -- a crash, a
    power cut, a Pi restart -- so Movie.stop_time is None for it. Treating an
    unknown end as 'still going' makes that recording overlap every later day:
    a video truncated in week one was being offered as material for day 38.

    The Pi records sequentially, so the next recording's start is when this one
    stopped, give or take the gap. The last one falls back to the end of the
    project.
    """
    stops: Dict[str, dt.datetime] = {}
    movies = sorted(log.movies, key=lambda m: m.start_time)
    for position, movie in enumerate(movies):
        if movie.stop_time is not None:
            stops[movie.h264_file] = movie.stop_time
        elif position + 1 < len(movies):
            stops[movie.h264_file] = movies[position + 1].start_time
        elif log.frames:
            stops[movie.h264_file] = log.frames[-1].time
        else:
            stops[movie.h264_file] = movie.start_time
    return stops


def _videos_for(log, day: Day) -> List[str]:
    """Project-relative mp4 names overlapping the day's window.

    Overlap, not 'started inside': a recording that began before the window
    still holds the first part of the day. The log's own movies_between is
    deliberately stricter, for a different question.
    """
    stops = _stops(log)
    out = []
    for movie in sorted(log.movies, key=lambda m: m.start_time):
        if movie.start_time <= day.last and stops[movie.h264_file] >= day.first:
            out.append(movie.mp4_file)
    return out


def _inferred(log) -> List[str]:
    """Recordings whose end had to be guessed, worth saying out loud."""
    return [m.mp4_file for m in log.movies if m.stop_time is None]


def _pose_dirs(layout: Layout, paths, analysis_id: str,
               extra: List[str]) -> List[Path]:
    """Where this project's parquet files might be.

    Two shapes, because they are named inconsistently: PoseData sits under the
    project the way everything else does, and it has also been seen under a
    flat directory joining the analysis and project with an underscore
    (YH_MC_Parentals_MC_959_t011_tr1/PoseData/days). Both are tried and the one
    that answered is reported, rather than a guess that silently finds nothing.
    """
    flat = '%s_%s' % (analysis_id, paths.project_id)
    found = [paths.root / POSE_SUBDIR,
             layout.local_root / flat / POSE_SUBDIR,
             layout.local_root / '__ProjectData' / flat / POSE_SUBDIR]
    return found + [Path(p) if Path(p).is_absolute() else paths.root / p
                    for p in extra]


def _parquet_for(cloud: Cloud, directories: List[Path],
                 stems: List[str]) -> tuple:
    """The parquet matching each video, as (absolute local path, name) pairs.

    Matched on the exact stem, not a substring: a stray '0001_vid_old.parquet'
    beside the real one would otherwise be published too.
    """
    wanted = set(stems)
    found, searched, extra = [], [], []
    for directory in directories:
        try:
            names = cloud.listdir(directory, files_only=True)
        except CloudError:
            continue
        searched.append(str(directory))
        if not names:
            continue
        for name in sorted(names):
            if not name.endswith('.parquet'):
                continue
            if Path(name).stem in wanted:
                found.append(directory / name)
            else:
                extra.append(name)
        if found:
            break          # the first directory that answers is the right one
    return found, searched, extra


def _copy(cloud: Cloud, source_local: Path, destination: str,
          go: bool) -> Optional[int]:
    """Server-side copy from the lab share to the external one."""
    source = cloud.layout.cloud(source_local)
    size = cloud.size(source_local)
    if not go:
        return size
    result = subprocess.run(['rclone', 'copyto', source, destination,
                             '--no-traverse'],
                            capture_output=True, encoding='utf-8')
    if result.returncode != 0:
        print('      FAILED: ' + (result.stderr or '').strip())
        return None
    return size


def _check_destination(dest: str) -> bool:
    result = subprocess.run(['rclone', 'lsd', dest], capture_output=True,
                            encoding='utf-8')
    if result.returncode == 0:
        return True
    print('The destination is not reachable:')
    print('  ' + (result.stderr or '').strip())
    print('\nCheck it with: rclone lsd ' + dest)
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('analysis_id')
    parser.add_argument('--dest', default=DEFAULT_DEST,
                        help='rclone address of the external share '
                             '(default: %(default)s)')
    parser.add_argument('--per-category', type=int, default=2)
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='minimum classifier probability (default 0.5)')
    parser.add_argument('--same-project-ok', action='store_true',
                        help='allow both days of a category to come from one project')
    parser.add_argument('--parquet-dir', action='append', default=[],
                        help='another directory to search for parquet, '
                             'project-relative or absolute; repeatable')
    parser.add_argument('--root', help='defaults to CICHLID_LOCAL_ROOT')
    parser.add_argument('--go', action='store_true',
                        help='actually copy; without it nothing is written')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    layout = (Layout(local_root=Path(args.root).expanduser()) if args.root
              else Layout.from_env())
    cloud = Cloud(layout=layout)
    print('Working directory: %s' % layout.local_root)
    print('Destination:       %s' % args.dest)
    print('Mode:              %s\n' % ('COPYING' if args.go else 'dry run'))

    if args.go and not _check_destination(args.dest):
        return 2

    states = AnalysisStates.load(layout.analysis(args.analysis_id), cloud)
    print('Counting spawns in %s' % args.analysis_id)
    days = _spawn_days(layout, cloud, states, args.analysis_id,
                       args.confidence, not args.quiet)
    if not days:
        print('\nNo days with cluster data. Nothing to do.')
        return 1

    chosen = _choose(days, args.per_category, not args.same_project_ok)
    if not any(chosen.values()):
        print('\nNo day in any category has a spawn above the confidence '
              'threshold. Try --confidence lower than %.2f.' % args.confidence)
        return 1

    print('\nChosen days')
    print('%-12s %-30s %5s %7s  %s' % ('category', 'project', 'day', 'spawns',
                                        'date'))
    for category, picked in chosen.items():
        for day in picked:
            print('%-12s %-30s %5d %7d  %s'
                  % (category, day.project_id, day.index, day.spawns, day.date))

    rows = []
    total = 0
    print('\nMaterial')
    for category, picked in chosen.items():
        for day in picked:
            paths = layout.project(day.project_id, args.analysis_id)
            if not paths.logfile.exists():
                cloud.download_optional(paths.logfile)
            log = parse_logfile(paths.logfile)
            videos = [paths.root / v for v in _videos_for(log, day)]
            stems = [v.stem for v in videos]
            parquet, searched, ignored = _parquet_for(
                cloud, _pose_dirs(layout, paths, args.analysis_id,
                                  args.parquet_dir), stems)

            target = '%s/%s/%s' % (args.dest.rstrip('/'), category, day.label)
            print('  %s  ->  %s' % (day.label, target))
            if not videos:
                print('      no video overlaps this day in the logfile')
            guessed = [g for g in _inferred(log)
                       if g in {str(v.relative_to(paths.root)) for v in videos}]
            if guessed:
                print('      end time not logged, inferred from the next '
                      'recording: ' + ', '.join(Path(g).name for g in guessed))
            if not parquet:
                print('      no parquet matched; looked in:')
                for place in (searched or ['nothing readable']):
                    print('        ' + place)
                if ignored:
                    print('      that directory holds parquet with other '
                          'names, e.g. ' + ', '.join(ignored[:3]))
            missing = sorted(set(stems) - {p.stem for p in parquet})
            if parquet and missing:
                print('      no parquet for: ' + ', '.join(missing))

            for source in videos + parquet:
                size = _copy(cloud, source,
                             target + '/' + source.name, args.go)
                if size is None and args.go:
                    continue
                total += size or 0
                print('      %-40s %8.1f MB' % (source.name,
                                                 (size or 0) / 1e6))
                rows.append({'category': category, 'projectID': day.project_id,
                             'day': day.index, 'trial': day.trial,
                             'date': day.date, 'spawns': day.spawns,
                             'file': source.name,
                             'bytes': size or 0,
                             'spawnTimes': ' '.join(t.strftime('%H:%M:%S')
                                                    for t in day.times)})

    index = Path('shared_spawn_days.csv')
    if rows:
        with open(index, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print('\nWrote %s (%d files, %.1f GB)' % (index, len(rows), total / 1e9))
        if args.go:
            _copy_file(index, args.dest.rstrip('/') + '/' + index.name)

    if not args.go:
        print('\nDry run — nothing was copied. Add --go to do it.')
    return 0


def _copy_file(local: Path, destination: str) -> None:
    """The index CSV is local, so it uploads rather than copying server-side."""
    result = subprocess.run(['rclone', 'copyto', str(local), destination],
                            capture_output=True, encoding='utf-8')
    if result.returncode != 0:
        print('Could not upload the index: ' + (result.stderr or '').strip())
    else:
        print('Index copied to the share, so recipients know what they have.')


if __name__ == '__main__':
    sys.exit(main())
