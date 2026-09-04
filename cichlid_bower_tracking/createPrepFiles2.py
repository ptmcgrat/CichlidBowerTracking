import argparse, datetime, json, os, subprocess, sys

import numpy as np

from helper_modules.file_manager import FileManager as FM

"""
Build PrepFiles2: properly time-matched image pairs for each trial.

    python createPrepFiles2.py YH_MC_Parentals
    python createPrepFiles2.py YH_MC_Parentals --ProjectIDs MC_920_t001_tr1

The Pi camera only writes a still when a video starts, so the usable pairs are
fixed by the video schedule: for each trial take the first and last Pi still
inside it, then find the depth frame closest in time to each. The existing
PrepFiles pairs the last Pi still with the last daylight depth frame instead,
which can be many hours apart, and the sand moves in between.

The depth stills live in Frames.tar next to the 2.5 MB .npy files, so the
archive runs to gigabytes. Dropbox serves ranged reads, so rather than download
it we seek: a tar member is preceded by a 512-byte header, and because the .npy
payloads are a constant size the offset of any frame can be estimated closely
enough to land a single read on it. Each read costs about three seconds of
latency, so the aim is a handful of reads per target rather than a walk.

Outputs, in PrepFiles2/ and uploaded to the cloud:
    Trial_<n>FirstPi.jpg     Trial_<n>LastPi.jpg
    Trial_<n>FirstDepth.jpg  Trial_<n>LastDepth.jpg
    Trial_<n>FirstDepth.npy  Trial_<n>LastDepth.npy
    pairs.json               what was chosen, and how far apart the two are
    frames_index.json        discovered tar offsets, so reruns skip the search
"""

BLOCK = 512
CHUNK = 6 * 1024 * 1024        # covers >2 frames, so a probe always finds headers
MAX_PROBES = 10


# ------------------------------------------------------------------ tar access

class RangeReader:
    """Reads byte ranges out of a cloud object with rclone."""

    def __init__(self, cloud_path):
        self.cloud_path = cloud_path
        self.requests = 0
        self.bytes = 0

    def size(self):
        out = subprocess.run(['rclone', 'lsjson', self.cloud_path],
                             capture_output=True, encoding='utf-8')
        if out.returncode != 0:
            raise RuntimeError('rclone lsjson failed: ' + out.stderr.strip())
        entries = json.loads(out.stdout)
        if not entries:
            raise FileNotFoundError(self.cloud_path)
        return int(entries[0]['Size'])

    def read(self, offset, count):
        self.requests += 1
        self.bytes += count
        out = subprocess.run(
            ['rclone', 'cat', '--offset', str(offset), '--count', str(count), self.cloud_path],
            capture_output=True)
        if out.returncode != 0:
            raise RuntimeError('rclone cat failed at offset ' + str(offset) + ': ' +
                               out.stderr.decode(errors='replace').strip())
        return out.stdout


def parse_header(block):
    """Return (name, size) for a tar header block, or None if it isn't one."""
    if len(block) < BLOCK or block[257:262] != b'ustar':
        return None
    name = block[0:100].split(b'\0')[0].decode('utf-8', 'replace')
    raw = block[124:136].split(b'\0')[0].strip()
    if not raw:
        return None
    try:
        size = int(raw, 8)
    except ValueError:
        return None
    return name, size


def scan_chunk(data, base):
    """Find every tar header in a chunk. Headers sit on 512-byte boundaries
    measured from the start of the archive, so base must be block aligned."""
    found = []
    for i in range(0, len(data) - BLOCK + 1, BLOCK):
        h = parse_header(data[i:i + BLOCK])
        if h:
            found.append({'offset': base + i, 'name': h[0], 'size': h[1]})
    return found


def frame_number(name):
    """Frames/Frame_000123.jpg -> 123. None for anything else."""
    base = name.split('/')[-1]
    if not base.startswith('Frame_'):
        return None
    stem = base.split('.')[0].replace('Frame_std_', '').replace('Frame_', '')
    return int(stem) if stem.isdigit() else None


class TarIndex:
    """Locates members inside a remote tar without reading it end to end."""

    def __init__(self, reader, cache=None):
        self.reader = reader
        self.total = reader.size()
        self.anchors = {}          # name -> {offset, size}
        if cache:
            self.anchors.update(cache)

    def _probe(self, offset):
        offset = max(0, min(self.total - BLOCK, (offset // BLOCK) * BLOCK))
        count = min(CHUNK, self.total - offset)
        found = scan_chunk(self.reader.read(offset, count), offset)
        for f in found:
            self.anchors[f['name']] = {'offset': f['offset'], 'size': f['size']}
        return found

    def _estimate(self, target_n):
        """Interpolate a byte offset for a frame number from known anchors."""
        pts = []
        for name, a in self.anchors.items():
            n = frame_number(name)
            if n is not None and name.endswith('.jpg'):
                pts.append((n, a['offset']))
        if not pts:
            return 0
        pts.sort()
        if len(pts) == 1:
            n0, o0 = pts[0]
            stride = self.total / max(1, max(p[0] for p in pts))
            return int(o0 + (target_n - n0) * stride)
        below = [p for p in pts if p[0] <= target_n]
        above = [p for p in pts if p[0] > target_n]
        if below and above:
            (n0, o0), (n1, o1) = below[-1], above[0]
        elif below:
            (n0, o0), (n1, o1) = (pts[-2], pts[-1]) if len(pts) > 1 else (pts[0], pts[0])
        else:
            (n0, o0), (n1, o1) = pts[0], pts[1]
        if n1 == n0:
            return o0
        return int(o0 + (target_n - n0) * (o1 - o0) / (n1 - n0))

    def ordered(self, samples=4):
        """Is the archive written in frame order?

        The pipeline tars with `tar -cvf`, which writes in readdir order — that
        is creation order on some filesystems and hash order on others, so this
        has to be measured rather than assumed. Returns (verdict, samples)."""
        seen = []
        for i in range(samples):
            offset = int(self.total * i / samples)
            for f in self._probe(offset):
                n = frame_number(f['name'])
                if n is not None:
                    seen.append((f['offset'], n))
                    break
        if len(seen) < 2:
            return False, seen
        seen.sort()
        verdict = all(seen[i][1] < seen[i + 1][1] for i in range(len(seen) - 1))
        return verdict, seen

    def _anchor_below(self, target_n):
        """Highest known .jpg anchor at or before the target frame.

        Only .jpg members count. Sorted archives put every Frame_std_*.npy after
        all the numbered frames, so a std file with a low frame number sits at a
        high offset and would send the walk far past the target."""
        best = None
        for name, a in self.anchors.items():
            if not name.endswith('.jpg'):
                continue
            n = frame_number(name)
            if n is not None and n <= target_n:
                if best is None or n > best[0]:
                    best = (n, a['offset'])
        return best

    def locate(self, member):
        """Return {offset, size} for a member name, searching if necessary.

        Interpolation gets close but can overshoot, so if the guess misses we
        walk forward in chunks from the nearest known header before the target.
        Every chunk advances past at least one member, so this terminates."""
        if member in self.anchors:
            return self.anchors[member]

        target_n = frame_number(member)
        if target_n is None:
            raise KeyError(member + ' is not a frame file')

        if not self.anchors:
            self._probe(0)
            if member in self.anchors:
                return self.anchors[member]

        # one interpolated probe, centred so an overshoot still brackets the target
        guess = self._estimate(target_n)
        self._probe(guess - CHUNK // 2)
        if member in self.anchors:
            return self.anchors[member]

        below = self._anchor_below(target_n)
        cursor = below[1] if below else 0
        for _ in range(MAX_PROBES):
            found = self._probe(cursor)
            if member in self.anchors:
                return self.anchors[member]
            if not found:
                cursor += CHUNK
            else:
                last = max(f['offset'] for f in found)
                cursor = last + BLOCK if last <= cursor else last
            if cursor >= self.total:
                break
        raise KeyError('could not locate ' + member + ' after ' + str(MAX_PROBES) + ' probes')

    def extract(self, member, dest):
        a = self.locate(member)
        data = self.reader.read(a['offset'] + BLOCK, a['size'])
        if len(data) != a['size']:
            raise RuntimeError('short read for ' + member)
        with open(dest, 'wb') as f:
            f.write(data)
        return a

    def cache(self):
        return self.anchors


def fetch_optional(cloud_path, local_path, directory=False):
    """Fetch something that may not be there yet.

    FileManager.downloadData runs `rclone lsf` on the parent directory first and
    drops into pdb.set_trace() if that fails, and allow_errors does not cover it
    — so a path whose parent does not exist in the cloud yet (PrepFiles2 on a
    first run) hangs at a debugger prompt. rclone copy just returns non-zero."""
    cmd = ['rclone', 'copy' if directory else 'copyto', cloud_path, local_path]
    out = subprocess.run(cmd, capture_output=True, encoding='utf-8')
    return out.returncode == 0 and os.path.exists(local_path)


def index_local_tar(path):
    """Walk a local archive once and record every member's offset and size.
    tarfile seeks between headers, so this is fast even on a huge file."""
    import tarfile
    index = {}
    with tarfile.open(path, 'r') as t:
        while True:
            m = t.next()
            if m is None:
                break
            if m.isfile():
                index[m.name] = {'offset': m.offset, 'size': m.size}
    return index


def extract_local(path, members, dests):
    """Pull specific members straight out of a local archive."""
    import tarfile
    with tarfile.open(path, 'r') as t:
        for member, dest in zip(members, dests):
            try:
                src = t.extractfile(member)
            except KeyError:
                src = None
            if src is None:
                raise FileNotFoundError(member + ' not found in ' + os.path.basename(path))
            with open(dest, 'wb') as f:
                f.write(src.read())


def stream_extract(cloud_path, members, dest_dir):
    """Fallback: pipe the whole archive through tar, keeping only what we want.
    Costs full bandwidth but never writes the archive to disk."""
    cat = subprocess.Popen(['rclone', 'cat', cloud_path], stdout=subprocess.PIPE)
    tar = subprocess.Popen(['tar', '-x', '-C', dest_dir] + list(members),
                           stdin=cat.stdout, stderr=subprocess.PIPE)
    cat.stdout.close()
    _, err = tar.communicate()
    cat.wait()
    if tar.returncode != 0:
        raise RuntimeError('stream extract failed: ' + err.decode(errors='replace'))


# --------------------------------------------------------------- pair building

def choose_pairs(lp, trial, index):
    """Pick the first and last Pi still inside the trial, and the depth frame
    nearest each in time."""
    movies = [m for m in trial.movies if m.startTime >= trial.startTime]
    if not movies:
        raise ValueError('trial ' + str(index) + ' has no video starting inside it '
                         '(' + str(trial.startTime) + ' to ' + str(trial.stopTime) + ')')

    frames = trial.daylight_frames or trial.frames
    if not frames:
        raise ValueError('trial ' + str(index) + ' has no depth frames')

    out = {}
    for label, movie in [('First', movies[0]), ('Last', movies[-1])]:
        nearest = min(frames, key=lambda f: abs((f.time - movie.startTime).total_seconds()))
        gap = abs((nearest.time - movie.startTime).total_seconds()) / 60.0
        out[label] = {
            'piFile': movie.pic_file, 'piTime': str(movie.startTime), 'movieIndex': movie.index,
            'depthPic': nearest.pic_file, 'depthNpy': nearest.npy_file,
            'depthTime': str(nearest.time), 'frameIndex': nearest.index,
            'lightsOn': bool(nearest.lof), 'gapMinutes': round(gap, 2),
        }
    return out


def build_project(fm_obj, projectID, args):
    fm_obj.setProjectID(projectID)
    lp = fm_obj.lp
    out_dir = fm_obj.localProjectDir + 'PrepFiles2/'
    fm_obj.createDirectory(out_dir)

    plan = {}
    for i, trial in enumerate(lp.trials, 1):
        plan[i] = choose_pairs(lp, trial, i)

    for i, sides in plan.items():
        for label, p in sides.items():
            flag = '' if p['gapMinutes'] <= 15 else '   <-- wide'
            print('    trial ' + str(i) + ' ' + label.lower() + ': ' +
                  p['piFile'].split('/')[-1] + ' + ' + p['depthPic'].split('/')[-1] +
                  ', ' + str(p['gapMinutes']) + ' min apart' + flag)

    if args.DryRun:
        return None

    # Pi stills are ordinary small files in Videos/, no tar involved
    for i, sides in plan.items():
        for label, p in sides.items():
            src = fm_obj.localProjectDir + p['piFile']
            fm_obj.downloadData(src)
            if not os.path.exists(src):
                raise FileNotFoundError(p['piFile'] + ' is not in the cloud')
            with open(src, 'rb') as f_in, \
                 open(out_dir + 'Trial_' + str(i) + label + 'Pi.jpg', 'wb') as f_out:
                f_out.write(f_in.read())

    wanted = []
    for i, sides in plan.items():
        for label, p in sides.items():
            wanted.append((p['depthPic'], out_dir + 'Trial_' + str(i) + label + 'Depth.jpg'))
            wanted.append((p['depthNpy'], out_dir + 'Trial_' + str(i) + label + 'Depth.npy'))

    cloud_tar = fm_obj.localFrameTarredDir.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir)
    index_path = out_dir + 'frames_index.json'
    members = [m for m, _ in wanted]
    dests = [d for _, d in wanted]

    # A cached index makes ordering irrelevant: exact offsets, one ranged read each.
    cache = {}
    if not args.Rebuild:
        cloud_index = index_path.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir)
        fetch_optional(cloud_index, index_path)
        if os.path.exists(index_path):
            try:
                with open(index_path) as f:
                    cache = json.load(f)
            except Exception:
                cache = {}
    have_all = cache and all(m in cache for m in members)

    if args.Stream:
        stream_extract(cloud_tar, members, fm_obj.localProjectDir)
        for member, dest in wanted:
            src = fm_obj.localProjectDir + member
            if not os.path.exists(src):
                raise FileNotFoundError(member + ' not found in Frames.tar')
            os.replace(src, dest)

    elif have_all:
        reader = RangeReader(cloud_tar)
        index = TarIndex(reader, cache)
        started = datetime.datetime.now()
        for member, dest in wanted:
            index.extract(member, dest)
        print('    cached index: ' + str(len(wanted)) + ' members in ' +
              str(reader.requests) + ' requests, ' + str(round(reader.bytes / 1e6, 1)) +
              ' MB, ' + str(round((datetime.datetime.now() - started).total_seconds())) + 's')

    else:
        reader = RangeReader(cloud_tar)
        index = TarIndex(reader, cache)
        is_ordered, samples = index.ordered()
        sample_text = ', '.join(['frame ' + str(n) + ' at ' + str(round(o / 1e9, 2)) + ' GB'
                                 for o, n in samples])
        if is_ordered:
            print('    archive is in frame order (' + sample_text + '), seeking')
            started = datetime.datetime.now()
            for member, dest in wanted:
                index.extract(member, dest)
            with open(index_path, 'w') as f:
                json.dump(index.cache(), f, indent=1)
            print('    ' + str(len(wanted)) + ' members in ' + str(reader.requests) +
                  ' requests, ' + str(round(reader.bytes / 1e6, 1)) + ' MB, ' +
                  str(round((datetime.datetime.now() - started).total_seconds())) + 's')
        else:
            # readdir order: nothing can be found without reading the whole archive,
            # so read it once, index it completely, and never pay this again.
            print('    archive is not in frame order (' + sample_text + ')')
            print('    downloading ' + str(round(index.total / 1e9, 2)) +
                  ' GB once to build an index', flush=True)
            started = datetime.datetime.now()
            tar_path = fm_obj.localFrameTarredDir
            fm_obj.downloadData(tar_path)
            if not os.path.exists(tar_path):
                raise FileNotFoundError('Frames.tar did not download')
            full = index_local_tar(tar_path)
            extract_local(tar_path, members, dests)
            with open(index_path, 'w') as f:
                json.dump(full, f, indent=1)
            print('    indexed ' + str(len(full)) + ' members in ' +
                  str(round((datetime.datetime.now() - started).total_seconds())) + 's')
            if not args.KeepTar:
                os.remove(tar_path)

    manifest = {
        'schema': 'cichlid-prepfiles2/1',
        'projectID': projectID, 'analysisID': fm_obj.analysisID, 'tankID': lp.tankID,
        'built': str(datetime.datetime.now().replace(microsecond=0)),
        'branch': fm_obj.branch_name,
        'trials': {str(i): sides for i, sides in plan.items()},
    }
    with open(out_dir + 'pairs.json', 'w') as f:
        json.dump(manifest, f, indent=1)

    if not args.NoUpload:
        fm_obj.uploadData(out_dir.rstrip('/'))
        print('    uploaded -> ' +
              out_dir.replace(fm_obj.localMasterDir, fm_obj.cloudMasterDir))

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description='Build PrepFiles2, time-matched Pi and depth stills for each trial')
    parser.add_argument('AnalysisID', type=str)
    parser.add_argument('--ProjectIDs', type=str, nargs='+',
                        help='Restrict to these projectIDs')
    parser.add_argument('--DryRun', action='store_true',
                        help='Report the chosen pairs without downloading anything')
    parser.add_argument('--Stream', action='store_true',
                        help='Pipe the archive through tar (fragile on large files)')
    parser.add_argument('--KeepTar', action='store_true',
                        help='Keep Frames.tar on disk after indexing it')
    parser.add_argument('--Rebuild', action='store_true',
                        help='Ignore any cached frames_index.json and rebuild it')
    parser.add_argument('--SkipExisting', action='store_true',
                        help='Leave projects that already have a PrepFiles2/pairs.json')
    parser.add_argument('--NoUpload', action='store_true')
    args = parser.parse_args()

    fm_obj = FM(args.AnalysisID)
    s_dt = fm_obj.s_dt
    projectIDs = s_dt[(s_dt.Prep == True) & (s_dt.RunAnalysis == True)].index.sort_values().to_list()
    if args.ProjectIDs:
        unknown = [p for p in args.ProjectIDs if p not in s_dt.index]
        if unknown:
            print('Unknown ProjectIDs: ' + ', '.join(unknown))
            return 1
        projectIDs = [p for p in args.ProjectIDs if p in projectIDs]
    if not projectIDs:
        print('No projects with Prep complete in ' + args.AnalysisID)
        return 1

    print('Building PrepFiles2 for ' + str(len(projectIDs)) + ' project(s)')
    failed = []
    for i, projectID in enumerate(projectIDs, 1):
        print('[' + str(i) + '/' + str(len(projectIDs)) + '] ' + projectID, flush=True)
        if args.SkipExisting and os.path.exists(
                fm_obj.localMasterDir + '__ProjectData/' + args.AnalysisID + '/' +
                projectID + '/PrepFiles2/pairs.json'):
            print('    already built, skipping')
            continue
        try:
            build_project(fm_obj, projectID, args)
        except Exception as e:
            print('    FAILED: ' + str(e))
            failed.append(projectID)

    print('\nDone. ' + str(len(projectIDs) - len(failed)) + ' built, ' +
          str(len(failed)) + ' failed.')
    if failed:
        print('Failed: ' + ', '.join(failed))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())