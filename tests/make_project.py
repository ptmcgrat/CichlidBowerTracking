"""Build a fake project in a fake cloud: logfile, Frames.tar, the lot."""
import io, tarfile, datetime as dt
from pathlib import Path
import numpy as np
from tests.make_log import write_log

def build(remote_root, project_id='MC_920_t001_tr1', days=6, resets=(2,4),
          H=48, W=64, with_std=True, drop_frames=(), analysis_id='YH_MC_Parentals'):
    proj = Path(remote_root)/'__ProjectData'/analysis_id/project_id
    proj.mkdir(parents=True, exist_ok=True)
    write_log(proj/'Logfile.txt', days=days, resets=resets)
    import sys; sys.path.insert(0,'/home/claude/cbc')
    from cichlid_bower_claude.logfile.parse import parse_logfile
    log = parse_logfile(proj/'Logfile.txt')
    rng = np.random.default_rng(4)
    yy,xx = np.mgrid[0:H,0:W]
    castle = np.exp(-(((xx-W*0.65)/8)**2 + ((yy-H*0.5)/7)**2))
    bad = ((xx-W*0.3)**2 + (yy-H*0.35)**2) < 16      # a churning patch
    span = (log.frames[-1].time - log.frames[0].time).total_seconds()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w') as tar:
        for f in log.frames:
            if f.index in drop_frames:
                continue
            frac = (f.time - log.frames[0].time).total_seconds()/span
            surface = 60 - 3*frac*castle + rng.normal(0, 0.02, (H,W))
            surface[bad] += rng.normal(0, 1.2, int(bad.sum()))
            surface[:3] = np.nan                       # a permanently dead strip
            for name, data in ((f.npy_file, surface),):
                raw = io.BytesIO(); np.save(raw, data.astype(np.float32))
                info = tarfile.TarInfo(name); info.size = raw.tell()
                raw.seek(0); tar.addfile(info, raw)
            jpg = io.BytesIO(b'\xff\xd8\xff' + bytes(200))
            info = tarfile.TarInfo(f.pic_file); info.size = len(jpg.getvalue())
            jpg.seek(0); tar.addfile(info, jpg)
            if with_std:
                raw = io.BytesIO()
                np.save(raw, np.full((H,W), 0.05, np.float32))
                info = tarfile.TarInfo(f.std_file); info.size = raw.tell()
                raw.seek(0); tar.addfile(info, raw)
    (proj/'Frames.tar').write_bytes(buf.getvalue())
    videos = proj/'Videos'
    videos.mkdir(exist_ok=True)
    for m in log.movies:
        (videos/Path(m.pic_file).name).write_bytes(b'\xff\xd8\xff' + bytes(300))
    return project_id, log
