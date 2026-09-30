"""Write a synthetic logfile matching the real format, for tests."""
import datetime as dt
from pathlib import Path

def write_log(path, days=6, resets=(2, 4), running=False, start=dt.datetime(2025,1,31,14,25)):
    lines = ['MasterStart: System: pi,,Device: realsense,,Camera: pi,,Uname: Linux,,'
             'TankID: t001,,ProjectID: MC_920_t001_tr1,,AnalysisID: YH_MC_Parentals,,SampleID: s1']
    lines.append('MasterRecordInitialStart: Time: ' + str(start))
    t = start
    frame_n, movie_n = 1, 1
    reset_days = set(resets)
    end = start + dt.timedelta(days=days)
    while t < end:
        lof = 8 <= t.hour < 18
        lines.append('FrameCaptured: NpyFile: Frames/Frame_%06d.npy,,PicFile: Frames/Frame_%06d.jpg,,'
                     'Time: %s,,AvgMed: 60.1,,AvgStd: 0.05,,GP: 0.98,,LOF: %s'
                     % (frame_n, frame_n, t, lof))
        frame_n += 1
        if lof and t.hour == 8 and t.minute == 0:
            lines.append('PiCameraStarted: Time: %s,,VideoFile: Videos/%04d_vid.h264,,'
                         'PicFile: Videos/%04d_pic.jpg,,FrameRate: 30,,Resolution: 1296x972'
                         % (t, movie_n, movie_n))
            lines.append('PiCameraStopped: Time: %s,,File: Videos/%04d_vid.h264'
                         % (t + dt.timedelta(hours=10), movie_n))
            movie_n += 1
        day_number = (t - start).days
        if day_number in reset_days and t.hour == 9 and t.minute == 10:
            lines.append('TankResetStart: Time: ' + str(t))
            lines.append('TankResetStop: Time: ' + str(t + dt.timedelta(minutes=80)))
        t += dt.timedelta(minutes=5)
    if not running:
        lines.append('MasterRecordStop: Time: ' + str(end))
    Path(path).write_text('\n'.join(lines) + '\n')
    return path
