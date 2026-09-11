"""
Daily endpoints bundle for the depth viewer.

Drop this into cichlid_bower_tracking/helper_modules/depth_endpoints.py and call
saveDailyEndpoints() from DepthPreparer.createSmoothedArray, right after
smoothDepthData is built and while rawDepthData is still in memory.

The viewer needs two frames per day, not the whole array: for a 14-day project
that is 28 frames instead of 8002, about 4 MB instead of 19.7 GB. Both the raw
and the smoothed frame are stored at the same indices so the pair shows exactly
what the interpolation did.
"""

import datetime
import json
import os

import numpy as np


def saveDailyEndpoints(fileManager, lp, rawDepthData, smoothDepthData,
                       out_dir=None, dtype=np.float16):
    """Write DepthFiles/daily_endpoints.npz and daily_endpoints.json.

    fileManager      the FileManager, for localProjectDir
    lp               fileManager.lp
    rawDepthData     (frames, H, W) before interpolation
    smoothDepthData  (frames, H, W) after
    out_dir          defaults to localProjectDir + 'DepthFiles/'

    Returns the path of the npz.
    """
    if out_dir is None:
        out_dir = fileManager.localProjectDir + 'DepthFiles/'
    os.makedirs(out_dir, exist_ok=True)

    raw_days, smooth_days, meta = [], [], []

    for t_num, trial in enumerate(lp.trials, 1):
        for start_f, stop_f in trial.days:
            # the lights-on frames of this day, by index into axis 0
            idx = [f.index for f in lp.frames
                   if start_f.index <= f.index <= stop_f.index and f.lof]
            if not idx:
                continue
            first, last = idx[0], idx[-1]

            # same indices for both arrays, so the pair is comparable
            raw_days.append(np.stack([rawDepthData[first], rawDepthData[last]]))
            smooth_days.append(np.stack([smoothDepthData[first], smoothDepthData[last]]))

            meta.append({
                'day': len(meta),
                'trial': t_num,
                'first_index': int(first),
                'last_index': int(last),
                'first_time': str(lp.frames[first].time),
                'last_time': str(lp.frames[last].time),
                'n_frames': len(idx),
                'raw_valid_first': float(np.isfinite(rawDepthData[first]).mean()),
                'raw_valid_last': float(np.isfinite(rawDepthData[last]).mean()),
            })

    if not meta:
        raise ValueError('no lights-on days found for ' + lp.projectID)

    raw_arr = np.stack(raw_days).astype(dtype)
    smooth_arr = np.stack(smooth_days).astype(dtype)

    npz_path = out_dir + 'daily_endpoints.npz'
    np.savez_compressed(npz_path, raw=raw_arr, smooth=smooth_arr)

    with open(out_dir + 'daily_endpoints.json', 'w') as f:
        json.dump({
            'schema': 'cichlid-depth-endpoints/1',
            'projectID': lp.projectID,
            'tankID': lp.tankID,
            'analysisID': fileManager.analysisID,
            'shape': list(raw_arr.shape),          # (n_days, 2, H, W)
            'dtype': str(dtype(0).dtype),
            'frameSize': [int(lp.width), int(lp.height)],
            'nTrials': len(lp.trials),
            'built': str(datetime.datetime.now().replace(microsecond=0)),
            'days': meta,
        }, f, indent=1)

    return npz_path