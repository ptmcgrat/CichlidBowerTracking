"""
Trial-level masking of churning pixels.

Save as cichlid_bower_tracking/helper_modules/depth_masking.py

Some pixels move constantly without going anywhere — specular highlights off a
rippling water surface are the usual cause. They are not building, and because
the water surface sits closer to the sensor than the sand they bias in one
direction, so averaging does not cancel them.

Travel — the summed absolute frame-to-frame movement — separates them cleanly.
Travel from sensor noise grows linearly with frame count, so dividing by
lights-on hours gives a rate that is comparable across trials of any length.
Subtracting |net change| removes the part of the movement that actually went
somewhere, so a pixel that genuinely built a lot can never be flagged.

Measured on simulated pixels over a 20-day trial:

    flat sand                 0.120 cm/h
    builds 3 cm steadily      0.108 cm/h
    builds 8 cm steadily      0.085 cm/h
    builds 6 cm in bursts     0.118 cm/h
    mild reflection           2.093 cm/h
    bad reflection           10.479 cm/h

The threshold is set from the data rather than fixed, because the noise floor
depends on water clarity, camera and tank. Working in logs, mask above
median x MAD^k. On real trials the masked fraction plateaus for k of 4 and
above, which is the signature of a genuinely separate population; k=5 sits
furthest from both edges of that plateau.

Usage inside DepthPreparer.createSmoothedArray, after the temporal pass and
before the spatial one:

    from helper_modules.depth_masking import trialChurnMask

    for trial in self.lp.trials:
        # pass one: temporal interpolation, per day
        for start_f, stop_f in trial.days:
            day = interpDepthData[start_f.index:stop_f.index + 1]
            day[:] = interpolate_time(day, min_good=goodDataCutoff)

        # the mask needs the whole trial, so it comes between the passes
        mask, stats = trialChurnMask(interpDepthData, self.lp, trial,
                                     tray_mask=self.tray_mask, k=5)
        interpDepthData[trial_slice][:, mask] = np.nan

        # pass two: spatial interpolation now fills the masked pixels too
        for start_f, stop_f in trial.days:
            day = interpDepthData[start_f.index:stop_f.index + 1]
            day[:] = interpolate_space(day, usable=self.tray_mask)
"""

import numpy as np
from scipy import ndimage


def trialTravelRate(data, lp, trial):
    """Excess travel per lights-on hour, per pixel, over one trial.

    Returns (excess_rate, travel_rate, hours). Travel is accumulated within each
    day only — the overnight gap is not a measurement — and divided by the total
    lights-on time.
    """
    total_travel = None
    hours = 0.0
    first_frame = last_frame = None

    for start_f, stop_f in trial.days:
        idx = [f.index for f in lp.frames
               if start_f.index <= f.index <= stop_f.index and f.lof]
        if len(idx) < 2:
            continue
        day = data[idx[0]:idx[-1] + 1]
        with np.errstate(invalid='ignore'):
            travel = np.nansum(np.abs(np.diff(day, axis=0)), axis=0)
        total_travel = travel if total_travel is None else total_travel + travel
        hours += (lp.frames[idx[-1]].time - lp.frames[idx[0]].time).total_seconds() / 3600.0
        if first_frame is None:
            first_frame = idx[0]
        last_frame = idx[-1]

    if total_travel is None or hours <= 0:
        return None, None, 0.0

    net = data[first_frame] - data[last_frame]
    return (total_travel - np.abs(net)) / hours, total_travel / hours, hours


def expectedFloor(std_mean, frames_per_hour=12.0):
    """The travel rate a clean tray should show, from the per-capture spread.

    Each stored frame is a median of ~30 captures, so its own noise is about
    std/sqrt(30); consecutive frames differ by 1.128 times that on average, and
    travel accumulates once per frame. Use the median of Frame_std across the
    tray as std.
    """
    sigma = float(np.nanmedian(std_mean)) / np.sqrt(30.0)
    return 1.128 * sigma * frames_per_hour


def churnMask(excess, tray_mask=None, k=5.0, max_fraction=0.25,
              expected_floor=None, floor_tolerance=4.0,
              close=3, open_=3, dilate=5):
    """Pixels whose excess travel rate is an outlier for this trial.

    k             cut at median x MAD^k, in log space
    max_fraction  refuse to mask more than this share of the tray. Above about a
                  third the median itself falls inside the bad population and the
                  rule silently inverts, so a large mask is a sign the statistic
                  has failed rather than that the trial is very bad.
    expected_floor  what a clean tray's median rate should be, from
                  expectedFloor(). If the observed median exceeds this by more
                  than floor_tolerance the whole tray is compromised: there are
                  no outliers to find because everything is an outlier, and the
                  mask would come back empty while looking like a pass. Optional
                  but strongly recommended for unattended runs — max_fraction
                  cannot catch this case.
    close/open_   reflections are contiguous patches: closing fills pinholes,
                  opening drops isolated single pixels
    dilate        widens the mask to catch the penumbra where a patch fades out

    Returns (mask, stats). mask is all-False if the cut would exceed
    max_fraction, and stats['applied'] says whether it was used.
    """
    stats = {'k': k, 'applied': False, 'masked_fraction': 0.0}
    if excess is None:
        stats['reason'] = 'no travel data'
        return np.zeros((1, 1), bool), stats

    region = np.isfinite(excess) & (excess > 0)
    if tray_mask is not None:
        region &= tray_mask
    v = excess[region]
    if v.size < 100:
        stats['reason'] = 'too few valid pixels'
        return np.zeros(excess.shape, bool), stats

    lv = np.log(v)
    lmed = float(np.median(lv))
    lmad = float(np.median(np.abs(lv - lmed))) * 1.4826
    threshold = float(np.exp(lmed + k * lmad))
    stats.update({'median': float(np.exp(lmed)), 'mad_factor': float(np.exp(lmad)),
                  'threshold': threshold,
                  'p99': float(np.percentile(v, 99)), 'max': float(v.max())})

    if lmad <= 0:
        stats['reason'] = 'no spread in travel rate'
        return np.zeros(excess.shape, bool), stats

    if expected_floor:
        stats['expected_floor'] = float(expected_floor)
        stats['floor_ratio'] = float(np.exp(lmed)) / float(expected_floor)
        if stats['floor_ratio'] > floor_tolerance:
            stats['reason'] = ('median travel rate is %.1fx the expected floor — the '
                               'whole tray is churning, so there are no outliers to '
                               'find. Masking nothing and flagging the trial.'
                               % stats['floor_ratio'])
            return np.zeros(excess.shape, bool), stats

    mask = (excess > threshold) & region
    if close:
        mask = ndimage.binary_closing(mask, np.ones((close, close)))
    if open_:
        mask = ndimage.binary_opening(mask, np.ones((open_, open_)))
    if dilate:
        mask = ndimage.binary_dilation(mask, np.ones((dilate, dilate)))
    if tray_mask is not None:
        mask &= tray_mask

    denom = region.sum()
    fraction = float(mask.sum()) / denom if denom else 0.0
    stats['masked_fraction'] = fraction

    if fraction > max_fraction:
        stats['reason'] = ('would mask %.1f%% of the tray, above the %.0f%% limit — '
                           'treating the trial as compromised and masking nothing'
                           % (100 * fraction, 100 * max_fraction))
        return np.zeros(excess.shape, bool), stats

    stats['applied'] = True
    return mask, stats


def trialChurnMask(data, lp, trial, tray_mask=None, k=5.0, max_fraction=0.25,
                   std_mean=None):
    """Convenience wrapper: travel rate then mask. Returns (mask, stats).

    std_mean  optional per-pixel Frame_std averaged over the trial, which lets
              the expected floor be computed and the whole-tray failure caught.
    """
    excess, rate, hours = trialTravelRate(data, lp, trial)
    floor = expectedFloor(std_mean) if std_mean is not None else None
    mask, stats = churnMask(excess, tray_mask=tray_mask, k=k,
                            max_fraction=max_fraction, expected_floor=floor)
    stats['hours'] = round(hours, 1)
    return mask, stats