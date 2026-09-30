"""Per-pixel statistics over one day of frames.

Pure functions over arrays: no files, no paths, no logfile. That makes them
testable against answers worked out by hand, which matters because these are
the numbers the mask is built from.

Everything is NaN-aware. There is no interpolation anywhere in this pipeline,
so missing pixels stay missing and a statistic computed over a pixel that was
absent half the day has to say so rather than quietly averaging fewer points.
"""

from __future__ import annotations

import contextlib
import warnings
from typing import Optional, Tuple

import numpy as np


@contextlib.contextmanager
def _suppress_all_nan():
    """All-NaN slices and empty means are expected here, not a problem."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        yield


def linear_fit(day: np.ndarray, min_valid: float = 0.5
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares line through each pixel's own time series.

    Returns (slope, intercept, valid_count). Pixels with fewer than min_valid
    of the day's frames come back as NaN in slope and intercept: a trend
    through two points is not a trend, and pretending otherwise is how a
    sparse pixel ends up looking like a steady builder.
    """
    frames = day.shape[0]
    t = np.arange(frames, dtype=np.float64)[:, None, None]
    valid = np.isfinite(day)
    n = valid.sum(axis=0).astype(np.float64)
    tv = np.where(valid, t, np.nan)

    with _suppress_all_nan():
        sum_t = np.nansum(tv, axis=0)
        sum_y = np.nansum(day, axis=0)
        sum_tt = np.nansum(tv * tv, axis=0)
        sum_ty = np.nansum(tv * day, axis=0)

    denominator = n * sum_tt - sum_t * sum_t
    safe = np.where(denominator == 0, 1.0, denominator)
    slope = np.where(denominator != 0, (n * sum_ty - sum_t * sum_y) / safe, 0.0)
    intercept = (sum_y - slope * sum_t) / np.where(n == 0, 1.0, n)

    too_sparse = n < min_valid * frames
    slope = np.where(too_sparse, np.nan, slope)
    intercept = np.where(too_sparse, np.nan, intercept)
    return slope, intercept, n


def residual(day: np.ndarray, min_valid: float = 0.5) -> np.ndarray:
    """RMS departure from each pixel's own fitted line, in cm.

    This is the statistic the mask is built from. A pixel that builds steadily
    scores near zero because the fit absorbs the build; one that churns without
    going anywhere scores high. Unlike total travel it does not grow with the
    number of frames, so a four-hour partial day and a ten-hour day give
    comparable numbers with no normalising.
    """
    frames = day.shape[0]
    slope, intercept, _ = linear_fit(day, min_valid=min_valid)
    t = np.arange(frames, dtype=np.float64)[:, None, None]
    fitted = intercept[None] + slope[None] * t
    with _suppress_all_nan():
        return np.sqrt(np.nanmean((day - fitted) ** 2, axis=0))


def theil_sen(day: np.ndarray, pairs: int = 64, seed: int = 0) -> np.ndarray:
    """Robust height change across the day, from the median of random slopes.

    Preferred to differencing the endpoints when the endpoints themselves may
    be disturbed — a fish resting on the sand at the wrong moment moves an
    endpoint but not the median of sixty-four slopes.
    """
    frames = day.shape[0]
    if frames < 2:
        return np.full(day.shape[1:], np.nan)
    rng = np.random.default_rng(seed)
    count = min(pairs, frames * (frames - 1) // 2)
    i = rng.integers(0, frames, count)
    j = rng.integers(0, frames, count)
    keep = i != j
    i, j = i[keep], j[keep]
    if not len(i):
        return np.full(day.shape[1:], np.nan)
    with _suppress_all_nan():
        slopes = (day[j] - day[i]) / (j - i)[:, None, None]
        return np.nanmedian(slopes, axis=0) * (frames - 1)


def travel(day: np.ndarray) -> np.ndarray:
    """Summed absolute movement between consecutive frames, in cm.

    Kept as a diagnostic rather than a filter: it counts a steady builder's
    movement the same as a flickering pixel's, and it grows with frame count,
    so days of different length are not comparable without dividing by hours.
    """
    with _suppress_all_nan():
        return np.nansum(np.abs(np.diff(day, axis=0)), axis=0)


def valid_fraction(day: np.ndarray) -> np.ndarray:
    """Share of the day's frames in which each pixel had a reading."""
    return np.isfinite(day).sum(axis=0) / float(day.shape[0])


def summarise_day(day: np.ndarray, std_stack: Optional[np.ndarray] = None,
                  min_valid: float = 0.5, seed: int = 0) -> dict:
    """Everything the collector keeps for one day, in one pass.

    std_stack is the per-capture spread written alongside each frame. It
    measures within-capture variation directly, where residual measures
    between-frame variation, so the two together separate a noisy sensor from
    a genuinely moving pixel.
    """
    out = {
        'residual': residual(day, min_valid=min_valid).astype(np.float32),
        'trend': theil_sen(day, seed=seed).astype(np.float32),
        'travel': travel(day).astype(np.float32),
        'valid': valid_fraction(day).astype(np.float32),
    }
    if std_stack is not None and len(std_stack):
        with _suppress_all_nan():
            out['std_mean'] = np.nanmean(std_stack, axis=0).astype(np.float32)
            out['std_max'] = np.nanmax(std_stack, axis=0).astype(np.float32)
    return out


def project_score(residual_maps, inside: Optional[np.ndarray] = None) -> np.ndarray:
    """One number per pixel for the whole project.

    Each day's residual is divided by that day's own median, which makes days
    comparable even though the noise floor drifts, and the median of those
    ratios is taken across every day. A pixel is flagged when it is *usually*
    far from its trend.

    A union of per-day masks was tried and rejected: it grows with the number
    of days and cannot distinguish a pixel that failed on two days out of
    thirty-four from one that failed on all of them.
    """
    ratios = []
    for day_residual in residual_maps:
        region = np.isfinite(day_residual) & (day_residual > 0)
        if inside is not None:
            region = region & inside
        values = day_residual[region]
        if values.size < 100:
            continue
        median = float(np.median(values))
        if median <= 0:
            continue
        ratios.append(day_residual / median)
    if not ratios:
        return np.full(np.shape(residual_maps[0]) if len(residual_maps) else (1, 1), np.nan)
    with _suppress_all_nan():
        return np.nanmedian(np.stack(ratios), axis=0)


def threshold(score: np.ndarray, k: float = 4.0,
              inside: Optional[np.ndarray] = None) -> Tuple[float, dict]:
    """The cut for a score map: the log-space median times MAD to the power k.

    Working in logs because the distribution is heavy-tailed on the right —
    the whole point is that a few pixels are orders of magnitude worse than
    typical, and a symmetric statistic would be dragged along by them.
    """
    region = np.isfinite(score) & (score > 0)
    if inside is not None:
        region = region & inside
    values = score[region]
    if values.size < 100:
        return float('inf'), {'reason': 'too few pixels to set a threshold'}
    logs = np.log(values)
    log_median = float(np.median(logs))
    log_mad = float(np.median(np.abs(logs - log_median))) * 1.4826
    cut = float(np.exp(log_median + k * log_mad))
    return cut, {'median': float(np.exp(log_median)),
                 'mad_factor': float(np.exp(log_mad)),
                 'threshold': cut, 'k': k, 'n': int(values.size)}
