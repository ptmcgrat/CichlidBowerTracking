"""
cichlid_pose.py
===============

Convert Ultralytics cichlid pose CSVs (one per tank-day) into Parquet, with
per-detection geometry computed at conversion time.

Supersedes the earlier ``pose_to_parquet.py`` / ``build_pose_dataset.py``:
no hive partitioning, no filename parsing, no across-day animal identity.

Layout on disk, where ``out_root`` is the project's own directory::

    <out_root>/
      tracks.parquet              one row per track, all days -- small
      pair_episodes.parquet       one row per pair episode, all days -- small
      days/
        0001_vid.parquet          one row per (frame, track)
        0001_vid_pairs.parquet    one row per (frame, track pair)
        0002_vid.parquet
        0002_vid_pairs.parquet

``project_id``, ``day_index`` and ``day_label`` are stored as columns inside
each file rather than encoded in directory names, so a file stays
self-describing if moved or shared. Files are built in a
temp directory and moved into place only once complete -- a sync client will
happily upload a half-written Parquet file, and a truncated Parquet file is
unreadable rather than partially readable.

Usage
-----
    from cichlid_pose import convert_day, build_project, open_project

    convert_day("d1.csv", project_id="MC_pair_017", day_index=0,
                day_label="0001_vid", fps=30.0,
                out_root="/dropbox/.../MC_pair_017")

    build_project(
        [("d1.csv", "0001_vid"), ("d2.csv", "0002_vid")],
        project_id="MC_pair_017", fps=30.0,
        out_root="/dropbox/.../MC_pair_017",
    )

    detections, tracks = open_project("/dropbox/.../MC_pair_017")
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

__version__ = "1.0.0"

# --------------------------------------------------------------------------
# Skeleton
# --------------------------------------------------------------------------

KEYPOINTS: tuple[str, ...] = (
    "Nose", "LeftEye", "RightEye", "Head",
    "Spine1", "Spine2", "Spine3", "Spine4",
    "Peduncle", "TailTip",
)

#: Head-to-tail chain. body_length_px is the sum of these segments, which is
#: invariant to body bending -- unlike a straight Nose-to-TailTip distance,
#: which shortens exactly when the fish curves during circling.
MIDLINE: tuple[str, ...] = (
    "Nose", "Head", "Spine1", "Spine2", "Spine3", "Spine4", "Peduncle", "TailTip",
)

#: Fitted for heading_rad. Deliberately excludes the tail, which swings during
#: quivering -- the frames where a stable heading matters most.
ANTERIOR: tuple[str, ...] = ("Nose", "Head", "Spine1", "Spine2")

#: Fitted for the posterior axis used to derive bend_angle_rad.
POSTERIOR: tuple[str, ...] = ("Spine3", "Spine4", "Peduncle", "TailTip")

#: Two keypoints closer than this (pixels) are counted as coincident -- the
#: signature of a collapsed pose.
COINCIDENT_EPS_PX = 1.0

#: TrackID restarts at 1 in every CSV. Offsetting by the day index makes
#: TrackUID unique across the whole project. Assumes fewer than this many
#: distinct TrackIDs in a single day.
TRACK_ID_BLOCK = 1_000_000


# --------------------------------------------------------------------------
# Tuple-string parsing
# --------------------------------------------------------------------------

# Unwraps np.float32(...) etc. [^()]* so it cannot swallow the outer parens.
_NP_SCALAR = re.compile(r"np\.(?:float|int|uint)\d*\(\s*([^()]*?)\s*\)")


def parse_point_column(s: pd.Series) -> pd.DataFrame:
    """Parse stringified ``(x, y)`` tuples into two float32 columns.

    Handles the ``np.float32(...)`` wrapper, plain tuples, and missing values.
    Vectorized -- no per-row eval. Unparseable cells become NaN rather than
    raising, so one malformed row costs a keypoint, not the whole file.
    """
    cleaned = (
        s.astype("string")
        .str.replace(_NP_SCALAR, r"\1", regex=True)
        .str.strip()
        .str.strip("()")
    )
    parts = cleaned.str.split(",", n=1, expand=True)
    if parts.shape[1] == 1:
        parts[1] = pd.NA

    out = pd.DataFrame(index=s.index)
    for i in (0, 1):
        out[i] = pd.to_numeric(parts[i].str.strip(), errors="coerce").astype("float32")
    return out


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _stack(df: pd.DataFrame, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (n, k) x and y arrays for the named keypoints."""
    x = np.column_stack([df[f"{n}_x"].to_numpy(dtype="float64") for n in names])
    y = np.column_stack([df[f"{n}_y"].to_numpy(dtype="float64") for n in names])
    return x, y


def principal_axis(
    x: np.ndarray, y: np.ndarray, nose_x: np.ndarray, nose_y: np.ndarray
) -> np.ndarray:
    """Orientation of the first principal axis of each row's point set.

    PCA on 2D points is total-least-squares line fitting: the first
    eigenvector of the 2x2 covariance is the direction of greatest spread,
    i.e. the long axis of the fish. Averaging over several keypoints suppresses
    the jitter that a two-point Nose-minus-Head vector would inherit whole.

    For a symmetric 2x2 there is a closed form, so this runs vectorized over
    every row at once rather than calling an eigensolver 45 million times::

        theta = 0.5 * arctan2(2*Sxy, Sxx - Syy)

    Eigenvectors are sign-ambiguous (v and -v both solve it), so the axis is
    then oriented to point anteriorly using the nose. Without that step the
    heading flips 180 degrees at random between frames and every angular
    velocity downstream is noise.

    Rows with fewer than 2 present keypoints yield NaN.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum(axis=1)

    xf = np.where(valid, x, np.nan)
    yf = np.where(valid, y, np.nan)
    with np.errstate(invalid="ignore"):
        mx = np.nanmean(xf, axis=1)
        my = np.nanmean(yf, axis=1)

    dx = np.where(valid, x - mx[:, None], 0.0)
    dy = np.where(valid, y - my[:, None], 0.0)

    sxx = (dx * dx).sum(axis=1)
    syy = (dy * dy).sum(axis=1)
    sxy = (dx * dy).sum(axis=1)

    theta = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)

    # Orient anteriorly: flip if the axis points away from the nose.
    ax = nose_x - mx
    ay = nose_y - my
    flip = (np.cos(theta) * ax + np.sin(theta) * ay) < 0
    theta = np.where(flip, theta + np.pi, theta)

    theta = wrap_angle(theta)
    return np.where(n >= 2, theta, np.nan)


def wrap_angle(a: np.ndarray) -> np.ndarray:
    """Wrap radians to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def compute_detection_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Add the row-local derived columns. Modifies and returns ``df``."""
    # --- body length: sum of midline segments -----------------------------
    mx, my = _stack(df, MIDLINE)
    seg = np.hypot(np.diff(mx, axis=1), np.diff(my, axis=1))
    # NaN if any midline keypoint is absent: a partial sum would silently
    # under-report length and skew everything normalized by it.
    complete = np.isfinite(mx).all(axis=1) & np.isfinite(my).all(axis=1)
    df["body_length_px"] = np.where(complete, seg.sum(axis=1), np.nan).astype("float32")

    # --- midline centroid -------------------------------------------------
    with np.errstate(invalid="ignore"):
        df["midline_centroid_x"] = np.nanmean(mx, axis=1).astype("float32")
        df["midline_centroid_y"] = np.nanmean(my, axis=1).astype("float32")

    nose_x = df["Nose_x"].to_numpy(dtype="float64")
    nose_y = df["Nose_y"].to_numpy(dtype="float64")

    # --- angles -----------------------------------------------------------
    ax_, ay_ = _stack(df, ANTERIOR)
    heading = principal_axis(ax_, ay_, nose_x, nose_y)
    df["heading_rad"] = heading.astype("float32")

    bx_, by_ = _stack(df, MIDLINE)
    df["body_axis_rad"] = principal_axis(bx_, by_, nose_x, nose_y).astype("float32")

    px_, py_ = _stack(df, POSTERIOR)
    posterior = principal_axis(px_, py_, nose_x, nose_y)
    # Signed, because quivering is an oscillation -- the sign carries the
    # phase that a magnitude would throw away.
    df["bend_angle_rad"] = wrap_angle(heading - posterior).astype("float32")

    # --- keypoint QC ------------------------------------------------------
    kx, ky = _stack(df, KEYPOINTS)
    present = np.isfinite(kx) & np.isfinite(ky)
    df["qc_n_missing_kp"] = (~present).sum(axis=1).astype("int8")

    k = len(KEYPOINTS)
    iu, ju = np.triu_indices(k, k=1)
    d = np.hypot(kx[:, iu] - kx[:, ju], ky[:, iu] - ky[:, ju])
    both = present[:, iu] & present[:, ju]
    df["qc_n_coincident_kp"] = (
        ((d < COINCIDENT_EPS_PX) & both).sum(axis=1).astype("int8")
    )

    return df


def compute_track_stats(
    df: pd.DataFrame, fps: float, gap_tolerance_s: float
) -> pd.DataFrame:
    """Add columns that need the neighbouring frame within the same track.

    Every derivative is computed strictly inside a track: never across a track
    boundary, and never across an internal gap longer than the tolerance.
    Tracks here fragment constantly because fish leave the field of view, so
    differencing across a gap would invent enormous speeds and angular
    velocities exactly where the data is weakest.
    """
    df = df.sort_values(["TrackUID", "FrameNum"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("TrackUID", sort=False)

    df["is_track_start"] = (g.cumcount() == 0)
    df["is_track_end"] = (g.cumcount(ascending=False) == 0)

    d_frames = g["FrameNum"].diff().to_numpy(dtype="float64")
    dt_s = d_frames / float(fps)
    usable = np.isfinite(dt_s) & (dt_s > 0) & (dt_s <= gap_tolerance_s)

    # Normalize speed by the track's median body length rather than the
    # per-frame value, so a single bad pose cannot rescale that frame.
    med_bl = g["body_length_px"].transform("median").to_numpy(dtype="float64")

    dx = g["midline_centroid_x"].diff().to_numpy(dtype="float64")
    dy = g["midline_centroid_y"].diff().to_numpy(dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.hypot(dx, dy) / dt_s / med_bl
    df["speed_bl_s"] = np.where(usable, speed, np.nan).astype("float32")

    dh = wrap_angle(g["heading_rad"].diff().to_numpy(dtype="float64"))
    with np.errstate(invalid="ignore", divide="ignore"):
        omega = dh / dt_s
    df["d_heading_rad_s"] = np.where(usable, omega, np.nan).astype("float32")

    return df


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def build_schema() -> pa.Schema:
    fields = [
        pa.field("project_id", pa.string()),
        pa.field("day_index", pa.int16()),
        pa.field("day_label", pa.string()),
        pa.field("FrameNum", pa.int64()),
        pa.field("Time_s", pa.float64()),
        pa.field("Timestamp", pa.timestamp("us")),
        pa.field("TrackID", pa.int32()),
        pa.field("TrackUID", pa.int64()),
        pa.field("X_center", pa.float32()),
        pa.field("Y_center", pa.float32()),
        pa.field("Width", pa.float32()),
        pa.field("Height", pa.float32()),
        pa.field("Sex", pa.string()),
        pa.field("SexID", pa.int8()),
    ]
    for kp in KEYPOINTS:
        fields.append(pa.field(f"{kp}_x", pa.float32()))
        fields.append(pa.field(f"{kp}_y", pa.float32()))
    fields += [
        pa.field("body_length_px", pa.float32()),
        pa.field("midline_centroid_x", pa.float32()),
        pa.field("midline_centroid_y", pa.float32()),
        pa.field("heading_rad", pa.float32()),
        pa.field("body_axis_rad", pa.float32()),
        pa.field("bend_angle_rad", pa.float32()),
        pa.field("qc_n_missing_kp", pa.int8()),
        pa.field("qc_n_coincident_kp", pa.int8()),
        pa.field("speed_bl_s", pa.float32()),
        pa.field("d_heading_rad_s", pa.float32()),
        pa.field("is_track_start", pa.bool_()),
        pa.field("is_track_end", pa.bool_()),
    ]
    return pa.schema(fields)


def _file_metadata(fps: float, extra: dict[str, Any] | None = None) -> dict[bytes, bytes]:
    meta: dict[str, Any] = {
        "schema_version": "1",
        "converter_version": __version__,
        "fps": fps,
        "keypoints": list(KEYPOINTS),
        "midline": list(MIDLINE),
        "anterior_fit": list(ANTERIOR),
        "posterior_fit": list(POSTERIOR),
        "coordinate_space": "raw_pixels",
        "y_axis": "down",
        "units": "pixels",
        "bend_angle_definition": "wrap(heading_rad - posterior_axis_rad)",
    }
    if extra:
        meta.update(extra)
    return {
        k.encode(): (v if isinstance(v, str) else json.dumps(v)).encode()
        for k, v in meta.items()
    }


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


def convert_day(
    csv_path: str | Path,
    *,
    day_index: int,
    day_label: str,
    fps: float,
    out_root: str | Path,
    project_id: str | None = None,
    recording_start: Any = None,
    gap_tolerance_s: float = 0.2,
    compression: str = "zstd",
    compression_level: int = 5,
    row_group_size: int = 500_000,
    overwrite: bool = False,
) -> Path:
    """Convert one tank-day CSV to Parquet. Returns the output path.

    Parameters
    ----------
    day_index
        Position of this day in the project, 0-based. Used for ordering, for
        cross-project comparison (projects start on different calendar dates,
        so day index is the comparable axis), and to make TrackUID unique.
    day_label
        Stable identifier for the recording, e.g. ``"0001_vid"``. Used as the
        filename and as provenance back to the source video.
    project_id
        Defaults to the name of the ``out_root`` directory, since that folder
        already identifies the project. Kept as a column -- not a directory --
        so a day file stays identifiable if it is copied out of its folder or
        pooled with days from other projects.
    recording_start
        Wall-clock time of frame 0 -- a ``datetime``, ``pd.Timestamp``, or ISO
        string. Optional, but with weeks of recording it is what makes
        time-of-day questions answerable; ``Timestamp`` is null without it.
        A tz-aware value is reduced to local wall time rather than shifted to
        UTC: on a light cycle, "two hours after lights-on" is the meaningful
        axis, and a UTC shift would scramble it.
    gap_tolerance_s
        Derivatives are NaN across track gaps longer than this.
    """
    csv_path = Path(csv_path)
    out_root = Path(out_root)
    if project_id is None:
        project_id = out_root.resolve().name
    out_path = out_root / "days" / f"{day_label}.parquet"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists. Pass overwrite=True.")

    raw = pd.read_csv(csv_path)
    pose_cols = [f"Pose_{k}" for k in KEYPOINTS]
    missing = [c for c in pose_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"{csv_path.name} is missing pose columns: {missing}")

    df = pd.DataFrame(index=raw.index)
    df["project_id"] = str(project_id)
    df["day_index"] = np.int16(day_index)
    df["day_label"] = str(day_label)
    df["FrameNum"] = pd.to_numeric(raw["FrameNum"], errors="coerce").astype("int64")
    df["TrackID"] = pd.to_numeric(raw["TrackID"], errors="coerce").astype("int32")
    df["TrackUID"] = (
        int(day_index) * TRACK_ID_BLOCK + df["TrackID"].astype("int64")
    ).astype("int64")

    for col in ("X_center", "Y_center", "Width", "Height"):
        if col in raw.columns:
            df[col] = pd.to_numeric(raw[col], errors="coerce").astype("float32")
    if "Sex" in raw.columns:
        df["Sex"] = raw["Sex"].astype("string")
    if "SexID" in raw.columns:
        df["SexID"] = pd.to_numeric(raw["SexID"], errors="coerce").astype("Int8")

    for kp, col in zip(KEYPOINTS, pose_cols):
        xy = parse_point_column(raw[col])
        df[f"{kp}_x"] = xy[0]
        df[f"{kp}_y"] = xy[1]

    df["Time_s"] = df["FrameNum"] / float(fps)
    if recording_start is not None:
        start = pd.Timestamp(recording_start)
        if start.tzinfo is not None:
            start = start.tz_localize(None)
        # Round to us: at 30 fps the frame period is 33.333... ms, which has
        # no exact ns representation and will not cast losslessly otherwise.
        df["Timestamp"] = (
            start + pd.to_timedelta(df["Time_s"], unit="s")
        ).dt.round("us")
    else:
        df["Timestamp"] = pd.NaT

    df = compute_detection_stats(df)
    df = compute_track_stats(df, fps=fps, gap_tolerance_s=gap_tolerance_s)

    schema = build_schema()
    for name in schema.names:
        if name not in df.columns:
            df[name] = pd.NA
    df = df[schema.names]

    table = pa.Table.from_pandas(df, preserve_index=False).cast(schema)
    table = table.replace_schema_metadata(
        _file_metadata(fps, {"project_id": str(project_id),
                             "day_index": int(day_index),
                             "day_label": str(day_label),
                             "source_csv": csv_path.name})
    )

    _atomic_write(table, out_path, compression, compression_level, row_group_size)
    return out_path


#: Suffix distinguishing the pair table from the detections table. Both live
#: in days/, so anything that scans that directory must filter -- a naive
#: glob would try to unify two incompatible schemas into one dataset.
PAIRS_SUFFIX = "_pairs"


def day_detection_files(out_root: str | Path) -> list[Path]:
    """Every detections file in days/, excluding the pair tables."""
    d = Path(out_root) / "days"
    return sorted(
        f for f in d.glob("*.parquet") if not f.stem.endswith(PAIRS_SUFFIX)
    )


def day_pair_files(out_root: str | Path) -> list[Path]:
    """Every pair-table file in days/."""
    d = Path(out_root) / "days"
    return sorted(f for f in d.glob(f"*{PAIRS_SUFFIX}.parquet"))


def _atomic_write(
    table: pa.Table,
    out_path: Path,
    compression: str,
    compression_level: int,
    row_group_size: int,
) -> None:
    """Write to a temp file, then move into place.

    Dropbox will upload a Parquet file mid-write, and a truncated Parquet file
    is unreadable rather than partially readable.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(
            table, tmp, compression=compression,
            compression_level=compression_level, row_group_size=row_group_size,
        )
        shutil.move(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------
# Tracks table
# --------------------------------------------------------------------------


def summarize_tracks(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """One row per track. Small enough to sync and query on its own.

    Fragmentation is structural here -- fish leave the field of view -- so this
    table is the index that makes it navigable rather than a defect report.
    """
    g = df.groupby("TrackUID", sort=True)
    out = pd.DataFrame({
        "project_id": g["project_id"].first(),
        "day_index": g["day_index"].first(),
        "day_label": g["day_label"].first(),
        "TrackID": g["TrackID"].first(),
        "first_frame": g["FrameNum"].min(),
        "last_frame": g["FrameNum"].max(),
        "n_detections": g.size(),
    })
    span = out["last_frame"] - out["first_frame"] + 1
    out["span_frames"] = span
    out["duration_s"] = span / float(fps)
    # Detections present vs frames spanned. A track alive for a minute but
    # detected in 60% of it will produce spurious motion wherever it is
    # interpolated downstream.
    out["coverage"] = out["n_detections"] / span

    out["median_body_length_px"] = g["body_length_px"].median()
    out["median_x"] = g["midline_centroid_x"].median()
    out["median_y"] = g["midline_centroid_y"].median()
    out["median_speed_bl_s"] = g["speed_bl_s"].median()
    out["frac_kp_missing"] = g["qc_n_missing_kp"].mean() / len(KEYPOINTS)

    # Sex is a per-detection prediction, so it flickers within a track. Store
    # the consensus and the vote fraction; with one male among five fish, a
    # weak male call is more likely a misclassified female.
    if "Sex" in df.columns:
        sex = df.dropna(subset=["Sex"])
        if len(sex):
            counts = sex.groupby(["TrackUID", "Sex"]).size().unstack(fill_value=0)
            total = counts.sum(axis=1)
            out["sex_consensus"] = counts.idxmax(axis=1)
            out["sex_vote_frac"] = (counts.max(axis=1) / total).astype("float32")

    return out.reset_index()


def write_project_tracks(out_root: str | Path, fps: float) -> Path:
    """Rebuild ``<out_root>/tracks.parquet`` from every day file present."""
    proj = Path(out_root)
    files = day_detection_files(proj)
    if not files:
        raise FileNotFoundError(f"no day files under {proj / 'days'}")

    cols = [
        "project_id", "day_index", "day_label", "TrackUID", "TrackID", "FrameNum",
        "body_length_px", "midline_centroid_x", "midline_centroid_y",
        "speed_bl_s", "qc_n_missing_kp", "Sex",
    ]
    frames = [pq.read_table(f, columns=cols).to_pandas() for f in files]
    tracks = pd.concat(
        [summarize_tracks(fr, fps) for fr in frames], ignore_index=True
    ).sort_values(["day_index", "TrackUID"]).reset_index(drop=True)

    table = pa.Table.from_pandas(tracks, preserve_index=False)
    project_id = str(tracks["project_id"].iloc[0]) if len(tracks) else ""
    table = table.replace_schema_metadata(
        _file_metadata(fps, {"project_id": project_id, "n_days": len(files)})
    )
    out_path = proj / "tracks.parquet"
    _atomic_write(table, out_path, "zstd", 5, 500_000)
    return out_path


def build_project(
    days: Iterable[tuple[Any, ...]],
    *,
    fps: float,
    out_root: str | Path,
    project_id: str | None = None,
    recording_starts: dict[str, Any] | None = None,
    gap_tolerance_s: float = 0.2,
    make_pairs: bool = True,
    min_episode_frames: int = 5,
    max_pair_distance_px: float | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Convert every day of one project, then write the tracks table.

    ``days`` is an iterable of ``(csv_path, day_label)`` pairs -- day_index is
    then the position in the sequence -- or of explicit
    ``(csv_path, day_index, day_label)`` triples. project_id comes from your
    existing index, so nothing is parsed out of filenames.

    ``recording_starts`` is keyed by day_label. ``project_id`` defaults to the
    name of the ``out_root`` directory.
    """
    items = []
    for i, entry in enumerate(days):
        if len(entry) == 2:
            csv_path, label = entry
            items.append((csv_path, i, str(label)))
        else:
            csv_path, idx, label = entry
            items.append((csv_path, int(idx), str(label)))

    seen = {}
    for _, idx, label in items:
        if idx in seen:
            raise ValueError(f"day_index {idx} used by both {seen[idx]!r} and {label!r}")
        seen[idx] = label

    results = []
    for csv_path, idx, label in items:
        try:
            path = convert_day(
                csv_path, day_index=idx, day_label=label,
                fps=fps, out_root=out_root, project_id=project_id,
                gap_tolerance_s=gap_tolerance_s,
                recording_start=(recording_starts or {}).get(label),
                overwrite=overwrite,
            )
            n = pq.read_metadata(path).num_rows
            results.append({"day_index": idx, "day_label": label, "ok": True,
                            "n_rows": n, "error": None})
            print(f"  ok    {idx:>3}  {label:<16} {n:>10,} rows")
        except Exception as exc:
            results.append({"day_index": idx, "day_label": label, "ok": False,
                            "n_rows": 0, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAIL  {idx:>3}  {label:<16} {type(exc).__name__}: {exc}")

    if not any(r["ok"] for r in results):
        return pd.DataFrame(results)

    write_project_tracks(out_root, fps)

    if make_pairs:
        for r in results:
            if not r["ok"]:
                continue
            try:
                dest = write_day_pairs(
                    out_root, r["day_label"], fps,
                    gap_tolerance_s=gap_tolerance_s,
                    min_episode_frames=min_episode_frames,
                    max_distance_px=max_pair_distance_px,
                )
                n = pq.read_metadata(dest).num_rows if dest else 0
                r["n_pair_rows"] = n
                print(f"  pairs {r['day_index']:>3}  {r['day_label']:<16} {n:>10,} rows")
            except Exception as exc:
                r["n_pair_rows"] = 0
                r["error"] = f"pairs: {type(exc).__name__}: {exc}"
                print(f"  PAIRS FAIL {r['day_label']}: {exc}")
        write_pair_episodes(out_root, fps)

    return pd.DataFrame(results)


# --------------------------------------------------------------------------
# Pairwise interactions
# --------------------------------------------------------------------------

#: Keypoint-to-keypoint distances stored for every co-occurring pair, in raw
#: pixels. The trailing "of1"/"of2" names which fish the keypoint belongs to.
#: nose-to-tail and nose-to-Spine4 in both directions capture the asymmetric
#: configurations -- who is behind whom, and whose flank is being approached.
PAIR_DISTANCES: tuple[tuple[str, int, str, int, str], ...] = (
    ("Nose", 1, "Nose",    2, "dist_nose1_nose2_px"),
    ("Nose", 1, "TailTip", 2, "dist_nose1_tailtip2_px"),
    ("Nose", 2, "TailTip", 1, "dist_nose2_tailtip1_px"),
    ("Nose", 1, "Spine4",  2, "dist_nose1_spine4of2_px"),
    ("Nose", 2, "Spine4",  1, "dist_nose2_spine4of1_px"),
)

#: Columns pulled from the detections table before the self-join.
_PAIR_SOURCE_COLS = (
    "project_id", "day_index", "day_label", "FrameNum", "Time_s",
    "TrackID", "TrackUID", "Sex",
    "midline_centroid_x", "midline_centroid_y",
    "heading_rad", "d_heading_rad_s", "qc_n_missing_kp",
    "Nose_x", "Nose_y", "TailTip_x", "TailTip_y", "Spine4_x", "Spine4_y",
)


def circular_mean(a: np.ndarray) -> float:
    """Vector mean of angles in radians, NaN-safe.

    An arithmetic mean of angles straddling the +/-pi boundary collapses to
    roughly zero instead of pi -- a bug that produces plausible-looking
    numbers, so every angular summary here goes through this.
    """
    a = np.asarray(a, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def build_pair_frame(
    df: pd.DataFrame,
    *,
    fps: float,
    gap_tolerance_s: float = 0.2,
    min_episode_frames: int = 5,
    max_distance_px: float | None = None,
) -> pd.DataFrame:
    """Expand one day of detections into per-frame pair rows.

    Each unordered pair appears once, canonically ordered ``TrackUID_1 <
    TrackUID_2``. Directional quantities that are not recoverable by symmetry
    (the two bearings, the asymmetric keypoint distances) get their own
    columns rather than the table getting a second row per pair.

    A **pair episode** is a maximal run of frames in which both members of a
    specific pair are present, allowing internal gaps up to
    ``gap_tolerance_s``. Because tracks here fragment whenever a fish leaves
    the field of view, episodes are short and numerous -- and every derivative
    must stay inside one, since differencing across an episode boundary
    invents rotation over an interval where nothing was observed.
    """
    cols = [c for c in _PAIR_SOURCE_COLS if c in df.columns]
    sub = df[cols]

    merged = sub.merge(sub, on="FrameNum", suffixes=("_1", "_2"))
    merged = merged[merged["TrackUID_1"] < merged["TrackUID_2"]]
    if not len(merged):
        return pd.DataFrame()

    out = pd.DataFrame(index=merged.index)
    for c in ("project_id", "day_index", "day_label", "Time_s"):
        if f"{c}_1" in merged.columns:
            out[c] = merged[f"{c}_1"]
    out["FrameNum"] = merged["FrameNum"]
    out["TrackID_1"] = merged["TrackID_1"]
    out["TrackID_2"] = merged["TrackID_2"]
    out["TrackUID_1"] = merged["TrackUID_1"]
    out["TrackUID_2"] = merged["TrackUID_2"]

    # --- proximity, raw pixels ------------------------------------------
    x1 = merged["midline_centroid_x_1"].to_numpy("float64")
    y1 = merged["midline_centroid_y_1"].to_numpy("float64")
    x2 = merged["midline_centroid_x_2"].to_numpy("float64")
    y2 = merged["midline_centroid_y_2"].to_numpy("float64")
    dx, dy = x2 - x1, y2 - y1
    out["centroid_distance_px"] = np.hypot(dx, dy).astype("float32")

    for kp_a, fish_a, kp_b, fish_b, name in PAIR_DISTANCES:
        ax = merged[f"{kp_a}_x_{fish_a}"].to_numpy("float64")
        ay = merged[f"{kp_a}_y_{fish_a}"].to_numpy("float64")
        bx = merged[f"{kp_b}_x_{fish_b}"].to_numpy("float64")
        by = merged[f"{kp_b}_y_{fish_b}"].to_numpy("float64")
        out[name] = np.hypot(ax - bx, ay - by).astype("float32")

    # --- angles -----------------------------------------------------------
    # Image y increases downward, so increasing orbital_rad is clockwise
    # on screen.
    orbital = np.arctan2(dy, dx)
    h1 = merged["heading_rad_1"].to_numpy("float64")
    h2 = merged["heading_rad_2"].to_numpy("float64")

    out["orbital_rad"] = orbital.astype("float32")
    out["rel_heading_rad"] = wrap_angle(h1 - h2).astype("float32")
    # Egocentric: where the other fish sits from this fish's point of view.
    # 0 = dead ahead, +/-pi = directly behind.
    out["bearing_1_rad"] = wrap_angle(orbital - h1).astype("float32")
    out["bearing_2_rad"] = wrap_angle(orbital + np.pi - h2).astype("float32")

    # Carried through so the rigid-rotation test is one expression per row.
    out["d_heading_1_rad_s"] = merged["d_heading_rad_s_1"].astype("float32")
    out["d_heading_2_rad_s"] = merged["d_heading_rad_s_2"].astype("float32")

    # --- context ----------------------------------------------------------
    if "Sex_1" in merged.columns:
        s1 = merged["Sex_1"].astype("string")
        s2 = merged["Sex_2"].astype("string")
        out["sex_1"] = s1
        out["sex_2"] = s2
        # Sorted initials, so MF and FM collapse to one category.
        code = pd.Series(
            [
                "".join(sorted((str(a)[:1].upper(), str(b)[:1].upper())))
                if pd.notna(a) and pd.notna(b) else pd.NA
                for a, b in zip(s1, s2)
            ],
            index=out.index, dtype="string",
        )
        out["pair_sex"] = code
    out["qc_n_missing_kp_1"] = merged["qc_n_missing_kp_1"].astype("int8")
    out["qc_n_missing_kp_2"] = merged["qc_n_missing_kp_2"].astype("int8")

    if max_distance_px is not None:
        out = out[out["centroid_distance_px"] <= max_distance_px]
        if not len(out):
            return pd.DataFrame()

    out = out.sort_values(
        ["TrackUID_1", "TrackUID_2", "FrameNum"], kind="mergesort"
    ).reset_index(drop=True)

    # --- episodes ---------------------------------------------------------
    max_gap = max(1, int(round(gap_tolerance_s * float(fps))))
    pair_changed = (
        out["TrackUID_1"].ne(out["TrackUID_1"].shift())
        | out["TrackUID_2"].ne(out["TrackUID_2"].shift())
    )
    frame_gap = out["FrameNum"].diff()
    new_episode = pair_changed | (frame_gap > max_gap) | frame_gap.isna()
    local_id = new_episode.cumsum().astype("int64") - 1

    day_index = int(out["day_index"].iloc[0]) if "day_index" in out.columns else 0
    out["pair_episode_id"] = (day_index * TRACK_ID_BLOCK + local_id).astype("int64")
    out["dt_frames"] = np.where(new_episode, np.nan, frame_gap).astype("float32")

    if min_episode_frames > 1:
        sizes = out.groupby("pair_episode_id")["FrameNum"].transform("size")
        out = out[sizes >= min_episode_frames].reset_index(drop=True)
        if not len(out):
            return pd.DataFrame()

    # --- orbital rate, strictly within an episode -------------------------
    g = out.groupby("pair_episode_id", sort=False)
    d_orb = wrap_angle(g["orbital_rad"].diff().to_numpy("float64"))
    dt_s = out["dt_frames"].to_numpy("float64") / float(fps)
    with np.errstate(invalid="ignore", divide="ignore"):
        omega = d_orb / dt_s
    out["d_orbital_rad_s"] = omega.astype("float32")

    return out


def summarize_pair_episodes(pairs: pd.DataFrame, fps: float) -> pd.DataFrame:
    """One row per pair episode -- the browsable index of interactions.

    This is the analogue of ``tracks.parquet``: small enough to scan or sync
    on its own, and it is where circling candidates surface without touching
    the frame-level table.
    """
    if not len(pairs):
        return pd.DataFrame()

    g = pairs.groupby("pair_episode_id", sort=True)
    out = pd.DataFrame({
        "project_id": g["project_id"].first(),
        "day_index": g["day_index"].first(),
        "day_label": g["day_label"].first(),
        "TrackUID_1": g["TrackUID_1"].first(),
        "TrackUID_2": g["TrackUID_2"].first(),
        "first_frame": g["FrameNum"].min(),
        "last_frame": g["FrameNum"].max(),
        "n_frames": g.size(),
    })
    out["duration_s"] = (out["last_frame"] - out["first_frame"] + 1) / float(fps)

    out["min_centroid_distance_px"] = g["centroid_distance_px"].min()
    out["median_centroid_distance_px"] = g["centroid_distance_px"].median()
    for *_, name in PAIR_DISTANCES:
        out[f"min_{name}"] = g[name].min()

    # Cumulative rotation. net is signed, so a pair that swings one way and
    # back cancels; abs does not. Circling shows up as net ~= abs and large.
    step = pairs["d_orbital_rad_s"] * pairs["dt_frames"] / float(fps)
    step = step.groupby(pairs["pair_episode_id"], sort=True)
    out["net_orbital_turn_rad"] = step.sum(min_count=1)
    out["abs_orbital_turn_rad"] = step.apply(lambda s: s.abs().sum())
    out["n_revolutions"] = out["net_orbital_turn_rad"].abs() / (2.0 * np.pi)
    out["median_d_orbital_rad_s"] = g["d_orbital_rad_s"].median()

    # Rigid-rotation test: in true circling the orbital rate and both heading
    # rates coincide, so the worst gap between them is near zero. A fish
    # spinning beside a stationary one fails this; so does a straight chase.
    gap = np.maximum(
        (pairs["d_orbital_rad_s"] - pairs["d_heading_1_rad_s"]).abs(),
        (pairs["d_orbital_rad_s"] - pairs["d_heading_2_rad_s"]).abs(),
    )
    out["median_rate_gap_rad_s"] = gap.groupby(
        pairs["pair_episode_id"], sort=True
    ).median()

    for col, name in (
        ("rel_heading_rad", "circmean_rel_heading_rad"),
        ("bearing_1_rad", "circmean_bearing_1_rad"),
        ("bearing_2_rad", "circmean_bearing_2_rad"),
    ):
        out[name] = g[col].apply(lambda s: circular_mean(s.to_numpy()))

    if "pair_sex" in pairs.columns:
        mode = g["pair_sex"].agg(
            lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else pd.NA
        )
        out["pair_sex"] = mode
        out["pair_sex_frac"] = g["pair_sex"].agg(
            lambda s: (s == s.dropna().mode().iloc[0]).mean()
            if len(s.dropna()) else np.nan
        )

    out["frac_frames_kp_complete"] = g.apply(
        lambda d: float(
            ((d["qc_n_missing_kp_1"] == 0) & (d["qc_n_missing_kp_2"] == 0)).mean()
        ),
        include_groups=False,
    )

    return out.reset_index()


def write_day_pairs(
    out_root: str | Path,
    day_label: str,
    fps: float,
    *,
    gap_tolerance_s: float = 0.2,
    min_episode_frames: int = 5,
    max_distance_px: float | None = None,
    compression: str = "zstd",
    compression_level: int = 5,
    row_group_size: int = 500_000,
) -> Path | None:
    """Build ``days/<day_label>_pairs.parquet`` from ``days/<day_label>.parquet``."""
    out_root = Path(out_root)
    src = out_root / "days" / f"{day_label}.parquet"
    if not src.exists():
        raise FileNotFoundError(src)

    df = pq.read_table(src).to_pandas()
    pairs = build_pair_frame(
        df, fps=fps, gap_tolerance_s=gap_tolerance_s,
        min_episode_frames=min_episode_frames, max_distance_px=max_distance_px,
    )
    if not len(pairs):
        return None

    table = pa.Table.from_pandas(pairs, preserve_index=False)
    table = table.replace_schema_metadata(
        _file_metadata(fps, {
            "table": "pairs",
            "day_label": str(day_label),
            "min_episode_frames": min_episode_frames,
            "gap_tolerance_s": gap_tolerance_s,
            "pair_ordering": "TrackUID_1 < TrackUID_2",
            "bearing_definition": "bearing_1 = wrap(orbital - heading_1); "
                                  "bearing_2 = wrap(orbital + pi - heading_2)",
        })
    )
    dest = out_root / "days" / f"{day_label}{PAIRS_SUFFIX}.parquet"
    _atomic_write(table, dest, compression, compression_level, row_group_size)
    return dest


def write_pair_episodes(out_root: str | Path, fps: float) -> Path | None:
    """Rebuild ``<out_root>/pair_episodes.parquet`` from every pairs file."""
    out_root = Path(out_root)
    files = day_pair_files(out_root)
    if not files:
        return None

    frames = []
    for f in files:
        pairs = pq.read_table(f).to_pandas()
        ep = summarize_pair_episodes(pairs, fps)
        if len(ep):
            frames.append(ep)
    if not frames:
        return None

    episodes = pd.concat(frames, ignore_index=True).sort_values(
        ["day_index", "first_frame"]
    ).reset_index(drop=True)

    table = pa.Table.from_pandas(episodes, preserve_index=False)
    table = table.replace_schema_metadata(
        _file_metadata(fps, {"table": "pair_episodes", "n_days": len(files)})
    )
    dest = out_root / "pair_episodes.parquet"
    _atomic_write(table, dest, "zstd", 5, 500_000)
    return dest


def load_pairs(
    out_root: str | Path,
    day_label: str,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load one day of pair rows, sorted by episode then frame."""
    path = Path(out_root) / "days" / f"{day_label}{PAIRS_SUFFIX}.parquet"
    cols = list(columns) if columns else None
    if cols:
        for req in ("pair_episode_id", "FrameNum"):
            if req not in cols:
                cols.insert(0, req)
    return (
        pq.read_table(path, columns=cols)
        .to_pandas()
        .sort_values(["pair_episode_id", "FrameNum"])
        .reset_index(drop=True)
    )


def open_pairs(out_root: str | Path) -> tuple[pads.Dataset, pd.DataFrame]:
    """Return ``(pairs_dataset, pair_episodes_dataframe)`` for one project."""
    out_root = Path(out_root)
    ds = pads.dataset(day_pair_files(out_root), format="parquet")
    episodes = pq.read_table(out_root / "pair_episodes.parquet").to_pandas()
    return ds, episodes


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def open_project(out_root: str | Path) -> tuple[pads.Dataset, pd.DataFrame]:
    """Return ``(detections_dataset, tracks_dataframe)`` for one project.

    ``out_root`` is the project directory. The detections dataset is lazy and
    spans every day at once; the tracks table is small and eager::

        det, tracks = open_project("/dropbox/.../MC_pair_017")
        long_male = tracks.query("sex_consensus == 'male' and duration_s > 30")
        tbl = det.to_table(
            columns=["Time_s", "TrackUID", "heading_rad", "d_heading_rad_s"],
            filter=pads.field("TrackUID").isin(long_male.TrackUID.tolist()),
        )
    """
    proj = Path(out_root)
    det = pads.dataset(day_detection_files(proj), format="parquet")
    tracks = pq.read_table(proj / "tracks.parquet").to_pandas()
    return det, tracks


def load_day(
    out_root: str | Path,
    day_label: str,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load one tank-day by its label, sorted by track then frame."""
    path = Path(out_root) / "days" / f"{day_label}.parquet"
    cols = list(columns) if columns else None
    if cols:
        for req in ("FrameNum", "TrackUID"):
            if req not in cols:
                cols.insert(0, req)
    return (
        pq.read_table(path, columns=cols)
        .to_pandas()
        .sort_values(["TrackUID", "FrameNum"])
        .reset_index(drop=True)
    )


def _main() -> None:
    p = argparse.ArgumentParser(description="Convert a cichlid pose CSV to Parquet.")
    p.add_argument("csv")
    p.add_argument("--project-id", default=None,
                   help="defaults to the out-root directory name")
    p.add_argument("--day-index", type=int, required=True)
    p.add_argument("--day-label", required=True)
    p.add_argument("--fps", type=float, required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--recording-start", default=None)
    p.add_argument("--gap-tolerance-s", type=float, default=0.2)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    path = convert_day(
        a.csv, day_index=a.day_index, day_label=a.day_label, fps=a.fps,
        project_id=a.project_id,
        out_root=a.out_root, recording_start=a.recording_start,
        gap_tolerance_s=a.gap_tolerance_s, overwrite=a.overwrite,
    )
    write_project_tracks(a.out_root, a.fps)
    write_day_pairs(a.out_root, a.day_label, a.fps,
                    gap_tolerance_s=a.gap_tolerance_s)
    write_pair_episodes(a.out_root, a.fps)
    print(path)


if __name__ == "__main__":
    _main()