#!/usr/bin/env python3
"""
spawning_clips.py -- cut annotation clips where pose-based circling detection
and the sand-manipulation classifier agree or disagree.

    python spawning_clips.py VIDEO POSE.parquet CLUSTERS.csv OUTDIR -n 100

Five output classes, each its own subdirectory:

    AgreedSpawning     sand says spawn (s), pose says circling
    AgreedNoSpawning   sand says some other behaviour, pose says no circling
    SandOnly           sand says spawn, pose missed it
    PoseOnly           pose says circling, sand says some other behaviour
    PoseNoSandEvent    pose says circling, no sand event nearby at all

The fifth class matters: the sand classifier only fires when a fish brushes
the substrate, so a pose detection with no sand event is *unresolved*, not
wrong. It is also where pose false positives collect. Either way it needs
eyes on it, and it cannot be folded into the other four.

Coordinate systems
------------------
Sand-event CSV stores X as the row index and Y as the column; pose stores
(x, y) with x as the column. They are transposed relative to each other:

    pose_x  <-  CSV Y        pose_y  <-  CSV X

Confirmed on MC_920_t001_tr1/0028_vid, where pose x tops out at 1201 against
the CSV's Y max of 1200, and pose y at 971 against the CSV's X max of 969.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Categories from the 3D ResNet sand-manipulation classifier.
SPAWN_CODE = "s"
REFLECTION_CODE = "x"          # artefact, not a behaviour -- always dropped
SAND_EVENT_S = 2.0             # each event is 2 s, centred on t

# Columns read from the pose parquet. Reading a subset keeps a 200 MB+ day
# manageable; nothing here needs the raw keypoints.
POSE_COLUMNS = [
    "FrameNum", "TrackUID", "Sex",
    "midline_centroid_x", "midline_centroid_y",
    "heading_rad", "d_heading_rad_s",
    "body_length_px", "qc_n_missing_kp",
]


# ==========================================================================
# Pair table -- rederived here so the co-localization logic stays editable
# ==========================================================================

def wrap_angle(a):
    """Wrap radians to (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def build_pair_table(
    pose: pd.DataFrame,
    *,
    fps: float,
    max_distance_px: float | None = None,
    require_male_female: bool = True,
    gap_tolerance_s: float = 0.2,
) -> pd.DataFrame:
    """One row per (frame, unordered track pair) with distances and angles.

    Sex is a per-detection prediction and flickers within a track, so the
    male/female gate uses each track's **consensus** sex rather than the
    per-frame call -- otherwise a pair blinks in and out of eligibility as
    the classifier wavers.

    ``max_distance_px`` prunes distant pairs before anything expensive is
    computed. It is applied *after* the per-frame geometry but *before* the
    episode/derivative stage, so it never breaks temporal contiguity within
    an episode.
    """
    df = pose.dropna(subset=["midline_centroid_x", "midline_centroid_y"])

    # --- consensus sex per track -----------------------------------------
    sex = df["Sex"].astype(str).str.lower()
    consensus = (
        sex.groupby(df["TrackUID"])
        .agg(lambda s: s.value_counts().idxmax() if len(s) else "unknown")
    )
    df = df.assign(track_sex=df["TrackUID"].map(consensus))

    keep = [
        "FrameNum", "TrackUID", "track_sex",
        "midline_centroid_x", "midline_centroid_y",
        "heading_rad", "d_heading_rad_s", "body_length_px", "qc_n_missing_kp",
    ]
    sub = df[keep]

    # --- self-join on frame, canonical ordering --------------------------
    merged = sub.merge(sub, on="FrameNum", suffixes=("_1", "_2"))
    merged = merged[merged["TrackUID_1"] < merged["TrackUID_2"]]
    if require_male_female:
        a = merged["track_sex_1"].to_numpy()
        b = merged["track_sex_2"].to_numpy()
        merged = merged[((a == "male") & (b == "female"))
                        | ((a == "female") & (b == "male"))]
    if not len(merged):
        return pd.DataFrame()

    x1 = merged["midline_centroid_x_1"].to_numpy("float64")
    y1 = merged["midline_centroid_y_1"].to_numpy("float64")
    x2 = merged["midline_centroid_x_2"].to_numpy("float64")
    y2 = merged["midline_centroid_y_2"].to_numpy("float64")
    dx, dy = x2 - x1, y2 - y1

    out = pd.DataFrame({
        "FrameNum": merged["FrameNum"].to_numpy(),
        "TrackUID_1": merged["TrackUID_1"].to_numpy(),
        "TrackUID_2": merged["TrackUID_2"].to_numpy(),
        "sex_1": merged["track_sex_1"].to_numpy(),
        "sex_2": merged["track_sex_2"].to_numpy(),
        "distance_px": np.hypot(dx, dy),
        # Midpoint: where in the frame the interaction is, for clip cropping.
        "mid_x": 0.5 * (x1 + x2),
        "mid_y": 0.5 * (y1 + y2),
        "mean_body_length_px": np.nanmean(
            np.column_stack([merged["body_length_px_1"].to_numpy("float64"),
                             merged["body_length_px_2"].to_numpy("float64")]),
            axis=1,
        ),
        "qc_missing": (merged["qc_n_missing_kp_1"].to_numpy()
                       + merged["qc_n_missing_kp_2"].to_numpy()),
    })

    # Angles. Image y increases downward, so increasing orbital_rad is
    # clockwise on screen.
    h1 = merged["heading_rad_1"].to_numpy("float64")
    h2 = merged["heading_rad_2"].to_numpy("float64")
    orbital = np.arctan2(dy, dx)
    out["orbital_rad"] = orbital
    out["rel_heading_rad"] = wrap_angle(h1 - h2)
    out["bearing_1_rad"] = wrap_angle(orbital - h1)      # 0 = fish 2 dead ahead
    out["bearing_2_rad"] = wrap_angle(orbital + np.pi - h2)
    out["d_heading_1_rad_s"] = merged["d_heading_rad_s_1"].to_numpy("float64")
    out["d_heading_2_rad_s"] = merged["d_heading_rad_s_2"].to_numpy("float64")

    if max_distance_px is not None:
        out = out[out["distance_px"] <= max_distance_px]
        if not len(out):
            return pd.DataFrame()

    out = out.sort_values(["TrackUID_1", "TrackUID_2", "FrameNum"],
                          kind="mergesort").reset_index(drop=True)

    # --- pair episodes ----------------------------------------------------
    # A maximal run of frames in which this same pair is present. Tracks
    # fragment constantly here because fish leave the field of view, so
    # derivatives must stay inside one episode: differencing across a gap
    # measures rotation over an interval where nothing was observed.
    max_gap = max(1, int(round(gap_tolerance_s * fps)))
    changed = (out["TrackUID_1"].ne(out["TrackUID_1"].shift())
               | out["TrackUID_2"].ne(out["TrackUID_2"].shift()))
    frame_gap = out["FrameNum"].diff()
    new_ep = changed | (frame_gap > max_gap) | frame_gap.isna()
    out["pair_episode_id"] = new_ep.cumsum().astype("int64") - 1
    out["dt_frames"] = np.where(new_ep, np.nan, frame_gap)

    # --- orbital angular velocity, within an episode only ----------------
    d_orb = wrap_angle(
        out.groupby("pair_episode_id", sort=False)["orbital_rad"].diff()
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        out["d_orbital_rad_s"] = d_orb / (out["dt_frames"] / fps)

    return out


# ==========================================================================
# Pose-based circling detection
# ==========================================================================

def detect_pose_bouts(
    pairs: pd.DataFrame,
    *,
    fps: float,
    window_s: float = 2.0,
    min_turn_rad: float = np.pi,
    max_rate_gap_rad_s: float = 1.5,
    max_distance_bl: float = 2.5,
    merge_gap_s: float = 60.0,
    min_windows: int = 15,
) -> pd.DataFrame:
    """Find circling bouts. Two conditions must hold together.

    1. Sustained rotation -- the pair turns >= ``min_turn_rad`` in a window.
    2. Rigid rotation -- the orbital rate agrees with both fish's own
       heading rates (they orbit each other rather than one circling a
       stationary partner).

    Neither suffices alone: a straight chase satisfies (2) trivially because
    all three rates are ~0, and one fish spinning beside a motionless one
    satisfies (1).

    ``min_windows`` exploits the fact that only *occurrence* matters here. A
    real bout runs for minutes and gives the detector thousands of chances to
    fire; an isolated false positive gives it one. Requiring repeated
    evidence costs almost no recall and removes most scattered noise.
    """
    if not len(pairs):
        return pd.DataFrame()

    w = max(2, int(round(window_s * fps)))
    p = pairs.sort_values(["pair_episode_id", "FrameNum"]).reset_index(drop=True)

    step = p["d_orbital_rad_s"] * p["dt_frames"] / fps
    turn = (step.groupby(p["pair_episode_id"], sort=False)
                .rolling(w, min_periods=w).sum()
                .reset_index(level=0, drop=True))

    gap = np.maximum(
        (p["d_orbital_rad_s"] - p["d_heading_1_rad_s"]).abs(),
        (p["d_orbital_rad_s"] - p["d_heading_2_rad_s"]).abs(),
    )
    mean_gap = (gap.groupby(p["pair_episode_id"], sort=False)
                   .rolling(w, min_periods=w).mean()
                   .reset_index(level=0, drop=True))
    mean_dist = (p["distance_px"].groupby(p["pair_episode_id"], sort=False)
                   .rolling(w, min_periods=w).mean()
                   .reset_index(level=0, drop=True))

    p["turn"] = turn
    p["rate_gap"] = mean_gap
    p["mean_dist"] = mean_dist

    dist_limit = max_distance_bl * p["mean_body_length_px"]
    hit = ((p["turn"].abs() >= min_turn_rad)
           & (p["rate_gap"] <= max_rate_gap_rad_s)
           & (p["mean_dist"] <= dist_limit))
    passing = p[hit.fillna(False)]
    if not len(passing):
        return pd.DataFrame()

    # --- assemble bouts by time contiguity, not pair identity ------------
    # One real bout spans many pair episodes and several TrackUIDs as tracks
    # break and reform; following a single pair through would shatter it.
    q = passing.sort_values("FrameNum")
    merge_frames = int(round(merge_gap_s * fps))
    new_bout = q["FrameNum"].diff().fillna(1e18) > merge_frames
    q = q.assign(_bout=new_bout.cumsum())

    rows = []
    for _, g in q.groupby("_bout", sort=True):
        if len(g) < min_windows:
            continue
        centre = int(g.loc[g["turn"].abs().idxmax(), "FrameNum"])
        rows.append({
            "start_frame": int(g["FrameNum"].min()) - w + 1,
            "end_frame": int(g["FrameNum"].max()),
            "centre_frame": centre,
            "n_windows": len(g),
            "peak_turn_rad": float(g["turn"].abs().max()),
            "median_turn_rad": float(g["turn"].abs().median()),
            "median_rate_gap": float(g["rate_gap"].median()),
            "median_distance_px": float(g["mean_dist"].median()),
            "mid_x": float(g["mid_x"].median()),
            "mid_y": float(g["mid_y"].median()),
            "n_pair_episodes": int(g["pair_episode_id"].nunique()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["start_s"] = out["start_frame"] / fps
        out["end_s"] = out["end_frame"] / fps
    return out


# ==========================================================================
# Sand events
# ==========================================================================

def load_sand_bouts(
    csv_path: str | Path, *, fps: float, merge_gap_s: float = 30.0,
    frame_range: tuple[int, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(spawn_bouts, other_events)`` from the cluster CSV.

    Rows with a null Prediction are clusters at the frame border where no
    clip could be created, so they were never classified -- they are
    *unknown*, not negative, and are dropped rather than counted as
    non-spawning. Reflections are an artefact class and are dropped too.

    ``frame_range`` clips events to the span the pose file actually covers.
    Without it, a full-day cluster CSV paired with a partial-day parquet
    reports every sand event outside that span as a pose miss.
    """
    ev = pd.read_csv(csv_path, index_col=0)
    ev = ev[ev["Prediction"].notna()]
    ev = ev[ev["Prediction"] != REFLECTION_CODE]

    # CSV X is the row index, Y the column -- transposed relative to pose.
    ev = ev.assign(pose_x=ev["Y"].astype(float), pose_y=ev["X"].astype(float))
    ev["frame"] = (ev["t"].astype(float) * fps).round().astype("int64")

    if frame_range is not None:
        lo, hi = frame_range
        ev = ev[(ev["frame"] >= lo) & (ev["frame"] <= hi)]

    spawn = ev[ev["Prediction"] == SPAWN_CODE].sort_values("t")
    other = ev[ev["Prediction"] != SPAWN_CODE].sort_values("t")

    if not len(spawn):
        return pd.DataFrame(), other

    # Merge the 2 s events into bouts; both partners generate events, so a
    # single spawning yields many and must be deduplicated.
    new = spawn["t"].diff().fillna(1e18) > merge_gap_s
    spawn = spawn.assign(_b=new.cumsum())
    g = spawn.groupby("_b")
    bouts = pd.DataFrame({
        "start_s": g["t"].min() - SAND_EVENT_S / 2,
        "end_s": g["t"].max() + SAND_EVENT_S / 2,
        "n_events": g.size(),
        "centre_frame": (g["t"].median() * fps).round().astype("int64"),
        "mid_x": g["pose_x"].median(),
        "mid_y": g["pose_y"].median(),
        "n_male": g["Sex"].apply(lambda s: (s == "Male").sum()),
        "n_female": g["Sex"].apply(lambda s: (s == "Female").sum()),
    }).reset_index(drop=True)
    bouts["start_frame"] = (bouts["start_s"] * fps).round().astype("int64")
    bouts["end_frame"] = (bouts["end_s"] * fps).round().astype("int64")
    return bouts, other


def _overlaps(a_start, a_end, b_start, b_end, tol_s):
    """Do two intervals come within ``tol_s`` of each other?"""
    return (a_start - tol_s <= b_end) & (b_start - tol_s <= a_end)


def _near_px(ax, ay, bx, by, tol_px):
    """Are two points within ``tol_px``? Accepts arrays for b."""
    return np.hypot(np.asarray(bx) - ax, np.asarray(by) - ay) <= tol_px


# ==========================================================================
# Cross-classification
# ==========================================================================

def classify(
    pose_bouts: pd.DataFrame,
    sand_bouts: pd.DataFrame,
    sand_other: pd.DataFrame,
    *,
    fps: float,
    tolerance_s: float = 30.0,
    match_px: float = 300.0,
    n_per_class: int,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign candidates to the five classes and sample within each.

    Matching is temporal AND spatial. Sand events are dense -- roughly one
    every ten seconds across a whole day -- so on time alone every moment in
    the video counts as "near a sand event", and the no-event class can never
    populate. Requiring the event to be within ``match_px`` of the pair makes
    the comparison about *these* fish rather than about anything happening
    anywhere in the tank.
    """
    rng = np.random.default_rng(seed)
    rows = []

    if len(sand_other):
        other_t = sand_other["t"].to_numpy(dtype=float)
        other_x = sand_other["pose_x"].to_numpy(dtype=float)
        other_y = sand_other["pose_y"].to_numpy(dtype=float)
    else:
        other_t = other_x = other_y = np.array([])

    def near_other(t: float, x: float, y: float) -> bool:
        if not len(other_t):
            return False
        lo, hi = np.searchsorted(other_t, [t - tolerance_s, t + tolerance_s])
        if hi <= lo:
            return False
        return bool(_near_px(x, y, other_x[lo:hi], other_y[lo:hi], match_px).any())

    # --- pose bouts: matched against sand spawn bouts --------------------
    for r in pose_bouts.itertuples():
        matched = False
        if len(sand_bouts):
            matched = bool((
                _overlaps(r.start_s, r.end_s,
                          sand_bouts["start_s"].to_numpy(),
                          sand_bouts["end_s"].to_numpy(), tolerance_s)
                & _near_px(r.mid_x, r.mid_y, sand_bouts["mid_x"].to_numpy(),
                           sand_bouts["mid_y"].to_numpy(), match_px)
            ).any())
        if matched:
            cls = "AgreedSpawning"
        elif near_other(r.centre_frame / fps, r.mid_x, r.mid_y):
            cls = "PoseOnly"
        else:
            cls = "PoseNoSandEvent"
        rows.append({
            "cls": cls, "centre_frame": r.centre_frame,
            "mid_x": r.mid_x, "mid_y": r.mid_y,
            "source": "pose", "peak_turn_rad": r.peak_turn_rad,
            "n_windows": r.n_windows,
            "median_distance_px": r.median_distance_px,
        })

    # --- sand spawn bouts the pose detector missed -----------------------
    for r in sand_bouts.itertuples():
        hit = False
        if len(pose_bouts):
            hit = bool((
                _overlaps(r.start_s, r.end_s,
                          pose_bouts["start_s"].to_numpy(),
                          pose_bouts["end_s"].to_numpy(), tolerance_s)
                & _near_px(r.mid_x, r.mid_y, pose_bouts["mid_x"].to_numpy(),
                           pose_bouts["mid_y"].to_numpy(), match_px)
            ).any())
        if not hit:
            rows.append({
                "cls": "SandOnly", "centre_frame": r.centre_frame,
                "mid_x": r.mid_x, "mid_y": r.mid_y,
                "source": "sand", "peak_turn_rad": np.nan,
                "n_windows": r.n_events, "median_distance_px": np.nan,
            })

    # --- AgreedNoSpawning: sand says another behaviour, pose says nothing -
    # Drawn from real sand events, not from empty stretches of video: a
    # negative is only informative if a fish was demonstrably doing
    # something at the time.
    if len(sand_other):
        cand = sand_other.sample(
            n=min(len(sand_other), n_per_class * 20), random_state=seed
        )
        for r in cand.itertuples():
            t = float(r.t)
            if len(pose_bouts) and (
                _overlaps(t, t, pose_bouts["start_s"].to_numpy(),
                          pose_bouts["end_s"].to_numpy(), tolerance_s)
                & _near_px(r.pose_x, r.pose_y, pose_bouts["mid_x"].to_numpy(),
                           pose_bouts["mid_y"].to_numpy(), match_px)
            ).any():
                continue
            if len(sand_bouts) and _overlaps(
                t, t, sand_bouts["start_s"].to_numpy(),
                sand_bouts["end_s"].to_numpy(), tolerance_s,
            ).any():
                continue
            rows.append({
                "cls": "AgreedNoSpawning", "centre_frame": int(t * fps),
                "mid_x": r.pose_x, "mid_y": r.pose_y,
                "source": "sand", "peak_turn_rad": np.nan,
                "n_windows": np.nan, "median_distance_px": np.nan,
            })

    if not rows:
        return pd.DataFrame()
    cand = pd.DataFrame(rows)

    # Sample within each class, spacing candidates so one long bout cannot
    # contribute many near-identical clips.
    picked = []
    for cls, g in cand.groupby("cls"):
        g = g.sample(frac=1.0, random_state=seed).sort_values("centre_frame")
        chosen, last = [], -10**9
        for row in g.itertuples():
            if row.centre_frame - last >= int(tolerance_s * fps):
                chosen.append(row.Index)
                last = row.centre_frame
        sel = cand.loc[chosen]
        if len(sel) > n_per_class:
            sel = sel.loc[rng.choice(sel.index, n_per_class, replace=False)]
        picked.append(sel)

    return pd.concat(picked).sort_values(["cls", "centre_frame"]).reset_index(drop=True)


# ==========================================================================
# Clip extraction
# ==========================================================================

def write_clip(video: Path, dest: Path, *, start_s, dur_s, cx, cy,
               crop_px: int, out_px: int) -> str:
    half = crop_px // 2
    vf = (f"crop={crop_px}:{crop_px}:"
          f"max(0\\,min(iw-{crop_px}\\,{int(cx) - half})):"
          f"max(0\\,min(ih-{crop_px}\\,{int(cy) - half})),"
          f"scale={out_px}:{out_px}")
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
           # -ss before -i seeks by keyframe rather than decoding from the
           # start of the file -- essential on a 10-hour recording.
           "-ss", f"{max(0.0, start_s):.3f}", "-i", str(video),
           "-t", f"{dur_s:.3f}", "-vf", vf,
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-crf", "24", "-an", str(dest)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return "ffmpeg not found"
    if p.returncode != 0:
        return f"ffmpeg: {p.stderr.strip()[:100]}"
    if not dest.exists() or dest.stat().st_size == 0:
        return "empty output"
    return "ok"


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("video")
    ap.add_argument("pose_parquet")
    ap.add_argument("clusters_csv")
    ap.add_argument("out_dir")
    ap.add_argument("-n", "--n-videos", type=int, default=100,
                    help="total clips across all classes")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--clip-s", type=float, default=10.0)
    ap.add_argument("--crop-px", type=int, default=500)
    ap.add_argument("--out-px", type=int, default=500)
    ap.add_argument("--tolerance-s", type=float, default=30.0,
                    help="pose/sand agreement window")
    ap.add_argument("--match-px", type=float, default=300.0,
                    help="spatial tolerance for pose/sand agreement")
    ap.add_argument("--sand-merge-s", type=float, default=30.0)
    ap.add_argument("--pose-merge-s", type=float, default=60.0)
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--min-turn-rad", type=float, default=np.pi)
    ap.add_argument("--max-rate-gap", type=float, default=1.5)
    ap.add_argument("--max-distance-bl", type=float, default=2.5)
    ap.add_argument("--min-windows", type=int, default=15)
    ap.add_argument("--pair-max-distance-px", type=float, default=600.0,
                    help="prune distant pairs before building the pair table")
    ap.add_argument("--any-sex", action="store_true",
                    help="do not require a male/female pair")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-tables", action="store_true",
                    help="write pair table and bout tables to OUTDIR")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    classes = ["AgreedSpawning", "AgreedNoSpawning",
               "SandOnly", "PoseOnly", "PoseNoSandEvent"]
    for c in classes:
        (out / c).mkdir(exist_ok=True)

    print("[1/5] reading pose parquet ...")
    pose = pq.read_table(args.pose_parquet, columns=POSE_COLUMNS).to_pandas()
    pair_frame_lo = int(pose.FrameNum.min())
    pair_frame_hi = int(pose.FrameNum.max())
    print(f"      {len(pose):,} detections, frames "
          f"{pair_frame_lo:,}-{pair_frame_hi:,} "
          f"({(pair_frame_hi - pair_frame_lo)/args.fps/60:.0f} min)")

    print("[2/5] building pair table ...")
    pairs = build_pair_table(
        pose, fps=args.fps, max_distance_px=args.pair_max_distance_px,
        require_male_female=not args.any_sex,
    )
    del pose
    print(f"      {len(pairs):,} pair rows, "
          f"{pairs.pair_episode_id.nunique() if len(pairs) else 0:,} episodes")

    print("[3/5] detecting circling ...")
    pose_bouts = detect_pose_bouts(
        pairs, fps=args.fps, window_s=args.window_s,
        min_turn_rad=args.min_turn_rad, max_rate_gap_rad_s=args.max_rate_gap,
        max_distance_bl=args.max_distance_bl, merge_gap_s=args.pose_merge_s,
        min_windows=args.min_windows,
    )
    print(f"      {len(pose_bouts)} pose bouts")

    frame_range = (int(pair_frame_lo), int(pair_frame_hi))
    sand_bouts, sand_other = load_sand_bouts(
        args.clusters_csv, fps=args.fps, merge_gap_s=args.sand_merge_s,
        frame_range=frame_range)
    print(f"      restricted to frames {frame_range[0]:,}-{frame_range[1]:,} "
          f"({frame_range[0]/args.fps/3600:.2f}-{frame_range[1]/args.fps/3600:.2f} h)")
    print(f"      {len(sand_bouts)} sand spawn bouts, "
          f"{len(sand_other):,} other sand events")

    if args.save_tables:
        pairs.to_parquet(out / "pair_table.parquet", index=False)
        pose_bouts.to_csv(out / "pose_bouts.csv", index=False)
        sand_bouts.to_csv(out / "sand_bouts.csv", index=False)

    print("[4/5] classifying ...")
    cand = classify(
        pose_bouts, sand_bouts, sand_other, fps=args.fps,
        tolerance_s=args.tolerance_s, match_px=args.match_px,
        n_per_class=max(1, args.n_videos // len(classes)), seed=args.seed,
    )
    if not len(cand):
        raise SystemExit("no candidates in any class")
    print(cand["cls"].value_counts().to_string())

    print("[5/5] writing clips ...")
    cand = cand.reset_index(drop=True)
    cand["clip"] = [f"{r.cls}/{r.cls}_{i:04d}.mp4"
                    for i, r in enumerate(cand.itertuples())]
    status = []
    for r in cand.itertuples():
        status.append(write_clip(
            Path(args.video), out / r.clip,
            start_s=r.centre_frame / args.fps - args.clip_s / 2,
            dur_s=args.clip_s, cx=r.mid_x, cy=r.mid_y,
            crop_px=args.crop_px, out_px=args.out_px,
        ))
    cand["status"] = status
    cand.to_csv(out / "candidates.csv", index=False)

    ok = (cand["status"] == "ok").sum()
    print(f"\nwrote {ok}/{len(cand)} clips -> {out}")
    bad = cand.loc[cand["status"] != "ok", "status"].value_counts()
    if len(bad):
        print(bad.to_string())


if __name__ == "__main__":
    main()