#!/usr/bin/env python3
"""
calibrate.py -- tune the circling detector against sand-manipulation events
across many days.

Two stages, because the expensive work does not depend on any threshold:

    # once -- builds pair tables and rolling window features, in parallel
    python calibrate.py precompute POSE_DIR CSV_DIR CACHE_DIR --workers 8

    # cheap -- grid-search thresholds against the cache, repeat freely
    python calibrate.py sweep CACHE_DIR --out results.parquet

    # summarise, with a project-level train/holdout split
    python calibrate.py report results.parquet --holdout-frac 0.3

Files are matched by stem: POSE_DIR/<base>.parquet with CSV_DIR/<base>.csv.
Pass the same path twice if both live in one folder.

Why per-project rather than per-day
-----------------------------------
Days within a project share fish, tank, camera and bower, so a day-level
train/test split leaks all of that and reports optimistic numbers that do not
survive a new project. Splits here are always by project, and results are
reported as the *distribution* of per-day recall rather than a pooled mean --
a parameter set averaging 70% that never drops below 55% is more useful than
one averaging 75% that fails completely on a fifth of projects.

What this can and cannot tell you
---------------------------------
The sand classifier only fires when a fish contacts the substrate, so it is
not a gold standard. Tuning to maximise agreement with it optimises for
spawnings involving sand contact -- exactly the subset pose was meant to
extend beyond. Use this to *rank* parameter sets, then hand-label clips
(especially PoseNoSandEvent and SandOnly) before treating any number as
performance.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import spawning_clips as sc

# Rows below this turn magnitude are never cached. The sweep refuses to test a
# min_turn_rad below it, since those rows are simply absent from the cache and
# would silently look like zero detections rather than an error.
PREFILTER_TURN_RAD = np.pi / 4
PREFILTER_DISTANCE_BL = 4.0

CACHE_COLUMNS = [
    "FrameNum", "pair_episode_id", "turn", "rate_gap",
    "mean_dist", "mean_bl", "mid_x", "mid_y",
]


# ==========================================================================
# Stage 1 -- precompute
# ==========================================================================

def compute_day_windows(
    pose: pd.DataFrame,
    *,
    fps: float,
    window_s: float,
    pair_max_distance_px: float,
    require_male_female: bool,
) -> tuple[pd.DataFrame, int, int]:
    """Pair table -> rolling window features, prefiltered.

    Returns ``(windows, n_pair_rows, n_episodes)``. The rolling statistics are
    computed here and cached, so a threshold sweep later is pure comparison.
    """
    pairs = sc.build_pair_table(
        pose, fps=fps, max_distance_px=pair_max_distance_px,
        require_male_female=require_male_female,
    )
    if not len(pairs):
        return pd.DataFrame(columns=CACHE_COLUMNS), 0, 0

    n_rows = len(pairs)
    n_eps = int(pairs["pair_episode_id"].nunique())
    w = max(2, int(round(window_s * fps)))
    g = pairs.groupby("pair_episode_id", sort=False)

    step = pairs["d_orbital_rad_s"] * pairs["dt_frames"] / fps
    turn = (step.groupby(pairs["pair_episode_id"], sort=False)
                .rolling(w, min_periods=w).sum()
                .reset_index(level=0, drop=True))

    gap = np.maximum(
        (pairs["d_orbital_rad_s"] - pairs["d_heading_1_rad_s"]).abs(),
        (pairs["d_orbital_rad_s"] - pairs["d_heading_2_rad_s"]).abs(),
    )
    rate_gap = (gap.groupby(pairs["pair_episode_id"], sort=False)
                   .rolling(w, min_periods=w).mean()
                   .reset_index(level=0, drop=True))
    mean_dist = (pairs["distance_px"].groupby(pairs["pair_episode_id"], sort=False)
                   .rolling(w, min_periods=w).mean()
                   .reset_index(level=0, drop=True))

    out = pd.DataFrame({
        "FrameNum": pairs["FrameNum"].to_numpy("int64"),
        "pair_episode_id": pairs["pair_episode_id"].to_numpy("int64"),
        "turn": turn.to_numpy("float32"),
        "rate_gap": rate_gap.to_numpy("float32"),
        "mean_dist": mean_dist.to_numpy("float32"),
        "mean_bl": pairs["mean_body_length_px"].to_numpy("float32"),
        "mid_x": pairs["mid_x"].to_numpy("float32"),
        "mid_y": pairs["mid_y"].to_numpy("float32"),
    })
    keep = (
        out["turn"].abs().ge(PREFILTER_TURN_RAD)
        & out["mean_dist"].le(PREFILTER_DISTANCE_BL * out["mean_bl"])
    )
    return out[keep.fillna(False)].reset_index(drop=True), n_rows, n_eps


def _precompute_one(job: dict) -> dict:
    """Process one day. Runs in a worker process."""
    base = job["base"]
    res = {"base_name": base, "ok": False, "error": None}
    try:
        pose = pq.read_table(job["parquet"], columns=sc.POSE_COLUMNS).to_pandas()
        if not len(pose):
            raise ValueError("empty pose file")
        lo, hi = int(pose.FrameNum.min()), int(pose.FrameNum.max())

        # project_id lives in the parquet, so a day is attributable even if
        # the filename says nothing about which project it came from.
        try:
            pid = str(pq.read_table(job["parquet"], columns=["project_id"])
                        .to_pandas()["project_id"].iloc[0])
        except Exception:
            pid = base

        win, n_rows, n_eps = compute_day_windows(
            pose, fps=job["fps"], window_s=job["window_s"],
            pair_max_distance_px=job["pair_max_distance_px"],
            require_male_female=job["require_male_female"],
        )
        del pose

        sand, other = sc.load_sand_bouts(
            job["csv"], fps=job["fps"], merge_gap_s=job["sand_merge_s"],
            frame_range=(lo, hi),
        )

        cache = Path(job["cache"])
        win.to_parquet(cache / "windows" / f"{base}.parquet", index=False)
        (sand if len(sand) else pd.DataFrame(
            columns=["start_s", "end_s", "mid_x", "mid_y", "n_events"]
        )).to_parquet(cache / "sand" / f"{base}.parquet", index=False)

        res.update(
            ok=True, project_id=pid, frame_lo=lo, frame_hi=hi,
            duration_s=(hi - lo) / job["fps"],
            n_pair_rows=n_rows, n_episodes=n_eps, n_windows_cached=len(win),
            n_sand_bouts=len(sand), n_sand_other=len(other),
        )
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["traceback"] = traceback.format_exc()[-400:]
    return res


def precompute(args) -> None:
    pose_dir, csv_dir = Path(args.pose_dir), Path(args.csv_dir)
    cache = Path(args.cache_dir)
    (cache / "windows").mkdir(parents=True, exist_ok=True)
    (cache / "sand").mkdir(parents=True, exist_ok=True)

    pairs = []
    for p in sorted(pose_dir.glob("*.parquet")):
        c = csv_dir / f"{p.stem}.csv"
        if c.exists():
            pairs.append((p.stem, p, c))
    if not pairs:
        raise SystemExit(f"no matched <base>.parquet / <base>.csv in "
                         f"{pose_dir} and {csv_dir}")

    unmatched = len(list(pose_dir.glob("*.parquet"))) - len(pairs)
    print(f"{len(pairs)} matched days" +
          (f" ({unmatched} parquet files without a CSV)" if unmatched else ""))

    if args.limit:
        pairs = pairs[:args.limit]
        print(f"limited to {len(pairs)}")

    jobs = [{
        "base": b, "parquet": str(p), "csv": str(c), "cache": str(cache),
        "fps": args.fps, "window_s": args.window_s,
        "pair_max_distance_px": args.pair_max_distance_px,
        "require_male_female": not args.any_sex,
        "sand_merge_s": args.sand_merge_s,
    } for b, p, c in pairs]

    if args.resume:
        have = {f.stem for f in (cache / "windows").glob("*.parquet")}
        jobs = [j for j in jobs if j["base"] not in have]
        print(f"resuming: {len(jobs)} remaining")
        if not jobs:
            print("nothing to do -- all days already cached")
            return

    results = []
    n_workers = args.workers or min(len(jobs), os.cpu_count() or 1)
    if n_workers > 1 and len(jobs) > 1:
        with cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
            for i, r in enumerate(pool.map(_precompute_one, jobs), 1):
                results.append(r)
                _log_precompute(r, i, len(jobs))
    else:
        for i, j in enumerate(jobs, 1):
            r = _precompute_one(j)
            results.append(r)
            _log_precompute(r, i, len(jobs))

    idx_path = cache / "index.parquet"
    new = pd.DataFrame(results)
    if args.resume and idx_path.exists():
        old = pd.read_parquet(idx_path)
        new = pd.concat([old[~old.base_name.isin(new.base_name)], new],
                        ignore_index=True)
    meta = {
        "prefilter_turn_rad": PREFILTER_TURN_RAD,
        "prefilter_distance_bl": PREFILTER_DISTANCE_BL,
        "window_s": args.window_s, "fps": args.fps,
        "pair_max_distance_px": args.pair_max_distance_px,
        "sand_merge_s": args.sand_merge_s,
        "require_male_female": not args.any_sex,
    }
    new.to_parquet(idx_path, index=False)
    (cache / "meta.json").write_text(json.dumps(meta, indent=2))

    ok = new[new.ok] if "ok" in new else new
    print(f"\n{len(ok)}/{len(new)} days cached")
    if len(ok):
        print(f"  {int(ok.n_sand_bouts.sum()):,} sand spawn bouts, "
              f"{ok.project_id.nunique()} projects, "
              f"{ok.duration_s.sum()/3600:.0f} h")
    bad = new[~new.ok] if "ok" in new else new.iloc[:0]
    if len(bad):
        print(f"  {len(bad)} failed:")
        for _, r in bad.head(5).iterrows():
            print(f"    {r.base_name}: {r.error}")


def _log_precompute(r: dict, i: int, n: int) -> None:
    if r["ok"]:
        print(f"  [{i}/{n}] {r['base_name']:<28} "
              f"{r['n_windows_cached']:>7,} windows  "
              f"{r['n_sand_bouts']:>3} sand bouts")
    else:
        print(f"  [{i}/{n}] {r['base_name']:<28} FAIL {r['error']}")


# ==========================================================================
# Stage 2 -- sweep
# ==========================================================================

def assemble_bouts(
    win: pd.DataFrame, *, fps: float, window_s: float,
    min_turn_rad: float, max_rate_gap: float, max_distance_bl: float,
    merge_gap_s: float, min_windows: int,
) -> pd.DataFrame:
    """Threshold cached windows and merge survivors into bouts.

    Bouts merge by time contiguity rather than pair identity: one real bout
    spans many pair episodes as tracks break and reform, so following a
    single pair through would shatter it.
    """
    hit = (
        win["turn"].abs().ge(min_turn_rad)
        & win["rate_gap"].le(max_rate_gap)
        & win["mean_dist"].le(max_distance_bl * win["mean_bl"])
    )
    q = win[hit.fillna(False)]
    if not len(q):
        return pd.DataFrame(columns=["start_s", "end_s", "mid_x", "mid_y"])

    q = q.sort_values("FrameNum")
    w = max(2, int(round(window_s * fps)))
    new = q["FrameNum"].diff().fillna(1e18) > merge_gap_s * fps
    q = q.assign(_b=new.cumsum())

    g = q.groupby("_b")
    out = pd.DataFrame({
        "start_s": (g["FrameNum"].min() - w + 1) / fps,
        "end_s": g["FrameNum"].max() / fps,
        "mid_x": g["mid_x"].median(),
        "mid_y": g["mid_y"].median(),
        "n_windows": g.size(),
    })
    return out[out["n_windows"] >= min_windows].reset_index(drop=True)


def score_day(
    win: pd.DataFrame, sand: pd.DataFrame, *, fps: float, window_s: float,
    tolerance_s: float, match_px: float, params: dict,
) -> dict:
    """Recall and matched-fraction for one day under one parameter set."""
    b = assemble_bouts(win, fps=fps, window_s=window_s, **params)

    n_sand, n_pose = len(sand), len(b)
    rec = mat = 0
    if n_sand and n_pose:
        bs, be = b["start_s"].to_numpy(), b["end_s"].to_numpy()
        bx, by = b["mid_x"].to_numpy(), b["mid_y"].to_numpy()
        ss, se = sand["start_s"].to_numpy(), sand["end_s"].to_numpy()
        sx, sy = sand["mid_x"].to_numpy(), sand["mid_y"].to_numpy()
        rec = sum(bool((sc._overlaps(a, z, bs, be, tolerance_s)
                        & sc._near_px(x, y, bx, by, match_px)).any())
                  for a, z, x, y in zip(ss, se, sx, sy))
        mat = sum(bool((sc._overlaps(a, z, ss, se, tolerance_s)
                        & sc._near_px(x, y, sx, sy, match_px)).any())
                  for a, z, x, y in zip(bs, be, bx, by))

    return {
        "n_sand_bouts": n_sand, "n_pose_bouts": n_pose,
        "n_recovered": rec, "n_matched": mat,
        "coverage_s": float((b["end_s"] - b["start_s"]).sum()) if n_pose else 0.0,
    }


def _grid(args) -> list[dict]:
    out = []
    for t in args.turn:
        for g in args.rate_gap:
            for d in args.distance_bl:
                for m in args.min_windows:
                    out.append({
                        "min_turn_rad": t, "max_rate_gap": g,
                        "max_distance_bl": d, "min_windows": m,
                        "merge_gap_s": args.merge_gap_s,
                    })
    return out


def sweep(args) -> None:
    cache = Path(args.cache_dir)
    meta = json.loads((cache / "meta.json").read_text())
    idx = pd.read_parquet(cache / "index.parquet")
    idx = idx[idx.ok]

    grid = _grid(args)
    below = [p for p in grid if p["min_turn_rad"] < meta["prefilter_turn_rad"]]
    if below:
        raise SystemExit(
            f"min_turn_rad below the cache prefilter "
            f"({meta['prefilter_turn_rad']:.3f}) -- those rows were never "
            f"cached and would look like zero detections. Re-run precompute "
            f"with a lower PREFILTER_TURN_RAD to explore that range."
        )
    print(f"{len(grid)} parameter sets x {len(idx)} days "
          f"({idx.project_id.nunique()} projects)")

    rows = []
    for n, r in enumerate(idx.itertuples(), 1):
        wf = cache / "windows" / f"{r.base_name}.parquet"
        sf = cache / "sand" / f"{r.base_name}.parquet"
        if not wf.exists():
            continue
        win = pd.read_parquet(wf)
        sand = pd.read_parquet(sf)
        for pi, params in enumerate(grid):
            s = score_day(win, sand, fps=meta["fps"],
                          window_s=meta["window_s"],
                          tolerance_s=args.tolerance_s,
                          match_px=args.match_px, params=params)
            rows.append({"param_id": pi, "base_name": r.base_name,
                         "project_id": r.project_id,
                         "duration_s": r.duration_s, **params, **s})
        if n % 25 == 0 or n == len(idx):
            print(f"  {n}/{len(idx)} days")

    res = pd.DataFrame(rows)
    res.to_parquet(args.out, index=False)
    print(f"\nwrote {len(res):,} rows -> {args.out}")
    _summarise(res, holdout_frac=0.0, seed=args.seed, top=args.top,
               max_cover=args.max_cover)


# ==========================================================================
# Reporting
# ==========================================================================

def _summarise(res: pd.DataFrame, *, holdout_frac: float, seed: int,
               top: int, max_cover: float = 0.05) -> None:
    """Rank parameter sets on per-day distributions, split by project."""
    projects = np.array(sorted(res.project_id.unique()))
    rng = np.random.default_rng(seed)
    if holdout_frac > 0 and len(projects) > 1:
        n_hold = max(1, int(round(len(projects) * holdout_frac)))
        hold = set(rng.choice(projects, n_hold, replace=False))
    else:
        hold = set()
    res = res.assign(split=np.where(res.project_id.isin(hold), "holdout", "train"))

    for split in ["train", "holdout"]:
        d = res[res.split == split]
        if not len(d):
            continue
        # Per-day rates first, then summarised -- pooling would hide a
        # parameter set that works on most projects and fails on a few.
        d = d.assign(
            recall=np.where(d.n_sand_bouts > 0, d.n_recovered / d.n_sand_bouts, np.nan),
            matched=np.where(d.n_pose_bouts > 0, d.n_matched / d.n_pose_bouts, np.nan),
            cover_frac=d.coverage_s / d.duration_s.clip(lower=1),
        )
        g = d.groupby(["param_id", "min_turn_rad", "max_rate_gap",
                       "max_distance_bl", "min_windows"], dropna=False)
        summ = g.agg(
            days=("base_name", "nunique"),
            recall_med=("recall", "median"),
            recall_q1=("recall", lambda s: s.quantile(0.25)),
            matched_med=("matched", "median"),
            cover_med=("cover_frac", "median"),
            sand=("n_sand_bouts", "sum"),
            pose=("n_pose_bouts", "sum"),
        ).reset_index()
        # Rank by the weaker of recall and the precision-like matched rate,
        # so a set cannot win by firing almost never. Coverage matters just
        # as much in the other direction: a detector active 20% of the day
        # matches everything by accident, so sets above max_cover are
        # excluded outright and coverage breaks ties among the rest.
        summ["score"] = np.minimum(summ.recall_med.fillna(0),
                                   summ.matched_med.fillna(0))
        eligible = summ[summ.cover_med <= max_cover]
        excluded = len(summ) - len(eligible)
        if len(eligible):
            summ = eligible
        summ = summ.sort_values(["score", "cover_med"],
                                ascending=[False, True])

        print(f"\n=== {split} ({d.project_id.nunique()} projects, "
              f"{d.base_name.nunique()} days) ===")
        if excluded:
            print(f"({excluded}/{excluded + len(summ)} sets excluded: "
                  f"coverage above {max_cover:.0%} of the day"
                  + ("" if len(eligible) else " -- none left, showing all")
                  + ")")
        cols = ["min_turn_rad", "max_rate_gap", "max_distance_bl",
                "min_windows", "recall_med", "recall_q1", "matched_med",
                "cover_med", "sand", "pose"]
        print(summ[cols].head(top).round(3).to_string(index=False))


def report(args) -> None:
    res = pd.read_parquet(args.results)
    _summarise(res, holdout_frac=args.holdout_frac, seed=args.seed,
               top=args.top, max_cover=args.max_cover)


# ==========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("precompute")
    p1.add_argument("pose_dir")
    p1.add_argument("csv_dir")
    p1.add_argument("cache_dir")
    p1.add_argument("--fps", type=float, default=30.0)
    p1.add_argument("--window-s", type=float, default=2.0)
    p1.add_argument("--pair-max-distance-px", type=float, default=600.0)
    p1.add_argument("--sand-merge-s", type=float, default=30.0)
    p1.add_argument("--any-sex", action="store_true")
    p1.add_argument("--workers", type=int, default=None)
    p1.add_argument("--limit", type=int, default=None)
    p1.add_argument("--resume", action="store_true")
    p1.set_defaults(func=precompute)

    p2 = sub.add_parser("sweep")
    p2.add_argument("cache_dir")
    p2.add_argument("--out", default="results.parquet")
    p2.add_argument("--turn", type=float, nargs="+",
                    default=[np.pi/2, np.pi*0.75, np.pi, np.pi*1.5])
    p2.add_argument("--rate-gap", type=float, nargs="+",
                    default=[1.5, 2.0, 2.5, 3.0])
    p2.add_argument("--distance-bl", type=float, nargs="+",
                    default=[1.5, 2.0, 2.5])
    p2.add_argument("--min-windows", type=int, nargs="+",
                    default=[5, 10, 20])
    p2.add_argument("--merge-gap-s", type=float, default=60.0)
    p2.add_argument("--tolerance-s", type=float, default=30.0)
    p2.add_argument("--match-px", type=float, default=300.0)
    p2.add_argument("--top", type=int, default=15)
    p2.add_argument("--max-cover", type=float, default=0.05,
                    help="reject sets whose bouts cover more than this "
                         "fraction of the day")
    p2.add_argument("--seed", type=int, default=0)
    p2.set_defaults(func=sweep)

    p3 = sub.add_parser("report")
    p3.add_argument("results")
    p3.add_argument("--holdout-frac", type=float, default=0.3)
    p3.add_argument("--top", type=int, default=15)
    p3.add_argument("--max-cover", type=float, default=0.05)
    p3.add_argument("--seed", type=int, default=0)
    p3.set_defaults(func=report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()