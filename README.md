# cichlid_bower_claude

Collects cichlid bower project data into a small, self-describing bundle, so
the review server can work from it without ever touching `Frames.tar` again.

## What it does

One archive read per project. Everything downstream — the crop, the residual
mask, the bower thresholds, the trial offsets — is applied when the server
draws, not baked into the data. Changing any of them costs a page rebuild
rather than a recollection.

## Install

    pip install numpy pandas
    export CICHLID_LOCAL_ROOT=~/Temp/CichlidAnalyzer

rclone must be configured with the `ptm_dropbox` remote. Nothing probes for
the working directory: if `CICHLID_LOCAL_ROOT` is unset, the commands say so
and stop.

## Use

    python -m cichlid_bower_claude analyses
    python -m cichlid_bower_claude status YH_MC_Parentals
    python -m cichlid_bower_claude collect YH_MC_Parentals
    python -m cichlid_bower_claude build   YH_MC_Parentals
    python -m cichlid_bower_claude serve   YH_MC_Parentals --port 8080

`collect` reads the archives; `build` turns a collected bundle into page
assets; `serve` puts them on HTTP and takes corrections back. `serve` builds a
page on first visit, so `build` is only needed to warm them in advance.

Useful flags: `--force` recollects even where the manifest is current,
`--keep-archive` leaves `Frames.tar` on disk, `--dry-run` makes no cloud
changes, and `--raise` lets an exception escape instead of being recorded as
a failed project.

## What it writes

Under `<root>/<projectID>/Collected/`:

| File | What it is |
| --- | --- |
| `manifest.json` | Written last, after everything else has uploaded. The completion marker. |
| `bundle.npz` | Eight per-day arrays, each `(days, height, width)`, float16 |
| `Bound_T<n>_<kind>_<offset>.npy` / `.jpg` | Boundary candidates for the trial-times tab |
| `Frame_NNNNNN.npy` | Day endpoint arrays |
| `Day_NN_first.jpg` / `Day_NN_last.jpg` | Day stills |

Bundle arrays: `first`, `last`, `residual`, `trend`, `travel`, `valid`,
`std_mean`, `std_max`.

About 80 MB per project for a 34-day recording at 640x480, against a
`Frames.tar` measured in gigabytes.

## Layout

| Module | Responsibility |
| --- | --- |
| `paths.py` | Where everything lives. Pure computation, no I/O |
| `cloud.py` | rclone. The only module that shells out |
| `states.py` | The analysis states CSV. The only module that imports pandas |
| `logfile/model.py` | `Frame`, `Movie`, `Trial`, `ProjectLog` |
| `logfile/parse.py` | Logfile to `ProjectLog`; records bad lines rather than raising |
| `logfile/days.py` | Day boundaries as a function of the trial offsets |
| `collect/archive.py` | Reading `Frames.tar`, streamed a day at a time |
| `collect/residual.py` | Per-pixel statistics. Arrays in, arrays out |
| `collect/frames.py` | Which frames to keep |
| `collect/bundle.py` | The bundle and its manifest |
| `collect/collector.py` | One project, end to end |
| `server/encode.py` | Depth arrays as PNGs the browser can read values from |
| `server/payload.py` | Bundle to page assets: a directory of PNGs and one JSON |
| `server/corrections.py` | `Corrections/prep.json`, the only thing the server writes |
| `server/app.py` | Flask: serves pages, accepts saves |
| `cli.py` | Entry points |

Three of those write nothing at all. That is deliberate: `paths`, `residual`
and `frames` are pure, so they are tested against answers worked out by hand
rather than against a network.

## Tests

    python -m pytest tests -q
    npm install jsdom && python -m pytest tests -q   # includes the page renders

`tests/make_log.py` writes a synthetic logfile in the real format;
`tests/make_project.py` builds a whole fake project — logfile, `Frames.tar`
with a growing castle and a churning patch — inside a directory standing in
for Dropbox. `cloud.FakeCloud` moves real bytes between them, so a test can
assert on what ended up where.

`tests/test_render.py` loads every page in a real DOM and fails if it reports
an error or renders nothing. It needs `npm install jsdom` and is skipped
without it. Checking each script on its own catches a typo and nothing else:
the faults that reach people need the whole page running — a name declared
in two scripts that share a scope, or a control referenced by an id that is not
in the markup. Both leave a blank screen.

Do not hold a real `Frames.tar` in memory. The test fixture does, which is
why it uses small frames; the collector itself streams.
