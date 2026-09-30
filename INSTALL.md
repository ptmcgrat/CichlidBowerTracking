# Starting the branch

The package is self-contained: no imports from the existing code, so it can
live beside it without either interfering with the other.

## 1. Branch from the existing repository

    cd ~/CichlidBowerTracking
    git checkout mcgrath_dev
    git pull
    git checkout -b claude_build

## 2. Drop the package in

Unpack the archive at the repository root. It adds:

    cichlid_bower_claude/     the package
    tests/                    pytest suite and fixtures
    pyproject.toml            dependencies and the console script
    README.md
    .gitignore

If the repository already has a `tests/` directory, unpack into a
subdirectory and move `tests/` in by hand rather than letting it merge.

## 3. Install and check

    pip install -e ".[dev]"
    python -m pytest tests -q

Twenty-one tests, a few seconds, no network. They build a synthetic logfile
and a whole fake project — `Frames.tar` included — inside a temporary
directory standing in for Dropbox.

## 4. Point it at the data

    export CICHLID_LOCAL_ROOT=~/Temp/CichlidAnalyzer

Put that in your shell profile on utaka. Nothing probes for the working
directory; if it is unset the commands say so and exit 2.

## 5. First run

    python -m cichlid_bower_claude analyses
    python -m cichlid_bower_claude status YH_MC_Parentals
    python -m cichlid_bower_claude collect YH_MC_Parentals --projects MC_920_t001_tr1

Start with one project. It downloads the whole `Frames.tar`, which is the
expensive step, and reports the size before pulling it.

    python -m cichlid_bower_claude collect YH_MC_Parentals

Then the sweep. A project that already has a current manifest is skipped, so
this is safe to rerun and safe to interrupt.

## 6. Commit

    git add cichlid_bower_claude tests pyproject.toml README.md .gitignore
    git commit -m "Collector: paths, cloud, logfile parsing, per-day bundle"
    git push -u origin claude_build

## What is not here yet

The page builders and the Flask server. Those still live in the old
`createServer.py` and `serveDashboard.py`, which read the old `FileManager`
and the old `PrepFiles2` layout, so they cannot run against this package's
output as they stand. The HTML templates carry over nearly unchanged; only
the payload builders need rewriting against `Collected/bundle.npz` and
`manifest.json`.

Also absent: anything that runs the depth, cluster or pose stages. This
package collects and will serve; it does not analyse.
