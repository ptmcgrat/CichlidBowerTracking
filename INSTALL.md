# Starting the branch

`claude_build` is an **orphan branch**: no shared history with `mcgrath_dev`,
so none of the old code comes with it. A normal `git checkout -b` would carry
the whole of `cichlid_bower_tracking/` across; this starts genuinely empty.

## 1. Clone somewhere separate

    git clone git@github.com:ptmcgrat/CichlidBowerTracking.git ~/CBC_build
    cd ~/CBC_build

Use a fresh clone. Step 2 deletes every tracked file from the working tree,
and doing that in your everyday checkout is a risk worth avoiding for the
price of one clone. `~/CichlidBowerTracking` stays untouched.

## 2. Make the orphan branch

    git checkout --orphan claude_build
    git rm -rf .

The working tree is now empty and nothing is staged.

## 3. Unpack

Unpack the archive at the root of the clone. It adds:

    cichlid_bower_claude/     the package
    tests/                    pytest suite and fixtures
    pyproject.toml            dependencies and the console script
    README.md
    INSTALL.md
    .gitignore

## 4. Install and check

    pip install -e ".[dev]"
    python -m pytest tests -q

Twenty-three tests, a few seconds, no network. They build a synthetic logfile
and a whole fake project — `Frames.tar` included — inside a temporary
directory standing in for Dropbox.

## 5. Point it at the data

    export CICHLID_LOCAL_ROOT=~/Temp/CichlidBowerClaude

Put it in your shell profile, or pass `--root` per command. Every command
prints the directory it resolved and where that came from, so a wrong one
shows up in the first line rather than as odd results later.

Nothing probes for it. The old code guessed by looking at mounted
directories, which depended on the machine and failed obscurely when it
guessed wrong.

## 6. First run

    python -m cichlid_bower_claude analyses
    python -m cichlid_bower_claude status YH_MC_Parentals
    python -m cichlid_bower_claude collect YH_MC_Parentals --projects MC_920_t001_tr1

Start with one project. It downloads the whole `Frames.tar`, which is the
expensive step, and reports the size before pulling it. The states file is
fetched automatically if it is not here yet; a misspelt analysisID lists the
ones that do exist.

    python -m cichlid_bower_claude collect YH_MC_Parentals

Then the sweep. A project with a current manifest is skipped, so this is safe
to rerun and safe to interrupt.

## 7. Commit and push

    git add .
    git commit -m "Collector: paths, cloud, logfile parsing, per-day bundle"
    git push -u origin claude_build

## On another machine

The branch exists now, so this is the whole of it:

    git clone --branch claude_build --single-branch \
        git@github.com:ptmcgrat/CichlidBowerTracking.git ~/CBC_build
    cd ~/CBC_build
    pip install -e ".[dev]"
    python -m pytest tests -q

`--single-branch` is worth the typing. Without it the clone drags in all of
`mcgrath_dev`'s history, which is most of the download and none of what a
machine that only runs the collector needs.

Then the two things that vary by machine:

    rclone listremotes                       # ptm_dropbox: must be there
    echo 'export CICHLID_LOCAL_ROOT=~/Temp/CichlidBowerClaude' >> ~/.bashrc

On a shared machine, point the root at a shared location rather than a home
directory — `/data/CichlidBowerClaude` or similar. The collector uploads what
it builds, so a project collected by one person should be downloaded once for
everyone rather than once per account.

### Updating

    git pull

That is usually all. The `-e` install imports straight from the clone, so code
changes take effect with no reinstall; rerun `pip install -e ".[dev]"` only if
the dependencies in `pyproject.toml` have changed.

## Living with an orphan branch

`git log` starts at that first commit. Switching between `claude_build` and
`mcgrath_dev` works normally and swaps the whole working tree. Merging in
either direction needs `--allow-unrelated-histories`, which is the trade for
a clean start — moving code between the branches means copying files, not
cherry-picking.

## What is not here yet

The page builders and the Flask server. Those still live in the old
`createServer.py` and `serveDashboard.py` on `mcgrath_dev`, which read the old
`FileManager` and the old `PrepFiles2` layout, so they cannot run against this
package's output as they stand. The HTML templates carry over nearly
unchanged; only the payload builders need rewriting against
`Collected/bundle.npz` and `manifest.json`.

Also absent: anything that runs the depth, cluster or pose stages. This
package collects, and will serve; it does not analyse.