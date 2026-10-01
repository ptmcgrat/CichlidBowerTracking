"""Sand-manipulation events, packed small.

The cluster stage writes one row per detected event with a position, a
behaviour code and a probability. A long project has hundreds of thousands, so
they are stored as typed columns rather than a table: eleven bytes an event
instead of a hundred, which is what lets the page hold every one of them and
re-filter without asking the server.

Positions are in Pi camera coordinates, and the file's X is the row index while
Y is the column — transposed relative to pose data. That has caused real errors,
so the transposition is undone here, once, and everything downstream sees
``x`` across and ``y`` down.
"""

from __future__ import annotations

import base64
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# the ten codes the classifier assigns, in a fixed order so the packed index
# means the same thing in every project
BIDS = ('c', 'p', 'b', 'f', 't', 'm', 's', 'd', 'o', 'x')
LABELS = {'c': 'bower scoop', 'p': 'bower spit', 'b': 'bower multiple',
          'f': 'feed scoop', 't': 'feed spit', 'm': 'feed multiple',
          's': 'spawn', 'd': 'drop sand', 'o': 'fish other',
          'x': 'no fish other'}
NO_PREDICTION = 255

FLAG_CLIP = 1          # a clip was cut, so the event could be classified
FLAG_CLUSTERED = 2     # LID is not -1: the detection joined a cluster
FLAG_CLUSTERED = 2     # LID is not -1: the transitions grouped into an event


class ClustersMissing(Exception):
    """No cluster file for this project."""


def read_clusters(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise ClustersMissing('no cluster file at ' + str(path))
    table = pd.read_csv(path, index_col='TimeStamp', parse_dates=True)
    for column in ('X', 'Y', 'Prediction'):
        if column not in table.columns:
            raise ClustersMissing('column ' + column + ' is missing from ' + str(path))
    return table.sort_index()


def pack(table: pd.DataFrame, days: List[dict]) -> Dict[str, object]:
    """Typed columns, base64 encoded, plus what the page needs to read them.

    Each event is assigned to the day whose lights-on window contains it. One
    that falls in no day — at night, or outside every trial — is given day 65535
    and can still be counted, because a cluster detected outside a trial is a
    fact about the project rather than something to hide.
    """
    times = table.index
    count = len(table)

    # undo the transposition here so nothing downstream has to remember it
    across = table['Y'].to_numpy(float)
    down = table['X'].to_numpy(float)

    codes = {bid: i for i, bid in enumerate(BIDS)}
    bid = table['Prediction'].map(codes).fillna(NO_PREDICTION).astype(np.uint8)

    probability = (table['Probability'] if 'Probability' in table.columns
                   else pd.Series(1.0, index=table.index))
    flags = np.zeros(count, np.uint8)
    if 'ClipCreated' in table.columns:
        flags |= (table['ClipCreated'] == 'Yes').to_numpy().astype(np.uint8) * FLAG_CLIP
    else:
        flags |= FLAG_CLIP

    # LID -1 marks a transition that never joined a cluster. Those are not
    # events and must not be plotted as if they were — but the proportion of
    # them is a direct measure of how well the clustering worked, so they are
    # kept and flagged rather than dropped here.
    if 'LID' in table.columns:
        flags |= (table['LID'].astype('Int64') != -1).fillna(False).to_numpy(
            dtype=bool).astype(np.uint8) * FLAG_CLUSTERED
    else:
        flags |= FLAG_CLUSTERED

    # LID of -1 means the transition never joined a cluster. Those rows are not
    # events and must stay out of any count of behaviour; the proportion of
    # them is a useful measure of how well the clustering worked, so they are
    # flagged rather than dropped.
    if 'LID' in table.columns:
        flags |= (table['LID'].to_numpy() != -1).astype(np.uint8) * FLAG_CLUSTERED
    else:
        flags |= FLAG_CLUSTERED

    day_index = np.full(count, 65535, np.uint16)
    trial = np.zeros(count, np.uint8)
    for day in days:
        first = dt.datetime.fromisoformat(day['first'])
        last = dt.datetime.fromisoformat(day['last'])
        inside = (times >= first) & (times <= last)
        day_index[inside] = day['index']
        trial[inside] = day['trial']

    def encode(values, dtype):
        return base64.b64encode(
            np.ascontiguousarray(values, dtype=dtype).tobytes()).decode('ascii')

    return {
        'n': int(count),
        'bids': list(BIDS),
        'labels': LABELS,
        'x': encode(np.clip(across, 0, 65535), np.uint16),
        'y': encode(np.clip(down, 0, 65535), np.uint16),
        'bid': encode(bid, np.uint8),
        'prob': encode(np.clip(probability.fillna(0) * 255, 0, 255), np.uint8),
        'flags': encode(flags, np.uint8),
        'day': encode(day_index, np.uint16),
        'trial': encode(trial, np.uint8),
        'hour': encode(times.hour.to_numpy(), np.uint8),
        'minute': encode(times.minute.to_numpy(), np.uint8),
    }


def summarise(table: pd.DataFrame) -> dict:
    """Counts worth having in the manifest without unpacking anything."""
    out = {'n': int(len(table))}
    if 'LID' in table.columns:
        unclustered = int((table['LID'] == -1).sum())
        out['unclustered'] = unclustered
        out['clustered'] = int(len(table)) - unclustered
    if 'Prediction' in table.columns:
        counts = table['Prediction'].value_counts()
        out['byBehaviour'] = {str(k): int(v) for k, v in counts.items()}
        out['noPrediction'] = int(table['Prediction'].isna().sum())
    if 'ClipCreated' in table.columns:
        out['noClip'] = int((table['ClipCreated'] != 'Yes').sum())
    if 'LID' in table.columns:
        unclustered = int((table['LID'].astype('Int64') == -1).sum())
        out['unclustered'] = unclustered
        out['clusteredFraction'] = round(1 - unclustered / max(1, len(table)), 4)
    if len(table):
        out['first'] = str(table.index.min())
        out['last'] = str(table.index.max())
    return out
