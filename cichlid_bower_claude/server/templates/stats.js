/* Cluster statistics, as a view inside the cluster page.
 *
 * Box plots rather than stacked bars: the question is how a quantity is
 * distributed across the days of a trial, and a stack hides that behind a
 * total. Each box is one hour of one category in one trial, over however many
 * days the trial has, with the days themselves drawn as points beside it —
 * six days is a summary of very little, and the points say so.
 *
 * Transitions that never joined a cluster (LID -1) are excluded from every
 * plot, because they are not events. Their proportion is reported instead,
 * since it measures how well the clustering worked.
 */

const STAT_GROUPS = [
  ['Building', ['c', 'p', 'b']],
  ['Feeding', ['f', 't', 'm']],
  ['Spawning', ['s']],
  ['Other', ['d', 'o']],
];
const HOURS = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
const HIGH = [111, 178, 232];
const LOW = [224, 105, 63];

function usableEvent(index) {
  return (EVENTS.flags[index] & 1) && (EVENTS.flags[index] & 2) &&
         EVENTS.bid[index] !== 255;
}

function quartiles(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const at = fraction => {
    const position = (sorted.length - 1) * fraction;
    const low = Math.floor(position), high = Math.ceil(position);
    return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
  };
  const q1 = at(0.25), median = at(0.5), q3 = at(0.75);
  const reach = 1.5 * (q3 - q1);
  const inside = sorted.filter(v => v >= q1 - reach && v <= q3 + reach);
  return { q1, median, q3, low: inside[0], high: inside[inside.length - 1],
           points: sorted, n: sorted.length };
}

function hourlyCounts(trial, codes, days, cut) {
  const above = HOURS.map(() => new Float64Array(days.length));
  const below = HOURS.map(() => new Float64Array(days.length));
  const slot = {};
  days.forEach((day, i) => { slot[day.index] = i; });
  for (let i = 0; i < EVENTS.n; i++) {
    if (EVENTS.trial[i] !== trial || !usableEvent(i)) continue;
    if (!codes.has(EVENTS.bid[i])) continue;
    const column = HOURS.indexOf(EVENTS.hour[i]);
    const row = slot[EVENTS.day[i]];
    if (column < 0 || row === undefined) continue;
    (EVENTS.prob[i] >= cut ? above : below)[column][row] += 1;
  }
  return { above, below };
}

// Each series is scaled to its own peak and reads against its own axis: low
// confidence events outnumber or undercount high confidence ones by a lot, and
// a shared axis would flatten one of them to nothing. The question here is
// whether the two have the same *shape* across the day, which a shared axis
// hides and separate ones show.
function boxPlot(series, title, width, height) {
  const pad = { left: 44, right: 44, top: 20, bottom: 28 };
  const plotWidth = width - pad.left - pad.right;
  const plotHeight = height - pad.top - pad.bottom;
  const slotWidth = plotWidth / HOURS.length;
  const boxWidth = slotWidth / (series.length + 1.2);

  series.forEach(one => {
    let peak = 0;
    one.boxes.forEach(box => {
      if (box && box.points.length)
        peak = Math.max(peak, box.points[box.points.length - 1]);
    });
    one.peak = peak || 1;
    one.y = value => pad.top + plotHeight - (value / one.peak) * plotHeight;
  });

  let body = '';
  for (let i = 0; i <= 4; i++) {
    const level = pad.top + plotHeight - (i / 4) * plotHeight;
    body += '<line x1="' + pad.left + '" y1="' + level + '" x2="' +
            (width - pad.right) + '" y2="' + level + '" stroke="#1d232c"/>';
    series.forEach((one, index) => {
      const value = Math.round(one.peak * i / 4);
      const colour = 'rgb(' + one.colour.join(',') + ')';
      if (index === 0)
        body += '<text x="' + (pad.left - 6) + '" y="' + (level + 4) + '" fill="' +
                colour + '" font-size="10" text-anchor="end">' + value + '</text>';
      else
        body += '<text x="' + (width - pad.right + 6) + '" y="' + (level + 4) +
                '" fill="' + colour + '" font-size="10">' + value + '</text>';
    });
  }

  HOURS.forEach((hour, column) => {
    series.forEach((one, index) => {
      const box = one.boxes[column];
      if (!box) return;
      const centre = pad.left + slotWidth * (column + 0.5) +
                     (index - (series.length - 1) / 2) * boxWidth * 1.1;
      const colour = 'rgb(' + one.colour.join(',') + ')';
      const y = one.y;
      body += '<line x1="' + centre + '" y1="' + y(box.low) + '" x2="' + centre +
              '" y2="' + y(box.high) + '" stroke="' + colour + '" stroke-width="1"/>';
      body += '<rect x="' + (centre - boxWidth / 2) + '" y="' + y(box.q3) +
              '" width="' + boxWidth + '" height="' +
              Math.max(1, y(box.q1) - y(box.q3)) + '" fill="' + colour +
              '" fill-opacity="0.5" stroke="' + colour + '" stroke-width="1">' +
              '<title>' + one.name + ', ' + hour + ':00 \u2014 median ' +
              box.median.toFixed(0) + ' over ' + box.n + ' days</title></rect>';
      body += '<line x1="' + (centre - boxWidth / 2) + '" y1="' + y(box.median) +
              '" x2="' + (centre + boxWidth / 2) + '" y2="' + y(box.median) +
              '" stroke="#e8ecf1" stroke-width="1.4"/>';
      box.points.forEach((value, k) => {
        const jitter = ((k * 37) % 100) / 100 - 0.5;
        body += '<circle cx="' + (centre + jitter * boxWidth * 0.7) + '" cy="' +
                y(value) + '" r="1.4" fill="#cfd6df" fill-opacity="0.7"/>';
      });
    });
    body += '<text x="' + (pad.left + slotWidth * (column + 0.5)) + '" y="' +
            (height - 8) + '" fill="#93a0b0" font-size="10" text-anchor="middle">' +
            hour + '</text>';
  });

  body += '<line x1="' + pad.left + '" y1="' + (height - pad.bottom) + '" x2="' +
          (width - pad.right) + '" y2="' + (height - pad.bottom) +
          '" stroke="#262d38"/>';
  body += '<text x="' + pad.left + '" y="' + (pad.top - 7) +
          '" fill="#e8ecf1" font-size="11" font-weight="600">' + title + '</text>';
  series.forEach((one, index) => {
    body += '<text x="' + (index === 0 ? pad.left + 70 : width - pad.right - 70) +
            '" y="' + (pad.top - 7) + '" fill="rgb(' + one.colour.join(',') +
            ')" font-size="10" text-anchor="' + (index === 0 ? 'start' : 'end') + '">' +
            one.name + ' \u00b7 peak ' + Math.round(one.peak) + '</text>';
  });

  const figure = el('figure');
  const holder = el('div');
  holder.innerHTML = '<svg viewBox="0 0 ' + width + ' ' + height +
                     '" style="width:100%;display:block">' + body + '</svg>';
  figure.appendChild(holder);
  return figure;
}

const SPLIT = 0.5;      // the two populations these plots compare

function statsView() {
  const box = el('div');
  const cut = Math.round(SPLIT * 255);
  const summary = EVENTS.summary || {};

  const note = el('p', 'sub');
  note.innerHTML = 'One box per hour over the days of that trial, with each day drawn ' +
    'as a point. Blue is confidence above ' + SPLIT + ', orange below it, each read ' +
    'against its own axis \u2014 left for blue, right for orange \u2014 so the two ' +
    'can be compared by shape rather than by height. If the orange boxes peak at the ' +
    'same hours as the blue, the events the classifier was unsure about are behaving ' +
    'like the ones it was sure about, which would mean the cut is discarding real ' +
    'events rather than noise.' +
    (summary.unclustered !== undefined
      ? ' <b>' + summary.unclustered + '</b> detections never joined a cluster, ' +
        (100 * (1 - (summary.clusteredFraction || 0))).toFixed(1) +
        '% of the file. They are excluded from every plot here; the proportion is ' +
        'itself a measure of how well the clustering worked.'
      : '');
  box.appendChild(note);

  const byTrial = {};
  D.days.forEach(day => { (byTrial[day.trial] = byTrial[day.trial] || []).push(day); });

  Object.keys(byTrial).sort((a, b) => a - b).forEach(trial => {
    const days = byTrial[trial];
    const section = el('div', 'trial');
    section.appendChild(el('h3', null, 'Trial ' + trial +
      '<span>' + days.length + ' days</span>'));
    box.appendChild(section);
    const grid = el('div', 'grid cols-4');
    section.appendChild(grid);

    STAT_GROUPS.forEach(([name, bids]) => {
      const codes = new Set(bids.map(bid => EVENTS.code[bid]));
      const counts = hourlyCounts(+trial, codes, days, cut);
      const series = [
        { name: 'above ' + SPLIT, colour: HIGH,
          boxes: counts.above.map(v => quartiles(Array.from(v))) },
        { name: 'below ' + SPLIT, colour: LOW,
          boxes: counts.below.map(v => quartiles(Array.from(v))) }];
      grid.appendChild(boxPlot(series, name, 420, 260));
    });
  });
  return box;
}