/* Trial-level metrics for every project in an analysis, by category.
 *
 * A trial is the unit, not a project: the tank is reset between trials, so
 * each is an independent observation of a fish building from flat sand.
 * Averaging a project's trials into one number throws away most of the
 * replication, so every trial is its own dot.
 *
 * A strip plot rather than a bar of means. With five to fifteen trials in a
 * category, the spread is the finding as often as the centre is, and a bar
 * hides it. The bar drawn here is the median, which is what you compare.
 */

let DATA = null;
const JITTER = 0.26;       // how far a dot may slide to escape its neighbours

// Which metrics get a panel, in reading order. The third element says how to
// label the axis; the fourth, whether zero is meaningful enough to draw.
const METRICS = [
  ['totalVolume', 'Total volume', 'cm³', false],
  ['castleVolume', 'Castle volume', 'cm³', false],
  ['pitVolume', 'Pit volume', 'cm³', false],
  ['bowerIndex', 'Bower index', '−1 pit … +1 castle', true],
  ['volumePerDay', 'Volume per day', 'cm³', false],
  ['grossVolume', 'Gross sand moved', 'cm³', false],
  ['nightShare', 'Built overnight', 'share of 24 h', false],
  ['nightRateRatio', 'Night rate ÷ day rate', '×', true],
  ['firstHalfIndex', 'Index, first half', '−1 … +1', true],
  ['secondHalfIndex', 'Index, second half', '−1 … +1', true],
  ['eventsPerDay', 'Events per day', 'events', false],
  ['scoopSpitRatio', 'Scoops ÷ spits', '×', true],
  ['scoopArea', 'Scoop cloud area', 'cm²', false],
  ['spitArea', 'Spit cloud area', 'cm²', false],
  ['scoopAspect', 'Scoop cloud aspect', '× (1 = round)', true],
  ['spitAspect', 'Spit cloud aspect', '× (1 = round)', true],
  ['centroidGap', 'Scoop to spit centroids', 'cm', false],
  ['spawnDepthMedian', 'Median spawn depth', 'cm', true],
  ['spawnOverCastle', 'Spawns over added sand', '%', false],
  ['daysTo90', 'Days to 90% of final', 'days', false],
];

// A colour per category, assigned in sorted order so the same category keeps
// the same colour between panels and between page loads.
const PALETTE = [[242, 163, 60], [111, 178, 232], [90, 168, 122],
                 [232, 132, 168], [198, 160, 246], [226, 198, 110],
                 [137, 196, 196], [224, 105, 63]];

function categories() {
  const seen = [];
  DATA.rows.forEach(row => {
    if (seen.indexOf(row.category) < 0) seen.push(row.category);
  });
  return seen.sort();
}

function colourFor(category, order) {
  return PALETTE[order.indexOf(category) % PALETTE.length];
}

function median(values) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle]
                           : (sorted[middle - 1] + sorted[middle]) / 2;
}

// A deterministic offset per dot. Math.random() would move every dot on every
// redraw, so the same trial would sit somewhere new each time you looked.
function offsetFor(row, index) {
  const key = (row.projectID || '') + ':' + row.trial + ':' + index;
  let hash = 0;
  for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) % 1009;
  return (hash / 1009 - 0.5) * 2 * JITTER;
}

function stripPlot(metric, label, units, showZero, order) {
  const box = el('div', 'strip');
  box.appendChild(el('h3', null, label));

  const groups = order.map(category => ({
    category,
    rows: DATA.rows.filter(r => r.category === category &&
                                typeof r[metric] === 'number' &&
                                isFinite(r[metric])),
  }));
  const present = groups.filter(g => g.rows.length);
  const values = [].concat.apply([], present.map(g => g.rows.map(r => r[metric])));
  if (!values.length) {
    box.appendChild(el('p', 'hint', 'nothing measured for this'));
    return box;
  }

  let lo = Math.min.apply(null, values), hi = Math.max.apply(null, values);
  if (showZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (lo === hi) { lo -= 0.5; hi += 0.5; }
  const pad = (hi - lo) * 0.08;
  lo -= pad; hi += pad;

  const counts = present.map(g => g.category + ' ' + g.rows.length).join(' · ');
  box.appendChild(el('p', 'hint', units + '  ·  ' + counts));

  const width = 420, height = 36 + present.length * 34;
  const left = 96, right = 10;
  const canvas = el('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  const X = v => left + (v - lo) / (hi - lo) * (width - left - right);

  ctx.fillStyle = '#0b0d11';
  ctx.fillRect(0, 0, width, height);

  // the axis, with the two ends labelled and zero marked where it matters
  ctx.font = '10px sans-serif';
  ctx.fillStyle = '#93a0b0';
  ctx.textAlign = 'left';
  ctx.fillText(trim(lo), left, height - 6);
  ctx.textAlign = 'right';
  ctx.fillText(trim(hi), width - right, height - 6);
  if (showZero && lo < 0 && hi > 0) {
    ctx.strokeStyle = 'rgba(147,160,176,.45)';
    ctx.beginPath();
    ctx.moveTo(X(0), 8); ctx.lineTo(X(0), height - 18);
    ctx.stroke();
  }

  present.forEach((group, row) => {
    const y = 20 + row * 34;
    const colour = colourFor(group.category, order);
    ctx.fillStyle = '#93a0b0';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(group.category.slice(0, 13), left - 8, y + 4);

    const mid = median(group.rows.map(r => r[metric]));
    if (mid !== null) {
      ctx.strokeStyle = 'rgb(' + colour.join(',') + ')';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(X(mid), y - 11); ctx.lineTo(X(mid), y + 11);
      ctx.stroke();
    }
    group.rows.forEach((r, i) => {
      ctx.fillStyle = 'rgba(' + colour.join(',') + ',.72)';
      ctx.beginPath();
      ctx.arc(X(r[metric]), y + offsetFor(r, i) * 22, 3, 0, Math.PI * 2);
      ctx.fill();
    });
  });
  box.appendChild(canvas);
  return box;
}

function trim(value) {
  const magnitude = Math.abs(value);
  if (magnitude >= 100) return value.toFixed(0);
  if (magnitude >= 10) return value.toFixed(1);
  return value.toFixed(magnitude >= 1 ? 2 : 3);
}

function draw() {
  const body = document.getElementById('body');
  body.textContent = '';
  const order = categories();

  if (!DATA.rows.length) {
    body.appendChild(el('div', 'note', 'No trials to compare. Either nothing ' +
      'in this analysis has been collected, or every trial is excluded.'));
    return;
  }

  const strips = el('div', 'strips');
  METRICS.forEach(([metric, label, units, showZero]) => {
    strips.appendChild(stripPlot(metric, label, units, showZero, order));
  });
  body.appendChild(strips);

  if (DATA.skipped && DATA.skipped.length) {
    body.appendChild(el('p', 'foot', DATA.skipped.length +
      ' project(s) left out: ' + DATA.skipped.map(
        s => s.projectID + ' (' + s.why + ')').join(', ')));
  }
}

function build() {
  const p = DATA.params || {};
  document.getElementById('title').textContent = DATA.analysisID;
  const trials = DATA.rows.length;
  const projects = DATA.rows.reduce((set, r) => set.add(r.projectID),
                                    new Set()).size;
  document.getElementById('subtitle').textContent =
    trials + ' trials from ' + projects + ' projects, by category. ' +
    'One dot per trial; the bar is the category median.';

  // The parameters are fixed here and the page says so. Recomputing every
  // project on a slider move is not a page load, so this runs at the
  // defaults -- and a workbook downloaded with the sliders elsewhere will
  // not match these numbers, which is why both state what they used.
  const params = el('p', 'params');
  params.innerHTML = 'Fixed settings for this page · ' +
    'confidence <b>' + p.confidence + '</b> · ' +
    'hours <b>' + p.hour_from + '–' + p.hour_to + '</b> · ' +
    'sand moved ≥ <b>' + p.threshold + '</b> cm · ' +
    'ellipse covers <b>' + Math.round(100 * p.coverage) + '%</b> · ' +
    'split at <b>' + Math.round(100 * p.split_at) + '%</b> · ' +
    'shape needs <b>' + p.min_for_shape + '</b> events<br>' +
    'A workbook downloaded at other settings will not match these numbers. ' +
    'Both record what they used.';
  const host = document.getElementById('body');
  host.parentNode.insertBefore(params, host);

  const bar = document.getElementById('controls');
  if (bar) {
    const link = el('a', null, 'Download CSV \u2193');
    link.href = 'trials.csv';
    link.className = 'act';
    link.style.cssText = 'display:inline-block;padding:6px 11px;border:1px ' +
      'solid var(--line);border-radius:5px;font-size:13px;background:var(--panel)';
    bar.appendChild(link);
    const note = el('span', null,
      'one row per trial \u00b7 trials are repeated measures of the same ' +
      'male, so collapsing to one row per fish is yours to decide');
    note.style.cssText = 'font-size:11.5px;color:var(--dim);margin-left:12px';
    bar.appendChild(note);
  }

  draw();
}

fetch('trials.json')
  .then(response => response.json())
  .then(payload => { DATA = payload; build(); })
  .catch(error => {
    const body = document.getElementById('body');
    if (body) body.appendChild(el('div', 'note',
      'Could not load the trial rows: ' + error.message));
  });

// el() lives in common.js on the project pages; this page is one level up and
// does not load it, so it has its own copy rather than a route that serves a
// shared file to a page that needs one function from it.
function el(tag, className, html) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (html !== undefined && html !== null) node.innerHTML = html;
  return node;
}
