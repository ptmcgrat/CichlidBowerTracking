/* Phenotypes: one row per trial, every project in an analysis, by category.
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
// Five rows of four, in the order asked for. The row is the grouping: depth
// first, then how much the fish did, then where it did it, then the shape of
// where. A flat auto-fit grid reflowed by window width and put unrelated
// panels side by side.
const ROWS = [
  ['Depth \u2014 how much', [
    ['totalVolume', 'Total volume', 'cm\u00b3', false],
    ['volumePerDay', 'Volume per day', 'cm\u00b3', false],
    ['pitVolume', 'Pit volume', 'cm\u00b3', false],
    ['castleVolume', 'Castle volume', 'cm\u00b3', false],
  ]],
  ['Depth \u2014 what shape', [
    ['bowerIndex', 'Bower index', '\u22121 pit \u2026 +1 castle', true],
    ['firstHalfIndex', 'Index, first half', '\u22121 \u2026 +1', true],
    ['secondHalfIndex', 'Index, second half', '\u22121 \u2026 +1', true],
    ['nightShare', 'Night share', 'night \u00f7 total sand moved', false],
  ]],
  ['Behaviour \u2014 how often', [
    ['eventsPerDay', 'Events per day', 'all events', false],
    ['buildEventsPerDay', 'Build events per day', 'scoop + spit + multiple', false],
    ['feedEventsPerDay', 'Feed events per day', 'feed scoop + spit + multiple', false],
    ['spawnEventsPerDay', 'Spawn events per day', 'spawns', false],
  ]],
  ['Clouds \u2014 how big', [
    ['scoopArea', 'Scoop cloud area', 'cm\u00b2 \u00b7 all events pooled', false],
    ['scoopAreaDaily', 'Scoop cloud area, daily mean', 'cm\u00b2', false],
    ['spitArea', 'Spit cloud area', 'cm\u00b2 \u00b7 all events pooled', false],
    ['spitAreaDaily', 'Spit cloud area, daily mean', 'cm\u00b2', false],
  ]],
  ['Clouds \u2014 what shape', [
    ['scoopSpitAreaIndex', 'Scoop \u00f7 spit area', '\u00d7 \u00b7 daily means', true],
    ['scoopAspectDaily', 'Scoop aspect, daily mean', '\u00d7 (1 = round)', true],
    ['spitAspectDaily', 'Spit aspect, daily mean', '\u00d7 (1 = round)', true],
    ['spawnDepthMedian', 'Median spawn depth', 'cm', true],
  ]],
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

  const hits = [];           // where each dot landed, for hit testing
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
      const cx = X(r[metric]), cy = y + offsetFor(r, i) * 22;
      ctx.fillStyle = 'rgba(' + colour.join(',') + ',.72)';
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, Math.PI * 2);
      ctx.fill();
      hits.push({ x: cx, y: cy, row: r });
    });
  });
  box.appendChild(canvas);
  attachHover(box, canvas, hits, metric, units);
  return box;
}

// --------------------------------------------------------------- hovering
// The dots are canvas pixels, not elements, so there is nothing for the
// browser to put a title on: finding the nearest dot to the pointer is the
// whole of it. The canvas is drawn at a fixed width and stretched by CSS, so
// client coordinates have to be scaled back into the space the dots were
// drawn in -- without that the hit test is right only at one window width.
const HOVER_RADIUS = 7;

function attachHover(box, canvas, hits, metric, units) {
  if (!hits.length) return;
  // a seam for the test: where each dot landed, in the canvas's own
  // coordinates. Without it a test can only check that hovering produces
  // some tooltip, which passes just as happily when the hit test forgets to
  // scale and every dot answers at the wrong place on screen.
  canvas._hits = hits;
  box.style.position = 'relative';
  const tip = el('div', 'tip');
  tip.style.display = 'none';
  box.appendChild(tip);

  const nearest = event => {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const px = (event.clientX - rect.left) * (canvas.width / rect.width);
    const py = (event.clientY - rect.top) * (canvas.height / rect.height);
    let best = null, bestGap = HOVER_RADIUS * HOVER_RADIUS;
    hits.forEach(hit => {
      const gap = (hit.x - px) * (hit.x - px) + (hit.y - py) * (hit.y - py);
      if (gap <= bestGap) { bestGap = gap; best = hit; }
    });
    return best;
  };

  canvas.addEventListener('mousemove', event => {
    const hit = nearest(event);
    if (!hit) { tip.style.display = 'none'; return; }
    const r = hit.row;
    const value = r[metric];
    tip.innerHTML = '<b>' + r.projectID + '</b>' +
      '<span>trial ' + r.trial + ' \u00b7 ' + r.category +
      (r.tankID ? ' \u00b7 tank ' + r.tankID : '') + '</span>' +
      '<span>' + (typeof value === 'number' ? trim(value) : value) +
      (units ? ' ' + units.split(' \u00b7 ')[0] : '') + '</span>';
    tip.style.display = 'block';
    // placed from the dot, then pulled back inside the panel: a dot at the
    // right-hand end would otherwise push the tooltip off the card
    const rect = canvas.getBoundingClientRect();
    const scale = rect.width / canvas.width;
    const left = Math.min(Math.max(hit.x * scale - 60, 0),
                          Math.max(0, rect.width - tip.offsetWidth));
    tip.style.left = left + 'px';
    tip.style.top = (canvas.offsetTop + hit.y * (rect.height / canvas.height)
                     - tip.offsetHeight - 8) + 'px';
  });
  canvas.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
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

  ROWS.forEach(([heading, metrics]) => {
    body.appendChild(el('h2', null, heading));
    const strips = el('div', 'strips');
    metrics.forEach(([metric, label, units, showZero]) => {
      strips.appendChild(stripPlot(metric, label, units, showZero, order));
    });
    body.appendChild(strips);
  });

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
    link.href = 'phenotypes.csv';
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

fetch('phenotypes.json')
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
