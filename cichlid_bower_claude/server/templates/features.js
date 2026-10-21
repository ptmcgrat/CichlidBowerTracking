/* The features page: one row per trial, five views of the same trial.
 *
 * Reading left to right: what the sand ended up like, how the structure got
 * there, where the fish worked, how its working spread out, and where it
 * spawned relative to what it built. Two of the five are trajectories rather
 * than summaries -- a point per day, joined in order -- because a bower index
 * of +0.6 reached steadily and one reached after a week of digging are
 * different animals, and a single number cannot tell them apart.
 *
 * Everything is derived from the depth bundle and the events, so it moves when
 * the crop, the threshold or the registration moves.
 */

let EVENTS = null;
let CONFIDENCE = 0.67;
let THRESHOLD = 0.6;     // cm before sand counts as moved
let COVERAGE = 0.68;     // the share of events an ellipse holds
let BUCKETS = null;      // events by trial, day and behaviour

const SCOOP = [242, 163, 60], SPIT = [111, 178, 232], SPAWN = [232, 132, 168];
const PANEL = 250;

const COLUMNS = [
  ['Total depth change', 'trial start to end', 'depth'],
  ['Volume and shape', 'per day · cumulative', 'shape'],
  ['Where it worked', 'scoops orange, spits blue', 'events'],
  ['Effort and spread', 'per day · cumulative', 'spread'],
  ['Spawning depth', 'sand under each spawn', 'spawn'],
];

function unpackColumn(text, Type) {
  const binary = atob(text);
  const buffer = new ArrayBuffer(binary.length);
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Type(buffer);
}

function loadEvents() {
  return fetch('clusters.json')
    .then(response => response.ok ? response.json() : null)
    .then(packed => {
      if (!packed) return null;
      const code = {};
      packed.bids.forEach((bid, i) => { code[bid] = i; });
      EVENTS = { n: packed.n, code, summary: packed.summary,
                 x: unpackColumn(packed.x, Uint16Array),
                 y: unpackColumn(packed.y, Uint16Array),
                 bid: unpackColumn(packed.bid, Uint8Array),
                 prob: unpackColumn(packed.prob, Uint8Array),
                 flags: unpackColumn(packed.flags, Uint8Array),
                 day: unpackColumn(packed.day, Uint16Array),
                 trial: unpackColumn(packed.trial, Uint8Array),
                 hour: unpackColumn(packed.hour, Uint8Array) };
      projectToDepth();
      return EVENTS;
    });
}

function projectToDepth() {
  EVENTS.dx = new Float32Array(EVENTS.n);
  EVENTS.dy = new Float32Array(EVENTS.n);
  EVENTS.placed = new Uint8Array(EVENTS.n);
  EVENTS.unplacedTrials = {};
  for (let i = 0; i < EVENTS.n; i++) {
    const H = transformFor(EVENTS.trial[i]);
    if (!H) {
      EVENTS.unplacedTrials[EVENTS.trial[i]] =
        (EVENTS.unplacedTrials[EVENTS.trial[i]] || 0) + 1;
      continue;
    }
    const point = applyH(H, [EVENTS.x[i], EVENTS.y[i]]);
    EVENTS.dx[i] = point[0];
    EVENTS.dy[i] = point[1];
    EVENTS.placed[i] = 1;
  }
}

function usable(index, cut) {
  // clipped, classified, clustered, placed, and confident enough
  return EVENTS.placed[index] && (EVENTS.flags[index] & 1) &&
         (EVENTS.flags[index] & 2) && EVENTS.bid[index] !== 255 &&
         EVENTS.prob[index] >= cut;
}

// One pass, rather than a scan of every event for every cell.
function bucketEvents(cut) {
  BUCKETS = {};
  for (let i = 0; i < EVENTS.n; i++) {
    if (!usable(i, cut)) continue;
    const byDay = BUCKETS[EVENTS.trial[i]] || (BUCKETS[EVENTS.trial[i]] = {});
    const byCode = byDay[EVENTS.day[i]] || (byDay[EVENTS.day[i]] = {});
    (byCode[EVENTS.bid[i]] || (byCode[EVENTS.bid[i]] = []))
      .push([EVENTS.dx[i], EVENTS.dy[i]]);
  }
}

function pointsUpTo(trial, code, upToDay) {
  const out = [];
  const byDay = (BUCKETS && BUCKETS[trial]) || {};
  Object.keys(byDay).forEach(day => {
    if (+day > upToDay) return;
    const found = byDay[day][code] || [];
    for (let i = 0; i < found.length; i++) out.push(found[i]);
  });
  return out;
}

// ---------------------------------------------------------------- geometry
function dispersion(points, coverage) {
  if (points.length < 10) return null;
  let mx = 0, my = 0;
  points.forEach(p => { mx += p[0]; my += p[1]; });
  mx /= points.length; my /= points.length;
  let sxx = 0, syy = 0, sxy = 0;
  points.forEach(p => {
    const dx = p[0] - mx, dy = p[1] - my;
    sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
  });
  sxx /= points.length; syy /= points.length; sxy /= points.length;
  const determinant = Math.max(0, sxx * syy - sxy * sxy);
  const k = -2 * Math.log(1 - coverage);
  const trace = sxx + syy;
  const root = Math.sqrt(Math.max(0, trace * trace / 4 - determinant));
  const cm = D.pixelLength || 0.1030168618;
  return { cx: mx, cy: my, n: points.length,
           area: Math.PI * k * Math.sqrt(determinant) * cm * cm,
           major: Math.sqrt(k * Math.max(0, trace / 2 + root)),
           minor: Math.sqrt(k * Math.max(0, trace / 2 - root)),
           angle: 0.5 * Math.atan2(2 * sxy, sxx - syy) };
}

function cropped(values, meta, trial) {
  const outside = cropMaskFor(meta, trial);
  if (!outside) return values;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = outside[i] ? NaN : values[i];
  return out;
}

// ------------------------------------------------------------------ panels
function panel(contents, foot) {
  const box = el('div');
  box.appendChild(contents);
  if (foot) box.appendChild(el('div', 'cellfoot', foot));
  return box;
}

function svgPanel(body, width, height) {
  const holder = el('div');
  holder.innerHTML = '<svg viewBox="0 0 ' + width + ' ' + height +
                     '" style="width:100%;display:block;background:#0b0d11">' +
                     body + '</svg>';
  return holder;
}

// A path through the days rather than a cloud of them: the order is the
// information. Early days are dim, the last day is a filled ring, so the
// direction of travel reads without an arrowhead.
function trajectory(points, options) {
  const width = PANEL, height = Math.round(PANEL * 0.82);
  const pad = { left: 38, right: 10, top: 12, bottom: 26 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const real = points.filter(p => p && isFinite(p.x) && isFinite(p.y));
  if (real.length < 2)
    return svgPanel('<text x="' + (width / 2) + '" y="' + (height / 2) +
                    '" fill="#6c7684" font-size="11" text-anchor="middle">' +
                    'not enough days</text>', width, height);

  let xMax = options.xMax;
  if (xMax === undefined) xMax = Math.max(...real.map(p => p.x)) * 1.08 || 1;
  let yLow = options.yMin, yHigh = options.yMax;
  if (yLow === undefined) {
    yLow = Math.min(...real.map(p => p.y));
    yHigh = Math.max(...real.map(p => p.y));
    const margin = (yHigh - yLow) * 0.15 || 0.1;
    yLow -= margin; yHigh += margin;
  }
  const x = v => pad.left + (v / xMax) * plotW;
  const y = v => pad.top + plotH - ((v - yLow) / (yHigh - yLow)) * plotH;

  let body = '';
  for (let i = 0; i <= 2; i++) {
    const value = yLow + (yHigh - yLow) * i / 2;
    body += '<line x1="' + pad.left + '" y1="' + y(value) + '" x2="' +
            (width - pad.right) + '" y2="' + y(value) + '" stroke="#1d232c"/>' +
            '<text x="' + (pad.left - 5) + '" y="' + (y(value) + 3.5) +
            '" fill="#93a0b0" font-size="9" text-anchor="end">' +
            (Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2)) +
            '</text>';
  }
  if (options.reference !== undefined && options.reference > yLow &&
      options.reference < yHigh)
    body += '<line x1="' + pad.left + '" y1="' + y(options.reference) + '" x2="' +
            (width - pad.right) + '" y2="' + y(options.reference) +
            '" stroke="#5c6673" stroke-dasharray="3 3"/>';

  let path = '';
  real.forEach((p, i) => { path += (i ? ' L' : 'M') + x(p.x) + ',' + y(p.y); });
  body += '<path d="' + path + '" fill="none" stroke="rgb(' +
          options.colour.join(',') + ')" stroke-width="1.4" ' +
          'stroke-opacity="0.65"/>';
  real.forEach((p, i) => {
    const last = i === real.length - 1;
    body += '<circle cx="' + x(p.x) + '" cy="' + y(p.y) + '" r="' +
            (last ? 3.6 : 2) + '" fill="' + (last ? 'rgb(' +
            options.colour.join(',') + ')' : '#0b0d11') + '" stroke="rgb(' +
            options.colour.join(',') + ')" stroke-width="1.1" fill-opacity="' +
            (last ? 1 : 0.9) + '" opacity="' +
            (0.35 + 0.65 * (i / Math.max(1, real.length - 1))).toFixed(2) +
            '"><title>' + p.label + '</title></circle>';
  });

  body += '<line x1="' + pad.left + '" y1="' + (height - pad.bottom) + '" x2="' +
          (width - pad.right) + '" y2="' + (height - pad.bottom) +
          '" stroke="#262d38"/>';
  body += '<text x="' + ((pad.left + width - pad.right) / 2) + '" y="' +
          (height - 8) + '" fill="#6c7684" font-size="9" text-anchor="middle">' +
          options.xLabel + '</text>';
  body += '<text x="' + pad.left + '" y="' + (pad.top - 3) +
          '" fill="#6c7684" font-size="9">' + options.yLabel + '</text>';
  return svgPanel(body, width, height);
}

function eventScatter(groups, meta, trial) {
  const width = PANEL;
  const height = Math.round(PANEL * D.depthSize[1] / D.depthSize[0]);
  const box = el('div');
  const stage = el('div', 'stage');
  const canvas = el('canvas');
  canvas.width = width; canvas.height = height;
  stage.appendChild(canvas);
  box.appendChild(stage);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0b0d11';
  ctx.fillRect(0, 0, width, height);
  const sx = width / D.depthSize[0], sy = height / D.depthSize[1];

  // the crop, so a point outside the tray is visibly outside it
  const crop = depthCropFor(trial);
  if (crop && crop.length >= 3) {
    ctx.strokeStyle = 'rgba(146,160,176,.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    crop.forEach((p, i) => {
      const px = p[0] * sx, py = p[1] * sy;
      if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
    });
    ctx.closePath();
    ctx.stroke();
  }

  groups.forEach(group => {
    ctx.fillStyle = 'rgb(' + group.colour.join(',') + ')';
    ctx.globalAlpha = group.points.length > 500 ? 0.3 : 0.6;
    group.points.forEach(point => {
      ctx.beginPath();
      ctx.arc(point[0] * sx, point[1] * sy, 1.2, 0, 6.2832);
      ctx.fill();
    });
    ctx.globalAlpha = 1;
    const shape = dispersion(group.points, COVERAGE);
    if (!shape) return;
    ctx.save();
    ctx.translate(shape.cx * sx, shape.cy * sy);
    ctx.rotate(shape.angle);
    ctx.strokeStyle = 'rgb(' + group.colour.join(',') + ')';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.ellipse(0, 0, shape.major * sx, shape.minor * sy, 0, 0, 6.2832);
    ctx.stroke();
    ctx.restore();
  });
  return box;
}

// Ported from the summary page, sized for a column. Each spawn is read against
// the sand as it was on its own day, and only those inside the ellipse are
// counted -- a spawn across the tank is a different event from one on the bower.
function spawnPanel(trial, days, meta, cumulativeByDay) {
  const width = PANEL, height = Math.round(PANEL * 0.82);
  const spawns = pointsUpToWithDay(trial, EVENTS.code.s, days);
  if (spawns.length < 5)
    return panel(svgPanel('<text x="' + (width / 2) + '" y="' + (height / 2) +
      '" fill="#6c7684" font-size="11" text-anchor="middle">' +
      spawns.length + ' spawns</text>', width, height), 'too few to plot');

  const shape = dispersion(spawns.map(s => [s.x, s.y]), COVERAGE);
  const inside = shape ? spawns.filter(spawn => {
    const dx = spawn.x - shape.cx, dy = spawn.y - shape.cy;
    const cos = Math.cos(-shape.angle), sin = Math.sin(-shape.angle);
    const u = (dx * cos - dy * sin) / shape.major;
    const v = (dx * sin + dy * cos) / shape.minor;
    return u * u + v * v <= 1;
  }) : spawns;

  const sx = meta.width / D.depthSize[0], sy = meta.height / D.depthSize[1];
  const values = [];
  // the cumulative map for a day is computed once and shared, not rebuilt for
  // every spawn on it: a full-frame difference per spawn is hundreds of passes
  // over the whole map to read one pixel each
  inside.forEach(spawn => {
    const cumulative = cumulativeByDay[spawn.day];
    if (!cumulative) return;
    const px = Math.round(spawn.x * sx), py = Math.round(spawn.y * sy);
    if (px < 0 || py < 0 || px >= meta.width || py >= meta.height) return;
    const depth = cumulative[py * meta.width + px];
    if (!Number.isNaN(depth)) values.push(depth);
  });
  if (values.length < 5)
    return panel(svgPanel('<text x="' + (width / 2) + '" y="' + (height / 2) +
      '" fill="#6c7684" font-size="11" text-anchor="middle">no measurable sand' +
      '</text>', width, height), String(spawns.length) + ' spawns');

  const bins = 20, span = 4;
  const counts = new Float32Array(bins);
  values.forEach(value => {
    counts[Math.min(bins - 1, Math.max(0,
      Math.floor((value + span) / (2 * span) * bins)))] += 1;
  });
  const peak = Math.max(...counts) || 1;
  const pad = { left: 24, right: 8, top: 12, bottom: 24 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  let body = '';
  for (let i = 0; i < bins; i++) {
    const size = (counts[i] / peak) * plotH;
    const value = -span + (2 * span) * (i + 0.5) / bins;
    body += '<rect x="' + (pad.left + plotW / bins * i + 0.6) + '" y="' +
            (height - pad.bottom - size) + '" width="' + (plotW / bins - 1.2) +
            '" height="' + size + '" fill="' +
            (value >= 0 ? 'rgb(' + SPIT.join(',') + ')'
                        : 'rgb(' + SCOOP.join(',') + ')') +
            '" opacity="0.85"><title>' + value.toFixed(1) + ' cm: ' +
            counts[i] + '</title></rect>';
  }
  const zero = pad.left + plotW * 0.5;
  body += '<line x1="' + zero + '" y1="' + (pad.top - 2) + '" x2="' + zero +
          '" y2="' + (height - pad.bottom) +
          '" stroke="#93a0b0" stroke-dasharray="3 3"/>';
  body += '<line x1="' + pad.left + '" y1="' + (height - pad.bottom) + '" x2="' +
          (width - pad.right) + '" y2="' + (height - pad.bottom) +
          '" stroke="#262d38"/>';
  body += '<text x="' + pad.left + '" y="' + (height - 8) +
          '" fill="#6c7684" font-size="9">−' + span + ' pit</text>';
  body += '<text x="' + (width - pad.right) + '" y="' + (height - 8) +
          '" fill="#6c7684" font-size="9" text-anchor="end">castle +' + span +
          '</text>';

  const sorted = values.slice().sort((a, b) => a - b);
  const median = sorted[sorted.length >> 1];
  const over = 100 * values.filter(v => v > 0).length / values.length;
  return panel(svgPanel(body, width, height),
    values.length + ' of ' + spawns.length + ' spawns · median ' +
    (median >= 0 ? '+' : '') + median.toFixed(2) + ' cm · ' +
    over.toFixed(0) + '% over added sand');
}

function pointsUpToWithDay(trial, code, days) {
  const out = [];
  const byDay = (BUCKETS && BUCKETS[trial]) || {};
  days.forEach(day => {
    const found = (byDay[day.index] || {})[code] || [];
    for (let i = 0; i < found.length; i++)
      out.push({ x: found[i][0], y: found[i][1], day: day.index });
  });
  return out;
}

// -------------------------------------------------------------------- rows
function renderTrial(trial, days, table) {
  const meta = days[0].firstPng;
  const cm3 = ' cm³';

  table.appendChild(el('div', 'rowlab', '<b>Trial ' + trial + '</b><span>' +
    days.length + ' days<br>' + days[0].date + '<br>to ' +
    days[days.length - 1].date + '</span>'));

  const cells = {};
  COLUMNS.forEach(([, , key]) => {
    const slot = el('div', 'cell');
    table.appendChild(slot);
    cells[key] = slot;
  });

  Promise.all([loadDepth(days[0].firstPng)].concat(
      days.map(day => loadDepth(day.lastPng)))).then(frames => {
    const start = frames[0];
    if (!start) return;

    // every day's cumulative change, computed once and used by three columns
    const cumulativeByDay = {};
    days.forEach((day, i) => {
      const end = frames[i + 1];
      if (end) cumulativeByDay[day.index] =
        cropped(difference(start, end), meta, trial);
    });

    // 1. the sand at the end of the trial
    const total = cumulativeByDay[days[days.length - 1].index];
    if (!total) return;
    const canvas = el('canvas');
    const stage = el('div', 'stage');
    stage.appendChild(canvas);
    paintMap(canvas, total, meta, { range: 4 });
    const final = volumes(total, THRESHOLD);
    cells.depth.appendChild(panel(stage,
      final.total.toFixed(0) + cm3 + ' moved · index ' +
      final.index.toFixed(2)));

    // 2. volume against shape, a point per day
    const shapePath = days.map(day => {
      const map = cumulativeByDay[day.index];
      if (!map) return null;
      const measured = volumes(map, THRESHOLD);
      return { x: measured.total, y: measured.index,
               label: day.date + ': ' + measured.total.toFixed(0) + cm3 +
                      ', index ' + measured.index.toFixed(2) };
    });
    cells.shape.appendChild(panel(trajectory(shapePath, {
      colour: SPAWN, xLabel: 'volume moved, cm³', yLabel: 'bower index',
      yMin: -1, yMax: 1, reference: 0 }),
      'ends at ' + final.index.toFixed(2)));

    // 3. where the fish actually worked
    const lastDay = days[days.length - 1].index;
    const scoops = pointsUpTo(trial, EVENTS.code.c, lastDay);
    const spits = pointsUpTo(trial, EVENTS.code.p, lastDay);
    cells.events.appendChild(panel(
      eventScatter([{ colour: SCOOP, points: scoops },
                    { colour: SPIT, points: spits }], meta, trial),
      scoops.length + ' scoops · ' + spits.length + ' spits'));

    // 4. effort against how spread out that effort was
    const spreadPath = days.map(day => {
      const a = dispersion(pointsUpTo(trial, EVENTS.code.c, day.index), COVERAGE);
      const b = dispersion(pointsUpTo(trial, EVENTS.code.p, day.index), COVERAGE);
      if (!a || !b || b.area <= 0) return null;
      return { x: a.n + b.n, y: a.area / b.area,
               label: day.date + ': ' + (a.n + b.n) + ' events, ratio ' +
                      (a.area / b.area).toFixed(2) };
    });
    const ends = spreadPath.filter(Boolean).pop();
    cells.spread.appendChild(panel(trajectory(spreadPath, {
      colour: SCOOP, xLabel: 'scoops + spits', yLabel: 'scoop area / spit area',
      reference: 1 }),
      ends ? 'ends at ' + ends.y.toFixed(2) : 'too few events'));

    // 5. where it spawned, against what it had built by then
    cells.spawn.appendChild(spawnPanel(trial, days, meta, cumulativeByDay));
  });
}

function draw() {
  const body = document.getElementById('body');
  body.textContent = '';
  const excluded = excludedBanner();
  if (excluded) body.appendChild(excluded);

  const unplaced = Object.keys(EVENTS.unplacedTrials || {});
  if (unplaced.length)
    body.appendChild(el('div', 'note', 'Trial ' + unplaced.join(', ') +
      ' has no registration, so its events cannot be placed in depth ' +
      'coordinates. Fit one on the prep page, or drop the override so it ' +
      'falls back to the project registration.'));

  bucketEvents(Math.round(CONFIDENCE * 255));

  const byTrial = daysByTrial(false);
  const trials = Object.keys(byTrial).sort((a, b) => a - b);
  if (!trials.length) {
    body.appendChild(el('div', 'note', 'No trials to show — every trial ' +
      'is excluded, or the project has not been collected.'));
    return;
  }

  const table = el('div', 'matrix');
  table.style.gridTemplateColumns = '116px repeat(5, minmax(0, 1fr))';
  body.appendChild(table);
  table.appendChild(el('div', 'blank'));
  COLUMNS.forEach(([name, hint]) => {
    table.appendChild(el('div', 'colhead',
      '<b>' + name + '</b><span>' + hint + '</span>'));
  });
  buildCodeSets();
  // the confidence slider changes which events exist, so the last redraw's
  // volumes and ellipses are not this redraw's
  VOLUME_CACHE = {};
  trials.forEach(trial => {
    renderTrial(+trial, byTrial[trial], table);
    renderTrialDetail(+trial, byTrial[trial], table);
  });
}

function build() {
  document.getElementById('title').textContent = D.projectID;
  document.getElementById('subtitle').textContent =
    'What this project amounts to, trial by trial.';
  document.getElementById('meta').innerHTML =
    [['Tank', D.tankID], ['Analysis', D.analysisID], ['Days', D.days.length],
     ['Trials', D.trials.length], ['Events', EVENTS ? EVENTS.n : 0]]
    .map(([k, v]) => '<div>' + k + '<b>' + v + '</b></div>').join('');

  const bar = document.getElementById('controls');
  bar.innerHTML =
    '<label>Confidence ≥ <b id="cv">' + CONFIDENCE.toFixed(2) + '</b></label>' +
    '<input type="range" id="cr" min="0" max="0.99" step="0.01" value="' +
      CONFIDENCE + '">' +
    '<label>Sand moved ≥ <b id="tv">' + THRESHOLD.toFixed(2) + '</b> cm</label>' +
    '<input type="range" id="tr" min="0.1" max="3" step="0.05" value="' +
      THRESHOLD + '">' +
    '<label>Ellipse covers <b id="ev">' + Math.round(100 * COVERAGE) +
    '</b>%</label>' +
    '<input type="range" id="er" min="50" max="99" step="1" value="' +
      Math.round(100 * COVERAGE) + '">';

  let pending = null;
  const later = () => {
    if (pending) clearTimeout(pending);
    pending = setTimeout(draw, 220);
  };
  bar.querySelector('#cr').addEventListener('input', event => {
    CONFIDENCE = parseFloat(event.target.value);
    bar.querySelector('#cv').textContent = CONFIDENCE.toFixed(2);
    later();
  });
  bar.querySelector('#tr').addEventListener('input', event => {
    THRESHOLD = parseFloat(event.target.value);
    bar.querySelector('#tv').textContent = THRESHOLD.toFixed(2);
    later();
  });
  bar.querySelector('#er').addEventListener('input', event => {
    COVERAGE = parseInt(event.target.value, 10) / 100;
    bar.querySelector('#ev').textContent = Math.round(100 * COVERAGE);
    later();
  });
  draw();
}

loadPayload(() => {
  loadEvents().then(events => {
    if (!events) {
      document.getElementById('body').appendChild(el('div', 'note',
        'No cluster data collected for this project.'));
      return;
    }
    build();
    document.getElementById('foot').textContent =
      'Collected ' + D.collected + ' · page built ' + D.built;
  });
});

// =====================================================================
// The second row per trial: five more columns under the first five.
//
// These answer questions about the trial as a whole rather than about one
// day, so they do not belong in the matrix above, whose columns are days.
// =====================================================================

const SPLIT_AT = 0.5;        // the share of total sand moved that ends half one
const MIN_FOR_SHAPE = 25;    // events before an ellipse's shape is reported
// Dot sizes. The radius was a fixed multiple of sqrt(count), which put a
// busy day at eighteen pixels and swallowed its neighbours. Normalising to
// the trial's busiest day keeps area proportional to count within the panel
// and bounds the result; the floor keeps a one-spawn day visible instead of
// vanishing into the axis.
const DOT_MIN = 1.7;
const DOT_MAX = 5.5;
const SCALE_PERCENTILE = 0.985;   // of the depth magnitudes, for the halves

// Bucket keys are the numeric behaviour codes, not the letters. Built lazily
// because EVENTS is not loaded when this file is evaluated. 'b', bower
// multiple, is deliberately in neither: it is both a scoop and a spit, so it
// would have to be counted twice or not at all, and counting it twice makes
// the two clouds artificially alike.
let SCOOP_CODE = null, SPIT_CODE = null;

function buildCodeSets() {
  SCOOP_CODE = EVENTS.code.c;
  SPIT_CODE = EVENTS.code.p;
}

// Sand moved between the endpoints the bundle already holds. The day's own
// window is first to last; the night is the previous day's last to this
// day's first — so overnight building is in the collected data already, and
// needed no new collection to get at.
let VOLUME_CACHE = {};

// Three panels want these numbers and the arithmetic is a pass over every
// pixel of every endpoint, so it happens once per trial and threshold.
function volumesFor(trial, days, meta) {
  const key = trial + ':' + THRESHOLD;
  if (!VOLUME_CACHE[key]) VOLUME_CACHE[key] = perDayVolumes(trial, days, meta);
  return VOLUME_CACHE[key];
}

function perDayVolumes(trial, days, meta) {
  return Promise.all(days.map(d =>
    Promise.all([loadDepth(d.firstPng), loadDepth(d.lastPng)])))
    .then(pairs => days.map((d, i) => {
      const [first, last] = pairs[i];
      const prevLast = i > 0 ? pairs[i - 1][1] : null;
      return {
        index: d.index, date: d.date, day: d,
        light: (first && last)
          ? volumes(cropped(difference(first, last), meta, trial), THRESHOLD)
          : null,
        dark: (prevLast && first)
          ? volumes(cropped(difference(prevLast, first), meta, trial), THRESHOLD)
          : null,
      };
    }));
}

// The day at which half the trial's sand has moved. Snapped to a day
// boundary, because days are the unit everywhere else; the fraction actually
// reached is reported rather than assumed to be a half.
function splitPoint(perDay) {
  const moved = perDay.map(v => (v.light ? v.light.total : 0) +
                                (v.dark ? v.dark.total : 0));
  const final = moved.reduce((a, b) => a + b, 0);
  if (!final) return null;
  let running = 0;
  for (let i = 0; i < moved.length; i++) {
    running += moved[i];
    if (running >= SPLIT_AT * final)
      return { at: i, fraction: running / final, total: final };
  }
  return { at: moved.length - 1, fraction: 1, total: final };
}

function hoursBetween(a, b) {
  const t0 = Date.parse(a), t1 = Date.parse(b);
  if (!isFinite(t0) || !isFinite(t1)) return null;
  const hours = (t1 - t0) / 3600000;
  // A negative span means the days are out of order or a clock moved, and a
  // span over two days is not a night. Either way it is not a measurement:
  // summing it blindly printed '-39.9 h dark' as though it were one.
  return (hours > 0 && hours < 48) ? hours : null;
}

// --------------------------------------------------------------- plotting
// A plotting frame, not a chart library: four small pictures do not justify
// the dependency, and every one of them is dots or bars against a day axis.
function frame(width, height, xLo, xHi, yLo, yHi) {
  const pad = { l: 46, r: 8, t: 8, b: 22 };
  const canvas = el('canvas');
  canvas.width = width; canvas.height = height;
  canvas.style.width = '100%';
  const ctx = canvas.getContext('2d');
  const X = v => pad.l + (v - xLo) / (xHi - xLo || 1) * (width - pad.l - pad.r);
  const Y = v => height - pad.b - (v - yLo) / (yHi - yLo || 1) * (height - pad.t - pad.b);

  ctx.fillStyle = '#0b0d11';
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = '#262d38';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, height - pad.b);
  ctx.lineTo(width - pad.r, height - pad.b);
  ctx.stroke();

  return {
    canvas, ctx, X, Y, pad, width, height,
    ticks(values, format) {
      ctx.fillStyle = '#93a0b0';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'right';
      values.forEach(v => {
        const y = Y(v);
        ctx.fillText(format ? format(v) : String(v), pad.l - 5, y + 3);
        ctx.strokeStyle = 'rgba(147,160,176,.16)';
        ctx.beginPath();
        ctx.moveTo(pad.l, y); ctx.lineTo(width - pad.r, y);
        ctx.stroke();
      });
    },
    xLabel(v, text) {
      ctx.fillStyle = '#93a0b0';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(text, X(v), height - pad.b + 13);
    },
    zero(v) {
      ctx.strokeStyle = 'rgba(147,160,176,.45)';
      ctx.beginPath();
      ctx.moveTo(pad.l, Y(v)); ctx.lineTo(width - pad.r, Y(v));
      ctx.stroke();
    },
    dot(x, y, colour, radius) {
      ctx.fillStyle = 'rgb(' + colour.join(',') + ')';
      ctx.beginPath();
      ctx.arc(X(x), Y(y), radius === undefined ? 2.6 : radius, 0, Math.PI * 2);
      ctx.fill();
    },
    line(points, colour) {
      if (points.length < 2) return;
      ctx.strokeStyle = 'rgba(' + colour.join(',') + ',.55)';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      points.forEach(([x, y], i) => {
        const px = X(x), py = Y(y);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    },
    tick(x, y, colour) {
      ctx.strokeStyle = colour;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(X(x) - 4, Y(y)); ctx.lineTo(X(x) + 4, Y(y));
      ctx.stroke();
    },
  };
}

function stripPanel(title, hint) {
  const box = el('figure');
  const caption = el('figcaption',
    null, '<b>' + title + '</b><br>' + (hint || ''));
  box._caption = caption;
  box._stage = el('div', 'stage');
  box.appendChild(box._stage);
  box.appendChild(caption);
  return box;
}

function nice(value, digits) {
  return (value === null || value === undefined || !isFinite(value))
    ? '—' : value.toFixed(digits === undefined ? 0 : digits);
}

// ------------------------------------------------- 1 and 2: volume halves
// Half of the sand moved, then the other half. Splitting on volume rather
// than on time compares two males at equal work done instead of equal
// elapsed time, which is the point: it takes the building rate out.
//
// The two indices do not average to the trial's index. Each describes the
// sand moved inside its own window, so they are not a decomposition of the
// finished bower and should not be read as one.
function halfPanels(trial, days, meta) {
  const first = stripPanel('First half of the volume', 'cumulative change');
  const second = stripPanel('Second half of the volume', 'cumulative change');
  [first, second].forEach(p => p._stage.appendChild(el('div', 'cellnote', 'measuring…')));

  volumesFor(trial, days, meta).then(perDay => {
    const split = splitPoint(perDay);
    if (!split) {
      [first, second].forEach(p => {
        p._stage.textContent = '';
        p._stage.appendChild(el('div', 'cellnote', 'no sand moved above the threshold'));
      });
      return;
    }
    const cut = days[split.at];
    return Promise.all([loadDepth(days[0].firstPng), loadDepth(cut.lastPng),
                        loadDepth(days[days.length - 1].lastPng)])
      .then(([start, middle, end]) => {
        if (!start || !middle || !end) return;
        const early = cropped(difference(start, middle), meta, trial);
        const late = cropped(difference(middle, end), meta, trial);

        // One colour scale across both, so the halves can be compared by eye;
        // independent scales would make a quiet half look like a busy one.
        //
        // Not the largest pixel, though: a single spike sets the scale and
        // flattens everything real into the middle of the ramp. The high
        // percentile of the two halves together keeps them comparable while
        // leaving the structure visible, and the few pixels past it clip,
        // which is the right trade for a map read by eye.
        const magnitudes = [];
        [early, late].forEach(map => {
          for (let i = 0; i < map.length; i++) {
            const v = Math.abs(map[i]);
            if (!Number.isNaN(v) && v > 0) magnitudes.push(v);
          }
        });
        let span = 0.5, clipped = 0;
        if (magnitudes.length) {
          magnitudes.sort((a, b) => a - b);
          const at = Math.floor(magnitudes.length * SCALE_PERCENTILE);
          span = Math.max(0.5, magnitudes[Math.min(at, magnitudes.length - 1)]);
          clipped = magnitudes.length - at;
        }

        [[first, early, 0, split.at, split.fraction],
         [second, late, split.at, days.length - 1, 1 - split.fraction]]
          .forEach(([box, map, fromDay, toDay, share]) => {
            box._stage.textContent = '';
            const canvas = el('canvas');
            box._stage.appendChild(canvas);
            paintMap(canvas, map, meta, { range: span });
            const v = volumes(map, THRESHOLD);
            const spanDays = toDay - fromDay + (box === first ? 1 : 0);
            box._caption.innerHTML =
              '<b>' + (box === first ? 'First' : 'Second') + ' half of the volume</b><br>' +
              'net volume <b>' + nice(v.total) + '</b> cm³ · ' +
              'index <b>' + nice(v.index, 3) + '</b><br>' +
              'days ' + (days[fromDay].index + 1) + '–' +
              (days[toDay].index + 1) + ' · ' + spanDays + ' days · ' +
              nice(share * 100) + '% of gross sand moved<br>' +
              '<span style="color:#6c7684">net is across the window, gross is ' +
              'day by day — they part where sand went back</span><br>' +
              '<span style="color:#6c7684">scale ±' + nice(span, 2) +
              ' cm, shared' + (clipped ? ' · ' + clipped + ' px clipped' : '') +
              '</span>';
          });
      });
  });
  return [first, second];
}

// ------------------------------------------- 3: how big the clouds are
// Ellipse area per day, scoops against spits, with the pooled half-trial and
// whole-trial figures to the right of a gap. The pooled number is not the
// mean of the days: pooling adds the day-to-day wandering of the centre, so
// a male who works one spot a day but moves about over the trial pools much
// wider than he scatters. That gap is the interesting part.
function areaPanel(trial, days, split) {
  const box = stripPanel('Cloud size by day', 'ellipse area · scoop orange, spit blue');
  const perDay = days.map(d => ({
    index: d.index,
    scoop: dispersion(pointsOnDay(trial, SCOOP_CODE, d.index), COVERAGE),
    spit: dispersion(pointsOnDay(trial, SPIT_CODE, d.index), COVERAGE),
  }));
  const values = [];
  perDay.forEach(r => ['scoop', 'spit'].forEach(k => {
    if (r[k]) values.push(r[k].area);
  }));

  const pooled = {};
  const ranges = { H1: [0, split ? split.at : days.length - 1],
                   H2: [split ? split.at : 0, days.length - 1],
                   All: [0, days.length - 1] };
  Object.keys(ranges).forEach(label => {
    const [lo, hi] = ranges[label];
    pooled[label] = {};
    ['scoop', 'spit'].forEach(k => {
      const codes = k === 'scoop' ? SCOOP_CODE : SPIT_CODE;
      const points = [];
      for (let i = lo; i <= hi; i++)
        pointsOnDay(trial, codes, days[i].index).forEach(p => points.push(p));
      const shape = dispersion(points, COVERAGE);
      pooled[label][k] = shape;
      if (shape) values.push(shape.area);
    });
  });

  if (!values.length) {
    box._stage.appendChild(el('div', 'cellnote',
      'no day has ' + 10 + ' or more events of either kind'));
    return box;
  }
  const top = Math.max.apply(null, values) * 1.1;
  const slots = days.length + 4;
  const f = frame(420, 180, -0.5, slots - 0.5, 0, top);
  const step = top / 4;
  f.ticks([step, 2 * step, 3 * step, 4 * step], v => nice(v));
  ['scoop', 'spit'].forEach(k => {
    const colour = k === 'scoop' ? SCOOP : SPIT;
    const path = [];
    perDay.forEach((r, i) => {
      if (!r[k]) return;
      const y = r[k].area;
      f.dot(i, y, colour);
      path.push([i, y]);
    });
    f.line(path, colour);
    ['H1', 'H2', 'All'].forEach((label, j) => {
      const shape = pooled[label][k];
      if (shape) f.dot(days.length + 1 + j, shape.area, colour, 3.4);
    });
  });
  f.xLabel(0, '1');
  f.xLabel(days.length - 1, String(days.length));
  ['H1', 'H2', 'All'].forEach((label, j) =>
    f.xLabel(days.length + 1 + j, label));
  box._stage.appendChild(f.canvas);
  box._caption.innerHTML =
    '<b>Cloud size by day</b><br>ellipse area, cm² · ' +
    'scoop orange, spit blue<br>' +
    '<span style="color:#6c7684">H1, H2 and All pool their events, which is ' +
    'wider than the daily mean whenever the centre moves between days</span>';
  return box;
}

// This page's dispersion() returns area in cm2 already, unlike the summary
// sheet's, which returns pixels squared. Converting a second time here
// overstated every cloud by about ninety-four.
function pointsOnDay(trial, code, dayIndex) {
  const byDay = (BUCKETS && BUCKETS[trial]) || {};
  return (byDay[dayIndex] && byDay[dayIndex][code]) || [];
}

// --------------------------------------- 4: what shape the clouds are
// Aspect ratio of the same ellipses, which is the elongation question. Note
// this is NOT the correlation of x with y: a cloud stretched along the tank
// axis has zero x-y correlation and an aspect ratio of five, so correlating
// the coordinates would miss exactly the elongation it was meant to find.
//
// The grey ticks are the median aspect ratio a perfectly circular cloud of
// that day's event count would produce. Anything near its own tick is a
// count, not a shape.
function aspectPanel(trial, days, split) {
  const box = stripPanel('Cloud shape by day', 'aspect ratio · grey = circular at that n');
  const shapeOf = points => {
    const d = dispersion(points, COVERAGE);
    if (!d || d.n < MIN_FOR_SHAPE || d.minor <= 0) return null;
    return { ratio: d.major / d.minor, n: d.n,
             angle: d.angle * 180 / Math.PI };
  };
  const perDay = days.map(d => ({
    scoop: shapeOf(pointsOnDay(trial, SCOOP_CODE, d.index)),
    spit: shapeOf(pointsOnDay(trial, SPIT_CODE, d.index)),
  }));

  const pooled = {};
  const ranges = { H1: [0, split ? split.at : days.length - 1],
                   H2: [split ? split.at : 0, days.length - 1],
                   All: [0, days.length - 1] };
  Object.keys(ranges).forEach(label => {
    const [lo, hi] = ranges[label];
    pooled[label] = {};
    ['scoop', 'spit'].forEach(k => {
      const codes = k === 'scoop' ? SCOOP_CODE : SPIT_CODE;
      const points = [];
      for (let i = lo; i <= hi; i++)
        pointsOnDay(trial, codes, days[i].index).forEach(p => points.push(p));
      pooled[label][k] = shapeOf(points);
    });
  });

  const values = [];
  perDay.forEach(r => ['scoop', 'spit'].forEach(k => {
    if (r[k]) { values.push(r[k].ratio); values.push(nullAspect(r[k].n)); }
  }));
  Object.keys(pooled).forEach(label => ['scoop', 'spit'].forEach(k => {
    if (pooled[label][k]) values.push(pooled[label][k].ratio);
  }));
  if (!values.length) {
    box._stage.appendChild(el('div', 'cellnote',
      'no day reaches ' + MIN_FOR_SHAPE + ' events of either kind'));
    return box;
  }

  const top = Math.max(2, Math.max.apply(null, values) * 1.08);
  const slots = days.length + 4;
  const f = frame(420, 180, -0.5, slots - 0.5, 1, top);
  f.ticks([1.5, 2, 3, 4].filter(v => v < top), v => v.toFixed(1));
  f.zero(1);
  ['scoop', 'spit'].forEach(k => {
    const colour = k === 'scoop' ? SCOOP : SPIT;
    const path = [];
    perDay.forEach((r, i) => {
      if (!r[k]) return;
      f.tick(i, nullAspect(r[k].n), 'rgba(147,160,176,.5)');
      f.dot(i, r[k].ratio, colour);
      path.push([i, r[k].ratio]);
    });
    f.line(path, colour);
    ['H1', 'H2', 'All'].forEach((label, j) => {
      const shape = pooled[label][k];
      if (!shape) return;
      f.tick(days.length + 1 + j, nullAspect(shape.n), 'rgba(147,160,176,.5)');
      f.dot(days.length + 1 + j, shape.ratio, colour, 3.4);
    });
  });
  f.xLabel(0, '1');
  f.xLabel(days.length - 1, String(days.length));
  ['H1', 'H2', 'All'].forEach((label, j) => f.xLabel(days.length + 1 + j, label));
  box._stage.appendChild(f.canvas);

  const all = pooled.All;
  box._caption.innerHTML =
    '<b>Cloud shape by day</b><br>aspect ratio of the same ellipses · ' +
    'at least ' + MIN_FOR_SHAPE + ' events<br>' +
    'whole trial: scoop <b>' + (all.scoop ? nice(all.scoop.ratio, 2) : '—') +
    '</b> (circular would be ' +
    (all.scoop ? nice(nullAspect(all.scoop.n), 2) : '—') + ') · ' +
    'spit <b>' + (all.spit ? nice(all.spit.ratio, 2) : '—') + '</b> (' +
    (all.spit ? nice(nullAspect(all.spit.n), 2) : '—') + ')<br>' +
    'angle to the tray’s x axis: scoop ' +
    (all.scoop ? nice(all.scoop.angle) + '°' : '—') + ' · spit ' +
    (all.spit ? nice(all.spit.angle) + '°' : '—');
  return box;
}

// ----------------------------------------- 5: spawn depth, sized by count
// A day with no spawns is absent, not plotted at zero: zero spawns has no
// depth, and a dot at zero sits exactly where "spawned on flat sand" lives.
function spawnDepthPanel(trial, days, meta) {
  const box = stripPanel('Spawn depth by day', 'dot area = spawns that day');
  box._stage.appendChild(el('div', 'cellnote', 'measuring…'));
  const spawnCode = EVENTS.code.s;

  Promise.all(days.map(d => Promise.all([loadDepth(days[0].firstPng),
                                         loadDepth(d.lastPng)])))
    .then(pairs => {
      const rows = [];
      days.forEach((d, i) => {
        const points = pointsOnDay(trial, spawnCode, d.index);
        if (!points.length) return;
        const [start, end] = pairs[i];
        if (!start || !end) return;
        // against the sand as it stood that day, never the finished bower
        const map = cropped(difference(start, end), meta, trial);
        const depths = [];
        points.forEach(([x, y]) => {
          const px = Math.round(x / (D.downsample || 1));
          const py = Math.round(y / (D.downsample || 1));
          if (px < 0 || py < 0 || px >= meta.width || py >= meta.height) return;
          const v = map[py * meta.width + px];
          if (!Number.isNaN(v)) depths.push(v);
        });
        if (!depths.length) return;
        const mean = depths.reduce((a, b) => a + b, 0) / depths.length;
        let lo = depths[0], hi = depths[0];
        depths.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; });
        rows.push({ at: i, mean, lo, hi, n: points.length, used: depths.length });
      });

      box._stage.textContent = '';
      if (!rows.length) {
        box._stage.appendChild(el('div', 'cellnote', 'no spawns in this trial'));
        box._caption.innerHTML = '<b>Spawn depth by day</b><br>no spawns above the ' +
          'confidence threshold';
        return;
      }
      let lo = 0, hi = 0;
      rows.forEach(r => { if (r.lo < lo) lo = r.lo; if (r.hi > hi) hi = r.hi; });
      const pad = Math.max(0.4, (hi - lo) * 0.12);
      const f = frame(420, 180, -0.5, days.length - 0.5, lo - pad, hi + pad);
      const step = (hi - lo + 2 * pad) / 4;
      f.ticks([lo - pad + step, lo - pad + 2 * step, lo - pad + 3 * step],
              v => v.toFixed(1));
      f.zero(0);
      const biggest = Math.max.apply(null, rows.map(r => r.n));
      rows.forEach(r => {
        f.ctx.strokeStyle = 'rgba(232,132,168,.4)';
        f.ctx.lineWidth = 1;
        f.ctx.beginPath();
        f.ctx.moveTo(f.X(r.at), f.Y(r.lo));
        f.ctx.lineTo(f.X(r.at), f.Y(r.hi));
        f.ctx.stroke();
        // area proportional to the count, so radius goes as its square root
        f.dot(r.at, r.mean, SPAWN,
              Math.max(DOT_MIN, DOT_MAX * Math.sqrt(r.n / biggest)));
      });
      // Two legend dots, because one tells you nothing about the mapping:
      // the largest day and a quarter of it, which is half the radius.
      f.ctx.font = '10px sans-serif';
      f.ctx.textAlign = 'left';
      [[biggest, DOT_MAX], [Math.round(biggest / 4), DOT_MAX / 2]]
        .forEach(([count, radius], i) => {
          const cx = f.pad.l + 10 + i * 56;
          f.ctx.fillStyle = 'rgba(232,132,168,.85)';
          f.ctx.beginPath();
          f.ctx.arc(cx, 16, radius, 0, Math.PI * 2);
          f.ctx.fill();
          f.ctx.fillStyle = '#93a0b0';
          f.ctx.fillText(String(count), cx + DOT_MAX + 4, 19);
        });
      f.xLabel(0, '1');
      f.xLabel(days.length - 1, String(days.length));
      box._stage.appendChild(f.canvas);

      const spawns = rows.reduce((sum, r) => sum + r.n, 0);
      const weighted = rows.reduce((sum, r) => sum + r.mean * r.n, 0) / spawns;
      box._caption.innerHTML =
        '<b>Spawn depth by day</b><br>cumulative depth under each spawn, that ' +
        'day · whisker is the day’s range<br>' +
        spawns + ' spawns over ' + rows.length + ' days · ' +
        'mean <b>' + nice(weighted, 2) + '</b> cm<br>' +
        '<span style="color:#6c7684">dot area is the count, scaled to this ' +
        'trial’s busiest day, so sizes compare within the panel and not ' +
        'between trials; days with no spawns are absent, not zero</span>';
    });
  return box;
}

// -------------------------------------------- 6: how much is built at night
// The depth sensor is infrared and sees in the dark; the video does not, so
// this is the one build measure that can only come from depth and will never
// have events beside it.
function nightStrip(trial, days, meta) {
  const box = stripPanel('Built overnight', 'share of the day’s sand');
  box._stage.appendChild(el('div', 'cellnote', 'measuring…'));

  volumesFor(trial, days, meta).then(perDay => {
    const rows = [];
    let darkTotal = 0, lightTotal = 0, darkHours = 0, lightHours = 0;
    let litDays = 0, darkNights = 0;
    perDay.forEach((v, i) => {
      if (!v.light) return;
      lightTotal += v.light.total;
      const lit = hoursBetween(v.day.first, v.day.last);
      if (lit) { lightHours += lit; litDays++; }
      if (!v.dark) return;        // day one has no night before it
      darkTotal += v.dark.total;
      const dark = hoursBetween(days[i - 1].last, v.day.first);
      if (dark) { darkHours += dark; darkNights++; }
      // the 24 hour window for this day is the previous evening through to
      // this evening, which is exactly the night plus the day
      const both = v.dark.total + v.light.total;
      if (both > 0) rows.push({ at: i, share: v.dark.total / both,
                                night: v.dark.total, total: both });
    });

    box._stage.textContent = '';
    if (!rows.length) {
      box._stage.appendChild(el('div', 'cellnote', 'no overnight pair to compare'));
      return;
    }
    // Volume, not the share. A share on its own cannot tell a real night of
    // digging from a day where almost nothing moved and the little that did
    // happened to be at night — those look identical at 80%. Stacking the
    // night inside the 24 hour total shows both at once: how much moved, and
    // how much of it was dark.
    const top = Math.max.apply(null, rows.map(r => r.total)) * 1.08 || 1;
    const f = frame(420, 118, -0.5, days.length - 0.5, 0, top);
    const step = top / 3;
    f.ticks([step, 2 * step, 3 * step], v => nice(v));
    const width = Math.max(2, Math.min(9, 300 / days.length));
    rows.forEach(r => {
      const x = f.X(r.at), base = f.Y(0);
      f.ctx.fillStyle = 'rgba(147,160,176,.26)';
      f.ctx.fillRect(x - width / 2, f.Y(r.total), width, base - f.Y(r.total));
      f.ctx.fillStyle = 'rgba(111,178,232,.8)';
      f.ctx.fillRect(x - width / 2, f.Y(r.night), width, base - f.Y(r.night));
    });
    f.ctx.font = '10px sans-serif';
    f.ctx.textAlign = 'left';
    f.ctx.fillStyle = 'rgba(111,178,232,.9)';
    f.ctx.fillRect(f.pad.l + 4, 10, 7, 7);
    f.ctx.fillStyle = '#93a0b0';
    f.ctx.fillText('night', f.pad.l + 15, 17);
    f.ctx.fillStyle = 'rgba(147,160,176,.4)';
    f.ctx.fillRect(f.pad.l + 48, 10, 7, 7);
    f.ctx.fillStyle = '#93a0b0';
    f.ctx.fillText('24 h', f.pad.l + 59, 17);
    f.xLabel(0, '1');
    f.xLabel(days.length - 1, String(days.length));
    box._stage.appendChild(f.canvas);

    const share = darkTotal / (darkTotal + lightTotal || 1);
    // a 14 hour night against a 10 hour day makes a half share mean the night
    // rate is lower, so the rate ratio is given beside the raw share
    const rate = (darkHours > 0 && lightHours > 0)
      ? (darkTotal / darkHours) / (lightTotal / lightHours) : null;
    box._caption.innerHTML =
      '<b>Built overnight</b><br>night inside the 24 hour total, cm³<br>' +
      'trial <b>' + nice(share * 100, 1) + '%</b> of ' +
      nice(darkTotal + lightTotal) + ' cm³<br>' +
      'rate <b>' + (rate === null ? '—' : nice(rate, 2) + '×') +
      '</b> the daytime rate · ' +
      (darkNights ? nice(darkHours / darkNights, 1) + ' h dark' : 'dark span unknown') +
      ' vs ' + (litDays ? nice(lightHours / litDays, 1) + ' h light'
                        : 'light span unknown') + '<br>' +
      '<span style="color:#6c7684">depth only — the cameras cannot see ' +
      'in the dark, so no events accompany this</span>';
  });
  return box;
}

// Five more cells in the same matrix, so they line up under the first five.
// Each carries its own heading in its caption: the matrix has one header row
// and these columns do not mean what that row says.
//
// Six measurements in five columns -- the overnight share rides under the
// second half, which is the other volume panel, rather than taking a column
// of its own and breaking the alignment with the row above.
function renderTrialDetail(trial, days, table) {
  const meta = days[0].firstPng;
  table.appendChild(el('div', 'rowlab', '<b>Trial ' + trial +
    '</b><span>halves by volume,<br>clouds, spawn depth</span>'));

  const slots = [];
  for (let i = 0; i < 5; i++) {
    const slot = el('div', 'cell');
    table.appendChild(slot);
    slots.push(slot);
  }

  const halves = halfPanels(trial, days, meta);
  slots[0].appendChild(halves[0]);
  slots[1].appendChild(halves[1]);
  slots[1].appendChild(nightStrip(trial, days, meta));
  slots[4].appendChild(spawnDepthPanel(trial, days, meta));

  // The ellipse panels need the volume split, which is asynchronous.
  // Appending them when their promise settles put the columns in a different
  // order on every reload, so their places are held now.
  [2, 3].forEach(i => {
    const holding = stripPanel('\u2026', 'waiting for the volume split');
    holding._stage.appendChild(el('div', 'cellnote', 'measuring\u2026'));
    slots[i].appendChild(holding);
  });
  volumesFor(trial, days, meta).then(perDay => {
    const split = splitPoint(perDay);
    slots[2].textContent = '';
    slots[3].textContent = '';
    slots[2].appendChild(areaPanel(trial, days, split));
    slots[3].appendChild(aspectPanel(trial, days, split));
  });
}
