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
function pixelArea() {
  // a map pixel covers more than a sensor pixel: the maps are downsampled for
  // the browser, and forgetting that understates volume by the square
  const cm = D.pixelLength || 0.1030168618;
  const factor = D.downsample || 1;
  return (cm * factor) * (cm * factor);
}

function volumes(values, threshold) {
  const area = pixelArea();
  let castle = 0, pit = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isNaN(v)) continue;
    if (v >= threshold) castle += v * area;
    else if (v <= -threshold) pit += -v * area;
  }
  const total = castle + pit;
  return { castle, pit, total,
           index: total > 0 ? (castle - pit) / total : 0 };
}

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
  trials.forEach(trial => renderTrial(+trial, byTrial[trial], table));
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
