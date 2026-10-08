/* The features page: one project reduced to the numbers that describe it.
 *
 * Everything here is derived, not stored — volumes from the depth bundle,
 * ellipses from the events, spawn depths from both — so it changes when the
 * crop, the threshold or the registration changes, and never disagrees with
 * the page it came from.
 *
 * Excluded trials are left out throughout. That is the point of excluding
 * them: a trial with no building should not pull a project's bower index
 * toward zero.
 */

let EVENTS = null;
let CONFIDENCE = 0.67;
let THRESHOLD = 0.6;     // cm of height before sand counts as moved
let COVERAGE = 0.68;

const SCOOP = [242, 163, 60], SPIT = [111, 178, 232], SPAWN = [232, 132, 168];

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
  for (let i = 0; i < EVENTS.n; i++) {
    const H = transformFor(EVENTS.trial[i]);
    if (!H) continue;
    const point = applyH(H, [EVENTS.x[i], EVENTS.y[i]]);
    EVENTS.dx[i] = point[0];
    EVENTS.dy[i] = point[1];
    EVENTS.placed[i] = 1;
  }
}

function usable(index, cut) {
  return EVENTS.placed[index] && (EVENTS.flags[index] & 1) &&
         (EVENTS.flags[index] & 2) && EVENTS.bid[index] !== 255 &&
         EVENTS.prob[index] >= cut;
}

// --------------------------------------------------------------- geometry
// One map pixel covers more than one sensor pixel, because the maps are stored
// downsampled for the browser. Forgetting that understates every volume by the
// square of the factor, so the area comes from both numbers rather than from
// the pixel size alone.
function pixelArea() {
  const cm = D.pixelLength || 0.1030168618;
  const factor = D.downsample || 1;
  return (cm * factor) * (cm * factor);
}

function volumes(values, threshold) {
  // castle is sand added, pit is sand taken away; a pixel that moved less than
  // the threshold is noise and counts as neither
  const area = pixelArea();
  let castle = 0, pit = 0, castleArea = 0, pitArea = 0, valid = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isNaN(v)) continue;
    valid++;
    if (v >= threshold) { castle += v * area; castleArea += area; }
    else if (v <= -threshold) { pit += -v * area; pitArea += area; }
  }
  const total = castle + pit;
  return { castle, pit, total, castleArea, pitArea, valid,
           // -1 is a pure pit, +1 a pure castle, 0 as much dug as piled
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
  return { cx: mx, cy: my, n: points.length,
           area: Math.PI * k * Math.sqrt(determinant) };
}

function eventsUpTo(trial, code, cut, dayIndex) {
  const points = [];
  for (let i = 0; i < EVENTS.n; i++) {
    if (EVENTS.trial[i] !== trial || EVENTS.day[i] > dayIndex) continue;
    if (!usable(i, cut) || EVENTS.bid[i] !== code) continue;
    points.push([EVENTS.dx[i], EVENTS.dy[i]]);
  }
  return points;
}

function cropped(values, meta, trial) {
  const outside = cropMaskFor(meta, trial);
  if (!outside) return values;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = outside[i] ? NaN : values[i];
  return out;
}

// ------------------------------------------------------------------ charts
function lineChart(series, options) {
  const width = 760, height = 240;
  const pad = { left: 56, right: 14, top: 22, bottom: 30 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const columns = options.labels.length;
  let low = options.min, high = options.max;
  if (low === undefined || high === undefined) {
    low = Infinity; high = -Infinity;
    series.forEach(one => one.values.forEach(v => {
      if (v === null || Number.isNaN(v)) return;
      if (v < low) low = v;
      if (v > high) high = v;
    }));
    if (!isFinite(low)) { low = 0; high = 1; }
    if (high === low) high = low + 1;
    const margin = (high - low) * 0.1;
    low -= margin; high += margin;
  }
  const x = i => pad.left + (columns > 1 ? plotW * i / (columns - 1) : plotW / 2);
  const y = v => pad.top + plotH - (v - low) / (high - low) * plotH;

  let body = '';
  for (let i = 0; i <= 4; i++) {
    const value = low + (high - low) * i / 4;
    body += '<line x1="' + pad.left + '" y1="' + y(value) + '" x2="' +
            (width - pad.right) + '" y2="' + y(value) + '" stroke="#1d232c"/>' +
            '<text x="' + (pad.left - 7) + '" y="' + (y(value) + 4) +
            '" fill="#93a0b0" font-size="10" text-anchor="end">' +
            (Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(2)) + '</text>';
  }
  if (low < 0 && high > 0)
    body += '<line x1="' + pad.left + '" y1="' + y(0) + '" x2="' +
            (width - pad.right) + '" y2="' + y(0) +
            '" stroke="#3a4350" stroke-dasharray="4 3"/>';

  series.forEach(one => {
    const colour = 'rgb(' + one.colour.join(',') + ')';
    let path = '', open = false;
    one.values.forEach((value, i) => {
      if (value === null || Number.isNaN(value)) { open = false; return; }
      path += (open ? ' L' : ' M') + x(i) + ',' + y(value);
      open = true;
    });
    body += '<path d="' + path + '" fill="none" stroke="' + colour +
            '" stroke-width="1.8"/>';
    one.values.forEach((value, i) => {
      if (value === null || Number.isNaN(value)) return;
      body += '<circle cx="' + x(i) + '" cy="' + y(value) + '" r="2.4" fill="' +
              colour + '"><title>' + options.labels[i] + ' · ' + one.name +
              ': ' + value.toFixed(2) + '</title></circle>';
    });
  });

  options.labels.forEach((label, i) => {
    if (columns > 12 && i % Math.ceil(columns / 10)) return;
    body += '<text x="' + x(i) + '" y="' + (height - 9) +
            '" fill="#93a0b0" font-size="10" text-anchor="middle">' + label +
            '</text>';
  });
  body += '<line x1="' + pad.left + '" y1="' + (height - pad.bottom) + '" x2="' +
          (width - pad.right) + '" y2="' + (height - pad.bottom) +
          '" stroke="#262d38"/>';
  body += '<text x="' + pad.left + '" y="' + (pad.top - 8) +
          '" fill="#e8ecf1" font-size="11" font-weight="600">' + options.title +
          '</text>';
  let legendX = pad.left + 150;
  series.forEach(one => {
    body += '<text x="' + legendX + '" y="' + (pad.top - 8) + '" fill="rgb(' +
            one.colour.join(',') + ')" font-size="10">● ' + one.name + '</text>';
    legendX += 14 + one.name.length * 6.2;
  });

  const figure = el('figure');
  const holder = el('div');
  holder.innerHTML = '<svg viewBox="0 0 ' + width + ' ' + height +
                     '" style="width:100%;display:block">' + body + '</svg>';
  figure.appendChild(holder);
  if (options.caption) figure.appendChild(el('figcaption', null, options.caption));
  return figure;
}

function statTile(label, value, note) {
  const box = el('div', 'tile');
  box.innerHTML = '<span class="tlabel">' + label + '</span>' +
                  '<b class="tvalue">' + value + '</b>' +
                  (note ? '<span class="tnote">' + note + '</span>' : '');
  return box;
}

// ------------------------------------------------------------------- build
function renderTrial(trial, days, host) {
  const cut = Math.round(CONFIDENCE * 255);
  const meta = days[0].firstPng;
  const labels = days.map(day => day.date.slice(5));

  const section = el('div', 'trial');
  section.appendChild(el('h3', null, 'Trial ' + trial +
    '<span>' + days.length + ' days · ' + days[0].date + ' to ' +
    days[days.length - 1].date + '</span>'));
  host.appendChild(section);
  const tiles = el('div', 'tiles');
  section.appendChild(tiles);
  const charts = el('div');
  section.appendChild(charts);

  Promise.all([loadDepth(days[0].firstPng)].concat(
      days.map(day => loadDepth(day.lastPng)))).then(frames => {
    const start = frames[0];
    if (!start) return;

    const perDay = days.map((day, i) => {
      const end = frames[i + 1];
      if (!end) return null;
      return volumes(cropped(difference(start, end), meta, trial), THRESHOLD);
    });
    const last = perDay.filter(Boolean).pop();
    if (!last) return;

    // the ellipses the summary page draws, measured rather than only shown
    const cm2 = (D.pixelLength || 0.1030168618) *
                (D.pixelLength || 0.1030168618);
    const scoopArea = [], spitArea = [], ratio = [];
    days.forEach(day => {
      const a = dispersion(eventsUpTo(trial, EVENTS.code.c, cut, day.index), COVERAGE);
      const b = dispersion(eventsUpTo(trial, EVENTS.code.p, cut, day.index), COVERAGE);
      scoopArea.push(a ? a.area * cm2 : null);
      spitArea.push(b ? b.area * cm2 : null);
      ratio.push(a && b && a.area > 0 ? b.area / a.area : null);
    });

    const spawn = spawnSummary(trial, days, meta, cut, start, frames);

    tiles.appendChild(statTile('Total sand moved',
      last.total.toFixed(0) + ' cm³',
      'castle ' + last.castle.toFixed(0) + ' + pit ' + last.pit.toFixed(0)));
    tiles.appendChild(statTile('Bower index', last.index.toFixed(3),
      last.index > 0.33 ? 'castle' : (last.index < -0.33 ? 'pit' : 'mixed')));
    tiles.appendChild(statTile('Castle area',
      last.castleArea.toFixed(0) + ' cm²',
      (100 * last.castleArea / Math.max(1, last.valid * pixelArea())).toFixed(1) +
      '% of the tray'));
    tiles.appendChild(statTile('Pit area', last.pitArea.toFixed(0) + ' cm²',
      (100 * last.pitArea / Math.max(1, last.valid * pixelArea())).toFixed(1) +
      '% of the tray'));
    tiles.appendChild(statTile('Spawns',
      spawn ? String(spawn.n) : '—',
      spawn ? 'median ' + spawn.median.toFixed(2) + ' cm' : 'none recorded'));
    tiles.appendChild(statTile('Spawning over',
      spawn ? spawn.overCastle.toFixed(0) + '%' : '—',
      'sand that had been added'));

    charts.appendChild(lineChart([
      { name: 'castle', values: perDay.map(v => v && v.castle), colour: SPIT },
      { name: 'pit', values: perDay.map(v => v && v.pit), colour: SCOOP }],
      { labels, title: 'Volume moved, cumulative (cm³)',
        caption: 'Sand added and sand taken away, each measured from the trial ' +
          'start. Both rising together is a fish moving sand within the tray; ' +
          'one flat is a fish only digging, or only piling.' }));

    charts.appendChild(lineChart([
      { name: 'bower index', values: perDay.map(v => v && v.index), colour: SPAWN }],
      { labels, title: 'Bower index', min: -1, max: 1,
        caption: '(castle − pit) / total. +1 is a pure castle, −1 a ' +
          'pure pit, 0 as much dug as piled. The early days are noisy because ' +
          'the denominator is small; it settles as the structure grows.' }));

    charts.appendChild(lineChart([
      { name: 'spits', values: spitArea, colour: SPIT },
      { name: 'scoops', values: scoopArea, colour: SCOOP }],
      { labels, title: 'Spatial spread of building (cm²)',
        caption: 'Area of the ellipse holding ' + Math.round(100 * COVERAGE) +
          '% of each behaviour, accumulating. A fish working one structure ' +
          'holds a small area; one that moves to a new site shows a jump as ' +
          'the ellipse stretches to cover both.' }));
  });
}

function spawnSummary(trial, days, meta, cut, start, frames) {
  const spawns = [];
  for (let i = 0; i < EVENTS.n; i++) {
    if (EVENTS.trial[i] !== trial || !usable(i, cut)) continue;
    if (EVENTS.bid[i] !== EVENTS.code.s) continue;
    spawns.push({ x: EVENTS.dx[i], y: EVENTS.dy[i], day: EVENTS.day[i] });
  }
  if (spawns.length < 5) return null;
  const shape = dispersion(spawns.map(s => [s.x, s.y]), COVERAGE);
  const sx = meta.width / D.depthSize[0], sy = meta.height / D.depthSize[1];
  const position = {};
  days.forEach((day, i) => { position[day.index] = i; });

  const values = [];
  spawns.forEach(spawn => {
    const slot = position[spawn.day];
    if (slot === undefined) return;
    const end = frames[slot + 1];
    if (!end) return;
    // each spawn against the sand as it was on its own day, which is the
    // comparison the summary page's histogram makes
    const cumulative = cropped(difference(start, end), meta, trial);
    const x = Math.round(spawn.x * sx), y = Math.round(spawn.y * sy);
    if (x < 0 || y < 0 || x >= meta.width || y >= meta.height) return;
    const depth = cumulative[y * meta.width + x];
    if (!Number.isNaN(depth)) values.push(depth);
  });
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  return { n: values.length, total: spawns.length,
           median: sorted[sorted.length >> 1],
           overCastle: 100 * values.filter(v => v > 0).length / values.length };
}

function draw() {
  const body = document.getElementById('body');
  body.textContent = '';
  const excluded = excludedBanner();
  if (excluded) body.appendChild(excluded);

  const byTrial = daysByTrial(false);
  const trials = Object.keys(byTrial).sort((a, b) => a - b);
  if (!trials.length) {
    body.appendChild(el('div', 'note', 'No trials to summarise — every ' +
      'trial is excluded, or the project has not been collected.'));
    return;
  }
  trials.forEach(trial => renderTrial(+trial, byTrial[trial], body));
}

function build() {
  document.getElementById('title').textContent = D.projectID;
  document.getElementById('subtitle').textContent =
    'What this project amounts to, trial by trial.';
  document.getElementById('meta').innerHTML =
    [['Tank', D.tankID], ['Analysis', D.analysisID],
     ['Days', D.days.length], ['Trials', D.trials.length],
     ['Events', EVENTS ? EVENTS.n : 0]]
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
  loadEvents().then(() => {
    build();
    document.getElementById('foot').textContent =
      'Collected ' + D.collected + ' · page built ' + D.built;
  });
});
