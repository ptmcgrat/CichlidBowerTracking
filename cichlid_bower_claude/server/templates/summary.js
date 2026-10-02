/* The summary sheet.
 *
 * Where the two streams meet. Cluster events are mapped into depth coordinates
 * through the saved registration, so a spit and the sand it moved can be
 * looked at in the same frame — which is the whole reason for registering the
 * cameras in the first place.
 *
 * Six rows per trial, one column per day, plus two figures underneath that ask
 * questions the maps cannot: where spawning happens relative to the bower, and
 * how tightly scoops and spits are each confined.
 */

let EVENTS = null;
let CONFIDENCE = 0.67;
let HOUR_FROM = 8, HOUR_TO = 18;
let COVERAGE = 0.9;      // the share of events an ellipse should contain
let SCALE = null;        // cm of depth per net event, fitted per trial

const ROWS = [
  ['Depth camera', 'the tray that day', 'still'],
  ['Depth change', 'cumulative, from the sensor', 'depth'],
  ['From events', 'cumulative, spits minus scoops', 'fromEvents'],
  ['Building', 'that day \u00b7 scoop orange, spit blue, multiple green', 'build'],
  ['Spawning', 'that day', 'spawn'],
];

const COLOURS = { scoop: [242, 163, 60], spit: [111, 178, 232],
                  multiple: [90, 168, 122], spawn: [232, 132, 168] };

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

// Every event in depth coordinates, computed once. Without a registration the
// two streams cannot be compared at all, so the page says so rather than
// drawing something meaningless.
function projectToDepth() {
  EVENTS.dx = new Float32Array(EVENTS.n);
  EVENTS.dy = new Float32Array(EVENTS.n);
  // each event through its own trial's registration, which matters only on a
  // project where a camera moved mid-way
  if (!PREP.transform && !Object.keys(PREP.overrides || {}).length) {
    EVENTS.projected = false;
    return;
  }
  for (let i = 0; i < EVENTS.n; i++) {
    const H = transformFor(EVENTS.trial[i]);
    if (!H) continue;
    const point = applyH(H, [EVENTS.x[i], EVENTS.y[i]]);
    EVENTS.dx[i] = point[0];
    EVENTS.dy[i] = point[1];
  }
  EVENTS.projected = true;
}

function keep(index, cut) {
  return (EVENTS.flags[index] & 1) && (EVENTS.flags[index] & 2) &&
         EVENTS.bid[index] !== 255 && EVENTS.prob[index] >= cut &&
         EVENTS.hour[index] >= HOUR_FROM && EVENTS.hour[index] < HOUR_TO;
}

// ------------------------------------------------------------------ ellipse
// The area containing a given share of the events, from the covariance. For a
// bivariate normal the ellipse at coverage p has area pi * k * sqrt(det), with
// k the chi-square quantile on two degrees of freedom — which for two degrees
// is just -2 ln(1 - p), no table needed.
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
  return { cx: mx, cy: my, n: points.length,
           area: Math.PI * k * Math.sqrt(determinant),
           major: Math.sqrt(k * Math.max(0, trace / 2 + root)),
           minor: Math.sqrt(k * Math.max(0, trace / 2 - root)),
           angle: 0.5 * Math.atan2(2 * sxy, sxx - syy) };
}

function eventsOf(trial, codes, cut, upToDay) {
  const points = [];
  for (let i = 0; i < EVENTS.n; i++) {
    if (EVENTS.trial[i] !== trial || !keep(i, cut)) continue;
    if (!codes.has(EVENTS.bid[i])) continue;
    if (upToDay !== undefined && EVENTS.day[i] > upToDay) continue;
    points.push([EVENTS.dx[i], EVENTS.dy[i]]);
  }
  return points;
}

// ------------------------------------------------------------------- panels
function scatterPanel(groups, meta, foot) {
  const box = el('div');
  const stage = el('div', 'stage');
  const canvas = el('canvas');
  canvas.width = meta.width; canvas.height = meta.height;
  stage.appendChild(canvas);
  box.appendChild(stage);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0b0d11';
  ctx.fillRect(0, 0, meta.width, meta.height);
  const sx = meta.width / D.depthSize[0], sy = meta.height / D.depthSize[1];
  groups.forEach(group => {
    ctx.fillStyle = 'rgb(' + group.colour.join(',') + ')';
    ctx.globalAlpha = group.points.length > 300 ? 0.4 : 0.75;
    group.points.forEach(point => {
      ctx.beginPath();
      ctx.arc(point[0] * sx, point[1] * sy, 1.4, 0, 6.2832);
      ctx.fill();
    });
  });
  ctx.globalAlpha = 1;
  if (foot) box.appendChild(el('div', 'cellfoot', foot));
  return box;
}

// Depth inferred from the events: a spit puts sand down, a scoop takes it
// away. The scale is fitted per trial so the totals match the sensor, which
// makes the constant itself informative — it is centimetres of sand per net
// event, and a wildly different value between trials means something changed.
function eventDepthMap(trial, upToDay, meta, cut) {
  const values = new Float32Array(meta.width * meta.height);
  const sx = meta.width / D.depthSize[0], sy = meta.height / D.depthSize[1];
  const spit = EVENTS.code.p, scoop = EVENTS.code.c;
  for (let i = 0; i < EVENTS.n; i++) {
    if (EVENTS.trial[i] !== trial || !keep(i, cut)) continue;
    if (EVENTS.day[i] > upToDay) continue;
    const code = EVENTS.bid[i];
    if (code !== spit && code !== scoop) continue;
    const x = Math.round(EVENTS.dx[i] * sx), y = Math.round(EVENTS.dy[i] * sy);
    if (x < 0 || y < 0 || x >= meta.width || y >= meta.height) continue;
    values[y * meta.width + x] += code === spit ? 1 : -1;
  }
  return blur(values, meta.width, meta.height, 3);
}

function blur(values, width, height, radius) {
  // events land on single pixels; without smoothing the map is dust rather
  // than a surface, and nothing can be compared with the sensor
  const out = new Float32Array(values.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0, count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          sum += values[ny * width + nx];
          count++;
        }
      }
      out[y * width + x] = count ? sum / count : 0;
    }
  }
  return out;
}

function fitScale(depthChange, eventMap) {
  // one number: the ratio that makes the event map's spread match the sensor's
  let a = 0, b = 0;
  for (let i = 0; i < eventMap.length; i++) {
    const measured = depthChange[i];
    if (Number.isNaN(measured)) continue;
    a += eventMap[i] * measured;
    b += eventMap[i] * eventMap[i];
  }
  return b > 0 ? a / b : 0;
}

function renderTrial(trial, days, host) {
  const cut = Math.round(CONFIDENCE * 255);
  const meta = days[0].firstPng;
  const code = EVENTS.code;

  for (let offset = 0; offset < days.length; offset += 8) {
    const block = days.slice(offset, offset + 8);
    const pad = 8 - block.length;
    const table = el('div', 'matrix');
    table.style.gridTemplateColumns = '132px repeat(8, minmax(0, 1fr))';
    host.appendChild(table);

    table.appendChild(el('div', 'blank'));
    block.forEach(day => table.appendChild(el('div', 'colhead',
      '<b>' + day.date.slice(5) + '</b><span>day ' + (day.index + 1) + '</span>')));
    for (let i = 0; i < pad; i++) table.appendChild(el('div', 'blank'));

    ROWS.forEach(([name, hint, key]) => {
      table.appendChild(el('div', 'rowlab',
        '<b>' + name + '</b><span>' + hint + '</span>'));
      block.forEach(day => {
        const slot = el('div', 'cell');
        table.appendChild(slot);
        fillCell(slot, key, trial, day, days, meta, cut);
      });
      for (let i = 0; i < pad; i++) table.appendChild(el('div', 'blank'));
    });
  }
}

function fillCell(slot, key, trial, day, days, meta, cut) {
  if (key === 'still') {
    if (day.firstJpg) {
      const stage = el('div', 'stage');
      const image = el('img');
      image.src = day.firstJpg;
      stage.appendChild(image);
      slot.appendChild(stage);
    } else {
      slot.appendChild(el('div', 'cellnote', 'no still'));
    }
    return;
  }

  if (key === 'depth') {
    Promise.all([loadDepth(days[0].firstPng), loadDepth(day.lastPng)])
      .then(([first, last]) => {
        if (!first || !last) return;
        const total = difference(first, last);
        const canvas = el('canvas');
        const stage = el('div', 'stage');
        stage.appendChild(canvas);
        slot.appendChild(stage);
        paintMap(canvas, total, meta, { range: 4 });
      });
    return;
  }

  if (key === 'fromEvents') {
    Promise.all([loadDepth(days[0].firstPng), loadDepth(day.lastPng)])
      .then(([first, last]) => {
        if (!first || !last) return;
        const total = difference(first, last);
        const events = eventDepthMap(trial, day.index, meta, cut);
        const scale = fitScale(total, events);
        SCALE = scale;
        const scaled = new Float32Array(events.length);
        for (let i = 0; i < events.length; i++) scaled[i] = events[i] * scale;
        const canvas = el('canvas');
        const stage = el('div', 'stage');
        stage.appendChild(canvas);
        slot.appendChild(stage);
        paintMap(canvas, scaled, meta, { range: 4 });
        slot.appendChild(el('div', 'cellfoot',
          scale ? scale.toFixed(3) + ' cm per net event' : 'no fit'));
      });
    return;
  }

  if (key === 'build') {
    const groups = [
      { colour: COLOURS.scoop,
        points: eventsOfDay(trial, new Set([EVENTS.code.c]), cut, day.index) },
      { colour: COLOURS.spit,
        points: eventsOfDay(trial, new Set([EVENTS.code.p]), cut, day.index) },
      { colour: COLOURS.multiple,
        points: eventsOfDay(trial, new Set([EVENTS.code.b]), cut, day.index) }];
    slot.appendChild(scatterPanel(groups, meta,
      groups.map(g => g.points.length).join(' / ')));
    return;
  }

  if (key === 'spawn') {
    const points = eventsOfDay(trial, new Set([EVENTS.code.s]), cut, day.index);
    slot.appendChild(scatterPanel([{ colour: COLOURS.spawn, points }], meta,
      String(points.length)));
  }
}

function eventsOfDay(trial, codes, cut, dayIndex) {
  const points = [];
  for (let i = 0; i < EVENTS.n; i++) {
    if (EVENTS.trial[i] !== trial || EVENTS.day[i] !== dayIndex) continue;
    if (!keep(i, cut) || !codes.has(EVENTS.bid[i])) continue;
    points.push([EVENTS.dx[i], EVENTS.dy[i]]);
  }
  return points;
}

// ------------------------------------------------- where spawning happens
// The question is whether spawning happens over the castle or the pit, so the
// depth under each spawn is read from the trial's own total change and
// histogrammed. A distribution centred on zero would mean spawning ignores the
// bower entirely, which is itself an answer.
function spawnHistogram(trial, days, meta, cut) {
  const figure = el('figure');
  figure.appendChild(el('figcaption', null, '<b>Depth under each spawn</b>'));
  Promise.all([loadDepth(days[0].firstPng),
               loadDepth(days[days.length - 1].lastPng)]).then(([first, last]) => {
    if (!first || !last) return;
    const total = difference(first, last);
    const sx = meta.width / D.depthSize[0], sy = meta.height / D.depthSize[1];
    const values = [];
    for (let i = 0; i < EVENTS.n; i++) {
      if (EVENTS.trial[i] !== trial || !keep(i, cut)) continue;
      if (EVENTS.bid[i] !== EVENTS.code.s) continue;
      const x = Math.round(EVENTS.dx[i] * sx), y = Math.round(EVENTS.dy[i] * sy);
      if (x < 0 || y < 0 || x >= meta.width || y >= meta.height) continue;
      const depth = total[y * meta.width + x];
      if (!Number.isNaN(depth)) values.push(depth);
    }
    if (values.length < 5) {
      figure.appendChild(el('figcaption', null, 'too few spawns to plot'));
      return;
    }
    const bins = 24, span = 4;
    const counts = new Float32Array(bins);
    values.forEach(value => {
      const bin = Math.min(bins - 1, Math.max(0,
        Math.floor((value + span) / (2 * span) * bins)));
      counts[bin] += 1;
    });
    const peak = Math.max(...counts) || 1;
    const width = 520, height = 200, pad = 36;
    let body = '';
    for (let i = 0; i < bins; i++) {
      const size = (counts[i] / peak) * (height - 2 * pad);
      const value = -span + (2 * span) * (i + 0.5) / bins;
      body += '<rect x="' + (pad + (width - 2 * pad) / bins * i + 1) + '" y="' +
              (height - pad - size) + '" width="' +
              ((width - 2 * pad) / bins - 2) + '" height="' + size +
              '" fill="' + (value >= 0 ? 'rgb(111,178,232)' : 'rgb(242,163,60)') +
              '" opacity="0.85"><title>' + value.toFixed(1) + ' cm: ' +
              counts[i] + ' spawns</title></rect>';
    }
    const zero = pad + (width - 2 * pad) * 0.5;
    body += '<line x1="' + zero + '" y1="' + (pad - 6) + '" x2="' + zero + '" y2="' +
            (height - pad) + '" stroke="#93a0b0" stroke-dasharray="3 3"/>';
    body += '<text x="' + pad + '" y="' + (height - 8) +
            '" fill="#93a0b0" font-size="10">\u2212' + span + ' cm, pit</text>' +
            '<text x="' + (width - pad) + '" y="' + (height - 8) +
            '" fill="#93a0b0" font-size="10" text-anchor="end">+' + span +
            ' cm, castle</text>';
    const sorted = values.slice().sort((a, b) => a - b);
    const median = sorted[sorted.length >> 1];
    const holder = el('div');
    holder.innerHTML = '<svg viewBox="0 0 ' + width + ' ' + height +
                       '" style="width:100%;display:block">' + body + '</svg>';
    figure.insertBefore(holder, figure.firstChild);
    figure.appendChild(el('figcaption', null,
      values.length + ' spawns \u00b7 median depth ' + median.toFixed(2) + ' cm \u00b7 ' +
      (100 * values.filter(v => v > 0).length / values.length).toFixed(0) +
      '% over sand that was added'));
  });
  return figure;
}

// ------------------------------------------------- how confined each is
function dispersionTable(trial, days) {
  const figure = el('figure');
  const rows = [];
  const windows = [[8, 18, 'all day'], [8, 13, '8 to 1'], [13, 18, '1 to 6']];
  const cuts = [0.3, 0.5, 0.67, 0.9];
  const savedFrom = HOUR_FROM, savedTo = HOUR_TO;

  cuts.forEach(confidence => {
    windows.forEach(([from, to, label]) => {
      HOUR_FROM = from; HOUR_TO = to;
      const cut = Math.round(confidence * 255);
      const scoops = eventsOf(trial, new Set([EVENTS.code.c]), cut);
      const spits = eventsOf(trial, new Set([EVENTS.code.p]), cut);
      const a = dispersion(scoops, COVERAGE), b = dispersion(spits, COVERAGE);
      if (!a || !b) return;
      const cm = D.pixelLength || 0.1030168618;
      const separation = Math.hypot(a.cx - b.cx, a.cy - b.cy) * cm;
      rows.push('<tr><td>' + confidence.toFixed(2) + '</td><td>' + label +
        '</td><td>' + a.n + '</td><td>' + (a.area * cm * cm).toFixed(0) +
        '</td><td>' + b.n + '</td><td>' + (b.area * cm * cm).toFixed(0) +
        '</td><td>' + (b.area / a.area).toFixed(2) + '</td><td>' +
        separation.toFixed(2) + '</td></tr>');
    });
  });
  HOUR_FROM = savedFrom; HOUR_TO = savedTo;

  const table = el('table', 'points');
  table.innerHTML = '<tr><th>confidence</th><th>hours</th><th>scoops</th>' +
    '<th>scoop area cm\u00b2</th><th>spits</th><th>spit area cm\u00b2</th>' +
    '<th>spit / scoop</th><th>centroids apart cm</th></tr>' + rows.join('');
  figure.appendChild(el('figcaption', null,
    '<b>How confined scoops and spits each are.</b> The area is the ellipse ' +
    'containing ' + (100 * COVERAGE).toFixed(0) + '% of the events, from their ' +
    'covariance. A ratio near one means the two are equally spread; a large ' +
    'separation means they happen in different places, which is what building a ' +
    'bower looks like.'));
  figure.appendChild(table);
  return figure;
}

function draw() {
  const body = document.getElementById('body');
  body.textContent = '';
  if (!EVENTS.projected) {
    body.appendChild(el('div', 'note', 'This project has no saved registration, so ' +
      'cluster events cannot be put into depth coordinates. Set one on the prep page.'));
    return;
  }
  const byTrial = daysByTrial(false);

  Object.keys(byTrial).sort((a, b) => a - b).forEach(trial => {
    const days = byTrial[trial];
    const section = el('div', 'trial');
    section.appendChild(el('h3', null, 'Trial ' + trial +
      '<span>' + days.length + ' days</span>'));
    body.appendChild(section);
    renderTrial(+trial, days, section);

    const cut = Math.round(CONFIDENCE * 255);
    const lower = el('div', 'grid cols-2');
    lower.appendChild(spawnHistogram(+trial, days, days[0].firstPng, cut));
    lower.appendChild(dispersionTable(+trial, days));
    section.appendChild(lower);
  });
}

function build() {
  document.getElementById('title').textContent = D.projectID;
  document.getElementById('subtitle').textContent =
    'Depth and behaviour in one frame.';
  document.getElementById('meta').innerHTML =
    [['Tank', D.tankID], ['Events', EVENTS.n], ['Days', D.days.length],
     ['Trials', D.trials.length],
     ['Registered', PREP.transform ? 'yes' : 'no']]
    .map(([k, v]) => '<div>' + k + '<b>' + v + '</b></div>').join('');

  const bar = document.getElementById('controls');
  bar.innerHTML =
    '<label>Confidence \u2265 <b id="cv">' + CONFIDENCE.toFixed(2) + '</b></label>' +
    '<input type="range" id="cr" min="0" max="0.99" step="0.01" value="' +
      CONFIDENCE + '">' +
    '<label>Hours <b id="hv">' + HOUR_FROM + '\u2013' + HOUR_TO + '</b></label>' +
    '<input type="range" id="h1" min="0" max="24" step="1" value="' + HOUR_FROM + '">' +
    '<input type="range" id="h2" min="0" max="24" step="1" value="' + HOUR_TO + '">' +
    '<label>Ellipse covers <b id="ev">' + (100 * COVERAGE).toFixed(0) +
    '</b>%</label>' +
    '<input type="range" id="er" min="50" max="99" step="5" value="' +
      (100 * COVERAGE) + '">';

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
  const hours = () => {
    bar.querySelector('#hv').textContent = HOUR_FROM + '\u2013' + HOUR_TO;
    later();
  };
  bar.querySelector('#h1').addEventListener('input', event => {
    HOUR_FROM = Math.min(parseInt(event.target.value, 10), HOUR_TO - 1);
    hours();
  });
  bar.querySelector('#h2').addEventListener('input', event => {
    HOUR_TO = Math.max(parseInt(event.target.value, 10), HOUR_FROM + 1);
    hours();
  });
  bar.querySelector('#er').addEventListener('input', event => {
    COVERAGE = parseInt(event.target.value, 10) / 100;
    bar.querySelector('#ev').textContent = (100 * COVERAGE).toFixed(0);
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
      'Collected ' + D.collected + ' \u00b7 page built ' + D.built;
  });
});
