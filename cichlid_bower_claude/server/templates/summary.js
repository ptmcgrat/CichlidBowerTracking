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
let COVERAGE = 0.68;     // the share of events an ellipse should contain
let SPREAD = 30;         // depth pixels each scoop or spit is spread over
// DIRTY and markDirty live in common.js: both scripts load into one scope, so
// declaring it again here was a SyntaxError that stopped the whole page

function markOf(dayIndex) {
  return (PREP.day_marks || {})[String(dayIndex)] || {};
}

function setMark(dayIndex, which, on) {
  PREP.day_marks = PREP.day_marks || {};
  const key = String(dayIndex);
  const entry = PREP.day_marks[key] || {};
  entry[which] = on;
  if (!entry.new_bower && !entry.wall_building && !entry.note)
    delete PREP.day_marks[key];
  else PREP.day_marks[key] = entry;
  markDirty();
}

// Two things a person can see and the analysis cannot: a bower restarted
// somewhere else, which makes the cumulative map two overlapping structures,
// and building against a wall, which is real sand movement the cluster
// detector misses because the fish is half out of frame.
function dayMarkControls(day) {
  const wrap = el('div');
  [['new_bower', 'new bower'], ['wall_building', 'wall building']]
    .forEach(([which, label]) => {
      const on = !!markOf(day.index)[which];
      const box = el('label', 'daymark' + (on ? ' set' : ''));
      const input = el('input');
      input.type = 'checkbox';
      input.checked = on;
      input.addEventListener('change', event => {
        setMark(day.index, which, event.target.checked);
        box.className = 'daymark' + (event.target.checked ? ' set' : '');
      });
      box.appendChild(input);
      box.appendChild(el('span', null, label));
      wrap.appendChild(box);
      wrap.appendChild(el('br'));
    });
  return wrap;
}
let SCALE = null;        // cm of depth per net event, fitted per trial

const ROWS = [
  ['Depth camera', 'the tray that day', 'still'],
  ['Depth change', 'cumulative, from the sensor', 'depth'],
  ['From events', 'cumulative, spits minus scoops', 'fromEvents'],
  ['24 hour change', 'this morning to the next', 'daily'],
  ['Building', 'that day \u00b7 scoop orange, spit blue, multiple green', 'build'],
  ['Spawning', 'that day', 'spawn'],
  ['Spread', 'cumulative \u00b7 ellipse holding the chosen share', 'spread'],
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
  // Which events have real coordinates. An event whose trial has no
  // registration used to keep dx = dy = 0 and be drawn at the top-left corner
  // while still being counted, so a trial with no transform looked like a
  // trial with no events — except that the counts said otherwise.
  EVENTS.placed = new Uint8Array(EVENTS.n);
  EVENTS.unplacedTrials = {};
  if (!PREP.transform && !Object.keys(PREP.overrides || {}).length) {
    EVENTS.projected = false;
    return;
  }
  for (let i = 0; i < EVENTS.n; i++) {
    const trial = EVENTS.trial[i];
    const H = transformFor(trial);
    if (!H) {
      EVENTS.unplacedTrials[trial] = (EVENTS.unplacedTrials[trial] || 0) + 1;
      continue;
    }
    const point = applyH(H, [EVENTS.x[i], EVENTS.y[i]]);
    EVENTS.dx[i] = point[0];
    EVENTS.dy[i] = point[1];
    EVENTS.placed[i] = 1;
  }
  EVENTS.projected = true;
}

function keep(index, cut) {
  return EVENTS.placed[index] &&
         (EVENTS.flags[index] & 1) && (EVENTS.flags[index] & 2) &&
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

// Outside the tray crop is not data: the sensor sees tank walls and floor
// there, and leaving it in makes every map's colour scale answer to something
// that is not sand.
function cropped(values, meta, trial) {
  const outside = cropMaskFor(meta, trial);
  if (!outside) return values;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++)
    out[i] = outside[i] ? NaN : values[i];
  return out;
}

// ------------------------------------------------------------------- panels
// The ellipse the table measures, drawn where it belongs. A number in a table
// says how spread out the events are; the outline says where, and whether the
// two behaviours sit in different places.
function drawEllipse(ctx, shape, colour, sx, sy) {
  if (!shape) return;
  ctx.save();
  ctx.translate(shape.cx * sx, shape.cy * sy);
  ctx.rotate(shape.angle);
  ctx.strokeStyle = 'rgb(' + colour.join(',') + ')';
  ctx.lineWidth = 1.2;
  ctx.globalAlpha = 0.95;
  ctx.beginPath();
  ctx.ellipse(0, 0, shape.major * sx, shape.minor * sy, 0, 0, 6.2832);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(0, 0, 2, 0, 6.2832);
  ctx.fillStyle = 'rgb(' + colour.join(',') + ')';
  ctx.fill();
  ctx.restore();
  ctx.globalAlpha = 1;
}

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
  groups.forEach(group => {
    if (group.ellipse === false) return;
    drawEllipse(ctx, dispersion(group.points, COVERAGE), group.colour, sx, sy);
  });
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
  // the spread is given in depth pixels, and the maps are downsampled
  const radius = Math.max(1, Math.round(SPREAD / 2 / (D.downsample || 1)));
  return blur(values, meta.width, meta.height, radius);
}

// A box blur done as two one-dimensional passes with a running sum, so the
// cost does not depend on the radius. The naive version was O(radius squared)
// per pixel, which at a thirty-pixel spread would be seventy million
// operations for one map.
function blur(values, width, height, radius) {
  if (radius < 1) return values;
  const pass = new Float32Array(values.length);
  const out = new Float32Array(values.length);
  const span = 2 * radius + 1;

  for (let y = 0; y < height; y++) {
    const row = y * width;
    let sum = 0;
    for (let x = -radius; x <= radius; x++)
      sum += values[row + Math.min(width - 1, Math.max(0, x))];
    for (let x = 0; x < width; x++) {
      pass[row + x] = sum / span;
      const leaving = row + Math.min(width - 1, Math.max(0, x - radius));
      const entering = row + Math.min(width - 1, Math.max(0, x + radius + 1));
      sum += values[entering] - values[leaving];
    }
  }

  for (let x = 0; x < width; x++) {
    let sum = 0;
    for (let y = -radius; y <= radius; y++)
      sum += pass[Math.min(height - 1, Math.max(0, y)) * width + x];
    for (let y = 0; y < height; y++) {
      out[y * width + x] = sum / span;
      const leaving = Math.min(height - 1, Math.max(0, y - radius)) * width + x;
      const entering = Math.min(height - 1, Math.max(0, y + radius + 1)) * width + x;
      sum += pass[entering] - pass[leaving];
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
    block.forEach(day => {
      const head = el('div', 'colhead',
        '<b>' + day.date.slice(5) + '</b><span>day ' + (day.index + 1) + '</span>');
      head.appendChild(dayMarkControls(day));
      table.appendChild(head);
    });
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
    // the depth camera's own JPEG, which comes out of the frame archive. A few
    // frames are written without one, so fall back to that day's video still
    // rather than leaving a hole, and say which is being shown.
    const source = day.firstJpg || day.videoJpg;
    if (source) {
      const stage = el('div', 'stage');
      const image = el('img');
      image.src = source;
      stage.appendChild(image);
      slot.appendChild(stage);
      if (!day.firstJpg)
        slot.appendChild(el('div', 'cellfoot', 'video still \u2014 no depth JPEG'));
    } else {
      slot.appendChild(el('div', 'cellnote', 'no still in the archive'));
    }
    return;
  }

  if (key === 'depth') {
    Promise.all([loadDepth(days[0].firstPng), loadDepth(day.lastPng)])
      .then(([first, last]) => {
        if (!first || !last) return;
        const canvas = el('canvas');
        const stage = el('div', 'stage');
        stage.appendChild(canvas);
        slot.appendChild(stage);
        paintMap(canvas, cropped(difference(first, last), meta, trial), meta,
                 { range: 4 });
      });
    return;
  }

  if (key === 'daily') {
    // this day's own morning against the next, which is the change the depth
    // page calls 24 hour: it isolates one day from the accumulated total
    const position = days.findIndex(d => d.index === day.index);
    const next = days[position + 1];
    if (!next) {
      slot.appendChild(el('div', 'cellnote', 'no following day in this trial'));
      return;
    }
    Promise.all([loadDepth(day.firstPng), loadDepth(next.firstPng)])
      .then(([morning, nextMorning]) => {
        if (!morning || !nextMorning) return;
        const canvas = el('canvas');
        const stage = el('div', 'stage');
        stage.appendChild(canvas);
        slot.appendChild(stage);
        paintMap(canvas, cropped(difference(morning, nextMorning), meta, trial),
                 meta, { range: 2 });
      });
    return;
  }

  if (key === 'fromEvents') {
    Promise.all([loadDepth(days[0].firstPng), loadDepth(day.lastPng)])
      .then(([first, last]) => {
        if (!first || !last) return;
        const total = cropped(difference(first, last), meta, trial);
        const events = eventDepthMap(trial, day.index, meta, cut);
        const scale = fitScale(total, events);
        SCALE = scale;
        const scaled = new Float32Array(events.length);
        for (let i = 0; i < events.length; i++) scaled[i] = events[i] * scale;
        const shown = cropped(scaled, meta, trial);
        const canvas = el('canvas');
        const stage = el('div', 'stage');
        stage.appendChild(canvas);
        slot.appendChild(stage);
        paintMap(canvas, shown, meta, { range: 4 });
        slot.appendChild(el('div', 'cellfoot',
          scale ? scale.toFixed(3) + ' cm per net event, ' + SPREAD + ' px spread'
                : 'no fit'));
      });
    return;
  }

  if (key === 'build') {
    const groups = [
      { colour: COLOURS.scoop,
        points: eventsOfDay(trial, new Set([EVENTS.code.c]), cut, day.index) },
      { colour: COLOURS.spit,
        points: eventsOfDay(trial, new Set([EVENTS.code.p]), cut, day.index) },
      // multiples are scoop and spit in one pass, so an ellipse over them
      // describes neither behaviour
      { colour: COLOURS.multiple, ellipse: false,
        points: eventsOfDay(trial, new Set([EVENTS.code.b]), cut, day.index) }];
    slot.appendChild(scatterPanel(groups, meta,
      groups.map(g => g.points.length).join(' / ')));
    return;
  }

  if (key === 'spawn') {
    const points = eventsOfDay(trial, new Set([EVENTS.code.s]), cut, day.index);
    slot.appendChild(scatterPanel([{ colour: COLOURS.spawn, points }], meta,
      String(points.length)));
    return;
  }

  if (key === 'spread') {
    // everything up to this day, so the ellipses settle as evidence builds
    const scoops = eventsOf(trial, new Set([EVENTS.code.c]), cut, day.index);
    const spits = eventsOf(trial, new Set([EVENTS.code.p]), cut, day.index);
    const cm = D.pixelLength || 0.1030168618;
    const a = dispersion(scoops, COVERAGE), b = dispersion(spits, COVERAGE);
    const foot = (a && b)
      ? (a.area * cm * cm).toFixed(0) + ' / ' + (b.area * cm * cm).toFixed(0) +
        ' cm\u00b2'
      : 'too few';
    slot.appendChild(scatterPanel(
      [{ colour: COLOURS.scoop, points: scoops, ellipse: false },
       { colour: COLOURS.spit, points: spits, ellipse: false }], meta, foot));
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

  // Spawns are gathered first so the ellipse can be fitted over all of them,
  // then each is read against the cumulative change up to its OWN day. Using
  // the whole trial's total would ask what the sand looked like at the end,
  // not what the fish was spawning over at the time.
  const spawns = [];
  for (let i = 0; i < EVENTS.n; i++) {
    if (EVENTS.trial[i] !== trial || !keep(i, cut)) continue;
    if (EVENTS.bid[i] !== EVENTS.code.s) continue;
    spawns.push({ x: EVENTS.dx[i], y: EVENTS.dy[i], day: EVENTS.day[i] });
  }
  if (spawns.length < 5) {
    figure.appendChild(el('figcaption', null, 'too few spawns to plot'));
    return figure;
  }

  // Only the spawns inside the ellipse. A spawn across the tank is a different
  // event from one on the bower, and including it moves the distribution
  // toward zero for a reason that has nothing to do with bower shape.
  const shape = dispersion(spawns.map(s => [s.x, s.y]), COVERAGE);
  const inside = shape ? spawns.filter(spawn => {
    const dx = spawn.x - shape.cx, dy = spawn.y - shape.cy;
    const cos = Math.cos(-shape.angle), sin = Math.sin(-shape.angle);
    const u = (dx * cos - dy * sin) / shape.major;
    const v = (dx * sin + dy * cos) / shape.minor;
    return u * u + v * v <= 1;
  }) : spawns;

  const byDay = {};
  inside.forEach(spawn => { (byDay[spawn.day] = byDay[spawn.day] || []).push(spawn); });
  const sx = meta.width / D.depthSize[0], sy = meta.height / D.depthSize[1];

  const wanted = Object.keys(byDay)
    .map(index => days.find(day => day.index === +index))
    .filter(Boolean);

  Promise.all([loadDepth(days[0].firstPng)].concat(
      wanted.map(day => loadDepth(day.lastPng)))).then(results => {
    const start = results[0];
    if (!start) return;
    const values = [];
    wanted.forEach((day, position) => {
      const end = results[position + 1];
      if (!end) return;
      const cumulative = cropped(difference(start, end), meta, trial);
      byDay[day.index].forEach(spawn => {
        const x = Math.round(spawn.x * sx), y = Math.round(spawn.y * sy);
        if (x < 0 || y < 0 || x >= meta.width || y >= meta.height) return;
        const depth = cumulative[y * meta.width + x];
        if (!Number.isNaN(depth)) values.push(depth);
      });
    });
    if (values.length < 5) {
      figure.appendChild(el('figcaption', null, 'too few spawns on measurable sand'));
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
      values.length + ' of ' + spawns.length + ' spawns, those inside the ' +
      Math.round(100 * COVERAGE) + '% ellipse \u00b7 median ' +
      median.toFixed(2) + ' cm \u00b7 ' +
      (100 * values.filter(v => v > 0).length / values.length).toFixed(0) +
      '% over sand that had been added by that day. Each spawn is read against ' +
      'the change accumulated up to its own day, not the trial total.'));
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
  // a trial whose events could not be placed says so, instead of drawing an
  // empty panel beside a count of events that are really there
  const unplaced = Object.keys(EVENTS.unplacedTrials || {});
  if (unplaced.length) {
    body.appendChild(el('div', 'note',
      'Trial ' + unplaced.join(', ') + ' ' + (unplaced.length > 1 ? 'have' : 'has') +
      ' no registration, so ' +
      unplaced.reduce((sum, t) => sum + EVENTS.unplacedTrials[t], 0) +
      ' events cannot be placed in depth coordinates and are left out below. ' +
      'This happens when a trial was given its own settings before a ' +
      'registration existed to copy. Fit it on the prep page, or remove its ' +
      'override so it falls back to the project registration.'));
  }

  const excluded = excludedBanner();
  if (excluded) body.appendChild(excluded);
  const byTrial = daysByTrial(false);

  Object.keys(byTrial).sort((a, b) => a - b).forEach(trial => {
    const days = byTrial[trial];
    const section = el('div', 'trial');
    const missing = (EVENTS.unplacedTrials || {})[trial];
    section.appendChild(el('h3', null, 'Trial ' + trial +
      '<span>' + days.length + ' days' +
      (missing ? ' \u00b7 no registration, ' + missing + ' events unplaced' : '') +
      '</span>'));
    body.appendChild(section);
    renderTrial(+trial, days, section);

    const cut = Math.round(CONFIDENCE * 255);
    const lower = el('div', 'grid cols-2');
    lower.appendChild(spawnHistogram(+trial, days, days[0].firstPng, cut));
    lower.appendChild(dispersionTable(+trial, days));
    section.appendChild(lower);
  });
}

// Only the marks are sent, merged onto what is already saved: this page has
// no registration or crop controls, so posting the whole object would let a
// stale copy here overwrite a change made on the prep page.
function saveMarks(whoField) {
  const who = (whoField && whoField.value || '').trim();
  if (!who) {
    window.alert('Put your name in before saving, so the change can be traced back.');
    if (whoField) whoField.focus();
    return;
  }
  try { window.localStorage.setItem('cbc-who', who); } catch (e) { /* private mode */ }
  const button = document.getElementById('save');
  const state = document.getElementById('state');
  button.disabled = true;
  state.textContent = 'saving\u2026';
  fetch('marks', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                   body: JSON.stringify({ who: who, day_marks: PREP.day_marks || {} }) })
    .then(response => response.json().then(body => ({ ok: response.ok, body })))
    .then(({ ok, body }) => {
      if (!ok) throw new Error(body.error || 'the server refused the change');
      DIRTY = false;
      state.textContent = 'saved ' + (body.updated || '').slice(11, 16);
    })
    .catch(error => {
      button.disabled = false;
      state.textContent = 'not saved: ' + error.message;
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
    '<label>Ellipse covers <b id="ev">' + Math.round(100 * COVERAGE) +
    '</b>%</label>' +
    '<input type="range" id="er" min="50" max="99" step="1" value="' +
      Math.round(100 * COVERAGE) + '">' +
    '<label>Event spread <b id="sv">' + SPREAD + '</b> px</label>' +
    '<input type="range" id="sr" min="5" max="80" step="5" value="' + SPREAD + '">';

  const whoField = bar.querySelector('#who');
  try { whoField.value = PREP.who || window.localStorage.getItem('cbc-who') || ''; }
  catch (e) { whoField.value = PREP.who || ''; }
  bar.querySelector('#save').addEventListener('click', () => saveMarks(whoField));

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
  bar.querySelector('#sr').addEventListener('input', event => {
    SPREAD = parseInt(event.target.value, 10);
    bar.querySelector('#sv').textContent = SPREAD;
    later();
  });
  bar.querySelector('#sr').addEventListener('input', event => {
    SPREAD = parseInt(event.target.value, 10);
    bar.querySelector('#sv').textContent = SPREAD;
    later();
  });
  bar.querySelector('#er').addEventListener('input', event => {
    COVERAGE = parseInt(event.target.value, 10) / 100;
    bar.querySelector('#ev').textContent = (100 * COVERAGE).toFixed(0);
    later();
  });
  draw();
}

window.addEventListener('beforeunload', event => {
  if (DIRTY) { event.preventDefault(); event.returnValue = ''; }
});

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
