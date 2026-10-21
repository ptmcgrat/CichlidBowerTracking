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

// Events grouped by trial, day and behaviour, built once per redraw.
//
// Every cell used to scan the whole event array to find its own handful of
// points, which at thirty-four days and a hundred and sixty thousand events
// came to forty-six million iterations for one page. The events do not change
// between cells; only the question does.
let BUCKETS = null;

function bucketEvents(cut) {
  BUCKETS = {};
  for (let i = 0; i < EVENTS.n; i++) {
    if (!keep(i, cut)) continue;
    const trial = EVENTS.trial[i], day = EVENTS.day[i], code = EVENTS.bid[i];
    const byDay = BUCKETS[trial] || (BUCKETS[trial] = {});
    const byCode = byDay[day] || (byDay[day] = {});
    (byCode[code] || (byCode[code] = [])).push([EVENTS.dx[i], EVENTS.dy[i]]);
  }
}

function bucket(trial, day, code) {
  const byDay = BUCKETS && BUCKETS[trial];
  const byCode = byDay && byDay[day];
  return (byCode && byCode[code]) || [];
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
  const byDay = (BUCKETS && BUCKETS[trial]) || {};
  Object.keys(byDay).forEach(day => {
    if (upToDay !== undefined && +day > upToDay) return;
    codes.forEach(code => {
      const found = byDay[day][code] || [];
      for (let i = 0; i < found.length; i++) points.push(found[i]);
    });
  });
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
  const byDay = (BUCKETS && BUCKETS[trial]) || {};
  [[EVENTS.code.p, 1], [EVENTS.code.c, -1]].forEach(([code, sign]) => {
    Object.keys(byDay).forEach(day => {
      if (+day > upToDay) return;
      const found = byDay[day][code] || [];
      for (let i = 0; i < found.length; i++) {
        const x = Math.round(found[i][0] * sx), y = Math.round(found[i][1] * sy);
        if (x < 0 || y < 0 || x >= meta.width || y >= meta.height) continue;
        values[y * meta.width + x] += sign;
      }
    });
  });
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
  codes.forEach(code => {
    const found = bucket(trial, dayIndex, code);
    for (let i = 0; i < found.length; i++) points.push(found[i]);
  });
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
  bucketEvents(Math.round(CONFIDENCE * 255));
  buildCodeSets();
  // the confidence slider changes which events exist, so last redraw's
  // volumes and ellipses are not this redraw's
  VOLUME_CACHE = {};

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
    renderTrialStrip(+trial, days, section, cut);

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
    '<input type="range" id="sr" min="5" max="80" step="5" value="' + SPREAD + '">' +
    '<span class="spacer"></span>' +
    '<label for="who" class="stat">Your name</label>' +
    '<input id="who" placeholder="e.g. Emily Keaton" autocomplete="name">' +
    '<span class="stat" id="state"></span>' +
    '<button class="act primary" id="save" disabled>Save marks</button>';

  const whoField = bar.querySelector('#who');
  const saveButton = bar.querySelector('#save');
  if (whoField) {
    let remembered = '';
    try { remembered = window.localStorage.getItem('cbc-who') || ''; }
    catch (e) { remembered = ''; }          // private mode: no store to read
    whoField.value = PREP.who || remembered;
  }
  if (saveButton)
    saveButton.addEventListener('click', () => saveMarks(whoField));

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

// =====================================================================
// The per-trial strip: six panels below the day matrix.
//
// These answer questions about the trial as a whole rather than about one
// day, so they do not belong in the matrix above, whose columns are days.
// =====================================================================

let VOL_THRESHOLD = 0.6;     // cm before sand counts as moved
const SPLIT_AT = 0.5;        // the share of total sand moved that ends half one
const MIN_FOR_SHAPE = 25;    // events before an ellipse's shape is reported
const SPAWN_REF = 10;        // the legend dot's spawn count

// Bucket keys are the numeric behaviour codes, not the letters. Built lazily
// because EVENTS is not loaded when this file is evaluated. 'b', bower
// multiple, is deliberately in neither: it is both a scoop and a spit, so it
// would have to be counted twice or not at all, and counting it twice makes
// the two clouds artificially alike.
let SCOOP = null, SPIT = null;

function buildCodeSets() {
  SCOOP = new Set([EVENTS.code.c]);
  SPIT = new Set([EVENTS.code.p]);
}

// Sand moved between the endpoints the bundle already holds. The day's own
// window is first to last; the night is the previous day's last to this
// day's first — so overnight building is in the collected data already, and
// needed no new collection to get at.
let VOLUME_CACHE = {};

// Three panels want these numbers and the arithmetic is a pass over every
// pixel of every endpoint, so it happens once per trial and threshold.
function volumesFor(trial, days, meta) {
  const key = trial + ':' + VOL_THRESHOLD;
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
          ? volumes(cropped(difference(first, last), meta, trial), VOL_THRESHOLD)
          : null,
        dark: (prevLast && first)
          ? volumes(cropped(difference(prevLast, first), meta, trial), VOL_THRESHOLD)
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

function panel(title, hint) {
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
  const first = panel('First half of the volume', 'cumulative change');
  const second = panel('Second half of the volume', 'cumulative change');
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

        // one colour scale across both, so the halves can be compared by eye;
        // independent scales would make a quiet half look like a busy one
        let span = 0;
        [early, late].forEach(map => {
          for (let i = 0; i < map.length; i++) {
            const v = Math.abs(map[i]);
            if (!Number.isNaN(v) && v > span) span = v;
          }
        });
        span = Math.max(0.5, span);

        [[first, early, 0, split.at, split.fraction],
         [second, late, split.at, days.length - 1, 1 - split.fraction]]
          .forEach(([box, map, fromDay, toDay, share]) => {
            box._stage.textContent = '';
            const canvas = el('canvas');
            box._stage.appendChild(canvas);
            paintMap(canvas, map, meta, { range: span });
            const v = volumes(map, VOL_THRESHOLD);
            const spanDays = toDay - fromDay + (box === first ? 1 : 0);
            box._caption.innerHTML =
              '<b>' + (box === first ? 'First' : 'Second') + ' half of the volume</b><br>' +
              'net volume <b>' + nice(v.total) + '</b> cm³ · ' +
              'index <b>' + nice(v.index, 3) + '</b><br>' +
              'days ' + (days[fromDay].index + 1) + '–' +
              (days[toDay].index + 1) + ' · ' + spanDays + ' days · ' +
              nice(share * 100) + '% of gross sand moved<br>' +
              '<span style="color:#6c7684">net is the change across the window, ' +
              'gross is what moved day by day — they differ wherever sand was ' +
              'put back, so a half can hold much of the work and little of the ' +
              'net change</span><br>' +
              '<span style="color:#6c7684">scale ±' + nice(span, 1) +
              ' cm, shared</span>';
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
function areaPanel(trial, days, cut, split) {
  const box = panel('Cloud size by day', 'ellipse area · scoop orange, spit blue');
  const perDay = days.map(d => ({
    index: d.index,
    scoop: dispersion(eventsOfDay(trial, SCOOP, cut, d.index), COVERAGE),
    spit: dispersion(eventsOfDay(trial, SPIT, cut, d.index), COVERAGE),
  }));
  const area = eventPixelArea();
  const values = [];
  perDay.forEach(r => ['scoop', 'spit'].forEach(k => {
    if (r[k]) values.push(r[k].area * area);
  }));

  const pooled = {};
  const ranges = { H1: [0, split ? split.at : days.length - 1],
                   H2: [split ? split.at : 0, days.length - 1],
                   All: [0, days.length - 1] };
  Object.keys(ranges).forEach(label => {
    const [lo, hi] = ranges[label];
    pooled[label] = {};
    ['scoop', 'spit'].forEach(k => {
      const codes = k === 'scoop' ? SCOOP : SPIT;
      const points = [];
      for (let i = lo; i <= hi; i++)
        eventsOfDay(trial, codes, cut, days[i].index).forEach(p => points.push(p));
      const shape = dispersion(points, COVERAGE);
      pooled[label][k] = shape;
      if (shape) values.push(shape.area * area);
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
    const colour = COLOURS[k];
    const path = [];
    perDay.forEach((r, i) => {
      if (!r[k]) return;
      const y = r[k].area * area;
      f.dot(i, y, colour);
      path.push([i, y]);
    });
    f.line(path, colour);
    ['H1', 'H2', 'All'].forEach((label, j) => {
      const shape = pooled[label][k];
      if (shape) f.dot(days.length + 1 + j, shape.area * area, colour, 3.4);
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

// Events are projected into full-resolution depth pixels, so their areas take
// the plain pixel size -- NOT pixelArea(), whose downsample factor is for the
// half-size maps. Mixing the two overstates every cloud by four.
function eventPixelArea() {
  const cm = D.pixelLength || 0.1030168618;
  return cm * cm;
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
function aspectPanel(trial, days, cut, split) {
  const box = panel('Cloud shape by day', 'aspect ratio · grey = circular at that n');
  const shapeOf = points => {
    const d = dispersion(points, COVERAGE);
    if (!d || d.n < MIN_FOR_SHAPE || d.minor <= 0) return null;
    return { ratio: d.major / d.minor, n: d.n,
             angle: d.angle * 180 / Math.PI };
  };
  const perDay = days.map(d => ({
    scoop: shapeOf(eventsOfDay(trial, SCOOP, cut, d.index)),
    spit: shapeOf(eventsOfDay(trial, SPIT, cut, d.index)),
  }));

  const pooled = {};
  const ranges = { H1: [0, split ? split.at : days.length - 1],
                   H2: [split ? split.at : 0, days.length - 1],
                   All: [0, days.length - 1] };
  Object.keys(ranges).forEach(label => {
    const [lo, hi] = ranges[label];
    pooled[label] = {};
    ['scoop', 'spit'].forEach(k => {
      const codes = k === 'scoop' ? SCOOP : SPIT;
      const points = [];
      for (let i = lo; i <= hi; i++)
        eventsOfDay(trial, codes, cut, days[i].index).forEach(p => points.push(p));
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
    const colour = COLOURS[k];
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
function spawnDepthPanel(trial, days, meta, cut) {
  const box = panel('Spawn depth by day', 'dot area = spawns that day');
  box._stage.appendChild(el('div', 'cellnote', 'measuring…'));
  const spawnCode = new Set([EVENTS.code.s]);

  Promise.all(days.map(d => Promise.all([loadDepth(days[0].firstPng),
                                         loadDepth(d.lastPng)])))
    .then(pairs => {
      const rows = [];
      days.forEach((d, i) => {
        const points = eventsOfDay(trial, spawnCode, cut, d.index);
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
        f.dot(r.at, r.mean, COLOURS.spawn,
              Math.max(2, 3.2 * Math.sqrt(r.n / SPAWN_REF)));
      });
      // the legend dot, without which the areas mean nothing on their own
      f.ctx.fillStyle = 'rgba(232,132,168,.85)';
      f.ctx.beginPath();
      f.ctx.arc(f.width - 42, 22, 3.2, 0, Math.PI * 2);
      f.ctx.fill();
      f.ctx.fillStyle = '#93a0b0';
      f.ctx.font = '10px sans-serif';
      f.ctx.textAlign = 'left';
      f.ctx.fillText('= ' + SPAWN_REF + ' spawns', f.width - 34, 25);
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
        '<span style="color:#6c7684">dot area is the count; days with no ' +
        'spawns are absent, not zero</span>';
    });
  return box;
}

// -------------------------------------------- 6: how much is built at night
// The depth sensor is infrared and sees in the dark; the video does not, so
// this is the one build measure that can only come from depth and will never
// have events beside it.
function nightPanel(trial, days, meta) {
  const box = panel('Built overnight', 'share of the day’s sand');
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
      const both = v.dark.total + v.light.total;
      if (both > 0) rows.push({ at: i, share: v.dark.total / both });
    });

    box._stage.textContent = '';
    if (!rows.length) {
      box._stage.appendChild(el('div', 'cellnote', 'no overnight pair to compare'));
      return;
    }
    const f = frame(420, 180, -0.5, days.length - 0.5, 0, 1);
    f.ticks([0.25, 0.5, 0.75, 1], v => v.toFixed(2));
    rows.forEach(r => {
      f.ctx.fillStyle = 'rgba(111,178,232,.5)';
      const x = f.X(r.at), base = f.Y(0);
      f.ctx.fillRect(x - 3, f.Y(r.share), 6, base - f.Y(r.share));
    });
    f.xLabel(0, '1');
    f.xLabel(days.length - 1, String(days.length));
    box._stage.appendChild(f.canvas);

    const share = darkTotal / (darkTotal + lightTotal || 1);
    // a 14 hour night against a 10 hour day makes a half share mean the night
    // rate is lower, so the rate ratio is given beside the raw share
    const rate = (darkHours > 0 && lightHours > 0)
      ? (darkTotal / darkHours) / (lightTotal / lightHours) : null;
    box._caption.innerHTML =
      '<b>Built overnight</b><br>overnight sand as a share of the day’s total<br>' +
      'trial <b>' + nice(share * 100, 1) + '%</b> of ' +
      nice(darkTotal + lightTotal) + ' cm³<br>' +
      'rate <b>' + (rate === null ? '—' : nice(rate, 2) + '×') +
      '</b> the daytime rate · ' + nice(darkHours / rows.length, 1) +
      ' h dark vs ' + nice(lightHours / perDay.length, 1) + ' h light<br>' +
      '<span style="color:#6c7684">depth only — the cameras cannot see ' +
      'in the dark, so no events accompany this</span>';
  });
  return box;
}

function renderTrialStrip(trial, days, host, cut) {
  const strip = el('div', 'grid cols-3 strip');
  const meta = days[0].firstPng;
  host.appendChild(strip);

  // Slots first, in reading order. Three of the six panels cannot be built
  // until the volume split is known, which is asynchronous, and appending
  // them when their promise happens to settle put the sheet in a different
  // order on every reload.
  const slots = [];
  for (let i = 0; i < 6; i++) {
    const slot = el('div', 'slot');
    strip.appendChild(slot);
    slots.push(slot);
  }

  halfPanels(trial, days, meta).forEach((box, i) => slots[i].appendChild(box));
  slots[4].appendChild(spawnDepthPanel(trial, days, meta, cut));
  slots[5].appendChild(nightPanel(trial, days, meta));

  const waiting = [2, 3].map(i => {
    const box = panel('\u2026', 'waiting for the volume split');
    box._stage.appendChild(el('div', 'cellnote', 'measuring\u2026'));
    slots[i].appendChild(box);
    return box;
  });
  volumesFor(trial, days, meta).then(perDay => {
    const split = splitPoint(perDay);
    slots[2].textContent = '';
    slots[3].textContent = '';
    slots[2].appendChild(areaPanel(trial, days, cut, split));
    slots[3].appendChild(aspectPanel(trial, days, cut, split));
  });
}
