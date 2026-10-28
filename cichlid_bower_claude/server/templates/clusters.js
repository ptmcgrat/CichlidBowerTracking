/* The cluster page.
 *
 * Laid out like the depth page: one column per day, eight to a block. Maps are
 * cumulative — everything up to and including that day — because building is
 * rare enough that a single day is mostly empty tank, and because that is how
 * the depth page shows total change, so the two read against each other.
 *
 * Every event is held in the page as typed columns, about eleven bytes each,
 * so the sliders re-filter without asking the server.
 */

let EVENTS = null;
let CONFIDENCE = 0.67;
// 8am to 6pm: the lit period, and the same window features.py analyses. The
// page opened on 0 to 24 and so disagreed with every number computed from it.
const HOUR_MIN = 8, HOUR_MAX = 18;
let HOUR_FROM = HOUR_MIN, HOUR_TO = HOUR_MAX;
// 0 shows every event, 1 male, 2 female -- the codes the collector packs.
// Filtering to a sex drops the unattributed events rather than splitting
// them, so the two sexes do not add up to all, and the control says so.
let SEX = 0;
let BIN = 20;            // pixels per bin, in Pi coordinates
let VIEW = 'maps';       // maps | stats

const COLOURS = {
  scoop: [242, 163, 60],     // orange
  spit: [111, 178, 232],     // blue
  multiple: [90, 168, 122],  // green
  spawn: [232, 132, 168],    // pink
  noclip: [20, 20, 24],      // black
  cropped: [224, 105, 63],   // red
};

// The build map accumulates, because building is rare enough that one day is
// mostly empty tank and the question is where the bower has got to. Everything
// below it is that day alone, drawn as points rather than binned: at a few
// hundred events a day the individual positions are the information, and
// binning them only throws it away.
const ROWS = [
  ['Pi camera', 'the tank that day', 'still'],
  ['Building', 'cumulative \u00b7 spits minus scoops', 'buildNet'],
  ['Build by type, kept',
   'that day \u00b7 scoop orange, spit blue, multiple green \u00b7 passes every filter',
   'build'],
  ['Build by type, dropped',
   'that day \u00b7 the same three, from every build event the filters reject ' +
   '\u2014 the hour and fish filters included',
   'buildFail'],
  ['Feed by type', 'that day \u00b7 scoop orange, spit blue, multiple green', 'feed'],
  ['Spawning', 'that day \u00b7 quivering events', 'spawn'],
  ['Set aside', 'that day \u00b7 no clip black, outside the crop red ' +
   '\u00b7 answers to no filter', 'aside'],
];
const CUMULATIVE = new Set(['buildNet']);


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
      const index = {};
      packed.bids.forEach((bid, i) => { index[bid] = i; });
      EVENTS = {
        n: packed.n, bids: packed.bids, labels: packed.labels, code: index,
        x: unpackColumn(packed.x, Uint16Array),
        y: unpackColumn(packed.y, Uint16Array),
        bid: unpackColumn(packed.bid, Uint8Array),
        prob: unpackColumn(packed.prob, Uint8Array),
        flags: unpackColumn(packed.flags, Uint8Array),
        day: unpackColumn(packed.day, Uint16Array),
        trial: unpackColumn(packed.trial, Uint8Array),
        hour: unpackColumn(packed.hour, Uint8Array),
        // absent in anything collected before the cluster file's Sex column
        // was packed. Null, not a zero filled array: the control has to be
        // able to tell 'not collected' from 'every event unknown'.
        sex: packed.sex ? unpackColumn(packed.sex, Uint8Array) : null,
        sexes: packed.sexes || null,
        summary: packed.summary,
      };
      return EVENTS;
    });
}

// The video crop, tested in Pi coordinates. Recomputed from what is saved
// rather than read from the cluster file's own column: the point of showing it
// is what the crop as it stands now would discard.
function outsideVideoCrop(x, y, trial) {
  const points = trial === undefined ? PREP.video_crop : videoCropFor(trial);
  if (!points || points.length < 3) return false;
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const xi = points[i][0], yi = points[i][1];
    const xj = points[j][0], yj = points[j][1];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi)
      inside = !inside;
  }
  return !inside;
}

function binGrid(width, height) {
  return { across: Math.max(1, Math.ceil(width / BIN)),
           down: Math.max(1, Math.ceil(height / BIN)) };
}

// Whether each event falls outside the video crop, worked out once per redraw
// rather than once per cell. It is a point-in-polygon test, and there are a
// hundred and sixty thousand events.
let CROPPED = null;
function computeCropped() {
  CROPPED = new Uint8Array(EVENTS.n);
  for (let i = 0; i < EVENTS.n; i++)
    if (outsideVideoCrop(EVENTS.x[i], EVENTS.y[i], EVENTS.trial[i])) CROPPED[i] = 1;
}

// Event indices grouped by day, filtered once. The maps are cumulative, so a
// day is the previous day's bins plus this day's events — which means each
// event is visited once per row rather than once per cell. Scanning every
// event for every cell was thirty-four times more work for the same answer.
function bucketByDay(trial, days) {
  const buckets = days.map(() => []);
  const position = {};
  days.forEach((day, index) => { position[day.index] = index; });
  const cut = Math.round(CONFIDENCE * 255);
  for (let i = 0; i < EVENTS.n; i++) {
    if (EVENTS.trial[i] !== trial) continue;
    const slot = position[EVENTS.day[i]];
    if (slot === undefined) continue;
    // the hour window is NOT applied here. It used to be, which meant an
    // event outside it reached no row at all and so could not be reported as
    // dropped. It is a gate like any other now, tested per row.
    buckets[slot].push(i);
  }
  return { buckets, cut };
}

function paintBins(canvas, layers, grid, width, height) {
  canvas.width = grid.across;
  canvas.height = grid.down;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(grid.across, grid.down);
  // each layer is normalised to its own maximum: feeding outnumbers building
  // three to one, so a shared scale would leave the build maps looking empty
  const peaks = layers.map(layer => {
    let peak = 0;
    for (let i = 0; i < layer.counts.length; i++)
      if (layer.counts[i] > peak) peak = layer.counts[i];
    return peak;
  });
  for (let i = 0, p = 0; i < grid.across * grid.down; i++, p += 4) {
    let r = 11, g = 13, b = 17, weight = 0;
    layers.forEach((layer, index) => {
      const peak = peaks[index];
      if (!peak || !layer.counts[i]) return;
      // log scale: a bower core is orders of magnitude denser than its edge
      const fraction = Math.log1p(layer.counts[i]) / Math.log1p(peak);
      const colour = layer.colour;
      r += colour[0] * fraction; g += colour[1] * fraction; b += colour[2] * fraction;
      weight += fraction;
    });
    if (weight > 1) { r /= weight; g /= weight; b /= weight; }
    image.data[p] = Math.min(255, r);
    image.data[p+1] = Math.min(255, g);
    image.data[p+2] = Math.min(255, b);
    image.data[p+3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

// A diverging map for the signed build total, so it reads the same way as the
// depth page: sand added one colour, sand removed the other, and nothing in
// the middle.
function netCell(values, grid, width, height, foot) {
  const box = el('div');
  const stage = el('div', 'stage');
  const canvas = el('canvas');
  canvas.style.imageRendering = 'pixelated';
  canvas.width = grid.across;
  canvas.height = grid.down;
  stage.appendChild(canvas);
  box.appendChild(stage);
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(grid.across, grid.down);
  let peak = 0;
  for (let i = 0; i < values.length; i++)
    if (Math.abs(values[i]) > peak) peak = Math.abs(values[i]);
  for (let i = 0, p = 0; i < values.length; i++, p += 4) {
    const v = values[i];
    let r = 11, g = 13, b = 17;
    if (peak && v) {
      const fraction = Math.log1p(Math.abs(v)) / Math.log1p(peak);
      const colour = v > 0 ? COLOURS.spit : COLOURS.scoop;
      r += colour[0] * fraction; g += colour[1] * fraction; b += colour[2] * fraction;
    }
    image.data[p] = Math.min(255, r); image.data[p+1] = Math.min(255, g);
    image.data[p+2] = Math.min(255, b); image.data[p+3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  if (foot) box.appendChild(el('div', 'cellfoot', foot));
  return box;
}

// Points, not bins. Drawn into a canvas sized to the bin grid would lose
// them, so this one is drawn at a fixed working size and scaled by CSS.
function scatterCell(groups, width, height, foot) {
  const box = el('div');
  const stage = el('div', 'stage');
  const canvas = el('canvas');
  const W = 260, H = Math.max(1, Math.round(260 * height / width));
  canvas.width = W; canvas.height = H;
  stage.appendChild(canvas);
  box.appendChild(stage);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0b0d11';
  ctx.fillRect(0, 0, W, H);
  groups.forEach(group => {
    ctx.fillStyle = 'rgb(' + group.colour.join(',') + ')';
    ctx.globalAlpha = group.points.length > 400 ? 0.45 : 0.8;
    group.points.forEach(point => {
      ctx.beginPath();
      ctx.arc(point[0] / width * W, point[1] / height * H, 1.6, 0, 6.2832);
      ctx.fill();
    });
  });
  ctx.globalAlpha = 1;
  if (foot) box.appendChild(el('div', 'cellfoot', foot));
  return box;
}

function cell(layers, grid, width, height, foot) {
  const box = el('div');
  const stage = el('div', 'stage');
  const canvas = el('canvas');
  canvas.style.imageRendering = 'pixelated';
  stage.appendChild(canvas);
  box.appendChild(stage);
  paintBins(canvas, layers, grid, width, height);
  if (foot) box.appendChild(el('div', 'cellfoot', foot));
  return box;
}

// Predicates take one context object rather than a positional list. They used
// to take (bid, prob, cut, hasClip, cropped) and forward them by hand, and a
// caller that forwarded four of the five left `cropped` undefined -- so the
// crop silently passed everything. Adding two more filters by the same route
// would be adding two more chances at that bug.
function context(index, cut) {
  const hour = EVENTS.hour[index];
  return {
    bid: EVENTS.bid[index],
    hasClip: (EVENTS.flags[index] & 1) !== 0,
    cropped: CROPPED[index] === 1,
    confident: EVENTS.prob[index] >= cut,
    inHours: hour >= HOUR_FROM && hour < HOUR_TO,
    sexOk: !SEX || !!(EVENTS.sex && EVENTS.sex[index] === SEX),
  };
}

// Every gate a kept event clears. The crop is one of them: it was computed on
// every redraw and then used only to colour the set aside row, so every other
// row drew events from outside the tray, which is what put feeding spits on
// the glass.
function passes(c) {
  return c.hasClip && !c.cropped && c.confident && c.inHours && c.sexOk &&
         c.bid !== 255;
}

// Why an event was rejected, tested in gate order and attributed to the first
// cause that applies, so the causes sum to the events shown rather than
// counting one event under every test it fails.
const DROP_REASONS = [
  ['no clip', c => !c.hasClip],
  ['outside crop', c => c.cropped],
  ['low confidence', c => !c.confident],
  ['outside hours', c => !c.inHours],
  ['other sex', c => !c.sexOk],
];

function categoriesFor(key) {
  const code = EVENTS.code;
  // the same build events seen from the other side: predicted as building and
  // rejected by any gate at all, the sex and hour filters included, so the row
  // is a complete account of what is not in the row above it. Clustering is
  // still required, because an unclustered transition is not an event and
  // belongs in the set aside row.
  const dropped = c => !passes(c) && c.bid !== 255;
  if (key === 'buildFail')
    return { scoop: c => dropped(c) && c.bid === code.c,
             spit: c => dropped(c) && c.bid === code.p,
             multiple: c => dropped(c) && c.bid === code.b };
  if (key === 'buildNet')
    // signed: a spit puts sand down and a scoop takes it away, so summing them
    // shows where the bower is being built against where it is being dug out
    return { spit: c => passes(c) && c.bid === code.p,
             scoop: c => passes(c) && c.bid === code.c };
  if (key === 'build')
    return { scoop: c => passes(c) && c.bid === code.c,
             spit: c => passes(c) && c.bid === code.p,
             multiple: c => passes(c) && c.bid === code.b };
  if (key === 'feed')
    return { scoop: c => passes(c) && c.bid === code.f,
             spit: c => passes(c) && c.bid === code.t,
             multiple: c => passes(c) && c.bid === code.m };
  if (key === 'spawn')
    return { spawn: c => passes(c) && c.bid === code.s };
  // set aside reports what never became usable, so it answers to no filter
  return { noclip: c => !c.hasClip,
           cropped: c => c.hasClip && c.cropped };
}

const LAYER_COLOUR = { all: COLOURS.scoop, scoop: COLOURS.scoop, spit: COLOURS.spit,
                       multiple: COLOURS.multiple, spawn: COLOURS.spawn,
                       noclip: COLOURS.noclip, cropped: COLOURS.cropped };

function renderTrial(trial, days, host) {
  const width = D.videoSize[0], height = D.videoSize[1];
  const grid = binGrid(width, height);
  const cells = grid.across * grid.down;
  const { buckets, cut } = bucketByDay(+trial, days);

  // one slot per block, filled as the running totals reach that day
  const slots = {};
  for (let offset = 0; offset < days.length; offset += 8) {
    const block = days.slice(offset, offset + 8);
    const pad = 8 - block.length;
    const table = el('div', 'matrix');
    table.style.gridTemplateColumns = '132px repeat(8, minmax(0, 1fr))';
    host.appendChild(table);

    table.appendChild(el('div', 'blank'));
    block.forEach(day => table.appendChild(el('div', 'colhead',
      '<b>' + day.date.slice(5) + '</b><span>to here</span>')));
    for (let i = 0; i < pad; i++) table.appendChild(el('div', 'blank'));

    ROWS.forEach(([name, hint, key]) => {
      table.appendChild(el('div', 'rowlab',
        '<b>' + name + '</b><span>' + hint + '</span>'));
      block.forEach(day => {
        const slot = el('div', 'cell');
        table.appendChild(slot);
        if (key === 'still') {
          if (day.videoJpg) {
            const stage = el('div', 'stage');
            const image = el('img');
            image.src = day.videoJpg;
            stage.appendChild(image);
            slot.appendChild(stage);
          } else {
            slot.appendChild(el('div', 'cellnote', 'no video this day'));
          }
        } else {
          slots[key + ':' + day.index] = slot;
        }
      });
      for (let i = 0; i < pad; i++) table.appendChild(el('div', 'blank'));
    });
  }

  ROWS.forEach(([name, hint, key]) => {
    if (key === 'still') return;      // filled above; nothing to accumulate
    const categories = categoriesFor(key);
    const parts = Object.keys(categories);
    const running = {};
    const totals = {};
    parts.forEach(part => { running[part] = new Float32Array(cells); totals[part] = 0; });
    const accumulates = CUMULATIVE.has(key);

    const reasons = DROP_REASONS.map(() => 0);

    days.forEach((day, position) => {
      const points = {};
      parts.forEach(part => { points[part] = []; });
      if (!accumulates) parts.forEach(part => { running[part].fill(0); totals[part] = 0; });
      if (!accumulates) reasons.fill(0);
      // by position within the trial, which is how the buckets were filled.
      // Indexing by the global day index worked for trial one, whose days
      // start at zero, and silently emptied every trial after it.
      for (const i of buckets[position] || []) {
        const clustered = (EVENTS.flags[i] & 2) !== 0;
        if (!clustered && key !== 'aside') continue;
        const c = context(i, cut);
        const bin = Math.min(grid.down - 1, (EVENTS.y[i] / BIN) | 0) * grid.across +
                    Math.min(grid.across - 1, (EVENTS.x[i] / BIN) | 0);
        let counted = false;
        for (const part of parts) {
          if (!categories[part](c)) continue;
          running[part][bin] += 1;
          totals[part] += 1;
          counted = true;
          if (!accumulates) points[part].push([EVENTS.x[i], EVENTS.y[i]]);
        }
        if (counted && key === 'buildFail')
          for (let r = 0; r < DROP_REASONS.length; r++)
            if (DROP_REASONS[r][1](c)) { reasons[r] += 1; break; }
      }
      const slot = slots[key + ':' + day.index];
      if (!slot) return;
      slot.textContent = '';
      if (key === 'buildNet') {
        const net = new Float32Array(cells);
        for (let i = 0; i < cells; i++) net[i] = running.spit[i] - running.scoop[i];
        slot.appendChild(netCell(net, grid, width, height,
          '+' + totals.spit + ' spits, \u2212' + totals.scoop + ' scoops'));
        return;
      }
      let foot = parts.map(part => totals[part]).join(' / ');
      if (key === 'buildFail')
        // why, not just how many: at confidence zero the only cause left is
        // the crop, and the footer saying so is what makes that readable
        foot += ' — ' + DROP_REASONS
          .map(([name], r) => reasons[r] + ' ' + name).join(', ');
      slot.appendChild(scatterCell(
        parts.map(part => ({ points: points[part], colour: LAYER_COLOUR[part] })),
        width, height, foot));
    });
  });
}

function draw() {
  const body = document.getElementById('body');
  body.textContent = '';
  if (VIEW === 'stats') {
    body.appendChild(statsView());
    return;
  }
  computeCropped();
  const excluded = excludedBanner();
  if (excluded) body.appendChild(excluded);
  const byTrial = daysByTrial(false);
  Object.keys(byTrial).sort((a, b) => a - b).forEach(trial => {
    const days = byTrial[trial];
    const section = el('div', 'trial');
    section.appendChild(el('h3', null, 'Trial ' + trial +
      '<span>' + days.length + ' days \u00b7 cumulative to each column</span>'));
    body.appendChild(section);
    renderTrial(trial, days, section);
  });
}

function build() {
  document.getElementById('title').textContent = D.projectID;
  document.getElementById('subtitle').textContent =
    'Classified events, accumulated day by day.';
  const summary = EVENTS.summary || {};
  document.getElementById('meta').innerHTML =
    [['Tank', D.tankID], ['Events', EVENTS.n],
     ['No clip', summary.noClip || 0]]
    .concat(summary.bySex
      ? [['Male', summary.bySex.Male || 0],
         ['Female', summary.bySex.Female || 0]]
      : [])
    .concat([['Days', D.days.length], ['Trials', D.trials.length]])
    .map(([k, v]) => '<div>' + k + '<b>' + v + '</b></div>').join('');

  const bar = document.getElementById('controls');
  bar.innerHTML =
    '<label>Confidence \u2265 <b id="cv">' + CONFIDENCE.toFixed(2) + '</b></label>' +
    '<input type="range" id="cr" min="0" max="0.99" step="0.01" value="' +
      CONFIDENCE + '">' +
    '<label>Hours <b id="hv">' + HOUR_FROM + '\u2013' + HOUR_TO + '</b></label>' +
    '<input type="range" id="h1" min="' + HOUR_MIN + '" max="' + HOUR_MAX +
      '" step="1" value="' + HOUR_FROM + '">' +
    '<input type="range" id="h2" min="' + HOUR_MIN + '" max="' + HOUR_MAX +
      '" step="1" value="' + HOUR_TO + '">' +
    (EVENTS.sex
      ? '<label>Fish <select id="sx">' +
          '<option value="0">any</option>' +
          '<option value="1">male</option>' +
          '<option value="2">female</option>' +
        '</select></label>'
      : '<span class="stat">no sex column in this project’s cluster ' +
        'file — recollect to filter by fish</span>') +
    '<label>Bin <b id="bv">' + BIN + '</b> px</label>' +
    '<input type="range" id="br" min="5" max="80" step="5" value="' + BIN + '">' +
    '<span class="spacer"></span><span class="stat" id="note"></span>';

  let pending = null;
  const later = () => {
    if (pending) clearTimeout(pending);
    pending = setTimeout(draw, 200);
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
  bar.querySelector('#br').addEventListener('input', event => {
    BIN = parseInt(event.target.value, 10);
    bar.querySelector('#bv').textContent = BIN;
    later();
  });
  const sexControl = bar.querySelector('#sx');
  if (sexControl) sexControl.addEventListener('change', event => {
    SEX = parseInt(event.target.value, 10);
    draw();
  });

  const tabs = document.getElementById('tabs');
  [['Maps', 'maps'], ['Statistics', 'stats']].forEach(([label, key]) => {
    const button = el('button', null, label);
    button.setAttribute('role', 'tab');
    button.setAttribute('aria-selected', String(VIEW === key));
    button.addEventListener('click', () => {
      VIEW = key;
      Array.from(tabs.children).forEach(other =>
        other.setAttribute('aria-selected', String(other === button)));
      // the statistics split at a fixed half, so the map controls do not
      // apply there and are hidden rather than left to mislead
      bar.style.display = key === 'maps' ? '' : 'none';
      draw();
    });
    tabs.appendChild(button);
  });
  bar.querySelector('#note').innerHTML =
    'each map is normalised to its own peak \u2014 feeding outnumbers building ' +
    'about three to one, so a shared scale would leave the build maps empty';
  draw();
}

loadPayload(() => {
  loadEvents().then(events => {
    if (!events) {
      document.getElementById('body').appendChild(el('div', 'note',
        'No cluster data collected for this project. Run the cluster stage, then ' +
        'collect it again.'));
      return;
    }
    build();
    document.getElementById('foot').textContent =
      'Collected ' + D.collected + ' \u00b7 page built ' + D.built;
  });
});
