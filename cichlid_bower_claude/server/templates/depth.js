/* The depth page.
 *
 * One column per day, eight to a block, six rows. Everything is drawn from the
 * raw endpoint frames with the tray crop and the residual mask applied here,
 * so moving a threshold costs a redraw rather than a rerun.
 *
 * Nothing is interpolated anywhere in this pipeline, so a change map is
 * defined only where both frames have a reading. The valid pixel count is
 * reported beside every figure: a day measured on fewer pixels reports a
 * smaller bower for a reason that is not biological.
 */

let THRESHOLD = 0.6;     // cm of height for a pixel to count as bower
let MIN_PIXELS = 400;    // contiguous pixels for a region to be kept
let RANGE = 2;           // colour scale for a daily map; total uses twice this
let MASKS = null;        // {crop, residual} once the score is loaded

// The colour scale is a multiple of RANGE rather than a fixed number: change
// accumulates, so a total map wants twice the span a single day does, and
// holding that ratio means one control sets both.
const ROWS = [
  ['Total change', 'trial start to end of day', 'total', 2],
  ['24 hour change', 'morning to next morning', 'full', 1],
  ['Daylight change', 'morning to evening', 'daylight', 1],
  ['Night change', 'evening to next morning', 'night', 1],
  ['Bower, total', 'regions in total change', 'bowerTotal', 2],
  ['Bower, daylight', 'regions in daylight change', 'bowerDaily', 1],
];

// Connected components over the thresholded change, castle and pit labelled
// separately so a pit touching a castle is not one region. Anything smaller
// than MIN_PIXELS is dropped: a handful of adjacent pixels clearing the height
// is noise, not a bower.
function bowerRegions(values, width, height, threshold, minPixels) {
  const sign = new Int8Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isNaN(v)) continue;
    if (v >= threshold) sign[i] = 1;
    else if (v <= -threshold) sign[i] = -1;
  }
  const keep = new Uint8Array(values.length);
  const seen = new Uint8Array(values.length);
  const stack = new Int32Array(values.length);
  let castle = 0, pit = 0, castleVolume = 0, pitVolume = 0;

  for (let start = 0; start < sign.length; start++) {
    if (!sign[start] || seen[start]) continue;
    const which = sign[start];
    let top = 0, count = 0, sum = 0;
    const members = [];
    stack[top++] = start;
    seen[start] = 1;
    while (top > 0) {
      const p = stack[--top];
      members.push(p);
      count++;
      sum += Math.abs(values[p]);
      const x = p % width, y = (p / width) | 0;
      if (x > 0 && sign[p-1] === which && !seen[p-1]) { seen[p-1] = 1; stack[top++] = p-1; }
      if (x < width-1 && sign[p+1] === which && !seen[p+1]) { seen[p+1] = 1; stack[top++] = p+1; }
      if (y > 0 && sign[p-width] === which && !seen[p-width]) { seen[p-width] = 1; stack[top++] = p-width; }
      if (y < height-1 && sign[p+width] === which && !seen[p+width]) { seen[p+width] = 1; stack[top++] = p+width; }
    }
    if (count < minPixels) continue;
    for (const p of members) keep[p] = 1;
    if (which > 0) { castle += count; castleVolume += sum; }
    else { pit += count; pitVolume += sum; }
  }

  const notBower = new Uint8Array(values.length);
  for (let i = 0; i < keep.length; i++) if (!keep[i]) notBower[i] = 1;
  return { notBower, castle, pit, castleVolume, pitVolume };
}

function applyMasks(values) {
  // the crop and the residual mask are applied here, not baked into the data,
  // so either can change without anything being recomputed upstream
  if (!MASKS) return values;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    const dropped = (MASKS.crop && MASKS.crop[i]) ||
                    (MASKS.residual && MASKS.residual[i]);
    out[i] = dropped ? NaN : values[i];
  }
  return out;
}

function cellMap(values, meta, range, opts) {
  // no captions at eight columns; the hover readout carries the value
  opts = opts || {};
  const wrap = el('div', 'stage');
  const canvas = el('canvas');
  wrap.appendChild(canvas);
  const readout = el('span', 'readout', '\u2014');
  wrap.appendChild(readout);
  paintMap(canvas, values, meta, Object.assign({ range }, opts));
  wrap.addEventListener('mousemove', event => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / rect.width * meta.width);
    const y = Math.floor((event.clientY - rect.top) / rect.height * meta.height);
    if (x < 0 || y < 0 || x >= meta.width || y >= meta.height) return;
    const v = values[y * meta.width + x];
    readout.textContent = Number.isNaN(v) ? 'no data'
                        : (v >= 0 ? '+' : '') + v.toFixed(2) + ' cm';
  });
  return wrap;
}

function blockFor(days, trial, host) {
  const COLUMNS = 8;
  const pad = COLUMNS - days.length;
  const grid = el('div', 'matrix');
  grid.style.gridTemplateColumns = '132px repeat(' + COLUMNS + ', minmax(0, 1fr))';
  host.textContent = '';
  host.appendChild(grid);

  grid.appendChild(el('div', 'blank'));
  days.forEach(day => {
    grid.appendChild(el('div', 'colhead',
      '<b>' + day.date.slice(5) + '</b><span>' + (day.hours || 0).toFixed(1) + ' h</span>'));
  });
  for (let i = 0; i < pad; i++) grid.appendChild(el('div', 'blank'));

  const cells = {};
  ROWS.forEach(([name, hint, key]) => {
    grid.appendChild(el('div', 'rowlab', '<b>' + name + '</b><span>' + hint + '</span>'));
    days.forEach(day => {
      const slot = el('div', 'cell');
      grid.appendChild(slot);
      cells[key + ':' + day.index] = slot;
    });
    for (let i = 0; i < pad; i++) grid.appendChild(el('div', 'blank'));
  });
  return cells;
}

function renderTrial(trial, daysInTrial, host) {
  const baseline = endpointsFor(daysInTrial[0], daysInTrial).first;

  for (let offset = 0; offset < daysInTrial.length; offset += 8) {
    const block = daysInTrial.slice(offset, offset + 8);
    const blockHost = el('div', 'block');
    host.appendChild(blockHost);
    const cells = blockFor(block, trial, blockHost);

    block.forEach(day => {
      const ends = endpointsFor(day, daysInTrial);
      const position = daysInTrial.findIndex(d => d.index === day.index);
      const next = daysInTrial[position + 1];
      const nextEnds = next ? endpointsFor(next, daysInTrial) : null;

      const needed = [baseline, ends.first, ends.last];
      if (nextEnds) needed.push(nextEnds.first);

      Promise.all(needed.map(loadDepth)).then(arrays => {
        const [base, first, last, nextFirst] = arrays;
        if (!base || !first || !last) return;
        const meta = ends.first;

        const total = applyMasks(difference(base, last));
        const daylight = applyMasks(difference(first, last));
        const night = nextFirst ? applyMasks(difference(last, nextFirst)) : null;
        const full = nextFirst ? applyMasks(difference(first, nextFirst)) : null;

        const put = (key, values, range, opts) => {
          const slot = cells[key + ':' + day.index];
          if (!slot) return;
          slot.textContent = '';
          if (!values) {
            slot.appendChild(el('div', 'cellnote',
              'no following day in this trial'));
            return;
          }
          slot.appendChild(cellMap(values, meta, range, opts));
          if (opts && opts.foot) slot.appendChild(el('div', 'cellfoot', opts.foot));
        };

        put('total', total, 2 * RANGE);
        put('full', full, RANGE);
        put('daylight', daylight, RANGE);
        put('night', night, RANGE);

        [['bowerTotal', total, 2 * RANGE], ['bowerDaily', daylight, RANGE]]
          .forEach(([key, values, range]) => {
            const regions = bowerRegions(values, meta.width, meta.height,
                                         THRESHOLD, MIN_PIXELS);
            let valid = 0;
            for (let i = 0; i < values.length; i++)
              if (!Number.isNaN(values[i])) valid++;
            put(key, values, range, {
              mark: regions.notBower, markColour: [30, 34, 40],
              foot: 'castle ' + regions.castle + ' px, pit ' + regions.pit +
                    ' px \u00b7 ' + valid + ' valid',
            });
          });
      });
    });
  }
}

function draw() {
  const body = document.getElementById('body');
  body.textContent = '';

  const byTrial = {};
  D.days.forEach(day => { (byTrial[day.trial] = byTrial[day.trial] || []).push(day); });

  Object.keys(byTrial).sort((a, b) => a - b).forEach(trial => {
    const days = byTrial[trial];
    const section = el('div', 'trial');
    const times = PREP.trials[String(trial)] || PREP.trials['1'] || {};
    section.appendChild(el('h3', null, 'Trial ' + trial +
      '<span>' + days.length + ' days \u00b7 ' + days[0].date + ' to ' +
      days[days.length - 1].date +
      (times.start ? ' \u00b7 start +' + times.start + ' min' : '') +
      (times.stop ? ' \u00b7 stop \u2212' + times.stop + ' min' : '') + '</span>'));
    body.appendChild(section);
    renderTrial(trial, days, section);
  });
}

function build() {
  document.getElementById('title').textContent = D.projectID;
  document.getElementById('subtitle').textContent =
    'Change by day, and the bower regions it contains.';
  document.getElementById('meta').innerHTML =
    [['Tank', D.tankID], ['Analysis', D.analysisID],
     ['Depth', D.depthSize ? D.depthSize.join(' \u00d7 ') : '?'],
     ['Days', D.days.length], ['Trials', D.trials.length],
     ['Collected', (D.collected || '').slice(0, 16)]]
    .map(([k, v]) => '<div>' + k + '<b>' + v + '</b></div>').join('');

  const bar = document.getElementById('controls');
  bar.innerHTML =
    '<label>Bower height \u2265 <b id="tv">' + THRESHOLD.toFixed(2) + '</b> cm</label>' +
    '<input type="range" id="tr" min="0.1" max="3" step="0.05" value="' + THRESHOLD + '">' +
    '<label>Minimum region <b id="pv">' + MIN_PIXELS + '</b> px</label>' +
    '<input type="range" id="pr" min="0" max="2000" step="25" value="' + MIN_PIXELS + '">' +
    '<label>Colour scale \u00b1<b id="rv">' + RANGE.toFixed(1) +
    '</b> cm, total \u00b1<b id="rv2">' + (2 * RANGE).toFixed(1) + '</b></label>' +
    '<input type="range" id="rr" min="0.5" max="8" step="0.5" value="' + RANGE + '">' +
    '<span class="spacer"></span><span class="stat" id="maskNote"></span>';

  let pending = null;
  const later = () => {
    // redrawing every block is expensive; wait for the slider to settle
    if (pending) clearTimeout(pending);
    pending = setTimeout(draw, 180);
  };
  bar.querySelector('#tr').addEventListener('input', event => {
    THRESHOLD = parseFloat(event.target.value);
    bar.querySelector('#tv').textContent = THRESHOLD.toFixed(2);
    later();
  });
  bar.querySelector('#pr').addEventListener('input', event => {
    MIN_PIXELS = parseInt(event.target.value, 10);
    bar.querySelector('#pv').textContent = MIN_PIXELS;
    later();
  });
  bar.querySelector('#rr').addEventListener('input', event => {
    RANGE = parseFloat(event.target.value);
    bar.querySelector('#rv').textContent = RANGE.toFixed(1);
    bar.querySelector('#rv2').textContent = (2 * RANGE).toFixed(1);
    later();
  });

  const note = bar.querySelector('#maskNote');
  const meta = D.days.length ? D.days[0].firstPng : null;
  const crop = meta ? cropMaskFor(meta) : null;

  if (!D.residualScore || !meta) {
    MASKS = { crop, residual: null };
    note.textContent = crop ? 'tray crop applied' : 'no crop or mask set';
    draw();
    return;
  }
  loadDepth(D.residualScore).then(score => {
    const result = score ? scoreMask(score, PREP.residual_k) : { mark: null };
    MASKS = { crop, residual: result.mark };
    let cut = 0, total = 0;
    for (let i = 0; i < (score ? score.length : 0); i++) {
      total++;
      if ((crop && crop[i]) || (result.mark && result.mark[i])) cut++;
    }
    note.innerHTML = 'crop and residual mask at k = <b>' + PREP.residual_k +
      '</b> remove <b>' + (100 * cut / Math.max(1, total)).toFixed(1) + '%</b> of the frame';
    draw();
  });
}

loadPayload(() => {
  if (!D.days.length) {
    document.getElementById('body').appendChild(
      el('div', 'note', 'No days in this bundle.'));
    return;
  }
  build();
  document.getElementById('foot').textContent =
    'Collected ' + D.collected + ' \u00b7 page built ' + D.built;
});