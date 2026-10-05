/* The prep page: four tabs over one saved object. */

// --------------------------------------------------------------- homography
// Normalised DLT. Conditioning matters: raw pixel coordinates are in the
// hundreds, so the unnormalised system is badly scaled and the fit wanders.
function normalise(points) {
  let cx = 0, cy = 0;
  points.forEach(p => { cx += p[0]; cy += p[1]; });
  cx /= points.length; cy /= points.length;
  let mean = 0;
  points.forEach(p => { mean += Math.hypot(p[0] - cx, p[1] - cy); });
  mean /= points.length || 1;
  const scale = mean > 0 ? Math.SQRT2 / mean : 1;
  return { matrix: [[scale, 0, -scale * cx], [0, scale, -scale * cy], [0, 0, 1]],
           points: points.map(p => [(p[0] - cx) * scale, (p[1] - cy) * scale]) };
}

function solveLeastSquares(A, rows, cols) {
  // normal equations with partial pivoting; nine unknowns, so this is ample
  const N = [];
  for (let i = 0; i < cols; i++) {
    N.push(new Float64Array(cols));
    for (let j = 0; j < cols; j++) {
      let sum = 0;
      for (let r = 0; r < rows; r++) sum += A[r][i] * A[r][j];
      N[i][j] = sum;
    }
  }
  // smallest eigenvector by inverse iteration on N
  let v = new Float64Array(cols).fill(1 / Math.sqrt(cols));
  for (let iteration = 0; iteration < 200; iteration++) {
    const M = N.map(row => Float64Array.from(row));
    for (let i = 0; i < cols; i++) M[i][i] += 1e-9;
    const x = gaussian(M, v, cols);
    if (!x) break;
    let norm = Math.hypot(...x);
    if (!norm) break;
    v = x.map(value => value / norm);
  }
  return v;
}

function gaussian(M, b, n) {
  const a = M.map((row, i) => Float64Array.from([...row, b[i]]));
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++)
      if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) < 1e-14) return null;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = a[r][col] / a[col][col];
      for (let c = col; c <= n; c++) a[r][c] -= factor * a[col][c];
    }
  }
  return Array.from({ length: n }, (_, i) => a[i][n] / a[i][i]);
}

function homography(from, to) {
  if (from.length < 4) return null;
  const F = normalise(from), T = normalise(to);
  const A = [];
  for (let i = 0; i < from.length; i++) {
    const [x, y] = F.points[i], [u, v] = T.points[i];
    A.push([-x, -y, -1, 0, 0, 0, u*x, u*y, u]);
    A.push([0, 0, 0, -x, -y, -1, v*x, v*y, v]);
  }
  const h = solveLeastSquares(A, A.length, 9);
  if (!h) return null;
  const Hn = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]];
  const Ti = invert3(T.matrix);
  if (!Ti) return null;
  const H = multiply3(multiply3(Ti, Hn), F.matrix);
  const scale = H[2][2];
  return scale ? H.map(row => row.map(value => value / scale)) : H;
}

// -------------------------------------------------------------------- loupe
// Picking the same tray corner in two images is impossible at page scale, so
// a magnifier follows the cursor. It flips above or below so it never sits
// under the hand that is pointing.
function attachLoupe(stage, image, factor) {
  factor = factor || 5;
  const loupe = el('div', 'loupe');
  loupe.style.backgroundImage = 'url(' + image.src + ')';
  // a crosshair at the centre: without it the magnifier shows you the
  // neighbourhood but not which pixel a click would actually take
  loupe.appendChild(el('span', 'crosshair-h'));
  loupe.appendChild(el('span', 'crosshair-v'));
  stage.appendChild(loupe);
  stage.addEventListener('mousemove', event => {
    const rect = stage.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width;
    const y = (event.clientY - rect.top) / rect.height;
    if (x < 0 || y < 0 || x > 1 || y > 1) { loupe.style.display = 'none'; return; }
    loupe.style.display = 'block';
    loupe.style.backgroundSize = (rect.width * factor) + 'px ' +
                                 (rect.height * factor) + 'px';
    loupe.style.backgroundPosition =
      (-x * rect.width * factor + 70) + 'px ' + (-y * rect.height * factor + 70) + 'px';
    loupe.style.left = (event.clientX - rect.left - 70) + 'px';
    loupe.style.top = (y < 0.4 ? event.clientY - rect.top + 24
                               : event.clientY - rect.top - 164) + 'px';
  });
  stage.addEventListener('mouseleave', () => { loupe.style.display = 'none'; });
  return loupe;
}

// ------------------------------------------------------------- trial times
function viewTimes() {
  const box = el('div');
  box.appendChild(el('p', 'sub',
    'The logged trial start is often too early, before the sand has settled after a ' +
    'reset. Pick the first frame that looks settled. Every total-build figure for a ' +
    'trial is measured from its start frame, so this matters most there. The depth ' +
    'panel shows how far the sand moved since the zero-offset frame.'));

  const byTrial = {};
  D.candidates.forEach(c => {
    byTrial[c.trial] = byTrial[c.trial] || {};
    (byTrial[c.trial][c.kind] = byTrial[c.trial][c.kind] || []).push(c);
  });

  Object.keys(byTrial).sort((a, b) => a - b).forEach(trial => {
    const section = el('div', 'trial');
    const meta = D.trials.find(t => String(t.number) === String(trial)) || {};
    section.appendChild(el('h3', null, 'Trial ' + trial +
      '<span>logged ' + (meta.start || '').slice(0, 16) + ' to ' +
      (meta.stop || '').slice(0, 16) + '</span>'));
    box.appendChild(section);

    [['start', 'Trial start', 'later is safer: the sand settles'],
     ['stop', 'Trial stop', 'earlier is safer: the reset may have begun'],
     ['reset', 'After the final reset', 'the last picture of the bower']
    ].forEach(([kind, title, hint]) => {
      const items = (byTrial[trial][kind] || []).sort((a, b) => a.offset - b.offset);
      if (!items.length) return;
      section.appendChild(el('p', 'stat',
        '<b>' + title + '</b> \u2014 ' + hint +
        (items.length < D.offsets.length
          ? ' \u00b7 only ' + items.length + ' of ' + D.offsets.length +
            ' offsets exist here, the recording ends before the rest'
          : '')));
      const grid = el('div', 'grid cols-5');
      section.appendChild(grid);

      Promise.all(items.map(c => loadDepth(c.depthPng))).then(arrays => {
        const zero = arrays[0];
        items.forEach((c, i) => {
          const chosen = chosenOffset(trial, kind) === c.offset;
          const fig = el('figure', chosen ? 'picked' : null);
          fig.appendChild(el('div', null,
            '<div style="padding:8px 11px;font-size:12px;color:var(--dim)">' +
            '<b style="color:var(--ink);display:block">' + kindLabel(kind, c.offset) +
            '</b>' + c.time.slice(11, 16) + ' \u00b7 frame ' + c.index + '</div>'));
          if (c.jpg) {
            const stage = el('div', 'stage');
            const image = el('img');
            image.src = c.jpg;
            stage.appendChild(image);
            fig.appendChild(stage);
          }
          let moved = '';
          if (zero && arrays[i] && c.depthPng) {
            const values = difference(zero, arrays[i]);
            const stage = el('div', 'stage');
            const canvas = el('canvas');
            stage.appendChild(canvas);
            fig.appendChild(stage);
            paintMap(canvas, values, c.depthPng, { range: 1 });
            let n = 0, valid = 0;
            for (let k = 0; k < values.length; k++) {
              if (Number.isNaN(values[k])) continue;
              valid++;
              if (Math.abs(values[k]) > 0.15) n++;
            }
            moved = c.offset === 0 ? 'reference frame'
                  : (100 * n / Math.max(1, valid)).toFixed(1) +
                    '% of pixels moved more than 0.15 cm since the reference';
          }
          fig.appendChild(el('figcaption', null, moved));
          const button = el('button', 'pick', chosen ? 'Chosen' : 'Use this frame');
          button.addEventListener('click', () => {
            setOffset(trial, kind, c.offset);
            markDirty();
            show(0);
          });
          fig.appendChild(button);
          grid.appendChild(fig);
        });
      });
    });
  });
  return box;
}

function kindLabel(kind, offset) {
  if (kind === 'stop') return 'stop \u2212' + offset + ' min';
  return kind + ' +' + offset + ' min';
}

function chosenOffset(trial, kind) {
  const entry = PREP.trials[String(trial)] || PREP.trials['1'] || {};
  const value = entry[kind];
  return value === undefined || value === null ? 0 : value;
}

function setOffset(trial, kind, offset) {
  PREP.trials[String(trial)] = PREP.trials[String(trial)] ||
                               Object.assign({ start: 0, stop: 0, reset: 0 },
                                             PREP.trials['1'] || {});
  PREP.trials[String(trial)][kind] = offset;
}

// ------------------------------------------------------------- registration
// Points are kept per project, not per pair: the cameras do not move between
// trials, so a fit made on one pair applies to all of them. Switching pairs is
// how that gets confirmed, not how a second fit gets made.
let POINTS = [];
let PENDING = null;        // a click on one side waiting for its partner

function viewRegistration() {
  const box = el('div');
  if (!D.pairs.length) {
    box.appendChild(el('div', 'note', 'No registration pairs were collected for this ' +
      'project. It may have no video stills.'));
    return box;
  }
  box.appendChild(el('p', 'sub',
    'Click the same feature in both images \u2014 tray corners work well \u2014 four ' +
    'pairs at least, six for a fit worth trusting. The magnifier follows the cursor. ' +
    'The wipe below shows the result: tray edges should stay continuous across the ' +
    'divider.'));

  const bar = trialSelector(() => show(current));
  const clear = el('button', 'act', 'Clear points');
  const undo = el('button', 'act', 'Undo last');
  bar.appendChild(undo);
  bar.appendChild(clear);
  bar.appendChild(el('span', 'spacer'));
  const fitNote = el('span', 'stat');
  bar.appendChild(fitNote);
  box.appendChild(bar);

  const grid = el('div', 'grid cols-2');
  box.appendChild(grid);
  const lower = el('div', 'grid cols-2 lower');
  const previewSlot = el('div');
  const tableSlot = el('div');
  lower.appendChild(previewSlot);
  lower.appendChild(tableSlot);
  box.appendChild(lower);

  const pair = D.pairs[TRIAL_INDEX] || {};

  function currentFit() {
    const complete = POINTS.filter(p => p.pi && p.depth);
    if (complete.length < 4) return null;
    const H = homography(complete.map(p => p.pi), complete.map(p => p.depth));
    if (!H) return null;
    const errors = residuals(H, complete.map(p => p.pi), complete.map(p => p.depth));
    const rms = Math.sqrt(errors.reduce((sum, e) => sum + e*e, 0) / errors.length);
    return { H, errors, rms, n: complete.length };
  }

  function draw() {
    grid.textContent = '';
    const fit = currentFit();
    const active = fit ? fit.H : PREP.transform;

    fitNote.innerHTML = fit
      ? '<b>' + fit.n + '</b> pairs \u00b7 fit <b>' + fit.rms.toFixed(2) +
        '</b> px \u00b7 worst <b>' + Math.max(...fit.errors).toFixed(2) + '</b> px'
      : (PREP.transform
          ? 'showing the registration on file \u2014 pick four pairs to replace it'
          : 'pick four or more pairs');

    [['pi', 'Pi camera', pair.piJpg, D.videoSize],
     ['depth', 'Depth camera', pair.depthJpg, D.depthSize]
    ].forEach(([side, title, src, size]) => {
      if (!src) return;
      const fig = el('figure');
      const stage = el('div', 'stage');
      const img = el('img');
      img.src = src;
      stage.appendChild(img);
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', '0 0 ' + size[0] + ' ' + size[1]);
      svg.setAttribute('preserveAspectRatio', 'none');
      stage.appendChild(svg);
      fig.appendChild(stage);
      const waiting = PENDING && PENDING.side !== side;
      fig.appendChild(el('figcaption', null, '<b>' + title + '</b> \u2014 ' +
        POINTS.filter(p => p[side]).length + ' placed' +
        (waiting ? ' \u00b7 <span style="color:var(--tray)">now click the matching ' +
                   'point here</span>' : '')));
      grid.appendChild(fig);

      img.addEventListener('load', () => attachLoupe(stage, img, 5));
      if (img.complete) attachLoupe(stage, img, 5);

      stage.addEventListener('click', event => {
        const rect = stage.getBoundingClientRect();
        const point = [(event.clientX - rect.left) / rect.width * size[0],
                       (event.clientY - rect.top) / rect.height * size[1]];
        placePoint(side, point);
      });
      paintPoints(svg, side, size, fit);
    });

    drawPreview(previewSlot, pair, active);
    drawTable(tableSlot, fit);
  }

  function placePoint(side, point) {
    if (PENDING && PENDING.side !== side) {
      const entry = { [PENDING.side]: PENDING.point, [side]: point };
      POINTS.push(entry);
      PENDING = null;
    } else {
      PENDING = { side, point };
    }
    markDirty();
    draw();
  }

  function paintPoints(svg, side, size, fit) {
    const radius = Math.max(4, size[0] / 120);
    let markup = POINTS.map((p, i) => {
      if (!p[side]) return '';
      const bad = fit && fit.errors[i] !== undefined && fit.errors[i] > 3 * fit.rms;
      return '<circle cx="' + p[side][0] + '" cy="' + p[side][1] + '" r="' + radius +
             '" class="handle" style="fill:' + (bad ? 'var(--warn)' : 'var(--tray)') +
             '"/><text x="' + p[side][0] + '" y="' + p[side][1] + '" dy="' +
             (-radius * 1.6) + '" fill="#fff" font-size="' + (radius * 2.4) +
             '" text-anchor="middle">' + (i + 1) + '</text>';
    }).join('');
    if (PENDING && PENDING.side === side) {
      markup += '<circle cx="' + PENDING.point[0] + '" cy="' + PENDING.point[1] +
                '" r="' + radius + '" style="fill:none;stroke:var(--ok);stroke-width:2"/>';
    }
    svg.innerHTML = markup;
  }

  undo.addEventListener('click', () => {
    if (PENDING) PENDING = null; else POINTS.pop();
    markDirty();
    draw();
  });
  clear.addEventListener('click', () => {
    POINTS = []; PENDING = null;
    markDirty();
    draw();
  });
  draw();
  return box;
}

// The wipe: judging a registration by whether tray edges stay continuous
// across a moving divider is far better than reading a number, because it
// shows *where* a fit is wrong rather than only how much.
function drawPreview(slot, pair, H) {
  slot.textContent = '';
  if (!H || !pair.depthJpg || !pair.piJpg) return;
  slot.appendChild(el('h2', null, 'Preview'));
  const fig = el('figure', 'half');
  const stage = el('div', 'stage');
  const base = el('img');
  base.src = pair.depthJpg;
  stage.appendChild(base);

  const over = el('div', 'wipe');
  const warped = el('img');
  warped.src = pair.piJpg;
  warped.style.transformOrigin = '0 0';
  over.appendChild(warped);
  stage.appendChild(over);
  const divider = el('div', 'divider');
  stage.appendChild(divider);
  fig.appendChild(stage);
  fig.appendChild(el('figcaption', null,
    '<b>Depth frame with the Pi frame warped over it.</b> Move the pointer to wipe. ' +
    'Continuous tray edges across the divider mean the fit holds.'));
  slot.appendChild(fig);

  function place() {
    const rect = stage.getBoundingClientRect();
    if (!rect.width) return;
    const scaleX = rect.width / D.depthSize[0], scaleY = rect.height / D.depthSize[1];
    // CSS matrix3d is column-major, and maps Pi pixels to depth pixels before
    // the element is scaled to the box it is drawn in
    const M = [H[0][0], H[1][0], 0, H[2][0],
               H[0][1], H[1][1], 0, H[2][1],
               0, 0, 1, 0,
               H[0][2], H[1][2], 0, H[2][2]];
    warped.style.width = D.videoSize[0] + 'px';
    warped.style.height = D.videoSize[1] + 'px';
    warped.style.transform = 'scale(' + scaleX + ',' + scaleY + ') matrix3d(' +
                             M.join(',') + ')';
  }
  base.addEventListener('load', place);
  if (base.complete) place();
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(place).observe(stage);

  stage.addEventListener('mousemove', event => {
    const rect = stage.getBoundingClientRect();
    const fraction = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
    over.style.width = (fraction * 100) + '%';
    divider.style.left = (fraction * 100) + '%';
  });
  over.style.width = '50%';
  divider.style.left = '50%';
}

// Per-point residuals, so an outlier is visible rather than silently dragging
// the fit. Least squares spreads one bad pick across every point, which is why
// the table matters more than the summary number.
function drawTable(slot, fit) {
  slot.textContent = '';
  if (!fit) return;
  slot.appendChild(el('h2', null, 'Per-point error'));
  slot.appendChild(el('p', 'stat', 'Least squares spreads one bad pick across every ' +
    'point, so a single row well above the rest is the pick to redo \u2014 not ' +
    'evidence that the whole fit is poor.'));
  const table = el('table', 'points');
  table.innerHTML = '<tr><th>point</th><th>error px</th><th></th></tr>' +
    fit.errors.map((error, i) => {
      const bad = error > 3 * fit.rms;
      return '<tr><td>' + (i + 1) + '</td><td>' + error.toFixed(2) + '</td><td>' +
             (bad ? '<span style="color:var(--warn)">well above the rest \u2014 ' +
                    'check this pick</span>' : '') + '</td></tr>';
    }).join('');
  slot.appendChild(table);
}

// -------------------------------------------------------------------- crops
// One trial selector shared by the registration and crop tabs: the cameras do
// not move between trials, so a crop that fits one should fit them all, and
// stepping through is how that gets confirmed.
let TRIAL_INDEX = 0;

// Which trial the registration and crop tabs are editing. Changes land on the
// project by default; "just this trial" writes an override instead.
function currentTrial() {
  const pair = D.pairs[TRIAL_INDEX];
  return pair ? pair.trial : 1;
}

function editingOverride() {
  return !!(PREP.overrides || {})[String(currentTrial())];
}

function setOverride(on) {
  PREP.overrides = PREP.overrides || {};
  const key = String(currentTrial());
  if (on) {
    // deep copies: Object.assign shares the point arrays, so the override and
    // the project would be the same object and editing either would edit both.
    // That is why a trial's crop never looked any different from the project's.
    const clone = value => value ? JSON.parse(JSON.stringify(value)) : value;
    PREP.overrides[key] = Object.assign({
      transform: clone(PREP.transform), depth_crop: clone(PREP.depth_crop),
      video_crop: clone(PREP.video_crop), excluded: false, reason: '' },
      PREP.overrides[key] || {});
  } else if (PREP.overrides[key]) {
    const excluded = PREP.overrides[key].excluded;
    const reason = PREP.overrides[key].reason;
    if (excluded) PREP.overrides[key] = { excluded: true, reason: reason };
    else delete PREP.overrides[key];
  }
  markDirty();
}

function trialSelector(onChange) {
  const bar = el('div', 'bar');
  bar.appendChild(el('label', null, 'Pair'));
  const select = el('select');
  D.pairs.forEach((p, i) => {
    const option = el('option', null, 'Trial ' + p.trial + ' ' + p.side +
      ' \u00b7 matched to ' + p.gapMinutes + ' min');
    option.value = String(i);
    if (i === TRIAL_INDEX) option.selected = true;
    select.appendChild(option);
  });
  select.addEventListener('change', () => {
    TRIAL_INDEX = +select.value;
    onChange();
  });
  bar.appendChild(select);

  const own = el('button', 'act');
  own.textContent = editingOverride() ? 'Using its own settings'
                                      : 'Give this trial its own';
  own.setAttribute('aria-pressed', String(editingOverride()));
  own.addEventListener('click', () => {
    setOverride(!editingOverride());
    onChange();
  });
  bar.appendChild(own);

  const exclude = el('button', 'act');
  const trial = currentTrial();
  const excluded = isExcluded(trial);
  exclude.textContent = excluded ? 'Excluded from analysis' : 'Exclude this trial';
  exclude.setAttribute('aria-pressed', String(excluded));
  exclude.addEventListener('click', () => {
    PREP.overrides = PREP.overrides || {};
    const key = String(trial);
    const entry = PREP.overrides[key] || {};
    entry.excluded = !entry.excluded;
    if (entry.excluded && !entry.reason)
      entry.reason = window.prompt('Why is this trial excluded?', 'no building') || '';
    PREP.overrides[key] = entry;
    markDirty();
    onChange();
  });
  bar.appendChild(exclude);

  if (D.pairs.length > 1)
    bar.appendChild(el('span', 'stat', 'step through these to confirm the crops and ' +
      'registration hold \u2014 the cameras do not move, so one setting usually ' +
      'serves every trial'));
  return bar;
}

function viewCrops() {
  const box = el('div');
  box.appendChild(el('p', 'sub',
    'Drag a corner to move it. The depth crop bounds the sand the depth camera sees; ' +
    'the video crop bounds the arena in Pi coordinates. They are independent \u2014 ' +
    'the depth camera does not always see the whole video field.'));
  const bar = trialSelector(() => show(current));
  const reset = el('button', 'act', 'Reset both crops');
  reset.addEventListener('click', () => {
    PREP.depth_crop = null;
    PREP.video_crop = null;
    markDirty();
    show(current);
  });
  bar.appendChild(el('span', 'spacer'));
  bar.appendChild(reset);
  box.appendChild(bar);

  const grid = el('div', 'grid cols-2 threequarter');
  box.appendChild(grid);
  const changeSlot = el('div');
  box.appendChild(changeSlot);
  const pair = D.pairs[TRIAL_INDEX] || {};

  [['depth', 'Depth crop', pair.depthJpg, D.depthSize,
    'orange bounds the tray the depth camera sees'],
   ['video', 'Video crop', pair.piJpg, D.videoSize,
    'blue bounds the arena; the dashed orange outline is the depth crop mapped ' +
    'into Pi coordinates, and should sit inside it']
  ].forEach(([which, title, src, size, hint]) => {
    const fig = el('figure');
    const stage = el('div', 'stage');
    if (src) { const img = el('img'); img.src = src; stage.appendChild(img); }
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 ' + size[0] + ' ' + size[1]);
    svg.setAttribute('preserveAspectRatio', 'none');
    stage.appendChild(svg);
    fig.appendChild(stage);
    fig.appendChild(el('figcaption', null, '<b>' + title + '</b> \u2014 ' + hint));
    grid.appendChild(fig);
    mountCrop(svg, which, size,
              which === 'depth' ? () => repaintCropChange() : null);
  });

  drawCropChange(changeSlot, pair.trial);
  return box;
}

// The crop drawn on the trial's own total change, so what it excludes can be
// judged against the data rather than against a photograph. Change running up
// to the boundary means the crop is clipping part of the bower.
let CROP_CHANGE = null;     // {values, meta, canvas, caption, trial} once loaded

function drawCropChange(slot, trial) {
  slot.textContent = '';
  CROP_CHANGE = null;
  const days = D.days.filter(d => String(d.trial) === String(trial));
  if (days.length < 1) return;
  slot.appendChild(el('h2', null, 'Total change for this trial'));
  const grid = el('div', 'grid cols-2 threequarter');
  slot.appendChild(grid);

  Promise.all([loadDepth(days[0].firstPng),
               loadDepth(days[days.length - 1].lastPng)]).then(([a, b]) => {
    if (!a || !b) return;
    const total = difference(a, b);
    const meta = days[0].firstPng;
    grid.textContent = '';
    grid.appendChild(mapFigure(total, meta,
      '<b>Total change</b> \u2014 ' + days[0].date + ' to ' +
      days[days.length - 1].date + ', with no crop applied', { range: 2 }));

    // the second panel is kept so a crop change repaints it rather than
    // reloading and re-differencing two frames that have not changed
    const fig = el('figure');
    const stage = el('div', 'stage');
    const canvas = el('canvas');
    stage.appendChild(canvas);
    fig.appendChild(stage);
    const caption = el('figcaption');
    fig.appendChild(caption);
    grid.appendChild(fig);
    CROP_CHANGE = { values: total, meta, canvas, caption, trial };
    repaintCropChange();
  });
}

// Recomputing the mask is a point-in-polygon test per pixel, so this runs when
// a corner is released rather than on every mouse move: at 640x480 that is
// three hundred thousand tests, which is fine once and not fine at sixty a
// second.
function repaintCropChange() {
  if (!CROP_CHANGE) return;
  const { values, meta, canvas, caption, trial } = CROP_CHANGE;
  // with the trial, so an override is what gets drawn rather than the
  // project default it was edited away from
  const outside = cropMaskFor(meta, trial);
  let valid = 0, cut = 0;
  for (let i = 0; i < values.length; i++) {
    if (Number.isNaN(values[i])) continue;
    valid++;
    if (outside && outside[i]) cut++;
  }
  paintMap(canvas, values, meta, { range: 2, mark: outside });
  caption.innerHTML = '<b>What the depth crop excludes</b> \u2014 ' +
    (editingOverride() ? 'this trial\u2019s own crop \u00b7 ' : '') +
    (outside ? 'magenta, ' + cut + ' pixels, ' +
               (100 * cut / Math.max(1, valid)).toFixed(1) + '% of the frame. Change ' +
               'running up to the boundary means the crop is clipping the bower.'
             : 'no crop set yet.');
}

function cropTarget() {
  // where an edit lands: this trial's override if it has one, else the project
  return editingOverride() ? PREP.overrides[String(currentTrial())] : PREP;
}

function cropPoints(which, size) {
  const key = which === 'depth' ? 'depth_crop' : 'video_crop';
  const PREP = cropTarget();
  if (!PREP[key]) {
    const inset = 0.12;
    PREP[key] = [[inset, inset], [1-inset, inset], [1-inset, 1-inset], [inset, 1-inset]]
      .map(p => [Math.round(p[0]*size[0]), Math.round(p[1]*size[1])]);
  }
  return PREP[key];
}

// Built once and then only its attributes change. Rebuilding the whole SVG on
// every mousemove, as this did, re-attached the listeners each time and never
// removed them, so a single drag left hundreds of handlers all rebuilding the
// same element. That was the lag.
function mountCrop(svg, which, size, onSettled) {
  const points = cropPoints(which, size);
  const namespace = 'http://www.w3.org/2000/svg';
  const radius = Math.max(4, size[0] / 90);

  const polygon = document.createElementNS(namespace, 'polygon');
  if (which === 'video') polygon.setAttribute('class', 'video');
  svg.appendChild(polygon);

  let derived = null;
  if (which === 'video' && PREP.transform) {
    derived = document.createElementNS(namespace, 'polygon');
    derived.setAttribute('class', 'derived');
    svg.appendChild(derived);
  }

  const handles = points.map((point, index) => {
    const circle = document.createElementNS(namespace, 'circle');
    circle.setAttribute('class', 'handle');
    circle.setAttribute('r', radius);
    circle.dataset.index = String(index);
    svg.appendChild(circle);
    return circle;
  });

  function paint() {
    polygon.setAttribute('points', points.map(p => p.join(',')).join(' '));
    handles.forEach((circle, index) => {
      circle.setAttribute('cx', points[index][0]);
      circle.setAttribute('cy', points[index][1]);
    });
    if (derived && PREP.transform && PREP.depth_crop) {
      const inverse = invert3(PREP.transform);
      if (inverse) {
        derived.setAttribute('points', PREP.depth_crop
          .map(p => applyH(inverse, p).map(Math.round).join(',')).join(' '));
      }
    }
  }

  let dragging = null;
  svg.addEventListener('pointerdown', event => {
    const index = event.target.dataset && event.target.dataset.index;
    if (index === undefined) return;
    dragging = +index;
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener('pointermove', event => {
    if (dragging === null) return;
    const rect = svg.getBoundingClientRect();
    points[dragging] = [
      Math.round((event.clientX - rect.left) / rect.width * size[0]),
      Math.round((event.clientY - rect.top) / rect.height * size[1])];
    paint();                       // attributes only; nothing is rebuilt
  });
  const release = event => {
    if (dragging === null) return;
    dragging = null;
    markDirty();                   // once per drag, not once per pixel
    if (onSettled) onSettled();    // and redraw what the crop excludes
    if (event && event.pointerId !== undefined && svg.hasPointerCapture(event.pointerId))
      svg.releasePointerCapture(event.pointerId);
  };
  svg.addEventListener('pointerup', release);
  svg.addEventListener('pointercancel', release);
  paint();
}

// ------------------------------------------------------------ pixel quality
function viewQuality() {
  const box = el('div');
  if (!D.residualScore) {
    box.appendChild(el('div', 'note', 'No residual data in this bundle.'));
    return box;
  }
  box.appendChild(el('p', 'sub',
    'Residual is the RMS departure from a straight line fitted through a day, so a ' +
    'steadily building pixel scores near zero. Each day is cut against its own median ' +
    'and MAD \u2014 the noise floor drifts \u2014 and the median of those ratios is ' +
    'taken across the project. A pixel is masked when it is usually far from its ' +
    'trend, not when it was bad once.'));

  const bar = el('div', 'bar');
  bar.innerHTML = '<label>Residual k = <b id="kv">' + PREP.residual_k + '</b></label>' +
    '<input type="range" id="kr" min="1" max="8" step="0.5" value="' +
    PREP.residual_k + '">';
  const info = el('span', 'stat');
  bar.appendChild(el('span', 'spacer'));
  bar.appendChild(info);
  box.appendChild(bar);

  const body = el('div');
  box.appendChild(body);

  function draw() {
    loadDepth(D.residualScore).then(score => {
      if (!score) return;
      const meta = D.residualScore;
      const result = scoreMask(score, PREP.residual_k);
      const crop = cropMaskFor(meta, currentTrial());
      body.textContent = '';

      const byTrial = {};
      D.days.forEach(day => { (byTrial[day.trial] = byTrial[day.trial] || []).push(day); });

      Object.keys(byTrial).sort((a, b) => a - b).forEach(trial => {
        const days = byTrial[trial];
        const section = el('div', 'trial');
        section.appendChild(el('h3', null, 'Trial ' + trial +
          '<span>' + days.length + ' days \u00b7 ' + days[0].date + ' to ' +
          days[days.length - 1].date + '</span>'));
        body.appendChild(section);

        const grid = el('div', 'grid cols-5');
        section.appendChild(grid);

        Promise.all([loadDepth(days[0].firstPng),
                     loadDepth(days[days.length - 1].lastPng)]).then(([a, b]) => {
          if (!a || !b) return;
          const total = difference(a, b);
          const shape = days[0].firstPng;

          let valid = 0;
          for (let i = 0; i < total.length; i++) if (!Number.isNaN(total[i])) valid++;
          const count = mask => {
            if (!mask) return 0;
            let n = 0;
            for (let i = 0; i < total.length; i++)
              if (!Number.isNaN(total[i]) && mask[i]) n++;
            return n;
          };
          const both = new Uint8Array(total.length);
          for (let i = 0; i < both.length; i++)
            both[i] = ((result.mark && result.mark[i]) || (crop && crop[i])) ? 1 : 0;
          const nResidual = count(result.mark), nCrop = count(crop), nBoth = count(both);

          grid.appendChild(mapFigure(total, shape,
            '<b>Total change, raw</b> \u2014 first morning to last evening',
            { range: 2 }));
          grid.appendChild(mapFigure(total, shape,
            '<b>Residual, whole project</b> \u2014 k = ' + PREP.residual_k +
            '. Magenta removes ' + nResidual + ' pixels, ' +
            (100 * nResidual / Math.max(1, valid)).toFixed(2) + '%.',
            { range: 2, mark: result.mark }));
          grid.appendChild(mapFigure(total, shape,
            '<b>Tray crop</b> \u2014 ' + (crop
              ? 'magenta removes ' + nCrop + ' pixels, ' +
                (100 * nCrop / Math.max(1, valid)).toFixed(2) + '%.'
              : 'no crop set yet.'), { range: 2, mark: crop }));
          grid.appendChild(mapFigure(total, shape,
            '<b>Both</b> \u2014 together they remove ' + nBoth + ' pixels, ' +
            (100 * nBoth / Math.max(1, valid)).toFixed(2) + '%. Residual adds <b>' +
            (nBoth - nCrop) + '</b> beyond the crop. Blacked out is what the analysis ' +
            'would discard.',
            { range: 2, mark: both, markColour: [0, 0, 0] }));

          const worst = days.map(day => {
            const stats = day.residualStats;
            if (!stats || !stats.median) return null;
            return { date: day.date,
                     cut: stats.median * Math.pow(stats.madFactor, PREP.residual_k),
                     median: stats.median };
          }).filter(Boolean).sort((a, b) => a.median - b.median).reverse().slice(0, 5);
          const panel = el('figure');
          panel.appendChild(el('figcaption', null,
            '<b>Noisiest days</b><br>' + (worst.length
              ? worst.map(w => w.date.slice(5) + ': median ' + w.median.toFixed(3) +
                               ', cut ' + w.cut.toFixed(3)).join('<br>')
              : 'no residual statistics') +
            '<br><br>A mask driven by one bad day shows up here as a single ' +
            'day well above the rest.'));
          grid.appendChild(panel);

          // and the days themselves, so which ones drive the mask is visible
          section.appendChild(el('p', 'stat',
            'Each day\u2019s own change, with that day\u2019s own cut at k = ' +
            PREP.residual_k + ' in magenta. The project mask above is not their ' +
            'union \u2014 a pixel has to be bad on most days to be masked.'));
          const strip = el('div', 'grid cols-8');
          section.appendChild(strip);
          days.forEach(day => {
            Promise.all([loadDepth(day.firstPng), loadDepth(day.lastPng),
                         loadDepth(day.residualPng)]).then(([f, l, r]) => {
              if (!f || !l) return;
              const daily = difference(f, l);
              let mark = null, n = 0, seen = 0;
              const stats = day.residualStats;
              if (r && stats && stats.median) {
                const cut = stats.median * Math.pow(stats.madFactor, PREP.residual_k);
                mark = new Uint8Array(r.length);
                for (let i = 0; i < r.length; i++) {
                  if (Number.isNaN(r[i])) continue;
                  seen++;
                  if (r[i] > cut) { mark[i] = 1; n++; }
                }
              }
              strip.appendChild(mapFigure(daily, day.firstPng,
                '<b>' + day.date.slice(5) + '</b> ' +
                (seen ? (100 * n / seen).toFixed(2) + '% removed' : ''),
                { range: 2, mark: mark }));
            });
          });
        });
      });

      info.innerHTML = 'cut at <b>' + (result.cut || 0).toFixed(2) +
        '</b> \u00b7 median <b>' + (result.median || 0).toFixed(3) +
        '</b> \u00b7 MAD factor <b>\u00d7' + (result.madFactor || 0).toFixed(3) + '</b>';
    });
  }

  bar.querySelector('#kr').addEventListener('input', event => {
    PREP.residual_k = parseFloat(event.target.value);
    bar.querySelector('#kv').textContent = PREP.residual_k;
    markDirty();
    draw();
  });
  draw();
  return box;
}

// -------------------------------------------------------------------- shell
const VIEWS = [['Trial times', viewTimes], ['Registration', viewRegistration],
               ['Crops', viewCrops], ['Pixel quality', viewQuality]];
let current = 0;

function show(index) {
  current = index;
  const tabs = document.getElementById('tabs');
  Array.from(tabs.children).forEach((b, i) =>
    b.setAttribute('aria-selected', String(i === index)));
  const body = document.getElementById('body');
  body.textContent = '';
  body.appendChild(VIEWS[index][1]());
}

function flash(text, kind) {
  const box = document.getElementById('msg');
  box.textContent = '';
  box.appendChild(el('div', 'msg ' + kind, text));
}

// Behind a VPN there is no authenticated identity to read, so the page asks.
// It is remembered in the browser rather than on the server: the same person
// usually works from the same machine, and a field that has to be retyped
// every time ends up empty.
function rememberedName() {
  try { return window.localStorage.getItem('cbc-who') || ''; } catch (e) { return ''; }
}

function storeName(value) {
  try { window.localStorage.setItem('cbc-who', value); } catch (e) { /* private mode */ }
}

function save() {
  // the fit is computed here rather than server-side: the points were picked
  // in the browser and the transform is what they mean
  const complete = POINTS.filter(p => p.pi && p.depth);
  if (complete.length >= 4) {
    const H = homography(complete.map(p => p.pi), complete.map(p => p.depth));
    if (H) {
      const errors = residuals(H, complete.map(p => p.pi), complete.map(p => p.depth));
      const target = cropTarget();
      target.transform = H;
      PREP.points = complete;
      PREP.fit_rms_px = Math.sqrt(errors.reduce((s, e) => s + e*e, 0) / errors.length);
    }
  }
  const whoField = document.getElementById('who');
  const who = (whoField && whoField.value || '').trim();
  if (!who) {
    flash('Put your name in before saving, so the change can be traced back.', 'bad');
    if (whoField) whoField.focus();
    return;
  }
  storeName(who);
  PREP.who = who;
  const button = document.getElementById('save');
  button.disabled = true;
  document.getElementById('state').textContent = 'saving\u2026';
  fetch('save', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(PREP) })
    .then(response => response.json().then(body => ({ ok: response.ok, body })))
    .then(({ ok, body }) => {
      if (!ok) throw new Error(body.error || 'the server refused the change');
      DIRTY = false;
      document.getElementById('state').textContent = 'saved ' +
        (body.updated || '').slice(11, 16);
      flash('Saved.', 'ok');
    })
    .catch(error => {
      button.disabled = false;
      document.getElementById('state').textContent = 'unsaved changes';
      flash('Could not save: ' + error.message, 'bad');
    });
}

function build() {
  document.getElementById('title').textContent = D.projectID;
  document.getElementById('subtitle').textContent =
    'Trial times, registration, crops and the residual filter.';
  document.getElementById('meta').innerHTML =
    [['Tank', D.tankID], ['Analysis', D.analysisID],
     ['Depth', D.depthSize ? D.depthSize.join(' \u00d7 ') : '?'],
     ['Video', D.videoSize ? D.videoSize.join(' \u00d7 ') : '?'],
     ['Days', D.days.length], ['Trials', D.trials.length],
     ['Collected', (D.collected || '').slice(0, 16)]]
    .map(([k, v]) => '<div>' + k + '<b>' + v + '</b></div>').join('');

  if (D.logIssues && D.logIssues.length) {
    const note = el('div', 'note', 'The log parser flagged this project:<ul>' +
      D.logIssues.map(x => '<li>' + x + '</li>').join('') + '</ul>');
    document.getElementById('msg').appendChild(note);
  }

  const tabs = document.getElementById('tabs');
  VIEWS.forEach((view, i) => {
    const button = el('button', null, view[0]);
    button.setAttribute('role', 'tab');
    button.addEventListener('click', () => show(i));
    tabs.appendChild(button);
  });

  const whoField = document.getElementById('who');
  if (whoField) whoField.value = PREP.who || rememberedName();
  document.getElementById('save').addEventListener('click', save);
  document.getElementById('save').disabled = true;
  document.getElementById('state').textContent = PREP.updated
    ? 'last saved ' + PREP.updated.slice(0, 16) + (PREP.who ? ' by ' + PREP.who : '')
    : 'never saved';

  window.addEventListener('beforeunload', event => {
    if (DIRTY) { event.preventDefault(); event.returnValue = ''; }
  });

  show(0);
  document.getElementById('foot').textContent =
    'Collected ' + D.collected + ' \u00b7 page built ' + D.built;
}

loadPayload(() => {
  // the points that produced the saved fit, so a reload shows the work
  POINTS = Array.isArray(PREP.points) ? PREP.points.slice() : [];
  build();
});
