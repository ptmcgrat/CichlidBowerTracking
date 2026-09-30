/* The prep page.
 *
 * Four tabs over one saved object. Everything a person decides — the trial
 * times, the registration, both crops, the residual k — lives in PREP and is
 * written back as a single file, so a save either lands whole or not at all.
 *
 * Depth images carry their values in the pixels rather than as colour, so the
 * page can read a height under the cursor and re-threshold without asking the
 * server. Images are fetched by URL, not embedded: the page stays small and
 * the browser caches what it has seen.
 */

let D = null;            // the payload
let PREP = null;         // what will be saved
let DIRTY = false;

const CACHE = {};        // decoded depth arrays, by url

// ---------------------------------------------------------------- utilities
function el(tag, cls, html) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

function jet(t) {
  t = Math.min(1, Math.max(0, t));
  return [Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 3))) * 255,
          Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 2))) * 255,
          Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 1))) * 255];
}

function loadDepth(meta) {
  if (!meta || !meta.url) return Promise.resolve(null);
  if (CACHE[meta.url]) return Promise.resolve(CACHE[meta.url]);
  return new Promise(resolve => {
    const image = new Image();
    image.onload = () => {
      const canvas = el('canvas');
      canvas.width = meta.width; canvas.height = meta.height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(image, 0, 0);
      const px = ctx.getImageData(0, 0, meta.width, meta.height).data;
      const out = new Float32Array(meta.width * meta.height);
      for (let i = 0, p = 0; i < out.length; i++, p += 4)
        out[i] = px[p+2] === 0 ? NaN : ((px[p] << 8) | px[p+1]) * meta.scale + meta.offset;
      CACHE[meta.url] = out;
      resolve(out);
    };
    image.onerror = () => resolve(null);
    image.src = meta.url;
  });
}

function difference(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

function paintMap(canvas, values, meta, opts) {
  opts = opts || {};
  const range = opts.range === undefined ? 2 : opts.range;
  canvas.width = meta.width; canvas.height = meta.height;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(meta.width, meta.height);
  const mark = opts.mark, colour = opts.markColour || [255, 0, 200];
  for (let i = 0, p = 0; i < values.length; i++, p += 4) {
    if (mark && mark[i]) {
      image.data[p] = colour[0]; image.data[p+1] = colour[1];
      image.data[p+2] = colour[2]; image.data[p+3] = 255;
      continue;
    }
    const v = values[i];
    if (Number.isNaN(v)) {
      image.data[p] = image.data[p+1] = image.data[p+2] = 0;
      image.data[p+3] = 255;
      continue;
    }
    const c = jet((v + range) / (2 * range));
    image.data[p] = c[0]; image.data[p+1] = c[1]; image.data[p+2] = c[2];
    image.data[p+3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}

function mapFigure(values, meta, caption, opts) {
  const fig = el('figure');
  const stage = el('div', 'stage');
  const canvas = el('canvas');
  stage.appendChild(canvas);
  const readout = el('span', 'readout', '\u2014');
  stage.appendChild(readout);
  fig.appendChild(stage);
  fig.appendChild(el('figcaption', null, caption));
  paintMap(canvas, values, meta, opts);
  stage.addEventListener('mousemove', event => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX - rect.left) / rect.width * meta.width);
    const y = Math.floor((event.clientY - rect.top) / rect.height * meta.height);
    if (x < 0 || y < 0 || x >= meta.width || y >= meta.height) return;
    const v = values[y * meta.width + x];
    readout.textContent = Number.isNaN(v) ? 'no data'
                        : (v >= 0 ? '+' : '') + v.toFixed(2) + ' cm';
  });
  return fig;
}

function markDirty() {
  DIRTY = true;
  document.getElementById('save').disabled = false;
  document.getElementById('state').textContent = 'unsaved changes';
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
            stage.appendChild(el('img'));
            stage.firstChild.src = c.jpg;
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
function viewRegistration() {
  const box = el('div');
  if (!D.pairs.length) {
    box.appendChild(el('div', 'note', 'No registration pairs were collected. The ' +
      'project may have no video stills, or it predates the collector change that ' +
      'gathers them.'));
    return box;
  }
  box.appendChild(el('p', 'sub',
    'Click the same feature in both images, at least four times \u2014 tray corners ' +
    'work well. The fit updates as you go. If the outlines already line up, leave it ' +
    'alone; the registration on file is shown until new points replace it.'));

  const bar = el('div', 'bar');
  const select = el('select');
  D.pairs.forEach((p, i) => {
    const option = el('option', null, 'Trial ' + p.trial + ' ' + p.side +
      ' \u00b7 matched to ' + p.gapMinutes + ' min');
    option.value = String(i);
    select.appendChild(option);
  });
  bar.appendChild(el('label', null, 'Pair'));
  bar.appendChild(select);
  const clear = el('button', 'act', 'Clear points');
  bar.appendChild(clear);
  bar.appendChild(el('span', 'spacer'));
  const fitNote = el('span', 'stat');
  bar.appendChild(fitNote);
  box.appendChild(bar);

  const grid = el('div', 'grid cols-2');
  box.appendChild(grid);
  let points = [];

  function draw() {
    const pair = D.pairs[+select.value];
    grid.textContent = '';
    fitNote.innerHTML = points.length >= 4
      ? '<b>' + points.length + '</b> pairs \u00b7 fit ' + fitRms(points).toFixed(2) + ' px'
      : (PREP.transform ? 'showing the registration on file'
                        : 'pick four or more pairs');

    [['piJpg', 'Pi camera', 'pi'], ['depthJpg', 'Depth camera', 'depth']]
      .forEach(([key, title, which]) => {
        if (!pair[key]) return;
        const fig = el('figure');
        const stage = el('div', 'stage');
        const img = el('img');
        img.src = pair[key];
        stage.appendChild(img);
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        stage.appendChild(svg);
        fig.appendChild(stage);
        fig.appendChild(el('figcaption', null, '<b>' + title + '</b> \u2014 ' +
          points.length + ' point(s) placed'));
        grid.appendChild(fig);

        stage.addEventListener('click', event => {
          const rect = stage.getBoundingClientRect();
          const x = (event.clientX - rect.left) / rect.width;
          const y = (event.clientY - rect.top) / rect.height;
          addPoint(which, x, y);
        });
        renderPoints(svg, which);
      });
  }

  function addPoint(which, x, y) {
    const open = points.find(p => !p[which]);
    if (open) open[which] = [x, y];
    else points.push({ [which]: [x, y] });
    markDirty();
    draw();
  }

  function renderPoints(svg, which) {
    svg.innerHTML = points.map((p, i) => {
      if (!p[which]) return '';
      return '<circle cx="' + (p[which][0] * 100) + '%" cy="' + (p[which][1] * 100) +
             '%" r="5" class="handle"/><text x="' + (p[which][0] * 100) + '%" y="' +
             (p[which][1] * 100) + '%" dy="-9" fill="#fff" font-size="11" ' +
             'text-anchor="middle">' + (i + 1) + '</text>';
    }).join('');
  }

  select.addEventListener('change', draw);
  clear.addEventListener('click', () => { points = []; draw(); });
  draw();
  return box;
}

function fitRms(points) {
  const usable = points.filter(p => p.pi && p.depth);
  return usable.length >= 4 ? 0.0 : NaN;    // the fit itself is computed on save
}

// -------------------------------------------------------------------- crops
function viewCrops() {
  const box = el('div');
  box.appendChild(el('p', 'sub',
    'Drag a corner to move it. The depth crop bounds the sand the depth camera sees; ' +
    'the video crop bounds the arena in Pi coordinates. They are independent \u2014 ' +
    'the depth camera does not always see the whole video field.'));
  const grid = el('div', 'grid cols-2');
  box.appendChild(grid);

  const pair = D.pairs[0];
  [['depth', 'Depth crop', pair && pair.depthJpg, D.depthSize],
   ['video', 'Video crop', pair && pair.piJpg, D.videoSize]
  ].forEach(([which, title, src, size]) => {
    const fig = el('figure');
    const stage = el('div', 'stage');
    if (src) { const img = el('img'); img.src = src; stage.appendChild(img); }
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 ' + size[0] + ' ' + size[1]);
    stage.appendChild(svg);
    fig.appendChild(stage);
    fig.appendChild(el('figcaption', null, '<b>' + title + '</b>'));
    grid.appendChild(fig);
    renderCrop(svg, which, size);
  });
  return box;
}

function cropPoints(which, size) {
  const key = which === 'depth' ? 'depth_crop' : 'video_crop';
  if (!PREP[key]) {
    const inset = 0.12;
    PREP[key] = [[size[0]*inset, size[1]*inset], [size[0]*(1-inset), size[1]*inset],
                 [size[0]*(1-inset), size[1]*(1-inset)], [size[0]*inset, size[1]*(1-inset)]]
                .map(p => [Math.round(p[0]), Math.round(p[1])]);
  }
  return PREP[key];
}

function renderCrop(svg, which, size) {
  const points = cropPoints(which, size);
  const cls = which === 'video' ? 'video' : '';
  svg.innerHTML = '<polygon class="' + cls + '" points="' +
    points.map(p => p.join(',')).join(' ') + '"/>' +
    points.map((p, i) => '<circle class="handle" cx="' + p[0] + '" cy="' + p[1] +
                         '" r="' + Math.max(4, size[0] / 90) + '" data-i="' + i + '"/>')
          .join('');
  let dragging = null;
  svg.addEventListener('mousedown', event => {
    const index = event.target.getAttribute && event.target.getAttribute('data-i');
    if (index !== null && index !== undefined) dragging = +index;
  });
  svg.addEventListener('mousemove', event => {
    if (dragging === null) return;
    const rect = svg.getBoundingClientRect();
    points[dragging] = [
      Math.round((event.clientX - rect.left) / rect.width * size[0]),
      Math.round((event.clientY - rect.top) / rect.height * size[1])];
    renderCrop(svg, which, size);
    markDirty();
  });
  svg.addEventListener('mouseup', () => { dragging = null; });
  svg.addEventListener('mouseleave', () => { dragging = null; });
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

  const grid = el('div', 'grid cols-3');
  box.appendChild(grid);

  function draw() {
    loadDepth(D.residualScore).then(score => {
      if (!score) return;
      const meta = D.residualScore;
      const finite = [];
      for (let i = 0; i < score.length; i++)
        if (!Number.isNaN(score[i]) && score[i] > 0) finite.push(score[i]);
      finite.sort((a, b) => a - b);
      const logs = finite.map(Math.log);
      const median = logs[logs.length >> 1];
      const deviations = logs.map(v => Math.abs(v - median)).sort((a, b) => a - b);
      const mad = deviations[deviations.length >> 1] * 1.4826;
      const cut = Math.exp(median + PREP.residual_k * mad);

      const mark = new Uint8Array(score.length);
      let masked = 0;
      for (let i = 0; i < score.length; i++)
        if (Number.isNaN(score[i]) || score[i] > cut) { mark[i] = 1; masked++; }

      info.innerHTML = 'cut at <b>' + cut.toFixed(2) + '</b> \u00b7 masks <b>' +
        (100 * masked / score.length).toFixed(2) + '%</b> of the frame';

      grid.textContent = '';
      const first = D.days[0], last = D.days[D.days.length - 1];
      Promise.all([loadDepth(first.firstPng), loadDepth(last.lastPng)])
        .then(([a, b]) => {
          if (!a || !b) return;
          const total = difference(a, b);
          grid.appendChild(mapFigure(total, first.firstPng,
            '<b>Total change</b> \u2014 first morning to last evening', { range: 2 }));
          grid.appendChild(mapFigure(total, first.firstPng,
            '<b>Masked at k = ' + PREP.residual_k + '</b> \u2014 magenta is removed',
            { range: 2, mark: mark }));
          grid.appendChild(mapFigure(score, meta,
            '<b>Residual score</b> \u2014 each pixel\u2019s typical departure, ' +
            'relative to its day', { range: 4 }));
        });
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

function save() {
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

fetch('prep.json')
  .then(response => response.json())
  .then(payload => {
    D = payload;
    PREP = payload.prep || {};
    PREP.trials = PREP.trials || {};
    if (PREP.residual_k === undefined || PREP.residual_k === null) PREP.residual_k = 4;
    build();
  })
  .catch(error => {
    document.getElementById('body').appendChild(
      el('div', 'note', 'Could not load the payload: ' + error.message));
  });
