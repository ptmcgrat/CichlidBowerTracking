/* Shared by every page.
 *
 * Depth images carry their values in the pixels rather than as colour, so a
 * page can read a height under the cursor and re-threshold without asking the
 * server. Images are fetched by URL, not embedded, so the browser caches what
 * it has already seen.
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

// Shared by every page, and only some of them can save, so the controls are
// optional rather than assumed.
function markDirty() {
  DIRTY = true;
  const save = document.getElementById('save');
  if (save) save.disabled = false;
  const state = document.getElementById('state');
  if (state) state.textContent = 'unsaved changes';
}



// ----------------------------------------------------------- the transform
// Moved here from the prep page: the summary needs to put cluster events into
// depth coordinates, which is the same arithmetic.
function multiply3(A, B) {
  const out = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    let sum = 0;
    for (let k = 0; k < 3; k++) sum += A[i][k] * B[k][j];
    out[i][j] = sum;
  }
  return out;
}

function invert3(M) {
  const [a,b,c] = M[0], [d,e,f] = M[1], [g,h,i] = M[2];
  const det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g);
  if (!det) return null;
  return [[(e*i-f*h)/det, (c*h-b*i)/det, (b*f-c*e)/det],
          [(f*g-d*i)/det, (a*i-c*g)/det, (c*d-a*f)/det],
          [(d*h-e*g)/det, (b*g-a*h)/det, (a*e-b*d)/det]];
}

function applyH(H, point) {
  const w = H[2][0]*point[0] + H[2][1]*point[1] + H[2][2];
  return [(H[0][0]*point[0] + H[0][1]*point[1] + H[0][2]) / w,
          (H[1][0]*point[0] + H[1][1]*point[1] + H[1][2]) / w];
}

function residuals(H, from, to) {
  return from.map((point, i) => {
    const mapped = applyH(H, point);
    return Math.hypot(mapped[0] - to[i][0], mapped[1] - to[i][1]);
  });
}


// ------------------------------------------------------- per-trial values
// One registration and one pair of crops normally serve a whole project,
// because the cameras do not move. A trial only has its own when somebody made
// it one, and these fall back rather than requiring every trial to be set.
function overrideFor(trial) {
  return (PREP.overrides || {})[String(trial)] || {};
}

function transformFor(trial) {
  return overrideFor(trial).transform || PREP.transform;
}

function depthCropFor(trial) {
  return overrideFor(trial).depth_crop || PREP.depth_crop;
}

function videoCropFor(trial) {
  return overrideFor(trial).video_crop || PREP.video_crop;
}

function isExcluded(trial) {
  return !!overrideFor(trial).excluded;
}

// Trials worth analysing. A trial with no building, or with data too poor to
// use, is excluded rather than silently averaged in.
function activeTrials() {
  const seen = [];
  D.days.forEach(day => {
    if (seen.indexOf(day.trial) < 0 && !isExcluded(day.trial)) seen.push(day.trial);
  });
  return seen.sort((a, b) => a - b);
}

function daysByTrial(includeExcluded) {
  const out = {};
  D.days.forEach(day => {
    if (!includeExcluded && isExcluded(day.trial)) return;
    (out[day.trial] = out[day.trial] || []).push(day);
  });
  return out;
}

// A trial left out of the analysis is invisible on every page that honours
// the exclusion, which is indistinguishable from a trial that was never
// recorded. This says which, and why.
function excludedBanner() {
  const out = [];
  const seen = [];
  D.days.forEach(day => {
    if (seen.indexOf(day.trial) >= 0) return;
    seen.push(day.trial);
    if (isExcluded(day.trial)) out.push(day.trial);
  });
  if (!out.length) return null;
  const reasons = out.map(trial => {
    const reason = overrideFor(trial).reason;
    return 'trial ' + trial + (reason ? ' (' + reason + ')' : '');
  });
  const banner = el('div', 'banner');
  banner.innerHTML = '<span>Excluded from this page: <b>' + reasons.join(', ') +
    '</b>. Unmark it on the prep page to bring it back.</span>';
  return banner;
}

// ----------------------------------------------------------------- the crop
function cropMaskFor(meta, trial) {
  // the crop is in full-resolution depth coordinates; a map may be smaller
  const points = trial === undefined ? PREP.depth_crop : depthCropFor(trial);
  if (!points || points.length < 3) return null;
  const sx = D.depthSize[0] / meta.width, sy = D.depthSize[1] / meta.height;
  const mask = new Uint8Array(meta.width * meta.height);
  for (let y = 0; y < meta.height; y++) {
    const py = (y + 0.5) * sy;
    for (let x = 0; x < meta.width; x++) {
      const px = (x + 0.5) * sx;
      let inside = false;
      for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
        const xi = points[i][0], yi = points[i][1];
        const xj = points[j][0], yj = points[j][1];
        if ((yi > py) !== (yj > py) &&
            px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside;
      }
      if (!inside) mask[y * meta.width + x] = 1;
    }
  }
  return mask;
}

function scoreMask(score, k) {
  const finite = [];
  for (let i = 0; i < score.length; i++)
    if (!Number.isNaN(score[i]) && score[i] > 0) finite.push(score[i]);
  if (!finite.length) return { mark: null };
  finite.sort((a, b) => a - b);
  const logs = finite.map(Math.log);
  const median = logs[logs.length >> 1];
  const deviations = logs.map(v => Math.abs(v - median)).sort((a, b) => a - b);
  const mad = deviations[deviations.length >> 1] * 1.4826;
  const cut = Math.exp(median + k * mad);
  const mark = new Uint8Array(score.length);
  for (let i = 0; i < score.length; i++)
    if (Number.isNaN(score[i]) || score[i] > cut) mark[i] = 1;
  return { mark, cut, median: Math.exp(median), madFactor: Math.exp(mad) };
}

// The endpoints a day actually has, once the saved trial offsets are applied.
//
// An offset moves only the first day's opening frame and the last day's
// closing frame; every intermediate day begins and ends at a lights-on
// boundary that no offset touches. The frames it moves them to were collected
// as boundary candidates, so this needs nothing the bundle does not hold.
function endpointsFor(day, daysInTrial) {
  const times = PREP.trials[String(day.trial)] || PREP.trials['1'] || {};
  let first = day.firstPng, last = day.lastPng;
  const isFirst = daysInTrial[0].index === day.index;
  const isLast = daysInTrial[daysInTrial.length - 1].index === day.index;
  if (isFirst && times.start) {
    const candidate = (D.candidates || []).find(
      c => String(c.trial) === String(day.trial) && c.kind === 'start' &&
           c.offset === times.start);
    if (candidate && candidate.depthPng) first = candidate.depthPng;
  }
  if (isLast && times.stop) {
    const candidate = (D.candidates || []).find(
      c => String(c.trial) === String(day.trial) && c.kind === 'stop' &&
           c.offset === times.stop);
    if (candidate && candidate.depthPng) last = candidate.depthPng;
  }
  return { first, last, moved: (isFirst && times.start) || (isLast && times.stop) };
}

function loadPayload(then) {
  fetch('page.json')
    .then(response => response.json())
    .then(payload => {
      D = payload;
      PREP = payload.prep || {};
      PREP.trials = PREP.trials || {};
      if (PREP.residual_k === undefined || PREP.residual_k === null) PREP.residual_k = 5;
      then();
    })
    .catch(error => {
      const body = document.getElementById('body');
      if (body) body.appendChild(el('div', 'note',
        'Could not load the payload: ' + error.message));
    });
}
