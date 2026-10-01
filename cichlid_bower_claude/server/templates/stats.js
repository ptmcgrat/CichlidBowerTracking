/* Cluster statistics.
 *
 * Five views over the same events, answering questions the maps cannot: when
 * things happen, how confident the classifier was, and whether the events it
 * was unsure about look like the ones it was sure about.
 *
 * There are no labels to check accuracy against, so the confidence question is
 * answered by a prior instead: real building happens at the bower. If
 * low-confidence building events land where high-confidence ones do, they are
 * probably right and the threshold is discarding good data.
 */

let EVENTS = null;
let CONFIDENCE = 0.67;

const GROUPS = [
  ['Building', ['c', 'p', 'b'], [111, 178, 232]],
  ['Feeding', ['f', 't', 'm'], [242, 163, 60]],
  ['Spawning', ['s'], [232, 132, 168]],
  ['Other', ['d', 'o'], [90, 168, 122]],
  ['No fish', ['x'], [138, 144, 153]],
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
      EVENTS = {
        n: packed.n, bids: packed.bids, labels: packed.labels, code,
        x: unpackColumn(packed.x, Uint16Array),
        y: unpackColumn(packed.y, Uint16Array),
        bid: unpackColumn(packed.bid, Uint8Array),
        prob: unpackColumn(packed.prob, Uint8Array),
        flags: unpackColumn(packed.flags, Uint8Array),
        day: unpackColumn(packed.day, Uint16Array),
        trial: unpackColumn(packed.trial, Uint8Array),
        hour: unpackColumn(packed.hour, Uint8Array),
        summary: packed.summary,
      };
      return EVENTS;
    });
}

// ------------------------------------------------------------------- charts
function chart(width, height, body, caption) {
  const figure = el('figure');
  const holder = el('div');
  holder.innerHTML = '<svg viewBox="0 0 ' + width + ' ' + height +
                     '" preserveAspectRatio="none" style="width:100%;display:block">' +
                     body + '</svg>';
  figure.appendChild(holder);
  figure.appendChild(el('figcaption', null, caption));
  return figure;
}

function axis(width, height, pad, labels, maxValue, units) {
  let out = '<line x1="' + pad + '" y1="' + (height - pad) + '" x2="' + (width - pad) +
            '" y2="' + (height - pad) + '" stroke="#262d38"/>';
  out += '<text x="' + pad + '" y="' + (pad - 10) + '" fill="#93a0b0" font-size="12">' +
         units + '</text>';
  labels.forEach(([x, text]) => {
    out += '<text x="' + x + '" y="' + (height - pad + 15) + '" fill="#93a0b0" ' +
           'font-size="11" text-anchor="middle">' + text + '</text>';
  });
  return out;
}

function stackedBars(counts, categories, width, height, pad, labels, units) {
  const columns = counts[categories[0].key].length;
  let peak = 0;
  for (let i = 0; i < columns; i++) {
    let total = 0;
    categories.forEach(category => { total += counts[category.key][i]; });
    if (total > peak) peak = total;
  }
  const barWidth = (width - 2 * pad) / columns;
  let out = '';
  for (let i = 0; i < columns; i++) {
    let base = height - pad;
    categories.forEach(category => {
      const value = counts[category.key][i];
      if (!value || !peak) return;
      const size = (value / peak) * (height - 2 * pad);
      out += '<rect x="' + (pad + i * barWidth + 1) + '" y="' + (base - size) +
             '" width="' + Math.max(1, barWidth - 2) + '" height="' + size +
             '" fill="rgb(' + category.colour.join(',') + ')" opacity="0.88">' +
             '<title>' + category.name + ', ' + labels[i][1] + ': ' + value +
             '</title></rect>';
      base -= size;
    });
  }
  return out + axis(width, height, pad, labels, peak, units + ' \u00b7 peak ' + peak);
}

// --------------------------------------------------------------------- views
function selected(index, cut) {
  return (EVENTS.flags[index] & 1) && EVENTS.bid[index] !== 255 &&
         EVENTS.prob[index] >= cut;
}

function viewByHour() {
  const cut = Math.round(CONFIDENCE * 255);
  const counts = {};
  const categories = GROUPS.map(([name, bids, colour]) => {
    counts[name] = new Float32Array(24);
    return { key: name, name, colour,
             codes: new Set(bids.map(bid => EVENTS.code[bid])) };
  });
  for (let i = 0; i < EVENTS.n; i++) {
    if (!selected(i, cut)) continue;
    const category = categories.find(c => c.codes.has(EVENTS.bid[i]));
    if (category) counts[category.key][EVENTS.hour[i]] += 1;
  }
  const labels = [];
  for (let hour = 0; hour < 24; hour++)
    labels.push([38 + (1000 - 76) / 24 * (hour + 0.5), hour % 3 === 0 ? String(hour) : '']);
  return chart(1000, 240,
    stackedBars(counts, categories, 1000, 240, 38, labels, 'events by hour of day'),
    'When events happen. Recording runs 8am to 6pm, so anything outside that is a ' +
    'clock or a classification problem rather than a fish.');
}

function viewByDay() {
  const cut = Math.round(CONFIDENCE * 255);
  const days = D.days.length;
  const counts = {};
  const categories = GROUPS.map(([name, bids, colour]) => {
    counts[name] = new Float32Array(days);
    return { key: name, name, colour,
             codes: new Set(bids.map(bid => EVENTS.code[bid])) };
  });
  for (let i = 0; i < EVENTS.n; i++) {
    if (!selected(i, cut)) continue;
    const day = EVENTS.day[i];
    if (day >= days) continue;
    const category = categories.find(c => c.codes.has(EVENTS.bid[i]));
    if (category) counts[category.key][day] += 1;
  }
  const labels = D.days.map((day, i) =>
    [38 + (1000 - 76) / days * (i + 0.5), i % 4 === 0 ? day.date.slice(5) : '']);
  return chart(1000, 240,
    stackedBars(counts, categories, 1000, 240, 38, labels, 'events by day'),
    'Whether building ramps up, holds steady or stops. A trial boundary shows as a ' +
    'step rather than a gap, because the days either side are still days.');
}

function viewConfidence() {
  // the whole distribution, not only what survives the cut: the question is
  // what the cut is discarding
  const bins = 20;
  const counts = {};
  const categories = GROUPS.map(([name, bids, colour]) => {
    counts[name] = new Float32Array(bins);
    return { key: name, name, colour,
             codes: new Set(bids.map(bid => EVENTS.code[bid])) };
  });
  for (let i = 0; i < EVENTS.n; i++) {
    if (!(EVENTS.flags[i] & 1) || EVENTS.bid[i] === 255) continue;
    const bin = Math.min(bins - 1, Math.floor(EVENTS.prob[i] / 255 * bins));
    const category = categories.find(c => c.codes.has(EVENTS.bid[i]));
    if (category) counts[category.key][bin] += 1;
  }
  const labels = [];
  for (let i = 0; i < bins; i++)
    labels.push([38 + (1000 - 76) / bins * (i + 0.5),
                 i % 4 === 0 ? (i / bins).toFixed(2) : '']);
  let body = stackedBars(counts, categories, 1000, 240, 38, labels,
                         'classifier confidence');
  const x = 38 + (1000 - 76) * CONFIDENCE;
  body += '<line x1="' + x + '" y1="20" x2="' + x + '" y2="220" stroke="#f2a33c" ' +
          'stroke-width="1.5" stroke-dasharray="4 4"/>' +
          '<text x="' + (x + 5) + '" y="32" fill="#f2a33c" font-size="11">cut</text>';
  return chart(1000, 240, body,
    'Where the classifier is unsure, and about what. Low confidence concentrated in ' +
    'the noise classes matters much less than low confidence about building.');
}

// The question with no labels to answer it: are the events the classifier was
// unsure about in the same places as the ones it was sure about? Real building
// happens at the bower, so if low-confidence building scatters across the tank
// it is noise, and if it lands on the bower the threshold is discarding data.
function viewCoherence() {
  const codes = ['c', 'p'].map(bid => EVENTS.code[bid]);
  const high = [];
  for (let i = 0; i < EVENTS.n; i++) {
    if (!(EVENTS.flags[i] & 1) || !codes.includes(EVENTS.bid[i])) continue;
    if (EVENTS.prob[i] < 0.9 * 255) continue;
    high.push(i);
  }
  if (high.length < 30)
    return chart(1000, 120, '', 'Too few confident building events to compare against.');

  let cx = 0, cy = 0;
  high.forEach(i => { cx += EVENTS.x[i]; cy += EVENTS.y[i]; });
  cx /= high.length; cy /= high.length;

  const deciles = 10;
  const distances = Array.from({ length: deciles }, () => []);
  for (let i = 0; i < EVENTS.n; i++) {
    if (!(EVENTS.flags[i] & 1) || !codes.includes(EVENTS.bid[i])) continue;
    const bin = Math.min(deciles - 1, Math.floor(EVENTS.prob[i] / 255 * deciles));
    distances[bin].push(Math.hypot(EVENTS.x[i] - cx, EVENTS.y[i] - cy));
  }
  const medians = distances.map(list => {
    if (!list.length) return 0;
    const sorted = list.slice().sort((a, b) => a - b);
    return sorted[sorted.length >> 1];
  });

  // what a scatter with no spatial structure would look like, for comparison
  const random = [];
  for (let i = 0; i < EVENTS.n; i += 7) {
    if (!(EVENTS.flags[i] & 1)) continue;
    random.push(Math.hypot(EVENTS.x[i] - cx, EVENTS.y[i] - cy));
  }
  random.sort((a, b) => a - b);
  const chance = random.length ? random[random.length >> 1] : 0;

  const width = 1000, height = 260, pad = 44;
  const peak = Math.max(chance, ...medians) * 1.15 || 1;
  const barWidth = (width - 2 * pad) / deciles;
  let body = '';
  medians.forEach((value, i) => {
    const size = (value / peak) * (height - 2 * pad);
    body += '<rect x="' + (pad + i * barWidth + 2) + '" y="' + (height - pad - size) +
            '" width="' + (barWidth - 4) + '" height="' + size +
            '" fill="rgb(111,178,232)" opacity="0.85"><title>confidence ' +
            (i / deciles).toFixed(1) + '\u2013' + ((i + 1) / deciles).toFixed(1) +
            ': median ' + value.toFixed(0) + ' px from the bower, ' +
            distances[i].length + ' events</title></rect>';
  });
  const chanceY = height - pad - (chance / peak) * (height - 2 * pad);
  body += '<line x1="' + pad + '" y1="' + chanceY + '" x2="' + (width - pad) +
          '" y2="' + chanceY + '" stroke="#e0693f" stroke-dasharray="5 4"/>' +
          '<text x="' + (width - pad) + '" y="' + (chanceY - 6) +
          '" fill="#e0693f" font-size="11" text-anchor="end">all events, any class</text>';
  const labels = medians.map((_, i) =>
    [pad + barWidth * (i + 0.5), (i / deciles).toFixed(1)]);
  body += axis(width, height, pad, labels, peak,
               'median distance from the bower, pixels');

  return chart(width, height, body,
    'Building events by confidence, against how far they land from where the ' +
    'confident ones are. Bars near the dashed line are no better placed than a ' +
    'random event, so that confidence band is noise. Bars well below it are landing ' +
    'on the bower, which means the classifier was right and unsure.');
}

function viewNoClip() {
  const width = D.videoSize[0], height = D.videoSize[1];
  const figure = el('figure');
  const stage = el('div', 'stage');
  const canvas = el('canvas');
  const W = 520, H = Math.round(520 * height / width);
  canvas.width = W; canvas.height = H;
  stage.appendChild(canvas);
  figure.appendChild(stage);
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0b0d11';
  ctx.fillRect(0, 0, W, H);
  let count = 0;
  ctx.fillStyle = 'rgb(224,105,63)';
  ctx.globalAlpha = 0.35;
  for (let i = 0; i < EVENTS.n; i++) {
    if (EVENTS.flags[i] & 1) continue;
    count++;
    ctx.fillRect(EVENTS.x[i] / width * W, EVENTS.y[i] / height * H, 1.5, 1.5);
  }
  ctx.globalAlpha = 1;
  figure.appendChild(el('figcaption', null,
    '<b>' + count + ' events with no clip</b>, so nothing to classify. These should ' +
    'form a band around the frame edge, where a two-second clip could not be cut. ' +
    'Anything in the middle of the tank is a different problem.'));
  return figure;
}

const VIEWS = [['By hour', viewByHour], ['By day', viewByDay],
               ['Confidence', viewConfidence], ['Is low confidence wrong?', viewCoherence],
               ['No clip', viewNoClip]];

function draw() {
  const body = document.getElementById('body');
  body.textContent = '';
  VIEWS.forEach(([name, render]) => {
    body.appendChild(el('h2', null, name));
    body.appendChild(render());
  });
}

function build() {
  document.getElementById('title').textContent = D.projectID;
  document.getElementById('subtitle').textContent =
    'When events happen, and how much the classifier can be believed.';
  const summary = EVENTS.summary || {};
  document.getElementById('meta').innerHTML =
    [['Tank', D.tankID], ['Events', EVENTS.n], ['No clip', summary.noClip || 0],
     ['Days', D.days.length], ['Trials', D.trials.length]]
    .map(([k, v]) => '<div>' + k + '<b>' + v + '</b></div>').join('');

  const bar = document.getElementById('controls');
  bar.innerHTML =
    '<label>Confidence \u2265 <b id="cv">' + CONFIDENCE.toFixed(2) + '</b></label>' +
    '<input type="range" id="cr" min="0" max="0.99" step="0.01" value="' +
      CONFIDENCE + '">' +
    '<span class="spacer"></span><span class="stat">the confidence chart shows the ' +
    'whole distribution, so the cut is drawn on it rather than applied</span>';
  let pending = null;
  bar.querySelector('#cr').addEventListener('input', event => {
    CONFIDENCE = parseFloat(event.target.value);
    bar.querySelector('#cv').textContent = CONFIDENCE.toFixed(2);
    if (pending) clearTimeout(pending);
    pending = setTimeout(draw, 160);
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
