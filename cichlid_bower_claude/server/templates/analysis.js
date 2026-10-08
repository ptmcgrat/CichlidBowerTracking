/* Across projects, grouped by category.
 *
 * One question at a time. This page starts with where spawning happens
 * relative to the sand the fish moved: a category that spawns over its castle
 * and one that spawns in its pit are doing different things with the same
 * behaviour, and that difference should be visible without reading
 * thirty-one summary pages.
 *
 * Every number here is computed on the server and cached per project, because
 * the alternative is a browser holding a gigabyte of depth maps.
 */

let DATA = null;
let MIN_SPAWNS = 20;     // a trial with a handful of spawns is not evidence

const CASTLE = [111, 178, 232];   // sand added
const PIT = [242, 163, 60];       // sand taken away

function el(tag, cls, html) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (html !== undefined) node.innerHTML = html;
  return node;
}

// A trial counts when it has a registration, enough spawns to mean something,
// and was not excluded. Silently averaging in the others is how a comparison
// becomes a comparison of data quality.
function usableTrials() {
  const out = [];
  (DATA.projects || []).forEach(project => {
    (project.trials || []).forEach(trial => {
      if (trial.excluded || !trial.spawnDepthHistogram) return;
      if ((trial.spawnsMeasured || 0) < MIN_SPAWNS) return;
      out.push({ project: project.projectID,
                 category: project.category || 'uncategorised',
                 tank: project.tankID, trial: trial.trial,
                 histogram: trial.spawnDepthHistogram,
                 n: trial.spawnsMeasured,
                 median: trial.spawnDepthMedian,
                 overCastle: trial.spawnOverCastle });
    });
  });
  return out;
}

function byCategory(trials) {
  const groups = {};
  trials.forEach(trial => {
    (groups[trial.category] = groups[trial.category] || []).push(trial);
  });
  return groups;
}

function pooled(trials, bins) {
  const counts = new Float64Array(bins);
  trials.forEach(trial => {
    for (let i = 0; i < bins; i++) counts[i] += trial.histogram[i] || 0;
  });
  return counts;
}

function centres(span, bins) {
  const width = 2 * span / bins;
  const out = [];
  for (let i = 0; i < bins; i++) out.push(-span + width * (i + 0.5));
  return out;
}

function quantileOf(counts, xs, fraction) {
  let total = 0;
  for (let i = 0; i < counts.length; i++) total += counts[i];
  if (!total) return null;
  let seen = 0;
  for (let i = 0; i < counts.length; i++) {
    seen += counts[i];
    if (seen >= total * fraction) return xs[i];
  }
  return xs[xs.length - 1];
}

// A ridgeline: one filled distribution per category, on a shared axis, so the
// comparison is a matter of looking down the column rather than holding two
// numbers in your head. Each is scaled to its own peak, because categories
// differ in how many spawns they have and the question is shape, not count.
function ridgeline(groups, span, bins, zoom) {
  const names = Object.keys(groups).sort();
  const xs = centres(span, bins);
  const rowHeight = 86;
  const pad = { left: 128, right: 20, top: 26, bottom: 36 };
  const width = 900;
  const height = pad.top + pad.bottom + names.length * rowHeight;
  const plotW = width - pad.left - pad.right;
  const x = value => pad.left + (value + zoom) / (2 * zoom) * plotW;

  let body = '';
  // the zero line is the whole reference: left of it the fish spawned over
  // sand it had removed, right of it over sand it had piled up
  const zero = x(0);
  body += '<line x1="' + zero + '" y1="' + (pad.top - 6) + '" x2="' + zero +
          '" y2="' + (height - pad.bottom) + '" stroke="#5c6673" ' +
          'stroke-dasharray="4 4"/>';
  body += '<text x="' + zero + '" y="' + (pad.top - 11) +
          '" fill="#93a0b0" font-size="10" text-anchor="middle">flat sand</text>';
  body += '<text x="' + (pad.left + 4) + '" y="' + (pad.top - 11) +
          '" fill="rgb(' + PIT.join(',') + ')" font-size="10">← over the pit</text>';
  body += '<text x="' + (width - pad.right - 4) + '" y="' + (pad.top - 11) +
          '" fill="rgb(' + CASTLE.join(',') + ')" font-size="10" text-anchor="end">' +
          'over the castle →</text>';

  names.forEach((name, row) => {
    const trials = groups[name];
    const counts = pooled(trials, bins);
    let peak = 0, total = 0;
    for (let i = 0; i < bins; i++) {
      if (counts[i] > peak) peak = counts[i];
      total += counts[i];
    }
    const base = pad.top + row * rowHeight + rowHeight - 24;
    const top = pad.top + row * rowHeight + 6;
    const amplitude = base - top;

    body += '<line x1="' + pad.left + '" y1="' + base + '" x2="' +
            (width - pad.right) + '" y2="' + base + '" stroke="#262d38"/>';

    if (peak > 0) {
      // drawn as two fills meeting at zero, so the side a category sits on is
      // carried by colour as well as position
      [['pit', PIT, v => v < 0], ['castle', CASTLE, v => v >= 0]]
        .forEach(([, colour, test]) => {
          let path = 'M' + x(test(-span) ? -zoom : 0) + ',' + base;
          let any = false;
          xs.forEach((value, i) => {
            if (!test(value)) return;
            path += ' L' + x(Math.max(-zoom, Math.min(zoom, value))) + ',' +
                    (base - amplitude * counts[i] / peak);
            any = true;
          });
          if (!any) return;
          path += ' L' + x(test(-span) ? 0 : zoom) + ',' + base + ' Z';
          body += '<path d="' + path + '" fill="rgb(' + colour.join(',') +
                  ')" fill-opacity="0.42" stroke="rgb(' + colour.join(',') +
                  ')" stroke-width="1.2" stroke-opacity="0.8"/>';
        });
    }

    // every trial's own median as a tick, so a category that looks tidy only
    // because one project dominates it does not look tidy here
    trials.forEach(trial => {
      if (trial.median === undefined || trial.median === null) return;
      const at = x(Math.max(-zoom, Math.min(zoom, trial.median)));
      body += '<line x1="' + at + '" y1="' + (base + 2) + '" x2="' + at +
              '" y2="' + (base + 11) + '" stroke="#cfd6df" stroke-width="1.4" ' +
              'stroke-opacity="0.8"><title>' + trial.project + ' trial ' +
              trial.trial + ' · ' + trial.n + ' spawns · median ' +
              trial.median.toFixed(2) + ' cm</title></line>';
    });

    const median = quantileOf(counts, xs, 0.5);
    if (median !== null) {
      const at = x(Math.max(-zoom, Math.min(zoom, median)));
      body += '<line x1="' + at + '" y1="' + top + '" x2="' + at + '" y2="' +
              base + '" stroke="#e8ecf1" stroke-width="1.6"/>';
    }

    let over = 0;
    for (let i = 0; i < bins; i++) if (xs[i] > 0) over += counts[i];
    body += '<text x="' + (pad.left - 12) + '" y="' + (base - 22) +
            '" fill="#e8ecf1" font-size="12" font-weight="600" text-anchor="end">' +
            name + '</text>';
    body += '<text x="' + (pad.left - 12) + '" y="' + (base - 8) +
            '" fill="#93a0b0" font-size="10" text-anchor="end">' +
            trials.length + ' trials · ' + total.toLocaleString() +
            ' spawns</text>';
    body += '<text x="' + (pad.left - 12) + '" y="' + (base + 6) +
            '" fill="#93a0b0" font-size="10" text-anchor="end">median ' +
            (median === null ? '—' : (median >= 0 ? '+' : '') +
             median.toFixed(2) + ' cm') + '</text>';
    body += '<text x="' + (width - pad.right) + '" y="' + (base - 8) +
            '" fill="#93a0b0" font-size="10" text-anchor="end">' +
            (total ? (100 * over / total).toFixed(0) : '0') +
            '% over the castle</text>';
  });

  for (let tick = -Math.floor(zoom); tick <= Math.floor(zoom); tick++) {
    body += '<text x="' + x(tick) + '" y="' + (height - 12) +
            '" fill="#93a0b0" font-size="10" text-anchor="middle">' +
            (tick > 0 ? '+' : '') + tick + '</text>';
  }
  body += '<text x="' + (width / 2) + '" y="' + (height - 1) +
          '" fill="#6c7684" font-size="10" text-anchor="middle">' +
          'depth change under each spawn, cm</text>';

  const holder = el('div');
  holder.innerHTML = '<svg viewBox="0 0 ' + width + ' ' + height +
                     '" style="width:100%;display:block">' + body + '</svg>';
  return holder;
}

function table(groups) {
  const names = Object.keys(groups).sort();
  const rows = names.map(name => {
    const trials = groups[name];
    const counts = pooled(trials, DATA.projects[0].depthBins || 64);
    const xs = centres(DATA.projects[0].depthSpan || 8,
                       DATA.projects[0].depthBins || 64);
    let total = 0, over = 0;
    for (let i = 0; i < counts.length; i++) {
      total += counts[i];
      if (xs[i] > 0) over += counts[i];
    }
    const projects = {};
    trials.forEach(t => { projects[t.project] = 1; });
    const medians = trials.map(t => t.median).filter(m => m !== null &&
                                                     m !== undefined)
                          .sort((a, b) => a - b);
    const spread = medians.length
      ? medians[0].toFixed(2) + ' to ' + medians[medians.length - 1].toFixed(2)
      : '—';
    return '<tr><td><b>' + name + '</b></td><td>' +
           Object.keys(projects).length + '</td><td>' + trials.length +
           '</td><td>' + total.toLocaleString() + '</td><td>' +
           (quantileOf(counts, xs, 0.25) || 0).toFixed(2) + '</td><td>' +
           (quantileOf(counts, xs, 0.5) || 0).toFixed(2) + '</td><td>' +
           (quantileOf(counts, xs, 0.75) || 0).toFixed(2) + '</td><td>' +
           (total ? (100 * over / total).toFixed(0) : '0') + '%</td><td>' +
           spread + '</td></tr>';
  });
  const node = el('table', 'points');
  node.innerHTML = '<tr><th>category</th><th>projects</th><th>trials</th>' +
    '<th>spawns</th><th>lower quartile</th><th>median</th><th>upper quartile</th>' +
    '<th>over castle</th><th>trial medians</th></tr>' + rows.join('');
  return node;
}

function draw() {
  const body = document.getElementById('body');
  body.textContent = '';

  const trials = usableTrials();
  if (!trials.length) {
    body.appendChild(el('div', 'note',
      'No trial has a registration and at least ' + MIN_SPAWNS + ' measured ' +
      'spawns yet. Spawn depth needs a registration, so the events can be ' +
      'placed on the depth map, and a crop, so the sand under them is real.'));
    return;
  }

  const groups = byCategory(trials);
  const span = DATA.projects[0].depthSpan || 8;
  const bins = DATA.projects[0].depthBins || 64;

  // zoom to where the spawns actually are, rather than the full stored span,
  // which is wide enough to hold outliers nobody wants to look at
  const all = pooled(trials, bins);
  const xs = centres(span, bins);
  const low = quantileOf(all, xs, 0.01), high = quantileOf(all, xs, 0.99);
  const zoom = Math.max(1, Math.ceil(Math.max(Math.abs(low), Math.abs(high))));

  body.appendChild(el('h2', null, 'Where spawning happens'));
  body.appendChild(el('p', 'sub',
    'For every spawn, the depth change under it on the day it happened, pooled ' +
    'by category. Each distribution is scaled to its own peak, so the comparison ' +
    'is of shape rather than of how many spawns a category has. The white line ' +
    'is the pooled median; the ticks below each curve are the individual trial ' +
    'medians, so a category carried by one project does not read as agreement.'));
  body.appendChild(ridgeline(groups, span, bins, zoom));
  body.appendChild(el('h2', null, 'By the numbers'));
  body.appendChild(table(groups));

  const skipped = [];
  (DATA.projects || []).forEach(project => {
    (project.trials || []).forEach(trial => {
      if (trial.excluded) return;
      if (!trial.spawnDepthHistogram)
        skipped.push(project.projectID + ' t' + trial.trial + ' (no spawn depths)');
      else if ((trial.spawnsMeasured || 0) < MIN_SPAWNS)
        skipped.push(project.projectID + ' t' + trial.trial +
                     ' (' + (trial.spawnsMeasured || 0) + ' spawns)');
    });
  });
  if (skipped.length) {
    body.appendChild(el('p', 'foot',
      'Left out: ' + skipped.slice(0, 12).join(', ') +
      (skipped.length > 12 ? ' and ' + (skipped.length - 12) + ' more' : '') +
      '. A trial needs a registration and at least ' + MIN_SPAWNS +
      ' measured spawns to appear.'));
  }
}

function build() {
  document.getElementById('subtitle').textContent =
    (DATA.projects || []).length + ' collected projects · sand counted as ' +
    'moved beyond ' + DATA.threshold + ' cm';
  const bar = document.getElementById('controls');
  bar.innerHTML =
    '<label>Minimum spawns per trial <b id="mv">' + MIN_SPAWNS + '</b></label>' +
    '<input type="range" id="mr" min="0" max="200" step="5" value="' +
      MIN_SPAWNS + '">' +
    '<span class="spacer"></span>' +
    '<span class="stat">computed per project and cached · ' +
    'rerun <code>features</code> after recollecting</span>';
  let pending = null;
  bar.querySelector('#mr').addEventListener('input', event => {
    MIN_SPAWNS = parseInt(event.target.value, 10);
    bar.querySelector('#mv').textContent = MIN_SPAWNS;
    if (pending) clearTimeout(pending);
    pending = setTimeout(draw, 150);
  });
  draw();
}

document.getElementById('body').appendChild(
  el('p', 'sub', 'Computing metrics for every project… the first load after a '
     + 'collection takes a while.'));
fetch('features.json')
  .then(response => response.json())
  .then(payload => {
    DATA = payload;
    if (!payload.projects || !payload.projects.length) {
      document.getElementById('body').textContent = '';
      document.getElementById('body').appendChild(el('div', 'note',
        'No projects have been collected in this analysis yet.'));
      return;
    }
    build();
  })
  .catch(error => {
    document.getElementById('body').textContent = '';
    document.getElementById('body').appendChild(
      el('div', 'note', 'Could not load the metrics: ' + error.message));
  });
