/* Run common.js's scoreMask on a score array and print the mask.
 *
 * Exists so the Python copy in features.py can be pinned to it. The two
 * decide which pixels every volume is measured on, and a mask that differs
 * between the page and the server means a number on screen and the same
 * number in the spreadsheet disagree for a reason nobody can see.
 *
 *   node tests/score_mask.js <scores.json> <k>
 *
 * scores.json holds {"values": [...], "nan": [indices]} because JSON has no
 * NaN; the indices are written back as NaN before the call, since NaN is one
 * of the two things the mask is for.
 */

const fs = require('fs');
const path = require('path');

const COMMON = path.join(__dirname, '..', 'cichlid_bower_claude', 'server',
                         'templates', 'common.js');

const input = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const k = parseFloat(process.argv[3]);

const score = new Float32Array(input.values);
(input.nan || []).forEach(i => { score[i] = NaN; });

// common.js is written for a page, and declares its own D, PREP and CACHE at
// the top level. Evaluating it inside a function gives those somewhere to
// live without passing them in -- a parameter named D collides with the file's
// own `let D` and the whole thing fails to parse.
const source = fs.readFileSync(COMMON, 'utf8');
const scoreMask = new Function(source + '\nreturn scoreMask;')();

const result = scoreMask(score, k);
process.stdout.write(JSON.stringify({
  cut: result.cut === undefined ? null : result.cut,
  marked: result.mark ? Array.from(result.mark) : null,
}));
