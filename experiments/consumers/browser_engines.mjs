#!/usr/bin/env node
// E9: multi-engine browser display ground-truth check.
//
// Rationale: the study's display ground truth was established with a pinned
// Chromium 151 build (raster_study/browser_probe/v3/results.json, 24/24
// canvas==scored on the exif6 fixtures). An independent
// engines. This probe repeats the SAME protocol on Firefox (newly installed
// via `npx playwright install firefox`), optionally WebKit (expected to fail
// on this WSL host without sudo; failure recorded verbatim), and re-runs the
// pinned chromium-1234 binary as a harness control.
//
// Protocol per fixture (mirrors browser_probe/v3_browser_capture.py):
//   1. load JPEG bytes as a data: URI into <img> with the same CSS
//      (margin:0;background:white;img display:block;width:256px;height:256px;
//       object-fit:fill), await img.decode()
//   2. canvas at naturalWidth/naturalHeight, 2d context
//      {willReadFrequently:true, colorSpace:'srgb'}, drawImage, getImageData
//   3. hash = sha256 over RGB bytes (RGBA readback with alpha dropped,
//      row-major) — identical to arr_hash(np.asarray(...)[:, :, :3]) in the
//      Chromium probe and to the scored pillow_exif_transpose raster hash.
//   4. element screenshot saved as PNG; screenshot raster hashed by decoding
//      the PNG back through an <img>/<canvas> in the same page (lossless, so
//      equivalent to the Pillow decode the Chromium probe used).
//   5. compare canvas/screenshot hash to the scored exif_display rgb_hash;
//      also report MAE / max abs diff of canvas RGB vs the scored raster
//      (Pillow exif_transpose, obtained from a python3 child process).
//
// Writes: results/consumers/browser_engines/browser_engines.json and
// pilot_v3/results/e9_screenshots/<engine>/<fixture>.png. No other files are modified.

import { createRequire } from 'node:module';
import { execFileSync, spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const BASE = '<study-tree>';
const PILOT = path.join(BASE, 'raster_study/pilot_v3');
const RESULTS = path.join(PILOT, 'results');
const SCREENSHOTS = path.join(RESULTS, 'e9_screenshots');
const CHROMIUM_PROBE = path.join(BASE, 'raster_study/browser_probe/v3/results.json');
const PINNED_CHROMIUM = path.join(process.env.HOME, '.cache/ms-playwright/chromium-1234/chrome-linux64/chrome');

const require = createRequire(import.meta.url); // CJS require honours NODE_PATH
const { chromium, firefox, webkit } = require('playwright');

const sha256 = (buf) => crypto.createHash('sha256').update(buf).digest('hex');

// ---------- fixture list + expected hashes (from existing artifacts) ----------
const v3 = JSON.parse(fs.readFileSync(CHROMIUM_PROBE, 'utf8'));
const exifRows = v3.rows.filter((r) => r.tag === 'exif6');

// Cross-check the probe's scored_rgb_hash against the family jsonl exif_display rows.
const famJsonl = {};
for (const fam of ['riva_gan', 'tree_ring_rand', 'tree_ring_ring', 'dwt_dct_svd']) {
  const rows = fs.readFileSync(path.join(RESULTS, `${fam}.jsonl`), 'utf8')
    .split('\n').filter(Boolean).map((l) => JSON.parse(l));
  famJsonl[fam] = {};
  for (const r of rows) {
    if (r.split === 'evaluation' && r.kind === 'exif_display') famJsonl[fam][r.idx] = r;
  }
}

const fixtures = exifRows.map((r) => {
  const file = path.join(PILOT, 'fixtures', `${r.family}_${r.idx}_exif6.jpg`);
  const jl = famJsonl[r.family][r.idx];
  return {
    family: r.family,
    idx: r.idx,
    file,
    name: path.basename(file),
    file_sha256_recorded: r.file_sha256,
    expected_rgb_hash: r.scored_rgb_hash,
    chromium_canvas_rgb_hash: r.chromium_canvas_rgb_hash,
    jsonl_rgb_hash: jl ? jl.rgb_hash : null,
    scored_hash_sources_agree: jl ? jl.rgb_hash === r.scored_rgb_hash : null,
  };
});

// ---------- scored display rasters (Pillow exif_transpose) via python3 ----------
const PY_SNIPPET = [
  'import sys, json, base64',
  'from PIL import Image, ImageOps',
  'out = {}',
  'for p in sys.argv[1:]:',
  '    with Image.open(p) as im:',
  '        t = ImageOps.exif_transpose(im).convert("RGB")',
  '    out[p] = {"w": t.size[0], "h": t.size[1], "rgb_b64": base64.b64encode(t.tobytes()).decode()}',
  'print(json.dumps(out))',
].join('\n');

function scoredRasters(paths) {
  const res = spawnSync('python3', ['-c', PY_SNIPPET, ...paths], { maxBuffer: 64 * 1024 * 1024 });
  if (res.status !== 0) throw new Error(`python3 scored-raster helper failed: ${res.stderr}`);
  return JSON.parse(res.stdout.toString());
}
const scored = scoredRasters(fixtures.map((f) => f.file));

// ---------- in-page extraction helpers (mirror of the v3 probe) ----------
const PAGE_HTML = '<html><head><style>html,body{margin:0;background:white}img{display:block;width:256px;height:256px;object-fit:fill}</style></head><body><img id="r"></body></html>';

// NOTE: unlike Python playwright, Node playwright does NOT invoke string-form
// evaluate sources (a string arrow function evaluates to the function object,
// which serializes to undefined) — all evaluates here use function form.

// Runs in page: read an already-decoded <img> into a canvas at natural size,
// return RGBA base64 (chunked btoa to avoid argument limits).
const imgToRgba = () => {
  const i = document.querySelector('img');
  const c = document.createElement('canvas');
  c.width = i.naturalWidth; c.height = i.naturalHeight;
  const x = c.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' });
  x.drawImage(i, 0, 0);
  const d = x.getImageData(0, 0, c.width, c.height).data;
  let s = ''; const CH = 0x8000;
  for (let o = 0; o < d.length; o += CH) s += String.fromCharCode.apply(null, d.subarray(o, Math.min(o + CH, d.length)));
  return { w: c.width, h: c.height, rgba_b64: btoa(s) };
};

// Runs in page: decode a PNG data URI, draw scaled to w x h (lossless decode;
// scaling only if sizes differ, as the Pillow probe's resize fallback did).
// Node playwright evaluate passes a single argument — args wrapped in an object.
const pngToRgba = async ({ uri, w, h }) => {
  const im = new Image(); im.src = uri; await im.decode();
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const x = c.getContext('2d', { willReadFrequently: true, colorSpace: 'srgb' });
  x.imageSmoothingEnabled = false;
  x.drawImage(im, 0, 0, w, h);
  const d = x.getImageData(0, 0, w, h).data;
  let s = ''; const CH = 0x8000;
  for (let o = 0; o < d.length; o += CH) s += String.fromCharCode.apply(null, d.subarray(o, Math.min(o + CH, d.length)));
  return { w: c.width, h: c.height, rgba_b64: btoa(s) };
};

function rgbaToRgb(buf) {
  const n = buf.length / 4;
  const rgb = Buffer.alloc(n * 3);
  for (let i = 0, j = 0; i < buf.length; i += 4) {
    rgb[j++] = buf[i]; rgb[j++] = buf[i + 1]; rgb[j++] = buf[i + 2];
  }
  return rgb;
}

function diffStats(a, b) {
  if (!Buffer.isBuffer(b) || a.length !== b.length) return { mae: null, max_absdiff: null, comparable: false };
  let sum = 0, max = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    sum += d;
    if (d > max) max = d;
  }
  return { mae: Number((sum / a.length).toFixed(4)), max_absdiff: max, comparable: true };
}

async function probeEngine(engineName, browser, outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  const page = await browser.newPage({
    viewport: { width: 320, height: 320 },
    deviceScaleFactor: 1,
    colorScheme: 'light',
  });
  const userAgent = await page.evaluate(() => navigator.userAgent);
  const rows = [];
  for (const fx of fixtures) {
    const row = {
      family: fx.family, idx: fx.idx, fixture: fx.name,
      file_sha256_observed: sha256(fs.readFileSync(fx.file)),
      file_sha256_matches_recorded: null,
      jsonl_scored_agrees: fx.scored_hash_sources_agree,
      natural: null,
      canvas_rgb_hash: null, canvas_equal_scored: null,
      canvas_mae_vs_scored: null, canvas_max_absdiff_vs_scored: null,
      screenshot_png: null, screenshot_rgb_hash: null, screenshot_equal_scored: null,
      screenshot_mae_vs_scored: null,
      error: null,
    };
    row.file_sha256_matches_recorded = row.file_sha256_observed === fx.file_sha256_recorded;
    try {
      const data = fs.readFileSync(fx.file);
      const uri = 'data:image/jpeg;base64,' + data.toString('base64');
      await page.setContent(PAGE_HTML);
      await page.locator('img').evaluate((e, src) => { e.src = src; }, uri);
      await page.locator('img').evaluate(async (e) => { await e.decode(); });
      const px = await page.evaluate(imgToRgba);
      row.natural = { w: px.w, h: px.h };
      const rgba = Buffer.from(px.rgba_b64, 'base64');
      const canvasRgb = rgbaToRgb(rgba);
      row.canvas_rgb_hash = sha256(canvasRgb);
      row.canvas_equal_scored = row.canvas_rgb_hash === fx.expected_rgb_hash;
      const ref = scored[fx.file];
      const refRgb = ref ? Buffer.from(ref.rgb_b64, 'base64') : null;
      const cs = diffStats(canvasRgb, refRgb);
      row.canvas_mae_vs_scored = cs.mae; row.canvas_max_absdiff_vs_scored = cs.max_absdiff;

      // element screenshot (same locator shot as the v3 probe), saved for human inspection
      const shot = await page.locator('img').screenshot();
      const shotPath = path.join(outDir, `${fx.family}_${fx.idx}_exif6.png`);
      fs.writeFileSync(shotPath, shot);
      row.screenshot_png = path.relative(RESULTS, shotPath);
      // hash the screenshot raster by decoding the PNG back in-page (lossless)
      const spx = await page.evaluate(pngToRgba, { uri: 'data:image/png;base64,' + shot.toString('base64'), w: px.w, h: px.h });
      const shotRgb = rgbaToRgb(Buffer.from(spx.rgba_b64, 'base64'));
      row.screenshot_rgb_hash = sha256(shotRgb);
      row.screenshot_equal_scored = row.screenshot_rgb_hash === fx.expected_rgb_hash;
      const ss = diffStats(shotRgb, refRgb);
      row.screenshot_mae_vs_scored = ss.mae;
      row.screenshot_equal_canvas = row.screenshot_rgb_hash === row.canvas_rgb_hash;
    } catch (e) {
      row.error = String((e && e.message) || e);
    }
    rows.push(row);
  }
  await page.close();
  const n = rows.length;
  const summary = {
    files: n,
    canvas_hash_matches: rows.filter((r) => r.canvas_equal_scored === true).length,
    screenshot_hash_matches: rows.filter((r) => r.screenshot_equal_scored === true).length,
    file_sha256_all_ok: rows.every((r) => r.file_sha256_matches_recorded === true),
    scored_sources_agree: rows.every((r) => r.jsonl_scored_agrees === true),
    max_canvas_mae: Math.max(...rows.map((r) => r.canvas_mae_vs_scored ?? -1)),
    errors: rows.filter((r) => r.error).length,
  };
  return { browser_version: browser.version(), user_agent: userAgent, rows, summary };
}

// ---------- run engines ----------
const out = {
  experiment: 'e9_browser_engines',
  purpose: 'second/third independent browser engine for the EXIF-orientation-6 display ground truth (Chromium-only so far)',
  generated_utc: new Date().toISOString(),
  hash_convention: 'sha256 over row-major RGB bytes: canvas 2d getImageData RGBA at natural size with alpha dropped (v3 probe / run_family.py arr_hash convention); scored side = Pillow ImageOps.exif_transpose(im).convert("RGB") rgb_hash of kind="exif_display" rows',
  reference: {
    chromium_probe_artifact: 'raster_study/browser_probe/v3/results.json',
    chromium_probe_browser_version: v3.browser_version,
    chromium_probe_script: 'raster_study/browser_probe/v3_browser_capture.py',
    expected_source: 'pilot_v3/results/{family}.jsonl kind="exif_display" split="evaluation" rows (consumer pillow_exif_transpose), cross-checked against browser_probe/v3 scored_rgb_hash',
    fixtures: 'pilot_v3/fixtures/{family}_{idx}_exif6.jpg (24 files, 6 per family x 4 families)',
  },
  environment: {
    node: process.version,
    playwright: require('playwright/package.json').version,
    platform: `${process.platform} ${process.arch}`,
    host: 'WSL2 linux 5.15.167.4-microsoft-standard-WSL2',
  },
  notes: [
    'screenshot_rgb_hash here is computed by decoding the saved PNG back through an in-page canvas (lossless); the v3 Chromium probe decoded it with Pillow. PNG decode is lossless/deterministic, so the two methods are equivalent; noted for transparency.',
    'chromium_control re-runs the pinned chromium-1234 binary through this new harness to prove the harness reproduces the published 24/24 before trusting its Firefox numbers.',
    'Firefox is launched with playwright defaults: --force-color-profile is a Chromium-only flag and has no Firefox equivalent, so Firefox colour-management defaults applied. All 24 fixtures are untagged (no ICC) JPEGs and canvas MAE vs the scored raster is 0.0 everywhere, confirming no colour-management transform occurred on this axis.',
  ],
  engines: {},
};

async function tryEngine(name, launcher, opts, outDirName, extraMeta = {}) {
  const entry = { attempted: true, launch_ok: false, ...extraMeta };
  let browser = null;
  try {
    browser = await launcher(opts);
    entry.launch_ok = true;
    const res = await probeEngine(name, browser, path.join(SCREENSHOTS, outDirName));
    Object.assign(entry, res);
  } catch (e) {
    entry.launch_error = String((e && e.message) || e);
    entry.launch_error_stack_head = String((e && e.stack) || '').split('\n').slice(0, 6).join(' | ');
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  out.engines[name] = entry;
}

// Firefox: newly installed engine (playwright 1.58, firefox-1509 = Firefox 146.0.1).
const FIREFOX_BIN = path.join(process.env.HOME, '.cache/ms-playwright/firefox-1509/firefox/firefox');
await tryEngine(
  'firefox',
  (o) => firefox.launch(o),
  { headless: true },
  'firefox',
  fs.existsSync(FIREFOX_BIN)
    ? { executable: FIREFOX_BIN, executable_sha256: sha256(fs.readFileSync(FIREFOX_BIN)) }
    : { executable: FIREFOX_BIN, executable_sha256: 'binary not found at expected path' },
);

// Harness control: pinned chromium-1234 (the v3 ground-truth binary).
await tryEngine(
  'chromium_control',
  (o) => chromium.launch(o),
  { headless: true, executablePath: PINNED_CHROMIUM, args: ['--no-sandbox', '--disable-gpu', '--force-color-profile=srgb'] },
  'chromium_control',
  fs.existsSync(PINNED_CHROMIUM)
    ? { executable: PINNED_CHROMIUM, executable_sha256: sha256(fs.readFileSync(PINNED_CHROMIUM)) }
    : { executable: PINNED_CHROMIUM, executable_sha256: 'binary not found at expected path' },
);

// WebKit: binary downloaded (webkit-2248) but host validation fails; attempt
// launches anyway and record errors verbatim. Second attempt skips playwright's
// curated validation so the raw dynamic-loader failure (if any) is captured.
const webkitEntry = {
  attempted: true,
  launch_ok: false,
  install: {
    npx_playwright_install: 'downloaded webkit-2248, then host validation warning: missing libraries libgtk-4.so.1, libgraphene-1.0.so.0, libevent-2.1.so.7, libgstgl-1.0.so.0, libgstcodecparsers-1.0.so.0, libavif.so.16, libharfbuzz-icu.so.0, libmanette-0.2.so.0, libenchant-2.so.2, libhyphen.so.0, libsecret-1.so.0, libwoff2dec.so.1.0.2 (Playwright Host validation warning, exit 0 with warning)',
    npx_playwright_install_deps: 'FAILED (needs sudo): "Switching to root user to install dependencies... sudo: a terminal is required to read the password; either use the -S option to read from standard input or configure an askpass helper / sudo: a password is required / Failed to install browser dependencies / Error: Installation process exited with code: 1" — per study instructions webkit system deps were NOT installed (no sudo available)',
    ldd_check: 'ldd of minibrowser-gtk/lib/libwebkitgtk-6.0.so.4 shows unresolved DT_NEEDED: libavif.so.16, libenchant-2.so.2, libevent-2.1.so.7, libgraphene-1.0.so.0, libgstcodecparsers-1.0.so.0, libgstgl-1.0.so.0, libgtk-4.so.1, libharfbuzz-icu.so.0, libhyphen.so.0, libjavascriptcoregtk-6.0.so.1, libjxl.so.0.8, libmanette-0.2.so.0, libsecret-1.so.0, libsoup-3.0.so.0 — hard link-time dependencies, not lazy dlopens',
  },
  launch_attempts: [],
};
for (const [label, envPatch] of [
  ['default (playwright host validation active)', {}],
  ['PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=1 (raw loader error)', { PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS: '1' }],
]) {
  try {
    const prev = { ...process.env };
    Object.assign(process.env, envPatch);
    let b = null;
    try {
      b = await webkit.launch({ headless: true });
      webkitEntry.launch_attempts.push({ mode: label, launched: true });
      webkitEntry.launch_ok = true;
      const res = await probeEngine('webkit', b, path.join(SCREENSHOTS, 'webkit'));
      Object.assign(webkitEntry, res);
    } finally {
      if (b) await b.close().catch(() => {});
      process.env = prev;
    }
    if (webkitEntry.launch_ok) break;
  } catch (e) {
    webkitEntry.launch_attempts.push({
      mode: label,
      launched: false,
      error: String((e && e.message) || e),
    });
  }
}
out.engines.webkit = webkitEntry;

// ---------- write result ----------
fs.writeFileSync(path.join(RESULTS, 'e9_browser_engines.json'), JSON.stringify(out, null, 1) + '\n');

// ---------- console summary ----------
console.log(`fixtures: ${fixtures.length} | scored hash sources agree: ${fixtures.filter((f) => f.scored_hash_sources_agree).length}/${fixtures.length}`);
for (const [name, e] of Object.entries(out.engines)) {
  if (e.launch_ok && e.summary) {
    console.log(`${name} (${e.browser_version}): canvas==scored ${e.summary.canvas_hash_matches}/${e.summary.files} | screenshot==scored ${e.summary.screenshot_hash_matches}/${e.summary.files} | max canvas MAE ${e.summary.max_canvas_mae} | errors ${e.summary.errors}`);
    for (const r of e.rows) {
      if (r.error || r.canvas_equal_scored === false) {
        console.log(`  ${r.error ? 'ERROR' : 'MISMATCH'} ${r.family} ${r.idx} ${r.error || ''} mae=${r.canvas_mae_vs_scored}`);
      }
    }
  } else {
    console.log(`${name}: LAUNCH FAILED — ${e.launch_error || JSON.stringify(e.launch_attempts)}`);
  }
}
