# Same Bytes, Different Evidence — experiment artifacts

Does a watermark verifier return the same verdict on the same encoded image file,
regardless of which standards-valid decoder produces its input raster? These experiments
measure that question end to end: one EXIF-orientation-6 JPEG, decoded by EXIF-ignoring
and EXIF-honouring consumers, scored by four watermark configurations at thresholds frozen
on calibration data before any evaluation score.

## What was found

| Finding | Result |
|---|---|
| Same-byte verdict split (held-out, n=6 per config) | 5/6, 6/6, 4/6, 6/6 (RivaGAN, Tree-Ring rand, Tree-Ring ring, dwtDctSvd); every miss a weak mark both paths rejected |
| False positives | 0/300 unmarked negatives per configuration (95% upper bound 0.99%); extension to 600: 0/600 for three configs (bound 0.50%), ring 1/600 (index 2609 at -26.484375 vs threshold -26.546875, bound 0.79%, inside the frozen 1.5% gate; max-rule 0/600) |
| Cause attribution | EXIF-honouring consumers (OpenCV default, Pillow exif_transpose) matched the display raster and its rejection in 24/24 pairs; EXIF-ignoring consumers (unmanaged Pillow, ImageMagick default) matched the raw raster and the acceptance; no mixed cases; baked-pixel controls reproduce display scores exactly |
| Browser display ground truth | Chromium 151 and Firefox 146.0.1 render the display raster identically (24/24 hashes, MAE 0) |
| Orientation breadth (n=30 per tag) | All seven non-identity tags split in both post-hoc configurations, except the classical mark surviving the vertical flip (orientation 4: both-accept 30/30; orientation 3, true 180°, splits 30/30); RivaGAN 26/30 at every tag |
| JPEG quality 60/75/85 | Split persists in all four configurations; display rejects 6/6 everywhere; losses at low quality are raw-path compression |
| Wrong-attribution | Fixed wrong message: 0/36 and 0/36; Tree-Ring cross-key: 0/6 and 0/6 |
| Threshold sensitivity | At every eligible calibration midpoint: FPR stays 0/300 and the split stays at least 1/6 |
| Loaders | Tested versions of torchvision, matplotlib, imageio decode EXIF-ignoring (24/24); OpenCV default honours (24/24); 6 of 7 inspected public watermark repositories load still images EXIF-ignoring (code inspection at pinned commits) |
| Colour axis (negative result) | ICC differentials are metadata-causal (baked-ICC 24/24, raw differs 23/24) but inadmissible: per-family minimum SSIM 0.777-0.842 below the frozen 0.90 rule; gamma rows carry a disclosed LUT direction bug; axis closed |
| Canonicalization service | Canonical path accepts 0/6 where raw accepts 4/6 (pilot 0.75 threshold): applying orientation rejects the rotated mark; idempotent 6/6, ~260-300 ms per request |

Every number above is re-derived from the raw rows in `results/` by
`experiments/verify_findings.py` (CPU-only, seconds):

    python3 experiments/verify_findings.py     # -> 86 PASS / 0 FAIL

## Repository layout

| Path | Contents |
|---|---|
| `experiments/` | Experiment code as executed: core scoring (`core/`), causal and breadth controls (`controls/`), sensitivity sweeps (`sensitivity/`), decode-consumer and browser probes (`consumers/`), canonicalization loopback service (`service/`), shared seed/prompt library and vendored Tree-Ring port (`src/`), and `verify_findings.py`. |
| `protocol/` | `frozen_protocol.json` (+ sha256), the five extension protocols, `environment_manifest.json`, `deviation_record.md`. |
| `results/` | Every raw artifact: per-configuration score rows, calibration and FPR rows, controls, sweeps, loader and browser-engine results (with screenshots), repository loader survey, service evaluation, `verification_report.json`, and `fixtures.sha256`. |

## The experiment

Four watermark configurations (RivaGAN post-hoc neural; Tree-Ring rand and Tree-Ring ring,
generator-native latent marks through a vendored port; dwtDctSvd classical transform) were
embedded in fresh Stable Diffusion 2.1-base 256x256 carriers generated from frozen
per-index seeds over a fixed 64-prompt pool. Identical EXIF-orientation-6 JPEG bytes
(quality 95, subsampling 0) were decoded by four consumers (unmanaged Pillow, Pillow
exif_transpose, OpenCV default imread, ImageMagick default convert) and scored at
thresholds frozen on calibration data before any evaluation score, with every decision
also recorded under a pre-specified maximum-rule alternative.

## Reproduction

- **Verify the findings (CPU, seconds).** `python3 experiments/verify_findings.py`.
- **Verify regenerated fixtures (CPU).** The ~2,500 carrier/marked/negative PNGs regenerate
  bit-for-bit from the frozen seeds (`experiments/src/common.py`, `derive_seed` with offset
  20281001, prompt = PROMPTS[idx mod 64]) via `experiments/core/run_family.py` and
  `experiments/core/gen_negatives.py`; check them with
  `cd results && sha256sum -c fixtures.sha256`.
- **Full re-run (GPU, ~5-6 h).** Execute core scoring, controls, sweeps and consumer
  probes in `experiments/`, then re-verify. Tree-Ring configurations need the SD 2.1-base
  snapshot pinned in `protocol/environment_manifest.json`; browser checks need Playwright
  with Chromium and Firefox. Scripts are archived as executed: internal directory names in
  path constants were neutralized to this layout, outputs regenerate into the working tree
  the scripts define, and execution-host path strings appear as `<study-tree>` / `<home>`
  placeholders.

## Protocol integrity

`protocol/frozen_protocol.json` is hash-bound into every calibration, FPR, and
verification artifact (field `protocol_sha256`):

    cd protocol && sha256sum -c frozen_protocol.sha256

The run's recorded deviations (GPU budget overrun, pre-row code fixes, the gamma LUT
direction bug, post-hoc SSIM admission computation, and the others) are in
`protocol/deviation_record.md`. `frozen_protocol.json` preserves its original internal
provenance strings verbatim, since its bytes are hash-bound into every results file.

## Environment determinism

The claim concerns decoding semantics, so decoder builds are part of the experiment:
Pillow 12.3.0 decodes JPEG through libjpeg-turbo 3.1.4.1 while OpenCV 5.0.0.93 bundles
3.1.2-70. Full provenance (versions, delegates, ONNX and port file hashes, SD 2.1-base
snapshot commit, RNG policy) is in `protocol/environment_manifest.json`.

## Results schema

All row files are JSON Lines; every row carries `idx`, `score`, `file_sha256` (the encoded
fixture's hash) and `rgb_hash` (the decoded raster's hash).

- `core/<config>/scores.jsonl`: the confirmatory rows (408 total). Fields: `family`,
  `kind` (marked, unmarked, wrong_key, identity, exif_verifier_raw, exif_display,
  exif_opencv, exif_magick, baked_visible, ordinary_jpeg, color_plain/icc_a98/gamma08_
  pillow_raw/managed), `split` (calibration/evaluation), `tensor_hash`, and a
  `validation{...}` block (format, size, orientation, cross-decoder hash) on the
  identity rows.
- `core/<config>/fpr_rows.jsonl` and `sensitivity/negatives_extension_rows.jsonl`: one
  row per unmarked negative with `accept_min_rule` / `accept_max_rule` booleans at the
  frozen thresholds (1,200 + 1,200 rows).
- `sensitivity/jpeg_quality_sweep.json`: per quality and lineage, two rows (`path`:
  raw/display) with `decision` and `decision_max_rule`.
- `sensitivity/wrong_message.json`: decoy rows with an `accept` boolean.
- `consumers/display_consumer_repeat.json`: per fixture and `consumer`, score and
  input-array hash equality.
- `controls/color_axis_ssim_admission.json`: per row `axis`, `ssim_managed_vs_plain`,
  `perceptual_ssim`, `perceptual_admitted`.
- Controls and sweep JSONs (`baked_icc_control`, `rotation_normalizing_control`,
  `orientation_breadth`, `threshold_sweep`) are family-keyed with per-lineage `rows`
  carrying raw/normalized/baked scores and accept booleans; calibration and FPR JSONs
  carry the frozen thresholds, rule definitions, and the protocol hash.
- `consumers/loader_survey/extended.json`: one row per fixture with per-consumer
  `equals_display` / `equals_raw` booleans (the fixture's `display_hash`/`raw_hash` are
  carried on each row).
- `consumers/browser_engines/browser_engines.json`: per engine and fixture: canvas and
  screenshot RGB hashes, equality flags against the scored display raster, and MAE.
- `consumers/repository_loader_survey.json`: per repository: pinned commit, verbatim
  loader quotes, and an `exif_behaviour` classification; `background_facts` records the
  decode-semantics background; the correction note documents the cv2.imread fix.
- `controls/latent_ring_check.json`: the shipped latent recomputation: marked-latent
  distances at identity/rot90/rot180 per pattern and the rot90 self-difference ratio.
- `results/service/`: the canonicalization-service evaluation: `http_rows.json` (per
  request: path, key, idx, source and returned hashes), `summary.json`, and
  `run_summary.md` (the shipped record for the idempotence and latency figures).

## Third-party code

`experiments/src/treering/` is a vendored port of the Tree-Ring reference implementation,
adapted for SD 2.1 at 256 px. The imwatermark package (RivaGAN, dwtDctSvd) is used from
PyPI (invisible-watermark 0.2.0). The SD 2.1-base checkpoint is fetched from its published
repository; its licence terms apply.

## Integrity

`results/MANIFEST.sha256` lists the sha256 of every shipped result file
(`verification_report.json` is excluded because the verifier regenerates it on each run):

    cd results && sha256sum -c MANIFEST.sha256

The frozen gate text contains two formulations, a 1.5% exact-upper-bound cap and
zero-accept language; under the executed strict greater-than convention the confirmatory
run satisfies both. The extension's single ring accept (1/600, bound 0.79%) satisfies the
1.5% cap while violating the literal zero-accept reading, which is the knife-edge
disclosure playing out under heavier sampling. Image fixtures themselves are not shipped:
they regenerate bit-for-bit from the frozen seeds (see Reproduction), and every fixture's
hash is recorded in the rows and in `results/fixtures.sha256`.

## License

Code is released under the MIT License (see LICENSE). Raw result rows and protocol records
are released under CC-BY-4.0 for verification and reuse with attribution.

## Citing

GitHub renders a "Cite this repository" button from CITATION.cff, equivalent to:

```bibtex
@misc{same-bytes-different-evidence,
  title  = {Same Bytes, Different Evidence (experiment artifacts)},
  author = {Srivastava, Yash and Singh, Amardeep and Singh, Monika},
  year   = {2026},
  note   = {Artifact release: experiment code, frozen protocol, and raw results}
}
```

## Scope and limitations

256-px carriers from one generator; four configurations across two codebases; decoder
versions as pinned in the environment manifest; browser evidence from Chromium and Firefox
(WebKit blocked by missing system libraries on the execution host); repository loader
evidence is code inspection at pinned commits, not execution.
