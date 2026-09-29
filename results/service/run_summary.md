# Canonicalization service evaluation summary (executed 2026-09-27, CPU, loopback only)

Local HTTP verification service (server.py, stdlib + onnxruntime CPU RivaGAN decoder) tested the canonical gateway idea end-to-end on pilot_v1 held-out EXIF fixtures (46-51) plus marked/unmarked/wrong-key controls, raw vs canonical ingestion paths, with a real pinned-Chromium browser UI consuming returned canonical PNGs (browser_ui.py; canvas matches canonical 12/12).

## Measured (evaluation_run.log, 64 HTTP rows)

- exif original raw: 4/6 accepted (v1 implemented threshold 0.75); literal frozen-rule sensitivity 0.984375 -> 2/6.
- exif canonical: 0/6 accepted (canonicalization applies orientation; watermark destroyed by rotation, score mean 0.479).
- marked (no EXIF) raw and canonical: 4/6 accepted both (canonical re-encode preserves RivaGAN bits).
- unmarked and wrong-key: 0/12 accepts either path.
- Idempotence: 6/6 repeated canonical PNG bytes exact; repeated scores/tensors exact.
- Latency: ~260-300 ms mean per request (CPU ONNX); canonical adds ~20 ms re-encode.
- 4 rows rejected by design (invalid/oversize/unsupported-color classes).

## Interpretation boundaries

This v1-fixture service run is EXPLORATORY: it inherits pilot_v1's threshold-rule deviation (documented in raster_study/pilot_v1/DEVIATIONS.md), so both implemented (0.75) and literal (0.984375) decision counts are reported. It demonstrates: a real service boundary that displays canonical PNG while a raw-path verifier scores differently on same uploaded bytes; canonicalization eliminates the EXIF raw/display split but at the cost of destroying the rotated watermark (clean marked TPR 4/6 vs 0/6 on EXIF originals) = the "rejection is not preservation" lesson with live model evidence. Not a deployed system; loopback only; Tree-Ring not served.

Reproduce: see evaluate.py header (starts server.py, runs browser_ui.py via /tmp/raster-browser-venv pinned Chromium).
