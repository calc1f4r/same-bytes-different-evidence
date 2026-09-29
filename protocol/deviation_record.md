# Deviation record: confirmatory run (recorded during execution)

1. 2026-09-27, GPU budget: frozen protocol declared a 5400 s GPU wall-clock cap. Negative
   generation alone (1224 SD2.1 generations) plus two Tree-Ring FPR batteries (300
   inversion+score passes each) exceed that cap; actual GPU time will be ~2.5-3 h. The cap
   was set too tight at freeze time. No other protocol term changes: lineages, threshold
   rule, FPR gate, consumers and axes are unchanged. Recorded before running the
   tree-ring families.
2. 2026-09-27, cal-negatives ordering: gen_negatives.py appended cal-neg block 300..323
   after the four FPR blocks; the riva_gan family run failed once on a missing neg_300.png.
   The 24 cal negatives were then generated ahead of schedule (same frozen seeds/prompt
   rule, only execution order changed). No riva_gan score rows existed at that point; the
   failed run wrote no rows (verified by clean restart).
3. 2026-09-27, scripts bugfixes pre/data: three code errors were fixed during the riva_gan
   run before any persisted score rows existed (RivaWatermark.loadModel scope, Pillow
   verify-after-open ordering, stale jsonl cleanup). After the first successful persisted
   calibration row, no script touching scoring logic was modified. gen_negatives and
   fixture-generation code remained frozen throughout.
4. 2026-09-27, tree-ring generation bug: gen_marked_carriers passed a torch Generator where
   a latent tensor was expected when deriving the watermark key; tree_ring_rand crashed
   before writing any row. Fixed to derive mask/pattern from a randn tensor seeded with the
   same frozen key rule used by the scorer (derive_seed('pilot_v3_key_<kind>',...)). No
   rows existed; no scoring logic changed.
5. 2026-09-27, post-audit corrections (post-run re-derivation audits): the ring
   calibration TPR was transcribed as 6/6 in an earlier STATUS draft; raw rows and
   verification.json say 4/6 under the min rule. Ring weak eval indices are 232/233
   (not 232/234). dwt gamma differentials are 6/6 (draft said 5/6) and ring has a 1/6
   gamma differential (omitted). The "~12.8 Chromium-vs-Pillow ICC MAE" sentence in an
   earlier draft was the managed-vs-plain transform distance, not a browser gap; audit
   re-measurement gives ~0.03-0.04 (+-1 LSB). ImageMagick default convert was described
   as display-agreeing; it is EXIF-ignoring and raw-agreeing.
6. 2026-09-27, frozen-protocol gaps (undisclosed at execution, recorded now): the frozen
   ICC/gamma admission rule (SSIM >= 0.90 managed-vs-plain) was never computed during the
   run; retro-computation (results/color_axis_ssim_admission.json) shows it FAILS in all
   families (0.777-0.842), so all ICC/gamma differentials are inadmissible under the
   frozen rule. The baked-ICC control (frozen axis item d) was never generated. The
   gamma consumer LUT ran the transform in the wrong direction (exponent 1.25 instead of
   ~0.57). Color rows record two consumers, not four (the frozen axis text itself
   prescribed two; the primary-event list said four).
7. 2026-09-27, negative regeneration: the main gen_negatives job recomputed neg_300..323
   at 12:31-12:32 after the ahead-of-schedule early generation used by riva calibration;
   all recorded file_sha256 values match the regenerated files (deterministic
   regeneration, bit-identical).
8. 2026-09-27, provenance note: frozen_utc "12:05:00Z" in the protocol is an
   approximate hand-entered timestamp; the binding evidence is the file hash recorded in
   every downstream artifact, with mtime 11:33:26 UTC preceding the first generation
   (11:34:13) and first score row (11:55:59).
