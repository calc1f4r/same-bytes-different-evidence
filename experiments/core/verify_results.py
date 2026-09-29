#!/usr/bin/env python3
"""v3 verifier: recompute every gate quantity from raw rows; write verification.json."""
import json, hashlib, sys
from pathlib import Path
OUT = Path(__file__).resolve().parents[1]
PROTO = json.loads((OUT/'manifests/frozen_protocol.json').read_text())
def sha(b): return hashlib.sha256(b).hexdigest()
res = {'protocol_sha256': sha((OUT/'manifests/frozen_protocol.json').read_bytes()), 'families': {}}
for fam in PROTO['families']:
    rp = OUT/'results'/f'{fam}.jsonl'
    if not rp.exists(): res['families'][fam] = {'status': 'not run'}; continue
    rows = [json.loads(l) for l in rp.read_text().splitlines() if l.strip()]
    cal = json.loads((OUT/'results'/f'{fam}_calibration.json').read_text())
    T = cal['threshold_min_rule']
    fprp = OUT/'results'/f'{fam}_fpr.json'
    fpr = json.loads(fprp.read_text()) if fprp.exists() else None
    ev = [r for r in rows if r['split'] == 'evaluation']
    kinds = {}
    for r in ev: kinds.setdefault(r['kind'], []).append(r)
    out = {
        'cal_threshold_min': cal['threshold_min_rule'], 'cal_threshold_max': cal['threshold_max_rule'],
        'cal_tpr': sum(s > cal['threshold_min_rule'] for s in cal['cal_marked'])/6,
        'cal_fpr': sum(s > cal['threshold_min_rule'] for s in cal['cal_negative'])/len(cal['cal_negative']),
        'identity_accepts': sum(1 for r in kinds.get('identity', []) if r['decision'] is True),
        'exif_raw_accepts': sum(1 for r in kinds.get('exif_verifier_raw', []) if r['decision'] is True),
        'exif_display_rejects': sum(1 for r in kinds.get('exif_display', []) if r['decision'] is False),
        'exif_opencv_decisions': [r['decision'] for r in kinds.get('exif_opencv', [])],
        'exif_magick_decisions': [r['decision'] for r in kinds.get('exif_magick', [])],
        'exif_raw_accepts_max_rule': sum(1 for r in kinds.get('exif_verifier_raw', []) if r.get('decision_max_rule') is True),
        'baked_rejects': sum(1 for r in kinds.get('baked_visible', []) if r['decision'] is False),
        'ordinary_accepts': sum(1 for r in kinds.get('ordinary_jpeg', []) if r['decision'] is True),
        'wrong_key_eval_accepts': sum(1 for r in kinds.get('wrong_key_eval', []) if r['decision'] is True),
        'ssim_all_admitted': all(r['perceptual_admitted'] for r in kinds.get('exif_verifier_raw', [])),
        'color_axis_rows': {k: {'decision_min': r['decision'], 'decision_max': r.get('decision_max_rule'), 'score': r['score']}
                            for k, r in [(r['kind'], r) for r in ev if r['kind'].startswith('color_')]},
        'same_byte_per_pair': all(
            a['file_sha256'] == b['file_sha256']
            for a, b in zip(kinds.get('exif_verifier_raw', []), kinds.get('exif_display', []))),
    }
    if fpr:
        out['fpr'] = {k: fpr[k] for k in ('n', 'false_accepts_min_rule', 'upper95_fpr_min_rule',
                                           'gate_1pct_min_rule', 'false_accepts_max_rule')}
    # ICC/gamma consumer differential: same file, pillow_raw vs pillow_managed decision differs
    diffs = []
    for axis in ('icc_a98', 'gamma08'):
        raw = next((r for r in ev if r['kind'] == f'color_{axis}_pillow_raw'), None)
        man = next((r for r in ev if r['kind'] == f'color_{axis}_pillow_managed'), None)
        if raw and man and raw['file_sha256'] == man['file_sha256'] and raw['decision'] != man['decision']:
            diffs.append({'axis': axis, 'raw': raw['decision'], 'managed': man['decision'],
                          'score_raw': raw['score'], 'score_managed': man['score']})
    out['icc_gamma_consumer_decision_differentials'] = diffs
    res['families'][fam] = out
(OUT/'results'/'verification.json').write_text(json.dumps(res, indent=2))
print(json.dumps({f: {k: v for k, v in d.items() if k not in ('color_axis_rows',)}
                  for f, d in res['families'].items()}, indent=1)[:3000])
