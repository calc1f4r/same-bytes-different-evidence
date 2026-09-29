#!/usr/bin/env python3
"""Experiment: rotation-normalizing verifier control.

Question: does the same-byte EXIF decision split disappear when the verifier
normalizes orientation before scoring (the obvious rotation-robust defense)?

Method: for each v3 EXIF fixture, score under (a) raw EXIF-ignoring decode
(the split condition) and (b) a verifier that applies exif_transpose first
(rotation normalization). All four configurations. CPU for two, GPU for two.
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parents[3]
OUT = BASE/'raster_study/pilot_v3'
sys.path.insert(0, str(OUT/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_family import make_family

res = {}
for fam in ('riva_gan', 'dwt_dct_svd', 'tree_ring_rand', 'tree_ring_ring'):
    cal = json.loads((OUT/'results'/f'{fam}_calibration.json').read_text())
    T = cal['threshold_min_rule']
    _, score, _ = make_family(fam)
    rows = []
    spec = json.loads((OUT/'manifests/frozen_protocol.json').read_text())['families'][fam]
    for idx in spec['marked_eval']:
        jpg = OUT/'fixtures'/f'{fam}_{idx}_exif6.jpg'
        with Image.open(jpg) as im:
            raw = np.asarray(im.convert('RGB')).copy()
            disp = np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
        s_raw, _ = score(raw, idx)
        s_norm, _ = score(disp, idx)   # normalized verifier = display-decode scoring
        rows.append({'idx': idx, 'raw_score': s_raw, 'raw_accept': s_raw > T,
                     'normalized_score': s_norm, 'normalized_accept': s_norm > T})
        print(fam, idx, f'raw={s_raw:.4f}({s_raw>T}) norm={s_norm:.4f}({s_norm>T})', flush=True)
    split = sum(1 for r in rows if r['raw_accept'] and not r['normalized_accept'])
    res[fam] = {'threshold': T, 'rows': rows,
                'raw_accept_normalized_reject': split,
                'both_accept': sum(1 for r in rows if r['raw_accept'] and r['normalized_accept']),
                'both_reject': sum(1 for r in rows if not r['raw_accept'] and not r['normalized_accept'])}
    print(fam, 'split-after-normalization:', split, '/6', flush=True)

(OUT/'results'/'rotation_normalizing_control.json').write_text(json.dumps(res, indent=1))
print('DONE -> results/rotation_normalizing_control.json')
