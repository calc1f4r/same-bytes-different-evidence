#!/usr/bin/env python3
"""Experiment: baked-ICC causal control.

Generate the missing frozen-protocol control (d): for each eval marked image,
produce a baked-ICC PNG = pixels pre-transformed through the SAME A98->sRGB
transform the managed decoder applies, stored WITHOUT the ICC profile.
If the ICC raw/managed decision differential is metadata-causal, the baked
control must equalize the two consumers' decisions and scores.
"""
import io, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageCms

BASE = Path(__file__).resolve().parents[3]
OUT = BASE/'raster_study/pilot_v3'
sys.path.insert(0, str(OUT/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

res = {}
# reuse the same A98 profile bytes the icc fixtures carry
src_profile = None
cand = sorted((BASE/'raster_study/color_axis_v1/fixtures').glob('*icc_a98*.png'))
with Image.open(cand[0]) as im0:
    src_profile = im0.info['icc_profile']
prof = ImageCms.ImageCmsProfile(io.BytesIO(src_profile))
srgb = ImageCms.createProfile('sRGB')

for fam in ('riva_gan', 'dwt_dct_svd', 'tree_ring_rand', 'tree_ring_ring'):
    cal = json.loads((OUT/'results'/f'{fam}_calibration.json').read_text())
    T = cal['threshold_min_rule']
    from run_family import make_family as mk
    _, score, _ = mk(fam)
    spec = json.loads((OUT/'manifests/frozen_protocol.json').read_text())['families'][fam]
    rows = []
    for idx in spec['marked_eval']:
        plain_p = OUT/'fixtures'/f'{fam}_{idx}_plain.png'
        icc_p = OUT/'fixtures'/f'{fam}_{idx}_icc_a98.png'
        baked_p = OUT/'fixtures'/f'{fam}_{idx}_baked_icc.png'
        if not baked_p.exists():
            with Image.open(icc_p) as im:
                baked = ImageCms.profileToProfile(im, prof, srgb, outputMode='RGB')
                baked.save(baked_p)  # NO icc_profile -> plain PNG, pixels pre-transformed
        a_plain = np.asarray(Image.open(plain_p).convert('RGB')).copy()
        a_icc_raw = a_plain  # raw consumer ignores profile
        with Image.open(icc_p) as im:
            a_icc_man = np.asarray(ImageCms.profileToProfile(im, prof, srgb, outputMode='RGB')).copy()
        a_baked = np.asarray(Image.open(baked_p).convert('RGB')).copy()
        s_raw, _ = score(a_icc_raw, idx)
        s_man, _ = score(a_icc_man, idx)
        s_baked, _ = score(a_baked, idx)
        rows.append({'idx': idx, 'icc_raw': s_raw, 'icc_managed': s_man, 'baked': s_baked,
                     'icc_raw_accept': s_raw > T, 'icc_managed_accept': s_man > T, 'baked_accept': s_baked > T,
                     'baked_equals_managed_score': s_baked == s_man})
        print(fam, idx, f'raw={s_raw:.4f} man={s_man:.4f} baked={s_baked:.4f}', flush=True)
    res[fam] = {'threshold': T, 'rows': rows}
    print(fam, 'baked==managed score:', sum(r['baked_equals_managed_score'] for r in rows), '/6', flush=True)

(OUT/'results'/'baked_icc_control.json').write_text(json.dumps(res, indent=1))
print('DONE -> results/baked_icc_control.json')
