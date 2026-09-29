#!/usr/bin/env python3
"""v3 repeat: fresh model load, exact score/tensor equality for first eval EXIF event per family."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_family import make_family
FIRST = {'riva_gan': 206, 'tree_ring_rand': 218, 'tree_ring_ring': 230, 'dwt_dct_svd': 242}
for family, idx in FIRST.items():
    rows = [json.loads(s) for s in (root/'results'/f'{family}.jsonl').read_text().splitlines()]
    lookup = {r['kind']: r for r in rows if r['split'] == 'evaluation' and r['idx'] == idx}
    _, detect = (None, None)
    e, s, c = make_family(family)
    jpg = root/'fixtures'/f'{family}_{idx}_exif6.jpg'
    with Image.open(jpg) as im:
        raw = np.asarray(im.convert('RGB')).copy()
        displayed = np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
    for kind, img in [('exif_verifier_raw', raw), ('exif_display', displayed)]:
        sc, th = s(img, idx)
        exp = lookup[kind]
        assert sc == exp['score'] and th == exp['tensor_hash'], (family, kind)
        print(f'{family} {idx} {kind} score={sc} tensor={th[:16]} PASS', flush=True)
