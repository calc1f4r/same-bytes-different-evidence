#!/usr/bin/env python3
"""v3 FPR scorer: score 300 independent unmarked negatives per family at frozen
min/max thresholds. Refuses overwrite. Writes results/{family}_fpr.json."""
import argparse, hashlib, json, sys, traceback
from pathlib import Path
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
PROTO = json.loads((OUT/'manifests/frozen_protocol.json').read_text())
NEG_BLOCKS = {'riva_gan': range(400, 700), 'tree_ring_rand': range(700, 1000),
              'tree_ring_ring': range(1000, 1300), 'dwt_dct_svd': range(1300, 1600)}

def sha(b): return hashlib.sha256(b).hexdigest()
def arr_hash(a): return sha(np.ascontiguousarray(a).tobytes())
def rgb(path):
    with Image.open(path) as im: return np.asarray(im.convert('RGB')).copy()

def main(family):
    from run_family import make_family
    cal = json.loads((OUT/'results'/f'{family}_calibration.json').read_text())
    t_min, t_max = cal['threshold_min_rule'], cal['threshold_max_rule']
    outp = OUT/'results'/f'{family}_fpr.json'
    if outp.exists(): raise RuntimeError('refusing overwrite FPR file')
    _, score, _ = make_family(family)
    rows = []
    import time; t0 = time.time()
    for j in NEG_BLOCKS[family]:
        p = OUT/'fixtures'/f'neg_{j}.png'
        rast = rgb(p)
        s, th = score(rast, j)   # wrong=False; key irrelevant for unmarked but keep idx deterministic
        rows.append({'idx': j, 'file_sha256': sha(p.read_bytes()), 'rgb_hash': arr_hash(rast),
                     'score': s, 'accept_min_rule': bool(s > t_min) if t_min is not None else None,
                     'accept_max_rule': bool(s > t_max) if t_max is not None else None})
        if (j - NEG_BLOCKS[family].start) % 25 == 0:
            print(f'{family} FPR {j} {time.time()-t0:.0f}s', flush=True)
    fa_min = sum(r['accept_min_rule'] for r in rows)
    fa_max = sum(1 for r in rows if r['accept_max_rule'])
    # exact one-sided 95% upper bound
    def upper(n, k):
        if k == 0: return 1 - 0.05**(1.0/n)
        from scipy.stats import beta
        return float(beta.ppf(0.95, k+1, n-k))
    res = {'family': family, 'n': len(rows), 'false_accepts_min_rule': fa_min,
           'false_accepts_max_rule': fa_max,
           'upper95_fpr_min_rule': upper(len(rows), fa_min),
           'upper95_fpr_max_rule': upper(len(rows), fa_max),
           'gate_1pct_min_rule': bool(fa_min == 0),
           'gate_1pct_max_rule': bool(fa_max == 0),
           'threshold_min_rule': t_min, 'threshold_max_rule': t_max,
           'protocol_sha256': sha((OUT/'manifests/frozen_protocol.json').read_bytes())}
    outp.write_text(json.dumps(res, indent=2))
    (OUT/'results'/f'{family}_fpr_rows.jsonl').write_text('\n'.join(json.dumps(r, sort_keys=True) for r in rows))
    print('FPR RESULT', json.dumps({k: res[k] for k in ('n', 'false_accepts_min_rule', 'upper95_fpr_min_rule', 'gate_1pct_min_rule')}))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--family', required=True)
    a = ap.parse_args()
    try: main(a.family)
    except Exception: traceback.print_exc(); sys.exit(1)
