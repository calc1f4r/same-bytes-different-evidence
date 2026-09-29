#!/usr/bin/env python3
"""E10 negatives extension: 300 ADDITIONAL independent unmarked negatives per family,
scored at the FROZEN pilot_v3 min-rule / max-rule thresholds (never recalibrated).

Conventions reused EXACTLY from pilot_v3 (no drift):
  generation  : gen_negatives.py  — SD2.1-base fp16, seed derive_seed('pilot_v3_neg',20281001,j),
               prompt PROMPTS[j % 64], 25 DDIM steps, guidance 6.0, 256x256, cuda generator,
               saved to fixtures/neg_<j>.png. Refuses overwrite.
  scoring     : run_fpr.py — raw/unmanaged Pillow decode (Image.open().convert('RGB')),
               score via run_family.make_family(<family>) with wrong=False,
               strict > comparison against results/<family>_calibration.json thresholds.
  hash record : per-row file_sha256 + rgb_hash (same keys as {family}_fpr_rows.jsonl,
               plus an explicit "family" field because rows live in one file).
  statistics  : exact one-sided 95% Clopper-Pearson upper bound, same upper() as run_fpr.py.

New DISJOINT index ranges (verified absent from fixtures/ and all prior lineages):
  riva_gan 1700..1999, dwt_dct_svd 2000..2299, tree_ring_rand 2300..2599, tree_ring_ring 2600..2899.

GPU sharing: the RTX 3070 Ti Laptop 8GB is shared. Before any GPU work the
script checks nvidia-smi; if free < 4096 MiB it sleeps 120 s and re-checks (max 10 waits,
then proceeds and relies on OOM retry). Batch size 1. On CUDA OOM: wait 90 s, retry the
single image (max 5 attempts), retry count recorded in the progress file.

Progress checkpoint after EVERY image: results/sensitivity/negatives_extension_progress.json
(atomic tmp+rename); score rows appended+flushed per row to
results/sensitivity/negatives_extension_rows.jsonl. Both survive interruption; re-running an
action skips already-done work.

Usage:
  e10_negatives_extension.py --generate --family riva_gan --limit 50
  e10_negatives_extension.py --score    --family riva_gan --limit 300
  e10_negatives_extension.py --finalize [--replace]
  e10_negatives_extension.py --status
"""
import argparse, hashlib, json, os, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import PROMPTS, derive_seed  # noqa: E402

FIX = OUT / 'fixtures'
RES = OUT / 'results'
PROGRESS_PATH = RES / 'e10_negatives_extension_progress.json'
ROWS_PATH = RES / 'e10_negatives_extension_rows.jsonl'
FINAL_PATH = RES / 'e10_negatives_extension.json'
GEN_MODEL = '<home>/.cache/huggingface/hub/models--WIBE-HuggingFace--stable-diffusion-2-1-base/snapshots/94e48088b3d3d6fee9e0176d12425770aaa4b2c4'

# E10 extension blocks — disjoint from each other and from every prior lineage
# (v1 32..63, v2 0..63/6..17, color_axis 46..48, v3 200..247/300..323/400..1599,
# breadth 1600..1659).
BLOCKS = {
    'riva_gan': range(1700, 2000),
    'dwt_dct_svd': range(2000, 2300),
    'tree_ring_rand': range(2300, 2600),
    'tree_ring_ring': range(2600, 2900),
}
GPU_FAMILIES = ('tree_ring_rand', 'tree_ring_ring')  # scoring needs GPU inversion
MIN_FREE_MIB = 4096
SEED_LABEL, SEED_EPOCH = 'pilot_v3_neg', 20281001


def utcnow():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def arr_hash(a):
    return sha(np.ascontiguousarray(a).tobytes())


def rgb(path):
    """raw/unmanaged Pillow decode — identical to run_family.rgb / run_fpr path."""
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).copy()


# ---------------- progress checkpoint (atomic, recomputable from disk) ----------------
def read_rows():
    rows = []
    if ROWS_PATH.exists():
        for line in ROWS_PATH.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_progress():
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {'extension': 'e10_negatives_extension', 'started_utc': utcnow(),
            'updated_utc': utcnow(), 'batches': [], 'notes': []}


def family_state(family):
    rows = [r for r in read_rows() if r['family'] == family]
    scored = sorted(r['idx'] for r in rows)
    generated = [j for j in BLOCKS[family] if (FIX / f'neg_{j}.png').exists()]
    return {
        'index_range': [BLOCKS[family].start, BLOCKS[family].stop - 1],
        'n_planned': len(BLOCKS[family]),
        'n_generated': len(generated),
        'n_scored': len(scored),
        'scored_indices': scored,
        'accepts_min_rule': sum(1 for r in rows if r['accept_min_rule']),
        'accepts_max_rule': sum(1 for r in rows if r['accept_max_rule']),
    }


def write_progress(progress, batch_record=None, note=None):
    if batch_record is not None:
        progress['batches'].append(batch_record)
        progress['batches'] = progress['batches'][-400:]
    if note and note not in progress['notes']:
        progress['notes'].append(note)
    progress['updated_utc'] = utcnow()
    progress['families'] = {f: family_state(f) for f in BLOCKS}
    tmp = PROGRESS_PATH.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(progress, indent=2))
    os.replace(tmp, PROGRESS_PATH)


# ---------------- GPU guard ----------------
def gpu_free_mib():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                              '--format=csv,noheader,nounits'], capture_output=True,
                             text=True, timeout=30, check=True).stdout.strip().splitlines()[0]
        used, total = (int(x.strip()) for x in out.split(','))
        return total - used, used, total
    except Exception:
        return None, None, None


def wait_for_gpu(progress, why, min_free_mib=MIN_FREE_MIB, max_waits=10):
    """Gate before model load: require min_free_mib free (sleep 120 s, re-check).
    Mid-run continuations pass a small headroom floor instead, because the resident
    fp16 pipeline itself occupies ~3.5 GB and would otherwise trip the gate forever."""
    waits = 0
    while True:
        free, used, total = gpu_free_mib()
        if free is None:
            progress_note_add(progress, f'gpu check failed ({why}); proceeding unguarded')
            return waits
        if free >= min_free_mib or waits >= max_waits:
            if waits:
                progress_note_add(progress, f'gpu wait x{waits} before {why} '
                                            f'(free {free} MiB at start, floor {min_free_mib})')
            return waits
        print(f'GPU busy: {free} MiB free < {MIN_FREE_MIB}; sleep 120s '
              f'({why}, wait #{waits + 1})', flush=True)
        time.sleep(120)
        waits += 1


def progress_note_add(progress, note):
    if note not in progress['notes']:
        progress['notes'].append(note)


def is_oom(exc):
    text = f'{type(exc).__name__}: {exc}'.lower()
    return 'out of memory' in text or 'outofmemory' in text


def with_oom_retry(fn, progress, batch):
    for attempt in range(5):
        try:
            return fn()
        except Exception as exc:
            if not is_oom(exc) or attempt == 4:
                raise
            batch['oom_retries'] += 1
            print(f'CUDA OOM (attempt {attempt + 1}); empty_cache, sleep 90s, retry',
                  flush=True)
            import torch
            torch.cuda.empty_cache()
            time.sleep(90)
    raise RuntimeError('unreachable')


# ---------------- generation (exact gen_negatives.py convention) ----------------
def generate(family, limit):
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    progress = load_progress()
    todo = [j for j in BLOCKS[family] if not (FIX / f'neg_{j}.png').exists()][:limit]
    if not todo:
        print(f'{family}: nothing to generate'); return
    batch = {'ts_utc': utcnow(), 'action': 'generate', 'family': family,
             'n_done': 0, 'oom_retries': 0, 'seconds': 0.0}
    t0 = time.time()
    waits = wait_for_gpu(progress, f'generate {family}')
    batch['gpu_wait_cycles'] = waits
    sch = DDIMScheduler.from_pretrained(GEN_MODEL, subfolder='scheduler', local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        GEN_MODEL, scheduler=sch, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    for n, j in enumerate(todo):
        def one():
            seed = derive_seed(SEED_LABEL, SEED_EPOCH, j)
            g = torch.Generator(device='cuda').manual_seed(seed)
            return pipe(PROMPTS[j % 64], num_inference_steps=25, guidance_scale=6.0,
                        height=256, width=256, generator=g).images[0]
        im = with_oom_retry(one, progress, batch)
        assert not (FIX / f'neg_{j}.png').exists(), f'refusing overwrite neg_{j}.png'
        im.save(FIX / f'neg_{j}.png')
        batch['n_done'] += 1
        batch['seconds'] = round(time.time() - t0, 1)
        write_progress(progress)
        if n % 25 == 0 or n == len(todo) - 1:
            free, _, _ = gpu_free_mib()
            print(f'{family} gen {n + 1}/{len(todo)} neg_{j} {time.time() - t0:.0f}s '
                  f'(free {free} MiB)', flush=True)
            if n < len(todo) - 1:
                wait_for_gpu(progress, f'generate {family} continuation',
                             min_free_mib=512, max_waits=3)
    batch['seconds'] = round(time.time() - t0, 1)
    write_progress(progress, batch_record=batch)
    print(f'{family} GENERATE BATCH DONE {batch["n_done"]} in {batch["seconds"]}s', flush=True)


# ---------------- scoring (exact run_fpr.py convention) ----------------
def frozen_thresholds(family):
    cal = json.loads((RES / f'{family}_calibration.json').read_text())
    t_min, t_max = cal['threshold_min_rule'], cal['threshold_max_rule']
    fpr = json.loads((RES / f'{family}_fpr.json').read_text())
    assert (t_min, t_max) == (fpr['threshold_min_rule'], fpr['threshold_max_rule']), \
        f'{family}: calibration/FPR threshold mismatch — refusing to score'
    return t_min, t_max


def score(family, limit):
    progress = load_progress()
    t_min, t_max = frozen_thresholds(family)
    done = {r['idx'] for r in read_rows() if r['family'] == family}
    todo = [j for j in BLOCKS[family]
            if j not in done and (FIX / f'neg_{j}.png').exists()][:limit]
    if not todo:
        print(f'{family}: nothing to score'); return
    batch = {'ts_utc': utcnow(), 'action': 'score', 'family': family,
             'n_done': 0, 'oom_retries': 0, 'seconds': 0.0}
    t0 = time.time()
    if family in GPU_FAMILIES:
        batch['gpu_wait_cycles'] = wait_for_gpu(progress, f'score {family}')
    from run_family import make_family
    _, score_fn, _ = make_family(family)
    with ROWS_PATH.open('a') as rf:
        for n, j in enumerate(todo):
            p = FIX / f'neg_{j}.png'
            rast = rgb(p)  # raw/unmanaged Pillow decode, same as run_fpr.py
            def one():
                return score_fn(rast, j)  # wrong=False
            s, _th = with_oom_retry(one, progress, batch)
            row = {'family': family, 'idx': j, 'file_sha256': sha(p.read_bytes()),
                   'rgb_hash': arr_hash(rast), 'score': s,
                   'accept_min_rule': bool(s > t_min) if t_min is not None else None,
                   'accept_max_rule': bool(s > t_max) if t_max is not None else None}
            rf.write(json.dumps(row, sort_keys=True) + '\n'); rf.flush()
            batch['n_done'] += 1
            batch['seconds'] = round(time.time() - t0, 1)
            write_progress(progress)
            if n % 25 == 0 or n == len(todo) - 1:
                print(f'{family} E10 FPR {j} score={s:.4f} {time.time() - t0:.0f}s',
                      flush=True)
                if family in GPU_FAMILIES and n < len(todo) - 1:
                    wait_for_gpu(progress, f'score {family} continuation',
                                 min_free_mib=512, max_waits=3)
    batch['seconds'] = round(time.time() - t0, 1)
    write_progress(progress, batch_record=batch)
    fa = sum(1 for r in read_rows() if r['family'] == family and r['accept_min_rule'])
    print(f'{family} SCORE BATCH DONE {batch["n_done"]} in {batch["seconds"]}s '
          f'(cumulative accepts_min_rule={fa})', flush=True)


# ---------------- finalize ----------------
def upper(n, k):
    """Exact one-sided 95% Clopper-Pearson upper bound — identical to run_fpr.py."""
    if k == 0:
        return 1 - 0.05 ** (1.0 / n)
    from scipy.stats import beta
    return float(beta.ppf(0.95, k + 1, n - k))


def dist_summary(scores):
    a = np.asarray(scores, dtype=float)
    return {'n': int(a.size), 'min': float(a.min()), 'q25': float(np.quantile(a, 0.25)),
            'median': float(np.median(a)), 'q75': float(np.quantile(a, 0.75)),
            'max': float(a.max()), 'mean': float(a.mean()), 'std': float(a.std(ddof=1))}


def versions():
    import torch, diffusers, scipy, cv2
    v = {'python': sys.version.split()[0], 'torch': torch.__version__,
         'cuda_available': bool(torch.cuda.is_available()),
         'diffusers': diffusers.__version__, 'Pillow': Image.__version__,
         'numpy': np.__version__, 'scipy': scipy.__version__, 'opencv': cv2.__version__}
    try:
        import onnxruntime
        v['onnxruntime'] = onnxruntime.__version__
    except Exception:
        pass
    try:
        import transformers
        v['transformers'] = transformers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        v['gpu'] = torch.cuda.get_device_name(0)
    return v


def finalize(replace=False):
    if FINAL_PATH.exists() and not replace:
        raise RuntimeError(f'refusing overwrite {FINAL_PATH} (pass --replace to rewrite)')
    progress = load_progress()
    rows = read_rows()
    t_start = time.time()
    per_family = {}
    for family in BLOCKS:
        t_min, t_max = frozen_thresholds(family)
        frows = sorted((r for r in rows if r['family'] == family), key=lambda r: r['idx'])
        orig = json.loads((RES / f'{family}_fpr.json').read_text())
        n_new, k_min = len(frows), sum(1 for r in frows if r['accept_min_rule'])
        k_max = sum(1 for r in frows if r['accept_max_rule'])
        n_tot = orig['n'] + n_new
        a_min = orig['false_accepts_min_rule'] + k_min
        a_max = orig['false_accepts_max_rule'] + k_max
        scores = [r['score'] for r in frows]
        per_family[family] = {
            'index_range_new': [BLOCKS[family].start, BLOCKS[family].stop - 1],
            'n_planned_new': len(BLOCKS[family]),
            'n_generated': family_state(family)['n_generated'],
            'n_scored_new': n_new,
            'threshold_min_rule_frozen': t_min,
            'threshold_max_rule_frozen': t_max,
            'accepts_min_rule_new': k_min,
            'accepts_max_rule_new': k_max,
            'combined_with_original': {
                'n_original': orig['n'],
                'accepts_min_rule_original': orig['false_accepts_min_rule'],
                'accepts_max_rule_original': orig['false_accepts_max_rule'],
                'n_total': n_tot,
                'accepts_min_rule_total': a_min,
                'accepts_max_rule_total': a_max,
                'upper95_fpr_min_rule_total': upper(n_tot, a_min),
                'upper95_fpr_max_rule_total': upper(n_tot, a_max),
            },
            'score_distribution_new': dist_summary(scores) if scores else None,
            'margin_min_rule': (t_min - max(scores)) if scores else None,
            'margin_max_rule': (t_max - max(scores)) if (scores and t_max is not None) else None,
        }
    gen_s = round(sum(b['seconds'] for b in progress['batches'] if b['action'] == 'generate'), 1)
    sco_s = round(sum(b['seconds'] for b in progress['batches'] if b['action'] == 'score'), 1)
    res = {
        'extension': 'e10_negatives_extension',
        'purpose': 'This extension doubles the negative population: 300 additional '
                   'independent unmarked negatives per family at the FROZEN pilot_v3 '
                   'thresholds (no recalibration).',
        'date_utc': utcnow(),
        'started_utc': progress['started_utc'],
        'seed_policy': "derive_seed('pilot_v3_neg', 20281001, j); prompt PROMPTS[j % 64]; "
                       "SD2.1-base fp16, 25 DDIM steps, guidance 6.0, 256x256 "
                       "(identical to gen_negatives.py)",
        'scoring_path': 'raw/unmanaged Pillow decode -> run_family.make_family score '
                        '(wrong=False) -> strict > frozen calibration threshold '
                        '(identical to run_fpr.py)',
        'thresholds_source': 'results/{family}_calibration.json (cross-checked against '
                             '{family}_fpr.json); NEVER recalibrated',
        'clopper_pearson_rule': 'one-sided 95% upper: k=0 -> 1-0.05**(1/n); else '
                                'scipy.stats.beta.ppf(0.95, k+1, n-k) (identical to run_fpr.py)',
        'families': per_family,
        'progress_notes': progress['notes'],
        'batch_log': progress['batches'],
        'runtime_seconds': {'generation_batches_total': gen_s,
                            'scoring_batches_total': sco_s,
                            'finalize': round(time.time() - t_start, 2)},
        'versions': versions(),
        'protocol_sha256': sha((OUT / 'manifests/frozen_protocol.json').read_bytes()),
        'calibration_sha256': {f: sha((RES / f'{f}_calibration.json').read_bytes())
                               for f in BLOCKS},
        'original_fpr_sha256': {f: sha((RES / f'{f}_fpr.json').read_bytes()) for f in BLOCKS},
        'rows_file': 'results/sensitivity/negatives_extension_rows.jsonl',
        'progress_file': 'results/sensitivity/negatives_extension_progress.json',
    }
    FINAL_PATH.write_text(json.dumps(res, indent=2))
    print('E10 FINAL', json.dumps({f: {'n_new': v['n_scored_new'],
          'accepts_min_new': v['accepts_min_rule_new'],
          'combined_min': f"{v['combined_with_original']['accepts_min_rule_total']}"
                          f"/{v['combined_with_original']['n_total']}",
          'upper95_min_total': round(v['combined_with_original']
                                     ['upper95_fpr_min_rule_total'], 6)}
          for f, v in per_family.items()}), flush=True)


def status():
    p = load_progress()
    fams = {f: {k: v for k, v in family_state(f).items() if k != 'scored_indices'}
            for f in BLOCKS}
    print(json.dumps({'updated_utc': p['updated_utc'], 'families': fams}, indent=2))
    print('notes:', json.dumps(p['notes'], indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--generate', action='store_true')
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--finalize', action='store_true')
    ap.add_argument('--replace', action='store_true',
                    help='allow finalize to rewrite the existing final json')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--family', required=True, choices=list(BLOCKS))
    ap.add_argument('--limit', type=int, default=50)
    a = ap.parse_args()
    try:
        if a.status:
            status()
        elif a.generate:
            generate(a.family, a.limit)
        elif a.score:
            score(a.family, a.limit)
        elif a.finalize:
            finalize(a.replace)
        else:
            ap.error('choose --generate / --score / --finalize / --status')
    except Exception:
        traceback.print_exc(); sys.exit(1)
