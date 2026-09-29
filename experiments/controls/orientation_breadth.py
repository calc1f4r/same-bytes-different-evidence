#!/usr/bin/env python3
"""Experiment: orientation breadth (EXIF 2-8) + larger n.

For CPU families (riva_gan, dwt_dct_svd): 30 fresh marked lineages each
(indices 1600..1629 / 1630..1659), every EXIF orientation 2..8 applied to each,
scored raw vs display. Threshold = frozen v3 min-rule threshold (preregistered
reuse; this is a breadth probe, not a new calibration).
"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity

BASE = Path(__file__).resolve().parents[3]
OUT = BASE/'raster_study/pilot_v3'
sys.path.insert(0, str(OUT/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_family import make_family, bits
from common import PROMPTS, derive_seed
import torch, cv2
from imwatermark import WatermarkEncoder

res = {}
for fam in ('riva_gan', 'dwt_dct_svd'):
    cal = json.loads((OUT/'results'/f'{fam}_calibration.json').read_text())
    T = cal['threshold_min_rule']
    _, score, _ = make_family(fam)
    wm = 'rivaGan' if fam == 'riva_gan' else 'dwtDctSvd'
    if fam == 'riva_gan':
        from imwatermark.rivaGan import RivaWatermark
        RivaWatermark.loadModel()
    rows = []
    base = 1600 if fam == 'riva_gan' else 1630
    for k in range(30):
        idx = base + k
        png = OUT/'fixtures'/f'breadth_{idx}_marked.png'
        if not png.exists():
            # generate carrier then mark (same frozen seed rule style)
            from diffusers import DDIMScheduler, StableDiffusionPipeline
            MODEL = '<home>/.cache/huggingface/hub/models--WIBE-HuggingFace--stable-diffusion-2-1-base/snapshots/94e48088b3d3d6fee9e0176d12425770aaa4b2c4'
            sch = DDIMScheduler.from_pretrained(MODEL, subfolder='scheduler', local_files_only=True)
            pipe = StableDiffusionPipeline.from_pretrained(MODEL, scheduler=sch, torch_dtype=torch.float16,
                safety_checker=None, requires_safety_checker=False, local_files_only=True).to('cuda')
            pipe.set_progress_bar_config(disable=True)
            g = torch.Generator(device='cuda').manual_seed(derive_seed('pilot_v3_breadth', 20281001, idx))
            im = pipe(PROMPTS[idx % 64], num_inference_steps=25, guidance_scale=6.0, height=256, width=256, generator=g).images[0]
            a = np.asarray(im.convert('RGB')).copy()
            e = WatermarkEncoder(); e.set_watermark('bits', bits(idx).tolist())
            m = cv2.cvtColor(e.encode(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), wm), cv2.COLOR_BGR2RGB)
            Image.fromarray(m, 'RGB').save(png)
            del pipe; torch.cuda.empty_cache()
        a = np.asarray(Image.open(png).convert('RGB')).copy()
        s_id, _ = score(a, idx)
        for orient in range(2, 9):
            jpg = OUT/'fixtures'/f'breadth_{idx}_exif{orient}.jpg'
            ex = Image.Exif(); ex[274] = orient
            Image.fromarray(a, 'RGB').save(jpg, format='JPEG', quality=95, subsampling=0, exif=ex)
            with Image.open(jpg) as im:
                raw = np.asarray(im.convert('RGB')).copy()
                disp = np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
            s_raw, _ = score(raw, idx)
            s_disp, _ = score(disp, idx)
            ssim = float(structural_similarity(disp, np.asarray(ImageOps.exif_transpose(Image.fromarray(a, 'RGB'))), channel_axis=2, data_range=255)) if orient != 1 else 1.0
            rows.append({'idx': idx, 'orient': orient, 'identity_score': s_id,
                         'raw_score': s_raw, 'raw_accept': s_raw > T,
                         'display_score': s_disp, 'display_accept': s_disp > T})
        if (k+1) % 5 == 0: print(fam, k+1, '/30 marked done', flush=True)
    # aggregates
    agg = {}
    for orient in range(2, 9):
        rr = [r for r in rows if r['orient'] == orient]
        agg[orient] = {'n': len(rr),
                       'split': sum(1 for r in rr if r['raw_accept'] and not r['display_accept']),
                       'both_accept': sum(1 for r in rr if r['raw_accept'] and r['display_accept']),
                       'both_reject': sum(1 for r in rr if not r['raw_accept'] and not r['display_accept'])}
    res[fam] = {'threshold': T, 'rows': rows, 'by_orientation': agg}
    print(fam, json.dumps(agg), flush=True)

(OUT/'results'/'orientation_breadth.json').write_text(json.dumps(res, indent=1))
print('DONE -> results/orientation_breadth.json')
