#!/usr/bin/env python3
"""v3 family runner. CPU families (riva_gan, dwt_dct_svd) run standalone; tree-ring
families require GPU model. All writes scoped to pilot_v3. --family <name>."""
import os
import argparse, hashlib, json, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from common import derive_seed, PROMPTS
PROTO = json.loads((OUT/'manifests/frozen_protocol.json').read_text())
(OUT/'results').mkdir(exist_ok=True)
(OUT/'fixtures').mkdir(exist_ok=True)

def sha(b): return hashlib.sha256(b).hexdigest()
def arr_hash(a): return sha(np.ascontiguousarray(a).tobytes())
def record(path, row):
    with path.open('a') as f: f.write(json.dumps(row, sort_keys=True)+'\n'); f.flush()
def rgb(path):
    with Image.open(path) as im: return np.asarray(im.convert('RGB')).copy()
def save(a, path, **kw): Image.fromarray(a, 'RGB').save(path, **kw)
def bits(idx, off=20281001): return np.random.default_rng(off+idx).integers(0, 2, 32, dtype=np.uint8)

def eligible_thresholds(pos, neg):
    scores = sorted(set(pos+neg))
    mids = [(x+y)/2 for x, y in zip(scores[:-1], scores[1:])]
    return [t for t in mids if all(n <= t for n in neg) and any(p > t for p in pos)]

def validate(path, orientation=None):
    with Image.open(path) as im:
        im.verify()
    with Image.open(path) as im:
        mode, fmt, size, orient = im.mode, im.format, list(im.size), im.getexif().get(274)
    import cv2
    cv = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert cv is not None
    return {'file_sha256': sha(path.read_bytes()), 'format': fmt, 'mode': mode, 'size': size,
            'orientation': orient, 'opencv_default_rgb_hash': arr_hash(cv2.cvtColor(cv, cv2.COLOR_BGR2RGB))}

# ---------- consumers ----------
def consumer_rasters(path):
    """Four scored consumer rasters from identical file bytes."""
    import cv2
    out = {}
    with Image.open(path) as im:
        out['pillow_raw'] = np.asarray(im.convert('RGB')).copy()
        out['pillow_exif_transpose'] = np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
    cv = cv2.imread(str(path), cv2.IMREAD_COLOR)
    out['opencv_default'] = cv2.cvtColor(cv, cv2.COLOR_BGR2RGB)
    # imagemagick default -> PNG -> pillow decode
    mag = subprocess.run(['convert', str(path), 'PNG:-'], capture_output=True, timeout=30, check=True)
    import io
    with Image.open(io.BytesIO(mag.stdout)) as im2:
        out['imagemagick_default'] = np.asarray(im2.convert('RGB')).copy()
    return out

# ---------- families ----------
def make_family(family):
    import cv2
    if family in ('riva_gan', 'dwt_dct_svd'):
        from imwatermark import WatermarkEncoder
        from imwatermark.rivaGan import RivaWatermark
        if family == 'riva_gan':
            RivaWatermark.loadModel()
            def detect_raw(bgr):
                ten = (np.asarray([bgr], dtype=np.float32)/127.5-1).transpose(3, 0, 1, 2)[None, :, :, :, :]
                return RivaWatermark.decoder.run(None, {'frame': ten})[0][0]
            def score(a, idx, wrong=False):
                bgr = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2BGR)
                raw = detect_raw(bgr)
                decoded = (raw > 0.52).astype(np.uint8)
                truth = bits(idx, 20291001 if wrong else 20281001)
                ten = (np.asarray([bgr], dtype=np.float32)/127.5-1).transpose(3, 0, 1, 2)[None, :, :, :, :]
                return float((decoded == truth).mean()), arr_hash(ten)
        else:
            wm = 'dwtDctSvd'
            from imwatermark import WatermarkDecoder
            def score(a, idx, wrong=False):
                bgr = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2BGR)
                d = WatermarkDecoder('bits', 32)
                dec = d.decode(bgr, 'dwtDctSvd')
                truth = bits(idx, 20291001 if wrong else 20281001)
                pseudo = np.asarray(dec, dtype=np.uint8)
                # score = bit accuracy; tensor proxy = exact bgr bytes hash
                return float((pseudo == truth).mean()), sha(bgr.tobytes())
        def embed(idx, carrier):
            e = WatermarkEncoder(); e.set_watermark('bits', bits(idx).tolist())
            return cv2.cvtColor(e.encode(cv2.cvtColor(carrier, cv2.COLOR_RGB2BGR), wm), cv2.COLOR_BGR2RGB)
        def carrier(idx):
            # fresh SD carrier must exist from gen step; generate inline if missing is NOT allowed (freeze) — assert
            p = OUT/'fixtures'/f'carrier_{idx}.png'
            assert p.exists(), f'missing {p}; run gen_marked first'
            return rgb(p)
        return embed, score, carrier
    # tree-ring variants
    import torch
    from diffusers import DDIMScheduler
    from treering import (InversableStableDiffusionPipeline, get_watermarking_mask,
                          get_watermarking_pattern, inject_watermark, transform_img, l1_complex_metric)
    pattern_kind = 'rand' if family == 'tree_ring_rand' else 'ring'
    model = Path(os.environ.get('SD21_SNAPSHOT', str(Path.home()/'.cache/huggingface/hub/models--WIBE-HuggingFace--stable-diffusion-2-1-base/snapshots/94e48088b3d3d6fee9e0176d12425770aaa4b2c4')))
    sch = DDIMScheduler.from_pretrained(str(model), subfolder='scheduler', local_files_only=True)
    pipe = InversableStableDiffusionPipeline.from_pretrained(str(model), scheduler=sch,
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    empty = pipe.get_text_embedding('')
    def key(idx, wrong=False):
        seed = derive_seed(f'pilot_v3_wrong_{pattern_kind}' if wrong else f'pilot_v3_key_{pattern_kind}', 20281001, idx)
        g = torch.Generator(device='cuda').manual_seed(seed)
        init = torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.float16, generator=g)
        return get_watermarking_mask(init, 0, 5, 'cuda', 'circle'), get_watermarking_pattern(init, pattern_kind, 5), seed
    def embed(idx, carrier=None):
        p = OUT/'fixtures'/f'marked_{idx}.png'
        assert p.exists(), f'missing {p}; run gen_marked first'
        return rgb(p)
    def score(a, idx, wrong=False):
        mask, pattern, seed = key(idx, wrong)
        t = transform_img(Image.fromarray(a, 'RGB'), 256).unsqueeze(0).to(empty.dtype).to('cuda')
        latent = pipe.get_image_latents(t, sample=False)
        inv = pipe.forward_diffusion(latents=latent, text_embeddings=empty, guidance_scale=1, num_inference_steps=25)
        distance = l1_complex_metric(inv, mask, pattern)
        tc = t.detach().cpu().contiguous().numpy()
        return float(-distance), arr_hash(tc)
    return embed, score, None

GEN_MODEL = os.environ.get('SD21_SNAPSHOT', str(Path.home()/'.cache/huggingface/hub/models--WIBE-HuggingFace--stable-diffusion-2-1-base/snapshots/94e48088b3d3d6fee9e0176d12425770aaa4b2c4'))

def gen_marked_carriers(family, indices):
    """Generate fresh carriers/marked images for the family (GPU for tree-ring, CPU-safe for post-hoc)."""
    need = [i for i in indices if not (OUT/'fixtures'/f'carrier_{i}.png').exists()]
    if not need: return
    if family in ('riva_gan', 'dwt_dct_svd'):
        import torch
        from diffusers import DDIMScheduler, StableDiffusionPipeline
        sch = DDIMScheduler.from_pretrained(GEN_MODEL, subfolder='scheduler', local_files_only=True)
        pipe = StableDiffusionPipeline.from_pretrained(GEN_MODEL, scheduler=sch, torch_dtype=torch.float16,
            safety_checker=None, requires_safety_checker=False, local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        for i in need:
            g = torch.Generator(device='cuda').manual_seed(derive_seed('pilot_v3_gen', 20281001, i))
            pipe(PROMPTS[i % 64], num_inference_steps=25, guidance_scale=6.0, height=256, width=256, generator=g).images[0].save(OUT/'fixtures'/f'carrier_{i}.png')
    else:
        import torch
        from diffusers import DDIMScheduler
        from treering import (InversableStableDiffusionPipeline, get_watermarking_mask,
                              get_watermarking_pattern, inject_watermark)
        pattern_kind = 'rand' if family == 'tree_ring_rand' else 'ring'
        sch = DDIMScheduler.from_pretrained(GEN_MODEL, subfolder='scheduler', local_files_only=True)
        pipe = InversableStableDiffusionPipeline.from_pretrained(GEN_MODEL, scheduler=sch,
            torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False, local_files_only=True).to('cuda')
        pipe.set_progress_bar_config(disable=True)
        for i in need:
            g = torch.Generator(device='cuda').manual_seed(derive_seed('pilot_v3_gen', 20281001, i))
            init = torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.float16, generator=g)
            kg = torch.Generator(device='cuda').manual_seed(derive_seed(f'pilot_v3_key_{pattern_kind}', 20281001, i))
            kinit = torch.randn(1, 4, 32, 32, device='cuda', dtype=torch.float16, generator=kg)
            mask = get_watermarking_mask(kinit, 0, 5, 'cuda', 'circle')
            pattern = get_watermarking_pattern(kinit, pattern_kind, 5)
            lat = inject_watermark(init, mask, pattern, 'complex')
            pipe(PROMPTS[i % 64], num_inference_steps=25, guidance_scale=6.0, height=256, width=256, latents=lat).images[0].save(OUT/'fixtures'/f'marked_{i}.png')

def main(family):
    spec = PROTO['families'][family]
    rowpath = OUT/'results'/f'{family}.jsonl'
    if rowpath.exists(): raise RuntimeError('refusing overwrite existing score rows')
    fam = family
    gen_marked_carriers(family, spec['marked_cal'] + spec['marked_eval'])
    # For post-hoc families: mark the carriers now (deterministic, before scoring)
    if family in ('riva_gan', 'dwt_dct_svd'):
        from imwatermark import WatermarkEncoder
        import cv2
        if family == 'riva_gan':
            from imwatermark.rivaGan import RivaWatermark
            RivaWatermark.loadModel()
        wm = 'rivaGan' if family == 'riva_gan' else 'dwtDctSvd'
        for i in spec['marked_cal'] + spec['marked_eval']:
            c = rgb(OUT/'fixtures'/f'carrier_{i}.png')
            e = WatermarkEncoder(); e.set_watermark('bits', bits(i).tolist())
            m = cv2.cvtColor(e.encode(cv2.cvtColor(c, cv2.COLOR_RGB2BGR), wm), cv2.COLOR_BGR2RGB)
            save(m, OUT/'fixtures'/f'marked_{i}.png')
    embed, score, _ = make_family(family)
    start = time.monotonic()

    def scored_row(kind, split, idx, raster, src_file, wrong=False, extra=None):
        s, th = score(raster, idx, wrong)
        r = {'family': fam, 'split': split, 'idx': idx, 'kind': kind,
             'file_sha256': sha(src_file.read_bytes()), 'rgb_hash': arr_hash(raster),
             'score': s, 'tensor_hash': th}
        if extra: r.update(extra)
        return r

    cal_pos, cal_neg = [], []
    # calibration: 6 marked, 6 unmarked (fresh negs 300..323 blocks), 6 wrong-key
    for idx, nidx in zip(spec['marked_cal'], spec['unmarked_cal']):
        a = rgb(OUT/'fixtures'/f'marked_{idx}.png')
        p = OUT/'fixtures'/f'marked_{idx}.png'
        r = scored_row('marked', 'calibration', idx, a, p)
        r['validation'] = validate(p); record(rowpath, r); cal_pos.append(r['score'])
        nimg = rgb(OUT/'fixtures'/f'neg_{nidx}.png')
        r = scored_row('unmarked', 'calibration', nidx, nimg, OUT/'fixtures'/f'neg_{nidx}.png')
        record(rowpath, r); cal_neg.append(r['score'])
        r = scored_row('wrong_key', 'calibration', idx, a, p, wrong=True)
        record(rowpath, r); cal_neg.append(r['score'])
        print(f'{fam} cal {idx} done {time.monotonic()-start:.0f}s', flush=True)
    elig = eligible_thresholds(cal_pos, cal_neg)
    t_min, t_max = (min(elig), max(elig)) if elig else (None, None)
    cal = {'family': fam, 'cal_marked': cal_pos, 'cal_negative': cal_neg,
           'threshold_min_rule': t_min, 'threshold_max_rule': t_max,
           'eligible_midpoints': elig, 'protocol_sha256': sha((OUT/'manifests/frozen_protocol.json').read_bytes())}
    with (OUT/'results'/f'{family}_calibration.json').open('x') as f:
        f.write(json.dumps(cal, indent=2)); f.flush()
    print('CALIBRATION', json.dumps({k: cal[k] for k in ('threshold_min_rule', 'threshold_max_rule')}), flush=True)
    T = t_min
    if T is None: raise RuntimeError('no eligible threshold; family uncalibrated per protocol')

    def decide(s): return s > T

    for idx in spec['marked_eval']:
        mp = OUT/'fixtures'/f'marked_{idx}.png'
        a = rgb(mp)
        # identity
        r = scored_row('identity', 'evaluation', idx, a, mp); r['decision'] = decide(r['score']); r['decision_max_rule'] = r['score'] > t_max if t_max is not None else None
        r['validation'] = validate(mp); record(rowpath, r)
        # EXIF same-byte fixtures
        jpg = OUT/'fixtures'/f'{fam}_{idx}_exif6.jpg'
        exif = Image.Exif(); exif[274] = 6
        Image.fromarray(a, 'RGB').save(jpg, format='JPEG', quality=95, subsampling=0, exif=exif)
        ordinary = OUT/'fixtures'/f'{fam}_{idx}_ordinary.jpg'
        Image.fromarray(a, 'RGB').save(ordinary, format='JPEG', quality=95, subsampling=0)
        cons = consumer_rasters(jpg)
        with Image.open(jpg) as im:
            display = np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
        baked = OUT/'fixtures'/f'{fam}_{idx}_baked_visible.png'; save(display, baked)
        display_ref = np.asarray(Image.fromarray(a, 'RGB').transpose(Image.Transpose.ROTATE_270))
        ssim = float(structural_similarity(display, display_ref, channel_axis=2, data_range=255))
        for cname, rast in cons.items():
            kind = {'pillow_raw': 'exif_verifier_raw', 'pillow_exif_transpose': 'exif_display',
                    'opencv_default': 'exif_opencv', 'imagemagick_default': 'exif_magick'}[cname]
            r = scored_row(kind, 'evaluation', idx, rast, jpg)
            r['consumer'] = cname; r['perceptual_ssim'] = ssim; r['perceptual_admitted'] = ssim >= 0.90
            r['decision'] = decide(r['score']); r['decision_max_rule'] = r['score'] > t_max if t_max is not None else None
            record(rowpath, r)
        for kind, src, f in [('baked_visible', rgb(baked), baked), ('ordinary_jpeg', rgb(ordinary), ordinary)]:
            r = scored_row(kind, 'evaluation', idx, src, f)
            r['perceptual_ssim'] = ssim; r['decision'] = decide(r['score']); r['decision_max_rule'] = r['score'] > t_max if t_max is not None else None
            record(rowpath, r)
        # ICC/gamma axis from the same marked image
        plain = OUT/'fixtures'/f'{fam}_{idx}_plain.png'; save(a, plain)
        icc = OUT/'fixtures'/f'{fam}_{idx}_icc_a98.png'
        src_icc = None
        a98_path = BASE/'raster_study/color_axis_v1/fixtures'
        cand = sorted(a98_path.glob('*icc_a98*.png')) if a98_path.exists() else []
        if cand:
            with Image.open(cand[0]) as im0:
                src_icc = im0.info.get('icc_profile')
        if src_icc:
            Image.fromarray(a, 'RGB').save(icc, format='PNG', icc_profile=src_icc)
        gamma = OUT/'fixtures'/f'{fam}_{idx}_gamma08.png'
        Image.fromarray(a, 'RGB').save(gamma, format='PNG')
        add_gamma(gamma, 0.8)
        for axis_file, kindsfx in [(plain, 'plain'), (icc if src_icc else None, 'icc_a98'), (gamma, 'gamma08')]:
            if axis_file is None: continue
            for cname in ('pillow_raw', 'pillow_managed'):
                rast = axis_decode(axis_file, cname)
                r = scored_row(f'color_{kindsfx}_{cname}', 'evaluation', idx, rast, axis_file)
                r['axis'] = kindsfx; r['consumer'] = cname
                r['decision'] = decide(r['score']); r['decision_max_rule'] = r['score'] > t_max if t_max is not None else None
                record(rowpath, r)
        print(f'{fam} eval {idx} done {time.monotonic()-start:.0f}s', flush=True)
    # wrong-key eval
    for idx in spec['marked_eval']:
        mp = OUT/'fixtures'/f'marked_{idx}.png'; a = rgb(mp)
        r = scored_row('wrong_key_eval', 'evaluation', idx, a, mp, wrong=True)
        r['decision'] = decide(r['score']); r['decision_max_rule'] = r['score'] > t_max if t_max is not None else None
        record(rowpath, r)
    print('EVAL BATTERY DONE', flush=True)

def add_gamma(path, gval):
    """Insert gAMA chunk into PNG (before IDAT)."""
    import struct, zlib
    data = path.read_bytes()
    pos = 8; out = bytearray(data[:8])
    while pos < len(data):
        ln = struct.unpack('>I', data[pos:pos+4])[0]
        typ = data[pos+4:pos+8]
        out += data[pos:pos+8+ln+4]
        if typ == b'IHDR':
            gma = struct.pack('>I', int(gval*100000))
            out += struct.pack('>I', 4) + b'gAMA' + gma + struct.pack('>I', zlib.crc32(b'gAMA'+gma) & 0xffffffff)
        pos += 12+ln
    path.write_bytes(bytes(out))

def axis_decode(path, consumer):
    if consumer == 'pillow_raw':
        with Image.open(path) as im: return np.asarray(im.convert('RGB')).copy()
    from PIL import ImageCms
    with Image.open(path) as im:
        if im.mode != 'RGB': im = im.convert('RGB')
        if 'icc_profile' in im.info:
            srgb = ImageCms.createProfile('sRGB')
            im = ImageCms.profileToProfile(im, ImageCms.ImageCmsProfile(io.BytesIO(im.info['icc_profile'])), srgb, outputMode='RGB')
        else:
            import PIL.ImageCms as IC
            g = im.info.get('gamma')
            if g:
                lut = [int(255*((i/255)**(1.0/float(g)))) for i in range(256)] if float(g) < 1 else [int(255*((i/255)**(1.0/float(g)))) for i in range(256)]
                im = im.point(lut*3)
        return np.asarray(im).copy()

import io

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--family', required=True,
        choices=['riva_gan', 'tree_ring_rand', 'tree_ring_ring', 'dwt_dct_svd'])
    a = ap.parse_args()
    try: main(a.family)
    except Exception: traceback.print_exc(); sys.exit(1)
