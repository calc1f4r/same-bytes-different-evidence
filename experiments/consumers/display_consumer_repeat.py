#!/usr/bin/env python3
"""E5 (2026-09-28): fifth-consumer display-policy re-score of the 24 v3 EXIF eval JPEGs.

Re-scores every pilot_v3 EXIF orientation-6 evaluation JPEG under a fifth
recorded consumer path: Pillow ImageOps.exif_transpose with no autocontrast or
any other enhancement (pure display decode), identical to the frozen
exif_display decode. Purpose: confirm display-policy agreement 24/24 by fresh
recomputation, including fresh model loads for the tree-ring families.
Compares scores and detector-input hashes against the stored exif_display rows.
Read-only on existing results; writes results/e5_display_consumer_repeat.json.
Protocol JSON written before execution. Run under the study virtualenv (GPU).
"""
import json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parents[1]
FIX = HERE / "fixtures"
RES = HERE / "results"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

(RES / "e5_protocol.json").write_text(json.dumps({
    "experiment": "E5 fifth-consumer display-policy re-score",
    "written_before": "execution", "date": "2026-09-28",
    "consumer": "PIL ImageOps.exif_transpose(im).convert('RGB'); no autocontrast, no enhancement",
    "rule": "re-score all 24 exif6 eval JPEGs; compare score and detector-input hash "
            "to stored exif_display rows; expect exact equality; no threshold changes",
}, indent=1))

FAMS = ["riva_gan", "tree_ring_rand", "tree_ring_ring", "dwt_dct_svd"]

def riva_score(a, idx):
    import cv2
    from imwatermark.rivaGan import RivaWatermark
    RivaWatermark.loadModel()
    bgr = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2BGR)
    truth = np.random.default_rng(20281001 + idx).integers(0, 2, 32, dtype=np.uint8)
    ten = (np.asarray([bgr], dtype=np.float32) / 127.5 - 1).transpose(3, 0, 1, 2)[None, :, :, :, :]
    raw = RivaWatermark.decoder.run(None, {"frame": ten})[0][0]
    decoded = (raw > 0.52).astype(np.uint8)
    return float((decoded == truth).mean()), ten

def dwt_score(a, idx):
    import cv2
    from imwatermark import WatermarkDecoder
    bgr = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2BGR)
    d = WatermarkDecoder("bits", 32)
    dec = d.decode(bgr, "dwtDctSvd")
    truth = np.random.default_rng(20281001 + idx).integers(0, 2, 32, dtype=np.uint8)
    return float((np.asarray(dec, dtype=np.uint8) == truth).mean()), bgr

def make_treering(kind):
    import torch
    from diffusers import DDIMScheduler
    from treering import (InversableStableDiffusionPipeline, get_watermarking_mask,
                          get_watermarking_pattern, transform_img, l1_complex_metric)
    from common import derive_seed
    model = Path.home() / ".cache/huggingface/hub/models--WIBE-HuggingFace--stable-diffusion-2-1-base/snapshots/94e48088b3d3d6fee9e0176d12425770aaa4b2c4"
    sch = DDIMScheduler.from_pretrained(str(model), subfolder="scheduler", local_files_only=True)
    pipe = InversableStableDiffusionPipeline.from_pretrained(str(model), scheduler=sch,
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
        local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    empty = pipe.get_text_embedding("")
    def score(a, idx):
        g = torch.Generator(device="cuda").manual_seed(derive_seed(f"pilot_v3_key_{kind}", 20281001, idx))
        init = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.float16, generator=g)
        mask = get_watermarking_mask(init, 0, 5, "cuda", "circle")
        pattern = get_watermarking_pattern(init, kind, 5)
        t = transform_img(Image.fromarray(a, "RGB"), 256).unsqueeze(0).to(empty.dtype).to("cuda")
        latent = pipe.get_image_latents(t, sample=False)
        inv = pipe.forward_diffusion(latents=latent, text_embeddings=empty, guidance_scale=1, num_inference_steps=25)
        return float(-l1_complex_metric(inv, mask, pattern)), t.detach().cpu().contiguous().numpy()
    return score

import hashlib
def th(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() if isinstance(x, np.ndarray) else th(np.asarray(x))

rows, treering_cache = [], {}
t0 = time.monotonic()
for fam in FAMS:
    stored = {}
    for line in (RES / f"{fam}.jsonl").open():
        r = json.loads(line)
        if r["kind"] == "exif_display":
            stored[r["idx"]] = r
    for pth in sorted(FIX.glob(f"{fam}_*_exif6.jpg")):
        idx = int(pth.name[len(fam) + 1:].split("_")[0])
        with Image.open(pth) as im:
            disp = np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()
        if fam == "riva_gan":
            s, t = riva_score(disp, idx)
        elif fam == "dwt_dct_svd":
            s, t = dwt_score(disp, idx)
        else:
            if fam not in treering_cache:
                treering_cache[fam] = make_treering("rand" if fam == "tree_ring_rand" else "ring")
            s, t = treering_cache[fam](disp, idx)
        st = stored[idx]
        rows.append({
            "family": fam, "idx": idx, "file": pth.name,
            "score_new": s, "score_stored": st["score"],
            "score_equal": s == st["score"],
            "tensor_hash_new": th(t), "tensor_hash_stored": st["tensor_hash"],
            "tensor_hash_equal": th(t) == st["tensor_hash"],
            "file_sha256_match": st["file_sha256"] == hashlib.sha256(pth.read_bytes()).hexdigest(),
        })
        print(f"{fam} {idx}: score {s} vs {st['score']} {'OK' if s == st['score'] else 'MISMATCH'} "
              f"({time.monotonic()-t0:.0f}s)", flush=True)
    if fam in treering_cache:
        del treering_cache[fam]
        import torch, gc
        gc.collect(); torch.cuda.empty_cache()

ok = sum(1 for r in rows if r["score_equal"] and r["tensor_hash_equal"])
doc = {"generated": "2026-09-28",
       "consumer": "Pillow ImageOps.exif_transpose, no autocontrast (pure display)",
       "families": FAMS, "n": len(rows),
       "score_and_tensor_equal": f"{ok}/{len(rows)}",
       "rows": rows}
(RES / "e5_display_consumer_repeat.json").write_text(json.dumps(doc, indent=1))
print(f"RESULT {ok}/{len(rows)}")
