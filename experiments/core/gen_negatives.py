#!/usr/bin/env python3
"""v3 step 1: generate 300+ fresh unmarked negatives per negative-block via SD2.1-base GPU.
Writes pilot_v3/fixtures/neg_<idx>.png for indices in frozen protocol. Refuses overwrite."""
import json, sys, time
from pathlib import Path
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import PROMPTS, derive_seed

PROTO = json.loads((OUT/'manifests/frozen_protocol.json').read_text())
BLOCKS = {
    'riva_gan': range(400, 700),
    'tree_ring_rand': range(700, 1000),
    'tree_ring_ring': range(1000, 1300),
    'dwt_dct_svd': range(1300, 1600),
}
CAL_NEG = list(range(300, 324))

MODEL = '<home>/.cache/huggingface/hub/models--WIBE-HuggingFace--stable-diffusion-2-1-base/snapshots/94e48088b3d3d6fee9e0176d12425770aaa4b2c4'

def main(only=None):
    fix = OUT/'fixtures'; fix.mkdir(exist_ok=True)
    todo = []
    for name, rng in BLOCKS.items():
        if only and name != only: continue
        for j in rng:
            if not (fix/f'neg_{j}.png').exists(): todo.append(j)
    for j in CAL_NEG:
        if not (fix/f'neg_{j}.png').exists(): todo.append(j)
    if not todo:
        print('nothing to generate'); return
    sch = DDIMScheduler.from_pretrained(MODEL, subfolder='scheduler', local_files_only=True)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, scheduler=sch, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False, local_files_only=True).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    t0 = time.time()
    for n, j in enumerate(todo):
        seed = derive_seed('pilot_v3_neg', 20281001, j)
        g = torch.Generator(device='cuda').manual_seed(seed)
        im = pipe(PROMPTS[j % 64], num_inference_steps=25, guidance_scale=6.0,
                  height=256, width=256, generator=g).images[0]
        im.save(fix/f'neg_{j}.png')
        if n % 25 == 0:
            print(f'{n+1}/{len(todo)} neg_{j} {time.time()-t0:.0f}s', flush=True)
    print(f'DONE {len(todo)} negatives in {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
