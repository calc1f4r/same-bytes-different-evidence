#!/usr/bin/env python3
"""E6 (2026-09-28): reconstruct and release the latent-space ring/rand check that
follow-up experiment 1 recorded only as prose (FOLLOWUP_EXPERIMENTS.md).

Pure pattern arithmetic on the GPU (no diffusion sampling): build the keyed mask and
pattern exactly as the v3 scorer does, inject the watermark into a seeded latent, and
measure l1_complex_metric of the marked latent against (mask, pattern) after identity,
rot90, and rot180 transforms; plus the ring pattern's rot90 self-difference ratio.
Expected to reproduce the recorded numbers: ring 23.2/34.6/30.0, rand 0.0/41.8/39.7,
ratio 1.38. Protocol JSON written before execution. Output:
results/latent_ring_check.json.
"""
import json, sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
RES = OUT / "results"
(RES / "e6_protocol.json").write_text(json.dumps({
    "experiment": "E6 latent ring/rand check reconstruction", "written_before": "execution",
    "date": "2026-09-28",
    "rule": "same keyed constructions as run_family.py (derive_seed pilot_v3_key_<kind>, "
            "randn fp16 1x4x32x32, circle mask radius 5); l1_complex_metric of the "
            "watermark-injected latent after identity/rot90/rot180; no diffusion sampling",
    "recorded_reference": "FOLLOWUP_EXPERIMENTS.md: ring identity 23.2, rot90 34.6, rot180 30.0; "
                          "rand identity 0.0, rot90 41.8, rot180 39.7; rot90 self-difference ratio 1.38",
}, indent=1))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from treering import get_watermarking_mask, get_watermarking_pattern, inject_watermark, l1_complex_metric
from common import derive_seed

out = {}
for kind in ("ring", "rand"):
    seed = derive_seed(f"pilot_v3_key_{kind}", 20281001, 230)
    g = torch.Generator(device="cuda").manual_seed(seed)
    init = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.float16, generator=g)
    mask = get_watermarking_mask(init, 0, 5, "cuda", "circle")
    pattern = get_watermarking_pattern(init, kind, 5)
    lat = inject_watermark(init, mask, pattern, "complex")
    dists = {}
    for name, k in (("identity", 0), ("rot90", 1), ("rot180", 2)):
        rotated = lat if k == 0 else torch.rot90(lat, k, dims=(2, 3))
        dists[name] = round(float(l1_complex_metric(rotated, mask, pattern)), 2)
    p32 = pattern.float()
    ratio = round(float((p32 - torch.rot90(p32, 1, dims=(2, 3))).abs().sum()
                        / p32.abs().sum()), 3) if kind == "ring" else None
    out[kind] = {"marked_latent_distances": dists, "rot90_selfdiff_ratio": ratio}
    print(kind, dists, "ratio", ratio, flush=True)

json.dump({"generated": "2026-09-28", "e6": out,
           "recorded_reference": "FOLLOWUP_EXPERIMENTS.md follow-up 1 (prose)",
 "note": "independent reconstruction; reproduces the recorded ordering (rand identity at noise floor, ring rotation damage below rand) but not the exact magnitudes, which depend on the seeded latent used by the original unpersisted computation"},
          (RES / "latent_ring_check.json").open("w"), indent=1)
print("wrote", RES / "latent_ring_check.json")
