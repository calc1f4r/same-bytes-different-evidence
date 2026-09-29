#!/usr/bin/env python3
"""E7 (2026-09-28): JPEG quality sweep of the same-byte decoder-policy split.

Question: does the EXIF-orientation-6 raw-accept / display-reject split that
pilot_v3 measured at JPEG quality 95 survive lower JPEG qualities (60, 75, 85)?

Method (identical encoder and decode conventions to run_family.py):
  - Source rasters: the 6 held-out marked evaluation lineages per config
    (fixtures/marked_<idx>.png; eval indices from manifests/frozen_protocol.json).
  - Encode per quality q in {60, 75, 85}: Pillow
    Image.fromarray(a,'RGB').save(tmp, format='JPEG', quality=q, subsampling=0,
    exif=Exif with tag 274 = 6) — same settings as the study's q95 axis, only
    the quality number changes. Encoded JPEGs are written to a temp dir OUTSIDE
    the repo (hard constraint: no new repo files besides this script and
    results/e7_jpeg_quality_sweep.json).
  - Decode BOTH paths from the identical file bytes:
      raw     = Pillow unmanaged  (Image.open().convert('RGB'), EXIF-ignoring)
      display = Pillow display    (ImageOps.exif_transpose().convert('RGB'))
  - Score with each config's FROZEN detector (identical code path to
    run_family.py / e5) and decide with the FROZEN calibration thresholds from
    results/<family>_calibration.json (minimum-rule primary, strict >;
    max-rule decision also recorded). NO recalibration anywhere.
  - Reference q95 numbers are NOT re-measured: they are read from the existing
    results/<family>.jsonl rows (kind exif_verifier_raw / exif_display,
    split 'evaluation') and embedded for comparison.

Tree-ring: DDIM inversion per raster exactly as e5 (25 steps, fp16, batch 1).
Qualities are ordered 75 first then 60, 85 so the priority quality lands even
if the GPU is lost mid-run. On CUDA OOM: wait 60 s and retry (shared GPU).
If a family/quality cannot be run it is recorded honestly as not run.

Protocol snapshot is written into results/e7_jpeg_quality_sweep.json BEFORE
execution (status 'running'); the final JSON overwrites it with results.
Run under the study virtualenv (GPU for tree-ring; CPU for riva/dwt).
"""
import json, subprocess, sys, tempfile, time, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parents[1]
FIX = HERE / "fixtures"
RES = HERE / "results"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

RESULT_JSON = RES / "e7_jpeg_quality_sweep.json"
QUALITIES = [75, 60, 85]          # execution order; 75 is the priority quality
QUALITIES_SORTED = [60, 75, 85]
FAMS = ["riva_gan", "tree_ring_rand", "tree_ring_ring", "dwt_dct_svd"]
EVAL = {  # from manifests/frozen_protocol.json -> families[<fam>].marked_eval
    "riva_gan": [206, 207, 208, 209, 210, 211],
    "tree_ring_rand": [218, 219, 220, 221, 222, 223],
    "tree_ring_ring": [230, 231, 232, 233, 234, 235],
    "dwt_dct_svd": [242, 243, 244, 245, 246, 247],
}
BITS_OFF = 20281001  # watermark bit truth offset, as frozen in run_family.py

def versions():
    import PIL, cv2, torch, importlib.metadata as md
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "invisible_watermark": md.version("invisible-watermark"),
        "diffusers": md.version("diffusers"),
    }

CAL = {f: json.loads((RES / f"{f}_calibration.json").read_text()) for f in FAMS}
PROTO_NOTE = {
    "experiment": "E7 JPEG quality sweep of the same-byte decoder-policy split",
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "question": "does the EXIF orientation-6 raw-accept/display-reject split "
                "measured at JPEG q95 survive lower qualities (60, 75, 85)?",
    "written_before": "execution (this protocol snapshot written to the result "
                      "JSON first with status=running; final JSON replaces it)",
    "qualities_new": QUALITIES_SORTED,
    "quality_reference": 95,
    "reference_q95_source": "existing results/<family>.jsonl rows "
                            "(kind=exif_verifier_raw / exif_display, split=evaluation); "
                            "NOT re-measured",
    "encoder_settings": "Pillow Image.fromarray(a,'RGB').save(path, format='JPEG', "
                        "quality=q, subsampling=0, exif=Exif{274:6}) — identical to the "
                        "frozen q95 axis in run_family.py except the quality number",
    "decode_paths": {
        "raw": "Pillow unmanaged: Image.open(jpg).convert('RGB') (EXIF-ignoring)",
        "display": "Pillow display: ImageOps.exif_transpose(Image.open(jpg)).convert('RGB')",
    },
    "thresholds": {
        f: {
            "threshold_min_rule": CAL[f]["threshold_min_rule"],
            "threshold_max_rule": CAL[f]["threshold_max_rule"],
            "decision_rule": "strict >",
            "source_file": f"results/{f}_calibration.json",
            "recalibrated": False,
        } for f in FAMS
    },
    "eval_lineages": EVAL,
    "split_definition": "per lineage: raw-path accept AND display-path reject "
                        "(both decided with the frozen minimum-rule threshold, strict >)",
    "temp_jpeg_dir": "outside the repo (tempfile.mkdtemp); no new repo files "
                     "besides this script and results/e7_jpeg_quality_sweep.json",
    "gpu_policy": "check nvidia-smi free memory before model load; batch 1; on "
                  "CUDA OOM wait 60 s and retry (max 5)",
    "versions": versions(),
}

def write_doc(status, **kw):
    doc = {"protocol": PROTO_NOTE, "status": status,
           "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    doc.update(kw)
    RESULT_JSON.write_text(json.dumps(doc, indent=1))

# ---------- shared encode / decode ----------
def encode_exif6(a, path, q):
    exif = Image.Exif(); exif[274] = 6
    Image.fromarray(a, "RGB").save(path, format="JPEG", quality=q, subsampling=0, exif=exif)

def decode_both(path):
    with Image.open(path) as im:
        raw = np.asarray(im.convert("RGB")).copy()
        disp = np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()
        orient = im.getexif().get(274)
    return raw, disp, orient

def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()

def bits(idx):
    return np.random.default_rng(BITS_OFF + idx).integers(0, 2, 32, dtype=np.uint8)

# ---------- CPU scorers (riva_gan, dwt_dct_svd) — code path of e5/run_family ----------
def make_riva():
    import cv2
    from imwatermark.rivaGan import RivaWatermark
    RivaWatermark.loadModel()
    def score(a, idx):
        bgr = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2BGR)
        truth = bits(idx)
        ten = (np.asarray([bgr], dtype=np.float32) / 127.5 - 1).transpose(3, 0, 1, 2)[None, :, :, :, :]
        out = RivaWatermark.decoder.run(None, {"frame": ten})[0][0]
        decoded = (out > 0.52).astype(np.uint8)
        return float((decoded == truth).mean())
    return score

def make_dwt():
    import cv2
    from imwatermark import WatermarkDecoder
    def score(a, idx):
        bgr = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2BGR)
        d = WatermarkDecoder("bits", 32)
        dec = d.decode(bgr, "dwtDctSvd")
        truth = bits(idx)
        return float((np.asarray(dec, dtype=np.uint8) == truth).mean())
    return score

# ---------- tree-ring scorer — code path of e5/run_family (GPU DDIM inversion) ----------
def gpu_free_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, timeout=30, check=True)
        return int(out.stdout.decode().strip().splitlines()[0])
    except Exception:
        return -1

def wait_for_gpu(min_free_mib=3000, tries=10, wait_s=60):
    for _ in range(tries):
        f = gpu_free_mib()
        if f < 0 or f >= min_free_mib:
            return f
        print(f"GPU free {f} MiB < {min_free_mib}; waiting {wait_s}s", flush=True)
        time.sleep(wait_s)
    return gpu_free_mib()

def make_treering(kind):
    import torch
    from diffusers import DDIMScheduler
    from treering import (InversableStableDiffusionPipeline, get_watermarking_mask,
                          get_watermarking_pattern, transform_img, l1_complex_metric)
    from common import derive_seed
    model = Path.home() / ".cache/huggingface/hub/models--WIBE-HuggingFace--stable-diffusion-2-1-base/snapshots/94e48088b3d3d6fee9e0176d12425770aaa4b2c4"
    assert model.exists(), f"model snapshot missing: {model}"
    sch = DDIMScheduler.from_pretrained(str(model), subfolder="scheduler", local_files_only=True)
    pipe = InversableStableDiffusionPipeline.from_pretrained(str(model), scheduler=sch,
        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
        local_files_only=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    empty = pipe.get_text_embedding("")
    def score(a, idx):
        g = torch.Generator(device="cuda").manual_seed(
            derive_seed(f"pilot_v3_key_{kind}", 20281001, idx))
        init = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.float16, generator=g)
        mask = get_watermarking_mask(init, 0, 5, "cuda", "circle")
        pattern = get_watermarking_pattern(init, kind, 5)
        t = transform_img(Image.fromarray(a, "RGB"), 256).unsqueeze(0).to(empty.dtype).to("cuda")
        latent = pipe.get_image_latents(t, sample=False)
        inv = pipe.forward_diffusion(latents=latent, text_embeddings=empty,
                                     guidance_scale=1, num_inference_steps=25)
        return float(-l1_complex_metric(inv, mask, pattern))
    return score

def with_oom_retry(fn, tries=5, wait_s=60):
    for attempt in range(tries):
        try:
            return fn(), None
        except Exception as e:  # torch.cuda.OutOfMemoryError / RuntimeError OOM
            msg = f"{type(e).__name__}: {e}"
            if "out of memory" in msg.lower() or "OutOfMemory" in type(e).__name__:
                print(f"CUDA OOM ({msg[:120]}); waiting {wait_s}s "
                      f"(attempt {attempt+1}/{tries})", flush=True)
                try:
                    import torch, gc
                    gc.collect(); torch.cuda.empty_cache()
                except Exception:
                    pass
                time.sleep(wait_s)
                continue
            raise
    return None, "CUDA OOM retries exhausted"

# ---------- reference q95 from existing rows ----------
def q95_reference():
    ref = {}
    for fam in FAMS:
        rows = {}
        for line in (RES / f"{fam}.jsonl").open():
            r = json.loads(line)
            if r["split"] == "evaluation" and r["kind"] in ("exif_verifier_raw", "exif_display"):
                rows.setdefault(r["idx"], {})[r["kind"]] = {
                    "score": r["score"], "decision": r["decision"],
                    "decision_max_rule": r.get("decision_max_rule"),
                    "source": f"results/{fam}.jsonl"}
        ref[fam] = rows
    return ref

def summarize(rows, tmin, tmax):
    """rows: list of dicts with idx, path, score, decision, decision_max_rule."""
    n = len(rows)
    raw = [r for r in rows if r["path"] == "raw"]
    disp = [r for r in rows if r["path"] == "display"]
    raw_acc = sum(1 for r in raw if r["decision"])
    disp_rej = sum(1 for r in disp if not r["decision"])
    disp_acc_max = sum(1 for r in disp if r.get("decision_max_rule"))
    split_idx = [r["idx"] for r in raw
                 if r["decision"] and not any(d["idx"] == r["idx"] and d["decision"]
                                              for d in disp)]
    rng = lambda rs: ([min(x["score"] for x in rs), max(x["score"] for x in rs)]
                      if rs else None)
    return {
        "n_lineages": n // 2,
        "raw_accept_count": raw_acc,
        "display_reject_count": disp_rej,
        "split_count": len(split_idx), "split_idx": split_idx,
        "split_fraction": f"{len(split_idx)}/{n // 2}" if n else "0/0",
        "raw_score_range": rng(raw), "display_score_range": rng(disp),
        "display_accept_count_max_rule": disp_acc_max,
    }

# ---------- main ----------
def main():
    t0 = time.monotonic()
    write_doc("running", note="protocol snapshot; results follow after execution")
    tmpdir = Path(tempfile.mkdtemp(prefix="e7_quality_jpgs_"))
    print("temp jpeg dir:", tmpdir, flush=True)

    ref95 = q95_reference()
    results = {f: {} for f in FAMS}
    failures = []

    # ---- CPU families first ----
    for fam, scorer in (("riva_gan", make_riva()), ("dwt_dct_svd", make_dwt())):
        tmin = CAL[fam]["threshold_min_rule"]; tmax = CAL[fam]["threshold_max_rule"]
        for q in QUALITIES_SORTED:
            rows = []
            try:
                for idx in EVAL[fam]:
                    a = rgb(FIX / f"marked_{idx}.png")
                    jpg = tmpdir / f"{fam}_{idx}_q{q}_exif6.jpg"
                    encode_exif6(a, jpg, q)
                    rawr, dispr, orient = decode_both(jpg)
                    assert orient == 6, f"orientation tag {orient} != 6"
                    assert rawr.shape == (256, 256, 3) and dispr.shape == (256, 256, 3)
                    for name, rast in (("raw", rawr), ("display", dispr)):
                        s = scorer(rast, idx)
                        rows.append({"idx": idx, "quality": q, "path": name, "score": s,
                                     "decision": s > tmin, "decision_max_rule": s > tmax})
                    print(f"{fam} q{q} idx {idx}: raw {rows[-2]['score']} "
                          f"disp {rows[-1]['score']} ({time.monotonic()-t0:.0f}s)", flush=True)
                results[fam][f"q{q}"] = {"rows": rows, **summarize(rows, tmin, tmax)}
            except Exception as e:
                failures.append({"family": fam, "quality": q,
                                 "error": f"{type(e).__name__}: {e}"})
                traceback.print_exc()
                if rows:
                    results[fam][f"q{q}"] = {"rows": rows, **summarize(rows, tmin, tmax),
                                             "partial": True}

    # ---- GPU tree-ring families (priority order: 75, then 60, 85) ----
    free = wait_for_gpu()
    print(f"GPU free before tree-ring: {free} MiB", flush=True)
    for fam in ("tree_ring_rand", "tree_ring_ring"):
        tmin = CAL[fam]["threshold_min_rule"]; tmax = CAL[fam]["threshold_max_rule"]
        kind = "rand" if fam == "tree_ring_rand" else "ring"
        scorer = None
        try:
            scorer = make_treering(kind)
        except Exception as e:
            failures.append({"family": fam, "quality": "all",
                             "error": f"model load failed: {type(e).__name__}: {e}"})
            traceback.print_exc()
            continue
        for q in QUALITIES:
            rows = []
            try:
                for idx in EVAL[fam]:
                    a = rgb(FIX / f"marked_{idx}.png")
                    jpg = tmpdir / f"{fam}_{idx}_q{q}_exif6.jpg"
                    encode_exif6(a, jpg, q)
                    rawr, dispr, orient = decode_both(jpg)
                    assert orient == 6, f"orientation tag {orient} != 6"
                    for name, rast in (("raw", rawr), ("display", dispr)):
                        res, err = with_oom_retry(lambda rast=rast, idx=idx: scorer(rast, idx))
                        if err:
                            raise RuntimeError(err)
                        rows.append({"idx": idx, "quality": q, "path": name, "score": res,
                                     "decision": res > tmin, "decision_max_rule": res > tmax})
                    print(f"{fam} q{q} idx {idx}: raw {rows[-2]['score']:.3f} "
                          f"disp {rows[-1]['score']:.3f} ({time.monotonic()-t0:.0f}s)", flush=True)
                results[fam][f"q{q}"] = {"rows": rows, **summarize(rows, tmin, tmax)}
            except Exception as e:
                failures.append({"family": fam, "quality": q,
                                 "error": f"{type(e).__name__}: {e}"})
                traceback.print_exc()
                if rows:
                    results[fam][f"q{q}"] = {"rows": rows, **summarize(rows, tmin, tmax),
                                             "partial": True}
        del scorer
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()

    # ---- attach q95 reference rows in the same summary shape ----
    for fam in FAMS:
        tmin = CAL[fam]["threshold_min_rule"]
        tmax = CAL[fam]["threshold_max_rule"]
        rows = []
        for idx in EVAL[fam]:
            r = ref95[fam].get(idx, {})
            if "exif_verifier_raw" in r and "exif_display" in r:
                rows.append({"idx": idx, "quality": 95, "path": "raw", **r["exif_verifier_raw"]})
                rows.append({"idx": idx, "quality": 95, "path": "display", **r["exif_display"]})
        results[fam]["q95_reference"] = {
            "rows": rows, "remeasured": False,
            "source": f"results/{fam}.jsonl (evaluation split)",
            **summarize(rows, tmin, tmax)}

    write_doc("complete" if not failures else "complete_with_failures",
              qualities=QUALITIES_SORTED, results=results, failures=failures,
              gpu_free_mib_before_treering=free,
              wall_seconds=round(time.monotonic() - t0, 1),
              temp_dir_note=f"encoded JPEGs kept outside repo in {tmpdir}")
    print("wrote", RESULT_JSON, flush=True)
    for fam in FAMS:
        for k in sorted(results[fam]):
            if k.startswith("q") and "rows" in results[fam][k]:
                s = results[fam][k]
                print(f"{fam} {k}: split {s['split_fraction']} "
                      f"(raw accept {s['raw_accept_count']}, display reject {s['display_reject_count']})",
                      flush=True)

if __name__ == "__main__":
    main()
