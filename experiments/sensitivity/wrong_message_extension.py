#!/usr/bin/env python3
"""E8 (2026-09-28): wrong-message / wrong-key extension on larger populations.

Extends the v3 wrong-key battery (which used only the 6 calibration images per
config, offset-20291001 per-index messages for imwatermark, per-family
'pilot_v3_wrong_*' keys for tree-ring) to every available marked lineage:

1. riva_gan / dwt_dct_svd (imwatermark): the 6 marked_eval lineages PLUS the 30
   orientation-breadth marked lineages per config (indices 1600..1629 / 1630..1659,
   embedded message bits(idx) via the frozen rule). Each file is decoded through the
   raw unmanaged-Pillow consumer path, bit accuracy is computed against ONE FIXED
   random 32-bit wrong message (seed recorded below), and the FROZEN v3 min-rule
   threshold (strict >, never recalibrated) decides accept/reject. Max-rule decisions
   recorded alongside, per the v3 sensitivity convention. Correct-message scores are
   recomputed as a decode-path fidelity check against the recorded identity rows.
2. tree_ring_rand / tree_ring_ring: marked evaluation latents are recomputed via the
   e6/run_family inversion machinery (get_image_latents + 25-step DDIM forward
   diffusion, batch 1) and scored under the WRONG key = the OTHER config's keyed
   pattern (ring-marked lineages vs the rand key's pattern and vice versa), decided
   against the lineage-owner config's frozen min-rule threshold. One identity
   re-score per family validates the machinery against the recorded v3 identity row.

Read-only on all existing files. Writes only results/e8_wrong_message_extension.json
(opened 'x'; refuses overwrite). Run under the study virtualenv. GPU section: waits for
>= 4500 MiB free before model load, batch 1, retries once-per-60s on CUDA OOM.
"""
import hashlib
import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parents[1]          # pilot_v3
FIX = HERE / "fixtures"
RES = HERE / "results"
BENCH = Path(__file__).resolve().parents[3] / "bench"
sys.path.insert(0, str(BENCH / "src"))
sys.path.insert(0, str(HERE / "scripts"))

from common import derive_seed  # noqa: E402
from run_family import bits     # noqa: E402  frozen embedded-message rule: bits(idx) = default_rng(20281001+idx)

PROTO_SHA = hashlib.sha256((HERE / "manifests" / "frozen_protocol.json").read_bytes()).hexdigest()
DATE = "2026-09-28"

# ---------------- fixed wrong message (single message for ALL imwatermark rows) ----------
WM_SEED_LABELS = ("e8_wrong_message", 20281001, 0)
WM_SEED = derive_seed(*WM_SEED_LABELS)
WRONG_MSG = np.random.default_rng(WM_SEED).integers(0, 2, 32, dtype=np.uint8)
WRONG_HEX = "".join(f"{int(''.join(str(b) for b in WRONG_MSG[i:i+8]), 2):02x}" for i in range(0, 32, 8))
WRONG_BITS_STR = "".join(str(b) for b in WRONG_MSG)

EVAL_POP = {"riva_gan": list(range(206, 212)), "dwt_dct_svd": list(range(242, 248))}
BREADTH_POP = {"riva_gan": list(range(1600, 1630)), "dwt_dct_svd": list(range(1630, 1660))}
TREE_EVAL = {"tree_ring_rand": list(range(218, 224)), "tree_ring_ring": list(range(230, 236))}
CROSS_KIND = {"tree_ring_rand": "ring", "tree_ring_ring": "rand"}  # owner -> wrong (other) pattern kind

failures = []


def sha(b):
    return hashlib.sha256(b).hexdigest()


def arr_hash(a):
    return sha(np.ascontiguousarray(a).tobytes())


def load_thr(fam):
    cal = json.loads((RES / f"{fam}_calibration.json").read_text())
    return cal["threshold_min_rule"], cal["threshold_max_rule"]


def pillow_raw(path):
    """Frozen raw unmanaged consumer: Image.open().convert('RGB'), EXIF-ignoring."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


# ---------------- recorded identity scores for fidelity comparison ----------------
def recorded_identity_scores():
    rec = {}
    for fam in ("riva_gan", "dwt_dct_svd"):
        for line in (RES / f"{fam}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "identity" and r.get("split") == "evaluation":
                rec[(fam, r["idx"])] = r["score"]
    ob = json.loads((RES / "orientation_breadth.json").read_text())
    for fam in ("riva_gan", "dwt_dct_svd"):
        for row in ob[fam]["rows"]:
            if row["orient"] == 2:  # identity_score is repeated per orient row; take once
                rec[(fam, row["idx"])] = row["identity_score"]
    return rec


# ---------------- imwatermark decoders (exact run_family decode paths) ----------------
def riva_decode(a):
    import cv2
    from imwatermark.rivaGan import RivaWatermark
    RivaWatermark.loadModel()
    bgr = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2BGR)
    ten = (np.asarray([bgr], dtype=np.float32) / 127.5 - 1).transpose(3, 0, 1, 2)[None, :, :, :, :]
    raw = RivaWatermark.decoder.run(None, {"frame": ten})[0][0]
    return (raw > 0.52).astype(np.uint8), ten


def dwt_decode(a):
    import cv2
    from imwatermark import WatermarkDecoder
    bgr = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2BGR)
    d = WatermarkDecoder("bits", 32)
    dec = d.decode(bgr, "dwtDctSvd")
    if dec is None:
        return None, bgr
    return np.asarray(dec, dtype=np.uint8), bgr


DECODE = {"riva_gan": riva_decode, "dwt_dct_svd": dwt_decode}


def imwatermark_block():
    rec = recorded_identity_scores()
    out = {}
    for fam in ("riva_gan", "dwt_dct_svd"):
        t_min, t_max = load_thr(fam)
        rows, skips = [], []
        pop = [(idx, FIX / f"marked_{idx}.png", "marked_eval") for idx in EVAL_POP[fam]]
        pop += [(idx, FIX / f"breadth_{idx}_marked.png", "breadth") for idx in BREADTH_POP[fam]]
        for idx, path, cohort in pop:
            row = {"idx": idx, "cohort": cohort, "file": str(path.relative_to(HERE))}
            try:
                if not path.exists():
                    row["status"] = "skipped_missing_file"
                    skips.append(row)
                    continue
                a = pillow_raw(path)
                dec, _inp = DECODE[fam](a)
                if dec is None:
                    row["status"] = "decode_failed"
                    row["decoded_bits"] = None
                    skips.append(row)
                    failures.append(f"{fam} idx {idx}: decoder returned None")
                    continue
                embedded = bits(idx)
                acc_wrong = float((dec == WRONG_MSG).mean())
                acc_right = float((dec == embedded).mean())
                row.update({
                    "status": "scored",
                    "file_sha256": sha(path.read_bytes()),
                    "rgb_hash": arr_hash(a),
                    "wrong_message_score": acc_wrong,
                    "decision": acc_wrong > t_min,
                    "decision_max_rule": acc_wrong > t_max,
                    "correct_message_score": acc_right,
                    "correct_message_decision": acc_right > t_min,
                    "embedded_message_hex": "".join(
                        f"{int(''.join(str(b) for b in embedded[i:i+8]), 2):02x}" for i in range(0, 32, 8)),
                    "hamming_wrong_vs_embedded": int((WRONG_MSG != embedded).sum()),
                })
                recd = rec.get((fam, idx))
                row["recorded_identity_score"] = recd
                row["matches_recorded_identity"] = (recd is not None and acc_right == recd)
            except Exception as e:  # record honestly, continue
                row["status"] = "error"
                row["error"] = f"{type(e).__name__}: {e}"
                failures.append(f"{fam} idx {idx}: {row['error']}")
                skips.append(row)
            rows.append(row)
        scored = [r for r in rows if r.get("status") == "scored"]
        out[fam] = {
            "population": {
                "marked_eval": len(EVAL_POP[fam]), "breadth": len(BREADTH_POP[fam]),
                "total": len(EVAL_POP[fam]) + len(BREADTH_POP[fam]),
                "scored": len(scored), "skipped": len(skips),
                "skipped_rows": [s.get("idx") for s in skips],
            },
            "embedded_message_rule": "bits(idx) = np.random.default_rng(20281001+idx).integers(0,2,32,uint8) "
                                     "(frozen v3 rule; recoverable for every lineage, none skipped for unrecoverable message)",
            "threshold_min_rule": t_min, "threshold_max_rule": t_max,
            "threshold_source": f"results/{fam}_calibration.json (frozen, reused, NOT recalibrated)",
            "wrong_message_accepts": sum(1 for r in scored if r["decision"]),
            "wrong_message_accepts_max_rule": sum(1 for r in scored if r["decision_max_rule"]),
            "correct_message_identity_accepts": sum(1 for r in scored if r["correct_message_decision"]),
            "fidelity_check": {
                "compared": sum(1 for r in scored if r["recorded_identity_score"] is not None),
                "exact_match": sum(1 for r in scored if r["matches_recorded_identity"]),
            },
            "rows": rows,
        }
        print(f"{fam}: wrong-message accepts {out[fam]['wrong_message_accepts']}/{len(scored)} "
              f"(max-rule {out[fam]['wrong_message_accepts_max_rule']}); "
              f"fidelity {out[fam]['fidelity_check']['exact_match']}/{out[fam]['fidelity_check']['compared']}",
              flush=True)
    return out


# ---------------- tree-ring cross-key block (GPU) ----------------
def gpu_free_mib():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return int(o.stdout.strip().splitlines()[0])
    except Exception as e:
        failures.append(f"nvidia-smi query failed: {e}")
        return -1


def treering_block(gate_mib=4500, patience_min=20):
    """gate_mib: required free MiB before model load. patience_min: total wait for the gate.
    A first full run used 4500/20; a merge-retry may lower the gate since the fp16
    SD2.1 pipeline + 25-step 256px inversion fits in ~3.3 GB."""
    block = {"status": "not_run", "reason": None, "gpu_free_mib_at_load": None, "gate_mib": gate_mib}
    deadline = time.monotonic() + patience_min * 60
    free = gpu_free_mib()
    while free < gate_mib and time.monotonic() < deadline:
        print(f"waiting for GPU (free {free} MiB < {gate_mib})", flush=True)
        time.sleep(30)
        free = gpu_free_mib()
    if free < gate_mib:
        block["reason"] = (f"GPU never had >= {gate_mib} MiB free within {patience_min} min "
                           f"(last {free} MiB); tree-ring wrong-key NOT run")
        failures.append(block["reason"])
        return block
    block["gpu_free_mib_at_load"] = free

    import torch
    from diffusers import DDIMScheduler
    from treering import (InversableStableDiffusionPipeline, get_watermarking_mask,
                          get_watermarking_pattern, transform_img, l1_complex_metric)
    model = Path.home() / ".cache/huggingface/hub/models--WIBE-HuggingFace--stable-diffusion-2-1-base/snapshots/94e48088b3d3d6fee9e0176d12425770aaa4b2c4"
    sch = DDIMScheduler.from_pretrained(str(model), subfolder="scheduler", local_files_only=True)
    pipe = None
    for attempt in range(3):
        try:
            pipe = InversableStableDiffusionPipeline.from_pretrained(
                str(model), scheduler=sch, torch_dtype=torch.float16,
                safety_checker=None, requires_safety_checker=False, local_files_only=True).to("cuda")
            break
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                raise
            msg = f"model-load OOM attempt {attempt+1} at {datetime.now(timezone.utc).isoformat()}; sleeping 60s"
            print(msg, flush=True)
            failures.append(msg)
            time.sleep(60)
    if pipe is None:
        block["reason"] = "model load failed after 3 OOM retries; tree-ring wrong-key NOT run"
        failures.append(block["reason"])
        return block
    pipe.set_progress_bar_config(disable=True)
    empty = pipe.get_text_embedding("")

    def key_for(kind, idx):
        """Exact run_family key construction, parameterised by pattern kind."""
        seed = derive_seed(f"pilot_v3_key_{kind}", 20281001, idx)
        g = torch.Generator(device="cuda").manual_seed(seed)
        init = torch.randn(1, 4, 32, 32, device="cuda", dtype=torch.float16, generator=g)
        return get_watermarking_mask(init, 0, 5, "cuda", "circle"), get_watermarking_pattern(init, kind, 5), seed

    def invert_distance(a, mask, pattern, attempt_log):
        """run_family scoring path: transform -> VAE latents -> 25-step DDIM inversion -> l1 distance."""
        for attempt in range(3):
            try:
                t = transform_img(Image.fromarray(a, "RGB"), 256).unsqueeze(0).to(empty.dtype).to("cuda")
                latent = pipe.get_image_latents(t, sample=False)
                inv = pipe.forward_diffusion(latents=latent, text_embeddings=empty,
                                             guidance_scale=1, num_inference_steps=25)
                return float(l1_complex_metric(inv, mask, pattern))
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if "out of memory" not in str(e).lower() and not isinstance(e, torch.cuda.OutOfMemoryError):
                    raise
                attempt_log.append(f"OOM attempt {attempt+1} at {datetime.now(timezone.utc).isoformat()}; sleeping 60s")
                torch.cuda.empty_cache()
                time.sleep(60)
        raise RuntimeError("3 OOM retries exhausted")

    out = {}
    for owner in ("tree_ring_rand", "tree_ring_ring"):
        t_min, t_max = load_thr(owner)
        other_t_min, _ = load_thr(f"tree_ring_{CROSS_KIND[owner]}")
        wrong_kind = CROSS_KIND[owner]
        rows, skips = [], []
        for n, idx in enumerate(TREE_EVAL[owner]):
            path = FIX / f"marked_{idx}.png"
            row = {"idx": idx, "file": str(path.relative_to(HERE)), "wrong_key_kind": wrong_kind}
            try:
                if not path.exists():
                    row["status"] = "skipped_missing_file"
                    skips.append(row)
                    continue
                row["file_sha256"] = sha(path.read_bytes())
                a = pillow_raw(path)
                mask, pattern, seed = key_for(wrong_kind, idx)
                row["wrong_key_seed"] = seed
                row["wrong_key_rule"] = f"derive_seed('pilot_v3_key_{wrong_kind}', 20281001, {idx}) -> randn fp16 -> circle mask r5 + {wrong_kind} pattern"
                oom_log = []
                dist = invert_distance(a, mask, pattern, oom_log)
                if oom_log:
                    row["oom_retries"] = oom_log
                score = -dist
                row.update({
                    "status": "scored",
                    "wrong_key_score": score,
                    "decision_owner_threshold": score > t_min,
                    "decision_owner_threshold_max_rule": score > t_max,
                    "decision_other_config_threshold": score > other_t_min,
                })
                # machinery validation: identity re-score of the FIRST eval image per family
                if n == 0:
                    m2, p2, s2 = key_for(owner.split("_")[-1], idx)
                    row["identity_revalidation"] = {
                        "key_rule": f"derive_seed('pilot_v3_key_{owner.split('_')[-1]}', 20281001, {idx})",
                        "identity_score": -invert_distance(a, m2, p2, oom_log),
                    }
            except Exception as e:
                row["status"] = "error"
                row["error"] = f"{type(e).__name__}: {e}"
                failures.append(f"{owner} idx {idx}: {row['error']}")
                skips.append(row)
            rows.append(row)
            print(f"{owner} idx {idx} done", flush=True)
        scored = [r for r in rows if r.get("status") == "scored"]
        # recorded identity rows for comparison
        rec_ident = {}
        for line in (RES / f"{owner}.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "identity" and r.get("split") == "evaluation":
                rec_ident[r["idx"]] = r["score"]
        for r in scored:
            if "identity_revalidation" in r:
                r["identity_revalidation"]["recorded_identity_score"] = rec_ident.get(r["idx"])
                r["identity_revalidation"]["matches_recorded"] = (
                    r["identity_revalidation"]["identity_score"] == rec_ident.get(r["idx"]))
        out[owner] = {
            "population": {"marked_eval": len(TREE_EVAL[owner]), "scored": len(scored), "skipped": len(skips),
                           "skipped_rows": [s.get("idx") for s in skips]},
            "wrong_key_rule": f"marked {owner} lineages scored against the OTHER config's keyed pattern "
                              f"({wrong_kind}); decision under {owner} frozen min-rule threshold (strict >)",
            "threshold_min_rule": t_min, "threshold_max_rule": t_max,
            "other_config_threshold_min_rule": other_t_min,
            "threshold_source": f"results/{owner}_calibration.json (frozen, reused, NOT recalibrated)",
            "wrong_key_accepts": sum(1 for r in scored if r["decision_owner_threshold"]),
            "wrong_key_accepts_max_rule": sum(1 for r in scored if r["decision_owner_threshold_max_rule"]),
            "wrong_key_accepts_other_threshold": sum(1 for r in scored if r["decision_other_config_threshold"]),
            "rows": rows,
        }
        print(f"{owner}: wrong-key accepts {out[owner]['wrong_key_accepts']}/{len(scored)}", flush=True)
    block["status"] = "run"
    block["configs"] = out
    block["identity_revalidation"] = {
        o: next((r["identity_revalidation"] for r in out[o]["rows"] if "identity_revalidation" in r), None)
        for o in out}
    return block


def versions():
    import importlib.metadata as md
    import PIL
    import numpy
    import cv2
    v = {
        "python": sys.version.split()[0],
        "invisible-watermark (imwatermark)": md.version("invisible-watermark"),
        "Pillow": PIL.__version__,
        "numpy": numpy.__version__,
        "opencv-python": cv2.__version__,
    }
    try:
        import torch
        v["torch"] = torch.__version__
    except Exception:
        v["torch"] = "not_imported"
    try:
        import diffusers
        v["diffusers"] = diffusers.__version__
    except Exception:
        v["diffusers"] = "not_imported"
    return v


def main():
    out_path = RES / "e8_wrong_message_extension.json"
    started = datetime.now(timezone.utc).isoformat()
    if out_path.exists():
        # Merge-retry mode: a previous invocation wrote the file but left tree-ring not_run
        # (shared-GPU gate). Re-run ONLY the tree-ring block at a lower gate and merge into
        # this file; imwatermark rows are preserved verbatim from the first run.
        doc = json.loads(out_path.read_text())
        if doc["tree_ring_wrong_key"].get("status") == "run":
            print("tree-ring already run; nothing to do", flush=True)
            return
        prev_reason = doc["tree_ring_wrong_key"].get("reason")
        print(f"merge-retry: previous tree-ring status not_run ({prev_reason}); retrying at gate 3300 MiB", flush=True)
        tree = treering_block(gate_mib=3300, patience_min=40)
        doc["tree_ring_wrong_key"] = tree
        doc.setdefault("attempt_history", []).append({
            "attempt": 2, "started_utc": started,
            "mode": "tree-ring-only merge retry",
            "previous_status_reason": prev_reason,
            "new_status": tree["status"],
        })
        doc["finished_utc"] = datetime.now(timezone.utc).isoformat()
        doc["failures"] = list(dict.fromkeys(doc.get("failures", []) + failures))
        doc["gpu"]["free_mib_before_tree_ring"] = tree.get("gpu_free_mib_at_load")
        tmp_write = out_path.with_suffix(".json.tmp")  # atomic replace of E8's own output file only
        tmp_write.write_text(json.dumps(doc, indent=1))
        tmp_write.replace(out_path)
        print("MERGED tree-ring retry into", out_path, flush=True)
        return
    imw = imwatermark_block()
    tree = treering_block()
    # hamming stats of the fixed wrong message vs every embedded lineage message
    hamming = {}
    for fam in ("riva_gan", "dwt_dct_svd"):
        ds = [int((WRONG_MSG != bits(i)).sum()) for i in EVAL_POP[fam] + BREADTH_POP[fam]]
        hamming[fam] = {"min": min(ds), "max": max(ds), "mean": float(np.mean(ds))}
    doc = {
        "experiment": "E8 wrong-message/wrong-key extension on larger populations",
        "date": DATE, "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "experiments/sensitivity/wrong_message_extension.py",
        "protocol_sha256": PROTO_SHA,
        "frozen_thresholds_note": "All thresholds are the frozen v3 minimum-rule midpoints loaded from "
                                  "results/{family}_calibration.json, strict >, REUSED NOT RECALIBRATED; "
                                  "max-rule decisions recorded per the v3 preregistered sensitivity convention.",
        "wrong_message": {
            "scope": "one FIXED 32-bit message for every imwatermark row (differs from the original v3 wrong-key "
                     "battery, which used per-index offset-20291001 messages on 6 calibration images)",
            "derivation": "np.random.default_rng(derive_seed('e8_wrong_message', 20281001, 0)).integers(0,2,32,dtype=np.uint8)",
            "seed": WM_SEED, "seed_labels": list(WM_SEED_LABELS),
            "hex": WRONG_HEX, "bits": WRONG_BITS_STR,
            "hamming_vs_embedded_messages": hamming,
            "note": "min Hamming distance to any embedded lineage message is recorded; a small distance would "
                    "weaken the wrong-message negative for that lineage",
        },
        "imwatermark_configs": imw,
        "tree_ring_wrong_key": tree,
        "versions": versions(),
        "gpu": {"free_mib_before_tree_ring": tree.get("gpu_free_mib_at_load"), "shared_gpu_policy": "batch 1, OOM retry after 60s x3"},
        "failures": failures,
    }
    with out_path.open("x") as f:
        json.dump(doc, f, indent=1)
        f.flush()
    print("WROTE", out_path, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
