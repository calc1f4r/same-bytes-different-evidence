#!/usr/bin/env python3
"""E2 (2026-09-28): decision-consistency score-distribution table (paper Table 3).

CPU-only, read-only over existing v3 rows; no model calls. For each of the four
v3 configurations, tabulates median / IQR / min / max of stored scores for
calibration marks, calibration negatives (unmarked, wrong-key, pooled), the 300
held-out FPR negatives, and held-out identity marks, alongside the frozen
thresholds. Writes results/table3_score_distributions.json and prints a
markdown table. Protocol note written before computing (no new measurements).
"""
import json, statistics
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
RES = OUT / "results"
FAMS = ["riva_gan", "tree_ring_rand", "tree_ring_ring", "dwt_dct_svd"]
LABEL = {"riva_gan": "RivaGAN", "tree_ring_rand": "Tree-Ring rand",
         "tree_ring_ring": "Tree-Ring ring", "dwt_dct_svd": "dwtDctSvd"}

(RES / "e2_protocol.json").write_text(json.dumps({
    "experiment": "E2 score-distribution table", "written_before": "computation",
    "date": "2026-09-28", "inputs": [f"results/{f}.jsonl" for f in FAMS] +
    [f"results/{f}_fpr_rows.jsonl" for f in FAMS] +
    [f"results/{f}_calibration.json" for f in FAMS],
    "rule": "descriptive statistics only; no new scores; no thresholds changed",
}, indent=1))

def q(xs, p):
    s = sorted(xs)
    if len(s) == 1: return s[0]
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)

def stats(xs):
    return {"n": len(xs), "median": round(statistics.median(xs), 6),
            "q1": round(q(xs, 0.25), 6), "q3": round(q(xs, 0.75), 6),
            "min": min(xs), "max": max(xs)}

def fmt(x):
    return f"{x:g}" if isinstance(x, (int, float)) else str(x)

rows = {}
for fam in FAMS:
    data = [json.loads(l) for l in (RES / f"{fam}.jsonl").open()]
    fpr = [json.loads(l) for l in (RES / f"{fam}_fpr_rows.jsonl").open()]
    cal = json.load((RES / f"{fam}_calibration.json").open())
    g = lambda kind, split="calibration": [r["score"] for r in data if r["kind"] == kind and r["split"] == split]
    rows[fam] = {
        "threshold_min_rule": cal["threshold_min_rule"],
        "threshold_max_rule": cal["threshold_max_rule"],
        "cal_marked": stats(g("marked")),
        "cal_unmarked": stats(g("unmarked")),
        "cal_wrong_key": stats(g("wrong_key")),
        "cal_negatives_pooled": stats(g("unmarked") + g("wrong_key")),
        "fpr_negatives": stats([r["score"] for r in fpr]),
        "eval_identity_marks": stats([r["score"] for r in data if r["kind"] == "identity"]),
        "eval_exif_raw": stats([r["score"] for r in data if r["kind"] == "exif_verifier_raw"]),
        "eval_exif_display": stats([r["score"] for r in data if r["kind"] == "exif_display"]),
    }

doc = {"generated": "2026-09-28", "source": "existing v3 jsonl/calibration/fpr rows (read-only)",
       "note": "IQR = q1..q3 (linear interpolation). Thresholds from frozen calibration files.",
       "families": rows}
(RES / "table3_score_distributions.json").write_text(json.dumps(doc, indent=1))

hdr = ["Config", "t (min/max rule)", "Cal marks (med, IQR)", "Cal negs pooled (med, IQR)",
       "FPR negs n=300 (med, IQR)", "Identity (med, IQR)", "EXIF raw (med, IQR)", "EXIF display (med, IQR)"]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for fam in FAMS:
    r = rows[fam]
    def ci(k): return f"{fmt(r[k]['median'])} ({fmt(r[k]['q1'])}..{fmt(r[k]['q3'])})"
    print("| " + " | ".join([
        LABEL[fam], f"{fmt(r['threshold_min_rule'])} / {fmt(r['threshold_max_rule'])}",
        ci("cal_marked"), ci("cal_negatives_pooled"), ci("fpr_negatives"),
        ci("eval_identity_marks"), ci("eval_exif_raw"), ci("eval_exif_display")]) + " |")
print("\nwrote", RES / "table3_score_distributions.json")
