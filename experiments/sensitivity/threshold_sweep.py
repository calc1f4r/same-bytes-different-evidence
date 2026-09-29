#!/usr/bin/env python3
"""E4 (2026-09-28): full threshold-sensitivity sweep over the v3 calibration rows.

CPU-only, read-only over existing rows; no model calls. For each configuration,
for EVERY adjacent-unique-score midpoint of the pooled calibration scores (not
only eligible ones), report: eligible under the frozen rule (zero empirical
calibration FPR), cal TPR, cal FPR, held-out FPR accepts (n=300), identity
accepts, EXIF raw accepts, EXIF display rejects, same-byte split count
(raw accept AND display reject), ordinary-JPEG accepts, wrong-key-eval accepts.
Decision rule as frozen: strict >. This closes the forking-path concern with
full transparency: every threshold the rule could have chosen is tabulated.

Writes results/threshold_sensitivity_sweep.json; prints a markdown table of
eligible midpoints (and the closest ineligible ones for context).
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
RES = OUT / "results"
FAMS = ["riva_gan", "tree_ring_rand", "tree_ring_ring", "dwt_dct_svd"]
LABEL = {"riva_gan": "RivaGAN", "tree_ring_rand": "Tree-Ring rand",
         "tree_ring_ring": "Tree-Ring ring", "dwt_dct_svd": "dwtDctSvd"}

(RES / "e4_protocol.json").write_text(json.dumps({
    "experiment": "E4 threshold-sensitivity sweep", "written_before": "computation",
    "date": "2026-09-28", "inputs": [f"results/{f}.jsonl" for f in FAMS] +
    [f"results/{f}_fpr_rows.jsonl" for f in FAMS] +
    [f"results/{f}_calibration.json" for f in FAMS],
    "rule": "sweep every adjacent-unique-score midpoint of pooled calibration "
            "scores; decision strict >; no thresholds changed; no new scores",
}, indent=1))

def load(fam):
    data = [json.loads(l) for l in (RES / f"{fam}.jsonl").open()]
    fpr = [json.loads(l) for l in (RES / f"{fam}_fpr_rows.jsonl").open()]
    cal = json.load((RES / f"{fam}_calibration.json").open())
    return data, fpr, cal

out = {}
for fam in FAMS:
    data, fpr, cal = load(fam)
    pos = cal["cal_marked"]; neg = cal["cal_negative"]
    scores = sorted(set(pos + neg))
    mids = [(x + y) / 2 for x, y in zip(scores[:-1], scores[1:])]
    by = lambda kind: {r["idx"]: r["score"] for r in data if r["kind"] == kind}
    ident = by("identity"); raw = by("exif_verifier_raw"); disp = by("exif_display")
    ordn = by("ordinary_jpeg"); wk = by("wrong_key_eval")
    fpr_scores = [r["score"] for r in fpr]
    sweep = []
    for t in mids:
        cal_tpr = sum(1 for p in pos if p > t)
        cal_fpr = sum(1 for n in neg if n > t)
        split = sum(1 for i in raw if raw[i] > t and not (disp[i] > t))
        sweep.append({
            "t": t,
            "eligible": cal_fpr == 0 and cal_tpr > 0,
            "cal_tpr": f"{cal_tpr}/6", "cal_fpr": f"{cal_fpr}/12",
            "fpr300_accepts": sum(1 for s in fpr_scores if s > t),
            "identity_accepts": sum(1 for i in ident if ident[i] > t),
            "exif_raw_accepts": sum(1 for i in raw if raw[i] > t),
            "exif_display_rejects": sum(1 for i in disp if not disp[i] > t),
            "same_byte_split": f"{split}/6",
            "ordinary_jpeg_accepts": sum(1 for i in ordn if ordn[i] > t),
            "wrong_key_eval_accepts": sum(1 for i in wk if wk[i] > t),
        })
    out[fam] = {
        "threshold_min_rule": cal["threshold_min_rule"],
        "threshold_max_rule": cal["threshold_max_rule"],
        "n_midpoints": len(mids), "n_eligible": sum(1 for s in sweep if s["eligible"]),
        "sweep": sweep,
        "split_survives_every_eligible_midpoint": all(
            int(s["same_byte_split"][0]) >= 1 for s in sweep if s["eligible"]),
    }

doc = {"generated": "2026-09-28",
       "source": "existing v3 rows (read-only); decision rule strict > as frozen",
       "definition": "same-byte split per idx: exif_verifier_raw accept AND exif_display reject",
       "families": out}
(RES / "threshold_sensitivity_sweep.json").write_text(json.dumps(doc, indent=1))

for fam in FAMS:
    o = out[fam]
    print(f"\n== {LABEL[fam]} (frozen {o['threshold_min_rule']:g} / {o['threshold_max_rule']:g}; "
          f"{o['n_eligible']} of {o['n_midpoints']} midpoints eligible) ==")
    print("| t | elig | calTPR | calFPR | FPR300 | ident | raw acc | disp rej | split | ordJPG | wrongKey |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in o["sweep"]:
        if s["eligible"] or (not s["eligible"] and s["cal_fpr"] == "0/12"):
            print(f"| {s['t']:g} | {'Y' if s['eligible'] else 'n'} | {s['cal_tpr']} | {s['cal_fpr']} | "
                  f"{s['fpr300_accepts']} | {s['identity_accepts']}/6 | {s['exif_raw_accepts']}/6 | "
                  f"{s['exif_display_rejects']}/6 | {s['same_byte_split']} | {s['ordinary_jpeg_accepts']}/6 | "
                  f"{s['wrong_key_eval_accepts']}/6 |")
print("\nwrote", RES / "threshold_sensitivity_sweep.json")
