#!/usr/bin/env python3
"""Verify the README findings table against this repository's raw rows.

Re-derives the core measured quantities from results/ and checks they match the
findings documented in README.md. Prints PASS/FAIL per check and a summary; writes
results/verification_report.json when that path is writable (it is regenerated on
every run, which is why it is excluded from MANIFEST.sha256). Run from the
repository root: python3 experiments/verify_findings.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "results/core"
CTRL = ROOT / "results/controls"
SENS = ROOT / "results/sensitivity"
CONS = ROOT / "results/consumers"
SVC = ROOT / "results/service"
NAME = {"riva_gan": "rivagan", "tree_ring_rand": "treering_rand",
        "tree_ring_ring": "treering_ring", "dwt_dct_svd": "dwtdctsvd"}
FAMS = list(NAME)

checks = []
def check(name, ok, detail=""):
    checks.append({"finding": name, "verdict": "PASS" if ok else "FAIL", "detail": str(detail)})
    if not ok:
        print("FAIL:", name, detail)

def rows(fam):
    return [json.loads(l) for l in (CORE / NAME[fam] / "scores.jsonl").open()]
def fpr(fam):
    return [json.loads(l) for l in (CORE / NAME[fam] / "fpr_rows.jsonl").open()]
def cal(fam):
    return json.load((CORE / NAME[fam] / "calibration.json").open())

# ---- core same-byte split, FPR, display rejection ----
EXPECT_SPLIT = {"riva_gan": 5, "tree_ring_rand": 6, "tree_ring_ring": 4, "dwt_dct_svd": 6}
for fam in FAMS:
    t = cal(fam)["threshold_min_rule"]
    r = rows(fam)
    raw = {x["idx"]: x["score"] for x in r if x["kind"] == "exif_verifier_raw"}
    disp = {x["idx"]: x["score"] for x in r if x["kind"] == "exif_display"}
    fh = {}
    for x in r:
        if x["kind"] in ("exif_verifier_raw", "exif_display"):
            fh.setdefault(x["idx"], set()).add(x["file_sha256"])
    same_file = all(len(v) == 1 for v in fh.values())
    split = sum(1 for i in raw if raw[i] > t and not disp[i] > t)
    check(f"{fam} same-byte split {EXPECT_SPLIT[fam]}/6", split == EXPECT_SPLIT[fam], split)
    check(f"{fam} EXIF pairs share one file hash", same_file)
    n_acc = sum(1 for x in fpr(fam) if x["score"] > t)
    check(f"{fam} FPR 0/300", n_acc == 0, n_acc)
    disp_rej = sum(1 for i in disp if not disp[i] > t)
    check(f"{fam} display rejects 6/6", disp_rej == 6, disp_rej)

# ---- consumer attribution: raw==magick, display==opencv hashes (24 pairs) ----
hh = {}
for fam in FAMS:
    for x in rows(fam):
        if x["kind"] in ("exif_verifier_raw", "exif_display", "exif_opencv", "exif_magick"):
            hh.setdefault((NAME[fam], x["idx"]), {})[x["kind"]] = x["rgb_hash"]
eq_rm = sum(1 for k in hh if hh[k]["exif_verifier_raw"] == hh[k]["exif_magick"])
eq_do = sum(1 for k in hh if hh[k]["exif_display"] == hh[k]["exif_opencv"])
check("raw==magick 24/24", eq_rm == 24, eq_rm)
check("display==opencv 24/24", eq_do == 24, eq_do)

# ---- negatives extension ----
ext = [json.loads(l) for l in (SENS / "negatives_extension_rows.jsonl").open()]
from collections import Counter
cnt, acc = Counter(), Counter()
for x in ext:
    cnt[x["family"]] += 1
    acc[x["family"]] += 1 if x["accept_min_rule"] else 0
for fam in FAMS:
    exp = 1 if fam == "tree_ring_ring" else 0
    check(f"{fam} extension {exp}/300 (combined {exp}/600)",
          cnt[fam] == 300 and acc[fam] == exp, f"{acc[fam]}/{cnt[fam]}")
check("max-rule extension accepts 0",
      sum(1 for x in ext if x.get("accept_max_rule")) == 0)
ring_hit = [x for x in ext if x["family"] == "tree_ring_ring" and x["accept_min_rule"]]
check("ring accept is idx 2609 at -26.484375",
      len(ring_hit) == 1 and ring_hit[0]["idx"] == 2609
      and abs(ring_hit[0]["score"] + 26.484375) < 1e-9)

# ---- orientation breadth (row-level) ----
ob = json.load((CTRL / "orientation_breadth.json").open())
for fam in ("riva_gan", "dwt_dct_svd"):
    per = {}
    for r in ob[fam]["rows"]:
        per.setdefault(r["orient"], []).append(r)
    for o, rs in sorted(per.items()):
        split = sum(1 for r in rs if r["raw_accept"] and not r["display_accept"])
        both = sum(1 for r in rs if r["raw_accept"] and r["display_accept"])
        if fam == "riva_gan":
            check(f"breadth riva orient {o}: 26/30 split", split == 26 and len(rs) == 30,
                  f"{split}/{len(rs)}")
        elif o == 4:
            check("breadth dwt orient 4 (vertical flip): 0/30 split, 30/30 both-accept",
                  split == 0 and both == 30, f"split {split}, both {both}/{len(rs)}")
        else:
            check(f"breadth dwt orient {o}: 30/30 split", split == 30 and len(rs) == 30,
                  f"{split}/{len(rs)}")

# ---- rotation-normalizing control (row-level) ----
rc = json.load((CTRL / "rotation_normalizing_control.json").open())
EXPECT_RC = {"riva_gan": 5, "tree_ring_rand": 6, "tree_ring_ring": 4, "dwt_dct_svd": 6}
total_flips = 0
for fam in FAMS:
    rs = rc[fam]["rows"]
    rawacc = [r for r in rs if r["raw_accept"]]
    flips = sum(1 for r in rawacc if not r["normalized_accept"])
    total_flips += flips
    check(f"normalization {fam}: {EXPECT_RC[fam]}/{len(rawacc)} raw-accepters flip to reject",
          flips == EXPECT_RC[fam] == len(rawacc), f"{flips}/{len(rawacc)}")
check("normalization total 21/21 flips to reject", total_flips == 21, total_flips)

# ---- threshold sweep (row-level over eligible midpoints) ----
ts = json.load((SENS / "threshold_sweep.json").open())
for fam in FAMS:
    fam_data = ts["families"][fam]
    elig = [e for e in fam_data["sweep"] if e["eligible"]]
    ok_fpr = all(e["fpr300_accepts"] == 0 for e in elig)
    ok_split = all(e["exif_raw_accepts"] >= 1 and e["exif_display_rejects"] >= 1 for e in elig)
    expected_elig = {"riva_gan": 3, "tree_ring_rand": 6, "tree_ring_ring": 4, "dwt_dct_svd": 2}[fam]
    check(f"sweep {fam}: {expected_elig} eligible midpoints (pinned), FPR 0/300, raw-accept and display-reject >= 1 each",
          ok_fpr and ok_split and len(elig) == expected_elig > 0
          and fam_data["n_eligible"] == expected_elig,
          f"eligible {len(elig)}")

# ---- JPEG quality sweep ----
qs = json.load((SENS / "jpeg_quality_sweep.json").open())
QS_EXP = {"riva_gan": {"q60": 4, "q75": 4, "q85": 5},
          "tree_ring_rand": {"q60": 5, "q75": 6, "q85": 6},
          "tree_ring_ring": {"q60": 2, "q75": 3, "q85": 3},
          "dwt_dct_svd": {"q60": 5, "q75": 5, "q85": 6}}
for fam, exp in QS_EXP.items():
    for q, e in exp.items():
        v = qs["results"][fam][q]
        rr = v["rows"]
        raw = {x["idx"]: x["decision"] for x in rr if x["path"] == "raw"}
        disp = {x["idx"]: x["decision"] for x in rr if x["path"] == "display"}
        split = sum(1 for i in raw if raw[i] and not disp[i])
        check(f"quality {fam} {q} split {e}/6, display rejects 6/6",
              split == e and sum(1 for i in disp if not disp[i]) == 6, split)

# ---- wrong-message / cross-key ----
wm = json.load((SENS / "wrong_message.json").open())
for fam in ("riva_gan", "dwt_dct_svd"):
    v = wm["imwatermark_configs"][fam]
    check(f"wrong-message {fam} 0/36",
          sum(1 for x in v["rows"] if x.get("accept")) == 0 and len(v["rows"]) == 36)
tkc = wm["tree_ring_wrong_key"]["configs"]
for fam in ("tree_ring_rand", "tree_ring_ring"):
    v = tkc[fam]
    check(f"cross-key {fam} 0/6",
          sum(1 for x in v["rows"] if x.get("accept")) == 0 and len(v["rows"]) == 6)

# ---- browser engines (row-level) ----
be = json.load((CONS / "browser_engines/browser_engines.json").open())
for eng in ("firefox", "chromium_control"):
    e = be["engines"][eng]
    check(f"{eng} 24/24 canvas + screenshot equal scored, MAE 0",
          sum(1 for x in e["rows"] if x["canvas_equal_scored"]) == 24
          and sum(1 for x in e["rows"] if x["screenshot_equal_scored"]) == 24
          and all(x["canvas_mae_vs_scored"] == 0 for x in e["rows"])
          and len(e["rows"]) == 24)

# ---- loader survey (row-level per consumer) ----
ls = json.load((CONS / "loader_survey/extended.json").open())
CONSUMERS = ["display_exif_transpose", "raw_pillow", "opencv_default",
             "opencv_ignore_orientation", "torchvision_decode_image",
             "torchvision_pil_loader", "matplotlib_imread",
             "pillow_exif_transpose_after_getexif", "imageio_imread",
             "tf_io_decode_image"]
honouring = set()
for consumer in CONSUMERS:
    n_d = sum(1 for r in ls["rows"] if r.get(f"{consumer}_equals_display") is True)
    n_r = sum(1 for r in ls["rows"] if r.get(f"{consumer}_equals_raw") is True)
    if consumer == "tf_io_decode_image":
        skipflag = all(r.get(f"{consumer}_equals_display") == "skipped: tensorflow not installed"
                       and r.get(f"{consumer}_equals_raw") is None for r in ls["rows"])
        check("loader tf_io unclassified (skip string recorded on all 24 rows, TensorFlow absent)",
              skipflag and len(ls["rows"]) == 24)
    else:
        check(f"loader {consumer}: 24/24 one-side match (counted from rows)",
              (n_d == 24) != (n_r == 24) and len(ls["rows"]) == 24,
              f"display {n_d}, raw {n_r}")
        if n_d == 24:
            honouring.add(consumer)
check("loader honouring set matches expected consumers",
      honouring == {"display_exif_transpose", "opencv_default",
                    "pillow_exif_transpose_after_getexif"}, sorted(honouring))

# ---- repository survey (corrected count) ----
rs_ = json.load((CONS / "repository_loader_survey.json").open())
repos = rs_.get("repos") or rs_.get("repositories")
beh = [str(r.get("exif_behaviour") or r.get("exif") or "") for r in repos]
n_ign = sum(1 for b in beh if b.startswith("ignores"))
n_hon = sum(1 for b in beh if b.startswith("honours"))
n_na = sum(1 for b in beh if b.startswith("not applicable"))
check("repository survey (counted from repo records): 8 inspected, 6 ignoring, 1 honouring, 1 n/a",
      len(repos) == 8 and n_ign == 6 and n_hon == 1 and n_na == 1,
      f"ignoring {n_ign}, honouring {n_hon}, n/a {n_na}")

# ---- baked-ICC + SSIM admission (row-level) ----
bi = json.load((CTRL / "baked_icc_control.json").open())
n_bi = eq_bi = 0
for fam in FAMS:
    for r in bi[fam]["rows"]:
        n_bi += 1
        eq_bi += 1 if r["baked_equals_managed_score"] else 0
check("baked-ICC equals managed 24/24", n_bi == 24 and eq_bi == 24, f"{eq_bi}/{n_bi}")
adm = json.load((CTRL / "color_axis_ssim_admission.json").open())
n_adm = sum(1 for fam in adm for r in adm[fam] if r["admitted"])
check("ICC admission 15/24 row-wise", n_adm == 15, n_adm)
fam_mins = [min(r["ssim_managed_vs_plain"] for r in adm[fam]) for fam in adm]
check("per-family minimum SSIM within 0.777-0.842",
      abs(min(fam_mins) - 0.777) < 0.001 and abs(max(fam_mins) - 0.842) < 0.001,
      f"family minima {[round(m, 4) for m in fam_mins]}")

# ---- latent check ----
lc = json.load((CTRL / "latent_ring_check.json").open())
ring_d = lc["e6"]["ring"]["marked_latent_distances"]
rand_d = lc["e6"]["rand"]["marked_latent_distances"]
check("latent ring identity < rot90/rot180; rand identity at noise floor",
      ring_d["identity"] < ring_d["rot90"] and ring_d["identity"] < ring_d["rot180"]
      and rand_d["identity"] < 1.0)
check("latent check artifact records rot90 self-difference ratio 1.388",
      abs(lc["e6"]["ring"]["rot90_selfdiff_ratio"] - 1.388) < 1e-9)

# ---- service: the run summary is the shipped record for these quantities ----
svc = (SVC / "run_summary.md").read_text()
check("service run summary states idempotence 6/6", "Idempotence: 6/6" in svc)
check("service run summary states latency ~260-300 ms", "~260-300 ms" in svc)
check("service run summary states raw 4/6 at 0.75", "4/6 accepted (v1 implemented threshold 0.75)" in svc)
_h = json.load((SVC / "http_rows.json").open())
_ok = sum(1 for r in _h if r.get("status") == "ok")
_rej = [r for r in _h if r.get("status") == "rejected"]
check("service http rows: 60 ok; 4 rejected rows are contract-rejection tests with documented errors",
      _ok == 60 and len(_rej) == 4
      and all(r.get("error") or (r.get("response") or {}).get("error") for r in _rej),
      f"ok {_ok}, rejected {len(_rej)}")

doc = {"generated": "2026-09-29", "checks": checks,
       "pass": sum(1 for c in checks if c["verdict"] == "PASS"),
       "fail": sum(1 for c in checks if c["verdict"] == "FAIL")}
try:
    (ROOT / "results/verification_report.json").write_text(json.dumps(doc, indent=1))
except OSError as e:
    print(f"note: results/ not writable ({e}); report printed only", file=sys.stderr)
print(f"{doc['pass']} PASS / {doc['fail']} FAIL of {len(checks)}")
sys.exit(1 if doc["fail"] else 0)
