#!/usr/bin/env python3
"""E3 (2026-09-28): benchmark-loader decode survey, extended.

Extends results/loader_survey_probe.json (torchvision decode_image, ImageFolder
pil_loader, matplotlib, OpenCV default, PIL raw) with:
  (a) PIL ImageOps.exif_transpose control (expected to equal the display raster)
  (b) cv2.imread with explicit IMREAD_IGNORE_ORIENTATION (expected to equal raw)
  (c) tf.io.decode_image if tensorflow is installed (skipped when absent)
  (d) PIL exif_transpose after forcing EXIF metadata parse/cache (getexif first)
  (e) imageio.imread if installed
Hash-compares each decode against the display raster and the raw raster for the
24 pilot_v3 EXIF orientation-6 eval JPEGs (6 per family). Read-only on fixtures;
writes results/loader_survey_extended.json. Run under the study virtualenv.
"""
import hashlib, json, platform
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

OUT = Path(__file__).resolve().parents[1]
FIX = OUT / "fixtures"
RES = OUT / "results"
import cv2, matplotlib, PIL, torchvision
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMS = ["riva_gan", "tree_ring_rand", "tree_ring_ring", "dwt_dct_svd"]
files = []
for fam in FAMS:
    files += sorted(FIX.glob(f"{fam}_*_exif6.jpg"))
assert len(files) == 24, len(files)

def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

(RES / "e3_protocol.json").write_text(json.dumps({
    "experiment": "E3 loader-survey extension", "written_before": "execution",
    "date": "2026-09-28", "fixtures": [f.name for f in files],
    "rule": "read-only decode probes; compare each loader raster hash against "
            "the display raster (ImageOps.exif_transpose) and the raw raster "
            "(Pillow convert RGB without transpose); no watermark scoring",
}, indent=1))

import io as _io

def decodes(path):
    out = {}
    with Image.open(path) as im:
        raw = np.asarray(im.convert("RGB")).copy()
        out["display_exif_transpose"] = np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()
    out["raw_pillow"] = raw
    out["opencv_default"] = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    out["opencv_ignore_orientation"] = cv2.cvtColor(
        cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION), cv2.COLOR_BGR2RGB)
    from torchvision.io import decode_image
    out["torchvision_decode_image"] = decode_image(str(path), mode=torchvision.io.ImageReadMode.RGB).permute(1, 2, 0).numpy()
    from torchvision.datasets.folder import pil_loader
    out["torchvision_pil_loader"] = np.asarray(pil_loader(str(path))).copy()
    out["matplotlib_imread"] = np.asarray(plt.imread(str(path))[:, :, :3] * 255).astype(np.uint8) \
        if str(path).lower().endswith((".png",)) else np.asarray(plt.imread(str(path)))
    # (d) exif_transpose after forcing EXIF parse/caching
    with Image.open(path) as im:
        _ = im.getexif()  # force parse/cache before transpose
        out["pillow_exif_transpose_after_getexif"] = np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()
    try:
        import imageio.v2 as iio
        out["imageio_imread"] = np.asarray(iio.imread(str(path)))[:, :, :3].copy()
    except Exception as e:
        out["imageio_imread"] = f"error: {e}"
    try:
        import tensorflow as tf  # noqa
        out["tf_io_decode_image"] = tf.io.decode_image(tf.io.read_file(str(path)), channels=3, expand_animations=False).numpy()
    except ImportError:
        out["tf_io_decode_image"] = "skipped: tensorflow not installed"
    return out

rows = []
for p in files:
    d = decodes(p)
    dh, rh = h(d["display_exif_transpose"]), h(d["raw_pillow"])
    row = {"file": p.name, "display_hash": dh, "raw_hash": rh,
           "orientation_tag": 6}
    for k, v in d.items():
        row[k + "_equals_display"] = (h(v) == dh) if isinstance(v, np.ndarray) else v
        row[k + "_equals_raw"] = (h(v) == rh) if isinstance(v, np.ndarray) else None
    rows.append(row)

def col(name): return {
    "equals_display": sum(1 for r in rows if r.get(name + "_equals_display") is True),
    "equals_raw": sum(1 for r in rows if r.get(name + "_equals_raw") is True),
    "neither": sum(1 for r in rows if r.get(name + "_equals_display") is False and r.get(name + "_equals_raw") is False),
    "n": len(rows)}

summary = {k: col(k) for k in
           ["display_exif_transpose", "raw_pillow", "opencv_default", "opencv_ignore_orientation",
            "torchvision_decode_image", "torchvision_pil_loader", "matplotlib_imread",
            "pillow_exif_transpose_after_getexif", "imageio_imread", "tf_io_decode_image"]}
doc = {"generated": "2026-09-28",
       "versions": {"python": platform.python_version(), "Pillow": PIL.__version__,
                    "opencv": cv2.__version__, "torchvision": torchvision.__version__,
                    "matplotlib": matplotlib.__version__},
       "n_fixtures": len(rows), "summary_equals_display_of_24": summary, "rows": rows}
(RES / "loader_survey_extended.json").write_text(json.dumps(doc, indent=1))
print(json.dumps(summary, indent=1))
print("wrote", RES / "loader_survey_extended.json")
