### Repository loader survey

We inspected the image-loading code used by detection and evaluation pipelines in eight public watermark repositories: Tree-Ring Watermarks, the WAVES benchmark, Stable Signature (the repository `facebookresearch/StableSignature` now resolves to `facebookresearch/stable_signature`), StegaStamp, invisible-watermark, RivaGAN, the WIBE framework, and DiffMark. Seven of the eight read still images, and six of the seven do so through decode paths that ignore EXIF orientation. Tree-Ring's JPEG-robustness test re-opens perturbed files with `Image.open(...)` (`optim_utils.py`), and its dataset loader applies `convert("RGB")` without any transpose. Stable Signature's `run_evals.py` and its documented decoding notebook load each file in an evaluation directory with `Image.open`. StegaStamp's `decode_image.py` feeds `ImageOps.fit(Image.open(filename).convert("RGB"), (400, 400))` to its detector; `ImageOps.fit` only resizes. WAVES and WIBE build source sets from `torchvision.datasets.ImageFolder` or HuggingFace datasets and re-decode JPEG-attacked buffers with `Image.open`. The invisible-watermark README instructs users to decode with `cv2.imread('test_wm.png')`, whose default honours the orientation tag (see the correction note below). A repository-wide search found no call to `ImageOps.exif_transpose` and no other orientation handling on any read path. RivaGAN, a video method, decodes frames with `cv2.VideoCapture`, where the EXIF question does not arise. These are code inspections of the eight repositories at the commits recorded in the table, not executions, so they establish which decode paths the pipelines traverse rather than measured detection outcomes.

| Repository | Load call (quoted) | File | EXIF behaviour | Commit |
|---|---|---|---|---|
| Tree-Ring Watermarks | `pil_image = Image.open(f)` | `guided_diffusion/image_datasets.py` | ignores | 3015283d |
| WAVES | `distorted_image = Image.open(buffered)` | `distortions/distortions.py` | ignores | 1477635d |
| Stable Signature | `pil_img = Image.open(os.path.join(img_dir, filename))` | `run_evals.py` | ignores | c91217c0 |
| StegaStamp | `image = Image.open(filename).convert("RGB")` | `decode_image.py` | ignores | c9844460 |
| invisible-watermark | `bgr = cv2.imread('test_wm.png')` | `README.md` | honours | 68d0376d |
| RivaGAN | `cap = cv2.VideoCapture(path)` | `rivagan/dataloader.py` | n/a (video) | efffa72a |
| WIBE | `to_tensor(data["image"].convert("RGB"))` | `src/wibench/datasets/mscoco/mscoco.py` | ignores | acf94ed8 |
| DiffMark | `pil_image = Image.open(f)` | `guided_diffusion/image_datasets.py` | ignores | 253a1135 |

Machine-readable evidence, including verbatim multi-line quotes and per-repository loader censuses, is in `repository_loader_survey.json`. IRIS (arXiv 2608.03539) could not be verified: the paper is withdrawn and no public code was found.


> Correction (2026-09-28): invisible-watermark's documented decode uses cv2.imread, whose default honours EXIF orientation (the study's own Section 6.2 measurement, 24/24). The still-image EXIF-ignoring count is 6 of 7, not 7 of 7.
