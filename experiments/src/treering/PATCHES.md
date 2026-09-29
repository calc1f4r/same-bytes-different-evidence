# PATCHES: vendored Tree-Ring adapter vs upstream

Upstream: `YuxinWenRick/tree-ring-watermark` @ `3015283d9cf82e90b628f02ad2121bd37408ca9a`
(the commit pinned as the `tree-ring-watermark` submodule in
`tools/wibe`; the submodule working tree was not checked out on disk, so the
three files needed were vendored from that exact commit).

Upstream targets torch 1.13 / transformers 4.23.1 / diffusers 0.11.1. This
harness pins torch 2.6.0 / transformers 4.46.3 / diffusers 0.31.0. Every
deviation from the upstream files is listed below. Watermark math (mask,
FFT pattern construction, injection, l1_complex measurement, DDIM inversion
algebra) is unchanged.

## modified_stable_diffusion.py

1. **`__init__` signature: `image_encoder` parameter added** (diffusers 0.31).
   0.31's `StableDiffusionPipeline.__init__` gained an `image_encoder`
   parameter between `feature_extractor` and `requires_safety_checker`, and it
   always registers an `image_encoder` component. Two failures follow from
   upstream's 8-positional-argument pass-through:
   (a) `requires_safety_checker=True` landed in `image_encoder` and crashed
   `register_modules` (`AttributeError: 'bool' object has no attribute '__module__'`);
   (b) even with keyword passing, the parent registers `image_encoder` while
   our subclass signature does not declare it, and the 0.31 `components`
   property (used by `_execution_device` on every pipeline call) raises
   `ValueError ... Expected {7 modules} to be defined, but dict_keys([...,
   'image_encoder']) are defined`. Both vendored `__init__`s therefore declare
   `image_encoder=None` explicitly and forward it as a keyword.
2. **`_encode_prompt` -> `encode_prompt`** (diffusers 0.31). The old private
   `_encode_prompt(prompt, device, num_images_per_prompt, do_cfg, negative_prompt)`
   is deprecated in 0.31 (warns on every call). Replaced with the public
   `encode_prompt`, which returns `(prompt_embeds, negative_prompt_embeds)`;
   the CFG batch is built manually with the same ordering the old API produced:
   `torch.cat([negative_prompt_embeds, prompt_embeds])` (uncond first), matching
   the `noise_pred.chunk(2)` guidance code below it.
3. **`reverse_process: True = False`** default annotation fixed to
   `reverse_process: bool = False` (in inverse_stable_diffusion.py): upstream
   had a nonsensical annotation (`True = False`); it worked in py3.8-era
   typing but is meaningless; behavior unchanged.
4. No functional changes otherwise. `BaseOutput` import
   (`from diffusers.utils import BaseOutput`), `check_inputs(4 positional)`,
   `prepare_latents`, `prepare_extra_step_kwargs`, `decode_latents`,
   `run_safety_checker`, `numpy_to_pil`, and the dataclass-subclass mechanism
   of `ModifiedStableDiffusionPipelineOutput` all still work on 0.31 as-is
   (verified by smoke test + full run).

## inverse_stable_diffusion.py

4. **Dropped unused imports** of `CLIPFeatureExtractor`, `CLIPTextModel`,
   `CLIPTokenizer`, `AutoencoderKL`, `UNet2DConditionModel`,
   `StableDiffusionSafetyChecker`, and `List/Union/Tuple` (deprecated import
   paths in transformers 4.46 / simply dead code). We always construct the
   pipeline with `safety_checker=None, feature_extractor=None`.
5. **Relative import** `from .modified_stable_diffusion import ...` so the two
   vendored files form a self-contained package (`src/treering/`).
6. Verified-on-0.31 (no patch needed, contrary to expectations):
   - `self.scheduler.alphas_cumprod[t]` with a 0-dim CUDA tensor `t` indexing a
     CPU tensor works in torch 2.6 (returns CPU 0-dim tensor);
   - 0-dim CPU float32 alpha scalars mixed with CUDA float16 latents keep the
     latents float16 (torch type promotion treats 0-dim tensors as
     lower-priority), so the fp16 denoising loop never trips a dtype error;
   - `self.unet.in_channels` and `self.vae_scale_factor` still exist in 0.31.

## optim_utils.py

7. **Removed heavy/dead dependencies**: `datasets`, `wandb`-adjacent CLIP
   helpers (`measure_similarity`, `get_dataset`), and `image_distortion`
   (this harness has its own frozen transform library in `src/transforms.py`).
8. **args-namespace -> explicit parameters** (`PATCH(harness)` in code):
   `get_watermarking_mask(init_latents, w_channel, w_radius, device, w_mask_shape)`,
   `inject_watermark(..., w_injection)`, etc. Same defaults as upstream
   (`w_channel=0`, `w_radius=10`, `w_pattern='rand'`, `w_mask_shape='circle'`,
   `w_injection='complex'`).
9. **`get_watermarking_pattern(gt_init, ...)`** now takes the pre-generated
   noise tensor instead of calling global `set_random_seed(args.w_seed)` +
   `pipe.get_random_latents()`. This harness derives all randomness from
   `common.derive_seed()` into explicit `torch.Generator` objects; the
   numerical branches are byte-identical.
10. **`l1_complex_metric(reversed_latents, mask, gt_patch)`**: single-image
    form of upstream `eval_watermark`'s `l1_complex` branch (upstream computed
    a no-watermark and a watermarked distance in one call because its runner
    always held both images).
11. **`transform_img` default `target_size=256`** (upstream 512; this bench is
    a 256px harness). Same transform body.
12. Upstream `eval_watermark`'s `NotImplementedError(...)`-without-raise typos
    fixed to actual `raise` statements in the kept functions.

## Scheduler choice (harness decision, not an upstream patch)

Upstream's runner and the WIBE wrapper pass `DPMSolverMultistepScheduler` for
*generation* while *inversion* always uses the hard-coded DDIM update in
`backward_diffusion`. On diffusers 0.31,
`DPMSolverMultistepScheduler` no longer exposes `final_alpha_cumprod`, which
the vendored DDIM inversion reads when a timestep reaches below 0. Rather than
shim that attribute, this adapter uses **`DDIMScheduler`** for both embedding
and detection: it is the model's own shipped scheduler config
(`scheduler_config.json` for `WIBE-HuggingFace/stable-diffusion-2-1-base` is a
DDIMScheduler config with `clip_sample=false`, matching the vendored
inversion's no-clip assumption), it makes generation and inversion share the
exact same deterministic DDIM algebra (lower inversion error), and it is
slower per step than DPMSolver but only 2 UNet evaluations/step, identical to
upstream's CFG cost. Deviation is recorded in the manifest rows
(`scheduler: "DDIMScheduler"`) and in results/treering_p1_report.md.

## Parameter scaling for 256px (harness decision)

Upstream default `w_radius=10` is tuned for 512px images (64x64 latents,
mask covers ~7.7% of the low-frequency plane). At 256px the latent is 32x32,
so the same relative ring is `10 * 32/64 = 5` → `w_radius=5` (~30.7% coverage
in absolute terms is *not* what happens: the ring covers the same relative
frequency band). `w_channel=0`, `w_pattern='rand'`, `w_mask_shape='circle'`,
`w_measurement='l1_complex'`, `w_injection='complex'` are unchanged from the
WIBE wrapper. The upstream repo ships no 256px-specific latent-SD guidance
(`256x256_diffusion.json` is the ImageNet guided-diffusion UNet config, not a
tree-ring config), so the detection threshold is calibrated on 16 unmarked
negatives instead of reusing WIBE's `77` (which is a 512px operating point).
