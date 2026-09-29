"""Vendored from YuxinWenRick/tree-ring-watermark @ 3015283d (optim_utils.py).

Only the watermarking utilities are kept (the training/CLIP/dataset helpers are
dropped). Adaptations for this harness are marked `PATCH(harness)`:

- functions take explicit keyword parameters instead of the upstream argparse
  namespace (`args.w_radius` -> `w_radius`, ...);
- `get_watermarking_pattern` takes a pre-generated noise tensor instead of
  calling the global `set_random_seed` + `pipe.get_random_latents()`, so all
  randomness flows through this project's derive_seed() into explicit
  torch.Generator objects;
- `l1_complex_metric` is the single-image form of upstream `eval_watermark`'s
  "w" branch (upstream computed both a no-watermark and a watermarked distance
  in one call because it always had both images);
- `transform_img` default target size is 256 here.

The numerical code (masks, FFT pattern construction, injection, measurement)
is byte-for-byte the upstream logic.
"""
import copy
import random

import numpy as np
import torch
from torchvision import transforms


def set_random_seed(seed=0):
    """Kept for parity with upstream; not used by this harness (we pass
    explicit torch.Generator objects seeded via common.derive_seed)."""
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=256):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    # reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2


# PATCH(harness): explicit params instead of (init_latents_w, args, device).
def get_watermarking_mask(init_latents_w, w_channel, w_radius, device, w_mask_shape='circle'):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=w_radius)
        torch_mask = torch.tensor(np_mask).to(device)

        if w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, w_channel] = torch_mask
    elif w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if w_channel == -1:
            # all channels
            watermarking_mask[:, :, anchor_p-w_radius:anchor_p+w_radius, anchor_p-w_radius:anchor_p+w_radius] = True
        else:
            watermarking_mask[:, w_channel, anchor_p-w_radius:anchor_p+w_radius, anchor_p-w_radius:anchor_p+w_radius] = True
    elif w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {w_mask_shape}')

    return watermarking_mask


# PATCH(harness): takes a pre-generated noise tensor `gt_init` (drawn with an
# explicit torch.Generator) instead of (pipe, args, device).
def get_watermarking_pattern(gt_init, w_pattern='rand', w_radius=10, w_pattern_const=0):
    if 'seed_ring' in w_pattern:
        gt_patch = gt_init

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(gt_init.device)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in w_pattern:
        gt_patch = gt_init
    elif 'rand' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += w_pattern_const
    elif 'ring' in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(gt_init.device)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    return gt_patch


# PATCH(harness): explicit param instead of args.w_injection.
def inject_watermark(init_latents_w, watermarking_mask, gt_patch, w_injection='complex'):
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))
    if w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        raise NotImplementedError(f'w_injection: {w_injection}')

    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w


# PATCH(harness): single-image form of upstream eval_watermark's 'l1_complex'
# branch (the upstream function required both a no-watermark and a watermarked
# reversed-latent tensor in the same call).
def l1_complex_metric(reversed_latents, watermarking_mask, gt_patch):
    reversed_latents_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))
    w_metric = torch.abs(reversed_latents_fft[watermarking_mask] - gt_patch[watermarking_mask]).mean().item()
    return w_metric
