"""Vendored Tree-Ring watermark adapter.

Source: YuxinWenRick/tree-ring-watermark @ 3015283d9cf82e90b628f02ad2121bd37408ca9a
(the commit pinned as the `tree-ring-watermark` submodule of the WIBE project;
the submodule working tree was not checked out, so the files are vendored
verbatim-with-patches here — see PATCHES.md).

Keeps the minimum needed for latent tree-ring watermarking on SD 2.1-base:
  - InversableStableDiffusionPipeline: SD pipeline with DDIM-style exact inversion
  - watermarking mask / pattern / injection / l1_complex measurement utilities
"""
from .inverse_stable_diffusion import InversableStableDiffusionPipeline, backward_ddim, forward_ddim
from .modified_stable_diffusion import ModifiedStableDiffusionPipeline
from .optim_utils import (
    circle_mask,
    get_watermarking_mask,
    get_watermarking_pattern,
    inject_watermark,
    l1_complex_metric,
    set_random_seed,
    transform_img,
)

__all__ = [
    "InversableStableDiffusionPipeline",
    "ModifiedStableDiffusionPipeline",
    "backward_ddim",
    "forward_ddim",
    "circle_mask",
    "get_watermarking_mask",
    "get_watermarking_pattern",
    "inject_watermark",
    "l1_complex_metric",
    "set_random_seed",
    "transform_img",
]
