"""LoRA (Low-Rank Adaptation) utilities for JAX/Flax UNet fine-tuning.

Matches the PyTorch reference: rank=4, alpha=4, gaussian init on A, zero init on B.
Zero-init on B ensures the initial LoRA delta (A @ B = 0) is zero, so training
starts from the exact pretrained weights.
Only targets attention projection layers: to_q, to_k, to_v, to_out.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

# Attention projection layer names to apply LoRA to
LORA_TARGET_KEYS = {"to_q", "to_k", "to_v", "to_out"}


def _iter_lora_paths(params, path=()):
    """Yield (path_tuple, kernel_array) for all LoRA target attention projection kernels.

    Handles both:
      - Dense modules: {key: {"kernel": array, ...}}
      - Modules with sub-layers (e.g. to_out): {key: {"0": {"kernel": array}, "1": {...}}}
    """
    if not isinstance(params, (dict, FrozenDict)):
        return
    for k, v in params.items():
        if k in LORA_TARGET_KEYS:
            if isinstance(v, (dict, FrozenDict)) and "kernel" in v:
                # Direct Dense: to_q/kernel, to_k/kernel, to_v/kernel
                yield path + (k, "kernel"), v["kernel"]
            elif isinstance(v, (dict, FrozenDict)):
                # Nested (e.g. to_out has index "0" for Dense, "1" for Dropout)
                for sk, sv in v.items():
                    if isinstance(sv, (dict, FrozenDict)) and "kernel" in sv:
                        yield path + (k, sk, "kernel"), sv["kernel"]
        else:
            yield from _iter_lora_paths(v, path + (k,))


def init_lora_params(frozen_unet_params, rank, rng):
    """Initialize LoRA parameters for all attention projection layers.

    Returns a flat dict: {path_str: {"A": array[in_dim, rank], "B": array[rank, out_dim]}}
    A: gaussian init (std=0.02). B: zero init. This ensures the initial LoRA delta
    A @ B = 0, so training starts from the exact pretrained weights.
    """
    lora_params = {}
    for path, kernel in _iter_lora_paths(frozen_unet_params):
        in_dim, out_dim = kernel.shape  # Flax Dense: [in_dim, out_dim]
        key_str = "/".join(str(p) for p in path)
        rng, rng_a = jax.random.split(rng)
        A = jax.random.normal(rng_a, (in_dim, rank), dtype=kernel.dtype) * 0.02
        B = jnp.zeros((rank, out_dim), dtype=kernel.dtype)
        lora_params[key_str] = {"A": A, "B": B}
    print(
        f"[ lora ] initialized {len(lora_params)} LoRA layers | "
        f"rank={rank} | "
        f"total params: {count_lora_params(lora_params):,}"
    )
    return lora_params


def apply_lora(frozen_params, lora_params, scale):
    """Return merged UNet params: frozen_params with LoRA deltas applied at target layers.

    merged_kernel = frozen_kernel + scale * A @ B
    where A: [in_dim, rank], B: [rank, out_dim], scale = alpha / rank.
    """
    def _merge(params, path=()):
        if isinstance(params, (dict, FrozenDict)):
            result = {k: _merge(v, path + (k,)) for k, v in params.items()}
            return FrozenDict(result) if isinstance(params, FrozenDict) else result
        # Leaf (array) — check if a LoRA delta exists for this path
        key_str = "/".join(str(p) for p in path)
        if key_str in lora_params:
            ab = lora_params[key_str]
            return params + scale * jnp.matmul(ab["A"], ab["B"])
        return params

    return _merge(frozen_params)


def count_lora_params(lora_params):
    """Return total number of LoRA trainable parameters."""
    return sum(x.size for x in jax.tree_util.tree_leaves(lora_params))
