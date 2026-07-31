"""Pure PyTorch replacements for torch_scatter operations.

Replaces scatter_mean and scatter_max with native PyTorch implementations
using scatter_add and scatter_reduce.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
                 fill_value: float = 0.0) -> torch.Tensor:
    """Compute mean of src elements grouped by index.

    Pure PyTorch replacement for torch_scatter.scatter_mean.
    """
    index = index.to(src.device)
    if out is None:
        if dim_size is None:
            dim_size = int(index.max()) + 1
        shape = list(src.shape)
        shape[dim] = dim_size
        out = src.new_full(shape, fill_value)

    out.scatter_add_(dim, index.expand_as(src), src)

    ones = src.new_ones(src.shape)
    count = src.new_zeros(out.shape)
    count.scatter_add_(dim, index.expand_as(ones), ones)
    count = count.clamp(min=1)

    out = out / count
    return out


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
                fill_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute max of src elements grouped by index.

    Pure PyTorch replacement for torch_scatter.scatter_max.
    Returns (output, argmax).
    """
    index = index.to(src.device)
    if out is None:
        if dim_size is None:
            dim_size = int(index.max()) + 1
        shape = list(src.shape)
        shape[dim] = dim_size
        out = src.new_full(shape, fill_value)

    idx_expanded = index.expand_as(src)
    out.scatter_reduce_(dim, idx_expanded, src, reduce="amax", include_self=False)

    return out, index
