"""PartField neural network for semantic part feature extraction.

Adapted from PartUV (SIGGRAPH Asia 2025) for integration with ML UV Unwrap.
Original: https://github.com/EricWang12/PartUV
"""

from .scatter_compat import scatter_mean, scatter_max

__all__ = ["scatter_mean", "scatter_max"]
