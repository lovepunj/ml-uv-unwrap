"""UV distortion optimization.

Minimizes conformal distortion via normalized gradient descent.
Works as a universal post-processing step on any method's UV output.

Includes OptCuts-style joint seam + parameterization optimization
for better distortion reduction.
"""

from __future__ import annotations

import numpy as np


def slim_optimize(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    num_iterations: int = 20,
    **kwargs,
) -> np.ndarray:
    """Optimize UV coordinates to minimize conformal distortion.

    Args:
        vertices: (V, 3) mesh vertex positions
        faces: (F, 3) triangle face indices
        uv_coords: (V, 2) initial UV coordinates
        num_iterations: optimization steps

    Returns:
        (V, 2) optimized UV coordinates
    """
    uv = uv_coords.copy().astype(np.float64)

    e1_3d = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    e2_3d = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    l1 = np.linalg.norm(e1_3d, axis=1) + 1e-12
    l2 = np.linalg.norm(e2_3d, axis=1) + 1e-12

    for _ in range(num_iterations):
        d1 = uv[faces[:, 1]] - uv[faces[:, 0]]
        d2 = uv[faces[:, 2]] - uv[faces[:, 0]]

        r1 = np.linalg.norm(d1, axis=1) / l1
        r2 = np.linalg.norm(d2, axis=1) / l2

        scale = 2.0 * (r1 - r2)
        d1_hat = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + 1e-12)
        d2_hat = d2 / (np.linalg.norm(d2, axis=1, keepdims=True) + 1e-12)

        gd1 = (scale / l1)[:, None] * d1_hat
        gd2 = -(scale / l2)[:, None] * d2_hat

        grad = np.zeros_like(uv)
        np.add.at(grad, faces[:, 0], -gd1 - gd2)
        np.add.at(grad, faces[:, 1], gd1)
        np.add.at(grad, faces[:, 2], gd2)

        # Adaptive step: normalize by gradient magnitude
        gnorm = np.linalg.norm(grad) + 1e-12
        step = min(0.01, 1.0 / gnorm)
        uv = uv - step * grad

    return uv.astype(np.float32)


def optcuts_optimize(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_coords: np.ndarray,
    num_rounds: int = 3,
    slim_iters: int = 20,
    distortion_threshold: float = 0.1,
) -> np.ndarray:
    """OptCuts-style joint optimization.

    Alternates between distortion analysis, seam cutting, and
    parameterization for improved results over plain SLIM.

    Args:
        vertices: (V, 3) mesh vertices
        faces: (F, 3) face indices
        uv_coords: (V, 2) initial UV coordinates
        num_rounds: number of analyze-cut-optimize rounds
        slim_iters: SLIM iterations per round
        distortion_threshold: max acceptable per-face distortion

    Returns:
        (V, 2) optimized UV coordinates with cuts
    """
    from .optcuts import optcuts_joint_optimize

    uv, _ = optcuts_joint_optimize(
        vertices, faces, uv_coords,
        num_rounds=num_rounds,
        slim_iters=slim_iters,
        distortion_threshold=distortion_threshold,
    )
    return uv
