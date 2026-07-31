from __future__ import annotations

"""Unsupervised per-mesh training loop for UV unwrapping."""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models import FlexParaUnwrapper


def train_unsupervised(
    model: FlexParaUnwrapper,
    points: torch.Tensor,
    num_iterations: int = 500,
    lr: float = 1e-3,
    weights: dict[str, float] | None = None,
    edges: torch.Tensor | None = None,
    faces: torch.Tensor | None = None,
    log_every: int = 100,
    device: str = "cpu",
    progress_callback: callable | None = None,
    partfield_features: torch.Tensor | None = None,
    ao_values: torch.Tensor | None = None,
) -> dict[str, list[float]]:
    """Unsupervised per-mesh optimization.

    This is the core training paradigm used by FAM/FlexPara/Nuvo:
    for each input mesh, we optimize the model parameters from scratch
    (or fine-tune from a pretrained initialization) to minimize
    cycle consistency + distortion losses.

    Args:
        model: UV unwrapping model
        points: (1, N, 3) surface points for this mesh
        num_iterations: optimization steps
        lr: learning rate
        weights: loss weight overrides
        edges: (E, 2) edge indices for distortion loss
        faces: (F, 3) face indices for distortion loss
        log_every: print loss every N steps
        device: 'cpu' or 'cuda'
        partfield_features: optional PartField conditioning
        ao_values: (N,) precomputed ambient occlusion values

    Returns:
        Dictionary of loss histories
    """
    model = model.to(device)
    points = points.to(device)

    # Clone and enable gradient
    points_opt = points.clone().detach().requires_grad_(False)

    # Move PartField features to device if provided
    pf_features = None
    if partfield_features is not None:
        pf_features = partfield_features.to(device).detach().requires_grad_(False)

    # Move AO values to device if provided
    ao_t = None
    if ao_values is not None:
        ao_t = ao_values.to(device).detach()

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=1e-5)

    history = {"total": [], "cycle": [], "chamfer": [], "distortion": [], "repulsion": []}

    if edges is not None:
        edges = edges.to(device)
    if faces is not None:
        faces = faces.to(device)

    start_time = time.time()

    for step in range(num_iterations):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(points_opt, partfield_features=pf_features)

        # Compute losses
        losses = model.compute_losses(
            points_opt,
            outputs,
            edges=edges,
            faces=faces,
            weights=weights,
            ao_values=ao_t,
        )

        # Backward
        losses["total"].backward()
        optimizer.step()
        scheduler.step()

        # Log
        for key in history:
            if key in losses:
                history[key].append(losses[key].item())

        if (step + 1) % log_every == 0:
            elapsed = time.time() - start_time
            loss_str = " | ".join(f"{k}: {v[-1]:.4f}" for k, v in history.items() if v)
            print(f"  [{step + 1}/{num_iterations}] {loss_str} | time: {elapsed:.1f}s")

        if progress_callback is not None:
            latest_losses = {k: v[-1] for k, v in history.items() if v}
            progress_callback(step + 1, num_iterations, latest_losses)

    return history


def train_supervised(
    model,
    dataloader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    weights: dict[str, float] | None = None,
    save_dir: str | Path | None = None,
    device: str = "cpu",
    log_every: int = 10,
    save_every: int = 10,
    progress_callback: callable | None = None,
) -> dict[str, list[float]]:
    """Train UV unwrapping model with ground-truth UV supervision.

    Uses a dataset of meshes with known UV maps to train the model
    in a supervised manner. The model learns to predict UV coordinates
    given a 3D mesh.

    Args:
        model: UV unwrapping model (e.g. ArtUVModel or similar)
        dataloader: yields dicts with 'vertices', 'faces', 'uv_coords'
        num_epochs: number of training epochs
        lr: learning rate
        weights: loss weight overrides
        save_dir: directory to save checkpoints
        device: 'cpu' or 'cuda'
        log_every: print loss every N epochs
        save_every: save checkpoint every N epochs
        progress_callback: callback(step, total_steps, losses)

    Returns:
        Dictionary of loss histories
    """
    model = model.to(device)
    save_dir = Path(save_dir) if save_dir else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    w = {
        "uv_l1": 1.0,
        "uv_l2": 0.5,
        "distortion": 0.1,
        "boundary": 0.05,
    }
    if weights:
        w.update(weights)

    history = {k: [] for k in ["total", "uv_l1", "uv_l2", "distortion", "boundary"]}
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_losses = {k: 0.0 for k in history}
        num_batches = 0

        for batch in dataloader:
            uv_gt = batch["uv_coords"].to(device)
            vertices = batch["vertices"].to(device)
            faces = batch["faces"].to(device)
            edges = batch.get("edges", None)
            if edges is not None:
                edges = edges.to(device)

            # Squeeze batch dimension from DataLoader (batch_size=1)
            if uv_gt.dim() == 3 and uv_gt.shape[0] == 1:
                uv_gt = uv_gt.squeeze(0)
            if vertices.dim() == 3 and vertices.shape[0] == 1:
                vertices = vertices.squeeze(0)
            if faces.dim() == 3 and faces.shape[0] == 1:
                faces = faces.squeeze(0)
            if edges is not None and edges.dim() == 3 and edges.shape[0] == 1:
                edges = edges.squeeze(0)

            optimizer.zero_grad()

            # Build initial UV (xatlas-style: project to best-fit plane)
            initial_uv = _compute_initial_uv(vertices)

            # Forward pass — try different model interfaces
            try:
                if edges is not None and edges.dim() == 2 and edges.shape[0] == 2:
                    edge_index = edges
                elif edges is not None and edges.dim() == 2:
                    # Convert (E, 2) to (2, E) for graph conv
                    edge_index = edges.T.contiguous()
                else:
                    edge_index = _build_edge_index(faces)

                result = model(initial_uv, vertices, faces, edge_index)
                pred_uv = result.get("uv_pred", result.get("uv_coords", result))
            except TypeError:
                # Fallback: try points-based forward
                try:
                    points = batch.get("points", vertices).to(device)
                    if points.dim() == 2:
                        points = points.unsqueeze(0)
                    result = model(points)
                    pred_uv = result.get("uv_coords", result.get("uv", result))
                except TypeError:
                    pred_uv = model(vertices, faces)

            if pred_uv is None:
                continue

            # Compute losses
            losses = _compute_supervised_losses(
                pred_uv, uv_gt, vertices, faces, w
            )

            losses["total"].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            for k in history:
                if k in losses:
                    epoch_losses[k] += losses[k].item()
            num_batches += 1

        scheduler.step()

        if num_batches > 0:
            for k in history:
                history[k].append(epoch_losses[k] / num_batches)

        # Log
        if (epoch + 1) % log_every == 0:
            elapsed = time.time() - start_time
            loss_str = " | ".join(f"{k}: {v[-1]:.4f}" for k, v in history.items() if v)
            print(f"  [{epoch + 1}/{num_epochs}] {loss_str} | lr: {scheduler.get_last_lr()[0]:.2e} | time: {elapsed:.1f}s")

        if progress_callback is not None:
            latest = {k: v[-1] for k, v in history.items() if v}
            progress_callback(epoch + 1, num_epochs, latest)

        # Save checkpoint
        if save_dir and (epoch + 1) % save_every == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "history": history,
            }
            torch.save(ckpt, save_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt")

    # Save final
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "model_final.pt")
        print(f"  Final model saved to {save_dir / 'model_final.pt'}")

    return history


def _compute_supervised_losses(
    pred_uv: torch.Tensor,
    gt_uv: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    """Compute supervised UV losses.

    Args:
        pred_uv: (V, 2) predicted UV coordinates
        gt_uv: (V, 2) ground-truth UV coordinates
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        weights: loss weights

    Returns:
        Dictionary of loss tensors including 'total'
    """
    # L1 loss on UV coordinates
    uv_l1 = torch.nn.functional.l1_loss(pred_uv, gt_uv)

    # L2 loss on UV coordinates
    uv_l2 = torch.nn.functional.mse_loss(pred_uv, gt_uv)

    # Distortion loss: penalize non-uniform stretching
    distortion = _face_distortion_loss(pred_uv, vertices, faces)

    # Boundary loss: encourage UVs to stay in [0, 1]
    boundary = _boundary_loss(pred_uv)

    total = (
        weights.get("uv_l1", 1.0) * uv_l1
        + weights.get("uv_l2", 0.5) * uv_l2
        + weights.get("distortion", 0.1) * distortion
        + weights.get("boundary", 0.05) * boundary
    )

    return {
        "total": total,
        "uv_l1": uv_l1.detach(),
        "uv_l2": uv_l2.detach(),
        "distortion": distortion.detach(),
        "boundary": boundary.detach(),
    }


def _face_distortion_loss(
    pred_uv: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Compute per-face UV distortion (isometric energy).

    Penalizes triangles where the ratio of 3D edge lengths to UV edge lengths
    differs from the mean ratio.
    """
    if faces.numel() == 0:
        return torch.tensor(0.0, device=pred_uv.device)

    try:
        v0 = pred_uv[faces[:, 0]]
        v1 = pred_uv[faces[:, 1]]
        v2 = pred_uv[faces[:, 2]]

        p0 = vertices[faces[:, 0]]
        p1 = vertices[faces[:, 1]]
        p2 = vertices[faces[:, 2]]

        # UV edge lengths
        uv_e1 = (v1 - v0).norm(dim=-1) + 1e-8
        uv_e2 = (v2 - v1).norm(dim=-1) + 1e-8
        uv_e3 = (v0 - v2).norm(dim=-1) + 1e-8

        # 3D edge lengths
        e3d_1 = (p1 - p0).norm(dim=-1) + 1e-8
        e3d_2 = (p2 - p1).norm(dim=-1) + 1e-8
        e3d_3 = (p0 - p2).norm(dim=-1) + 1e-8

        # Ratios
        r1 = uv_e1 / e3d_1
        r2 = uv_e2 / e3d_2
        r3 = uv_e3 / e3d_3

        # Penalize variance of ratios within each face
        mean_r = (r1 + r2 + r3) / 3.0
        distortion = ((r1 - mean_r) ** 2 + (r2 - mean_r) ** 2 + (r3 - mean_r) ** 2).mean()

        return distortion
    except (IndexError, RuntimeError):
        return torch.tensor(0.0, device=pred_uv.device)


def _boundary_loss(pred_uv: torch.Tensor) -> torch.Tensor:
    """Penalize UVs outside [0, 1] range."""
    below = torch.relu(-pred_uv).sum()
    above = torch.relu(pred_uv - 1.0).sum()
    return (below + above) / pred_uv.shape[0]


def train_batch(
    model: FlexParaUnwrapper,
    dataloader,
    num_iterations: int = 1600,
    lr: float = 1e-3,
    weights: dict[str, float] | None = None,
    save_dir: str | Path | None = None,
    device: str = "cpu",
    log_every: int = 100,
) -> dict[str, list[float]]:
    """Train on a batch of meshes (multiple per-mesh optimizations).

    For each mesh in the dataloader, we run an independent optimization.
    This simulates the "fine-tuning" paradigm where we adapt the model
    to a collection of shapes.

    Args:
        model: UV unwrapping model
        dataloader: iterable yielding dicts with 'points' key
        num_iterations: steps per mesh
        lr: learning rate
        weights: loss weights
        save_dir: directory to save checkpoints
        device: 'cpu' or 'cuda'
        log_every: log every N steps

    Returns:
        Aggregated loss history
    """
    save_dir = Path(save_dir) if save_dir else None

    all_history = {"total": [], "cycle": [], "chamfer": [], "distortion": []}

    for batch_idx, batch in enumerate(dataloader):
        points = batch["points"].to(device)
        if points.dim() == 2:
            points = points.unsqueeze(0)

        print(f"\nMesh {batch_idx + 1}: {batch.get('path', 'unknown')}")
        print(f"  Points: {points.shape[1]}")

        history = train_unsupervised(
            model=model,
            points=points,
            num_iterations=num_iterations,
            lr=lr,
            weights=weights,
            log_every=log_every,
            device=device,
        )

        # Accumulate histories
        for key in all_history:
            if key in history:
                all_history[key].extend(history[key])

        # Save checkpoint
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / f"model_mesh_{batch_idx:04d}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "mesh_path": batch.get("path", ""),
                "history": history,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    return all_history


def _compute_initial_uv(vertices: torch.Tensor) -> torch.Tensor:
    """Compute initial UV coordinates by projecting to best-fit plane.

    Uses PCA to find the dominant plane, then projects vertices to 2D.
    This provides a reasonable starting point for UV refinement.
    """
    if vertices.dim() == 3:
        batch_results = []
        for b in range(vertices.shape[0]):
            batch_results.append(_compute_initial_uv_single(vertices[b]))
        return torch.stack(batch_results)
    return _compute_initial_uv_single(vertices)


def _compute_initial_uv_single(vertices: torch.Tensor) -> torch.Tensor:
    """Compute initial UV for a single mesh (V, 3) -> (V, 2)."""
    v = vertices - vertices.mean(dim=0)

    # SVD for PCA
    try:
        _, S, Vt = torch.linalg.svd(v, full_matrices=False)
        # Project onto first two principal components
        proj = Vt[:2]  # (2, 3)
        uv = v @ proj.T  # (V, 2)
    except RuntimeError:
        # Fallback: just use x,y coordinates
        uv = v[:, :2]

    # Normalize to [0, 1]
    uv_min = uv.min(dim=0).values
    uv_max = uv.max(dim=0).values
    uv_range = uv_max - uv_min
    uv_range = torch.clamp(uv_range, min=1e-8)
    uv = (uv - uv_min) / uv_range

    return uv


def _build_edge_index(faces: torch.Tensor) -> torch.Tensor:
    """Build edge index tensor from faces for graph convolutions.

    Returns (2, E) tensor suitable for graph neural networks.
    Handles both (F, 3) and batched (B, F, 3) face tensors.
    """
    if faces.dim() == 3:
        faces = faces[0]

    edges = set()
    f_np = faces.detach().cpu().numpy()
    for face in f_np:
        n = len(face)
        for i in range(n):
            a, b = int(face[i]), int(face[(i + 1) % n])
            edges.add((min(a, b), max(a, b)))

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=faces.device)

    edge_list = sorted(edges)
    return torch.tensor(edge_list, dtype=torch.long, device=faces.device).T
