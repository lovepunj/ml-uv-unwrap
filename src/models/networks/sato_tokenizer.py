from __future__ import annotations

"""
SATO: Strips as Tokens tokenizer.

Serializes triangle/quad meshes into discrete token sequences using strip-based
traversal and three-level hierarchical quantization, with optional UV island
segmentation for chart-aware encoding.

Reference: SATO – Strips as Tokens (SIGGRAPH 2026).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_QUANTIZATION_LEVELS = 3
CODEBOOK_SIZE_C1 = 256
CODEBOOK_SIZE_C2 = 256
CODEBOOK_SIZE_C3 = 256

# Special token ranges – values above the normal codebook entries.
# Each "c1 star" token marks the start of a new strip.
# Each "c1 uv"  token marks the completion of a UV island.
STRIP_TRANSITION_BASE = CODEBOOK_SIZE_C1
UV_TRANSITION_BASE = CODEBOOK_SIZE_C1 + STRIP_TRANSITION_BASE


# ---------------------------------------------------------------------------
# Hierarchical quantization helpers
# ---------------------------------------------------------------------------


def _hierarchical_quantize(
    values: np.ndarray,
    c1_bins: int = CODEBOOK_SIZE_C1,
    c2_bins: int = CODEBOOK_SIZE_C2,
    c3_bins: int = CODEBOOK_SIZE_C3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-level hierarchical quantization.

    Parameters
    ----------
    values : (N,) float array in [0, 1].
    c1_bins, c2_bins, c3_bins : number of bins per level.

    Returns
    -------
    c1, c2, c3 : integer arrays in [0, bins-1].
    """
    c1 = np.clip((values * c1_bins).astype(np.int32), 0, c1_bins - 1)
    residual = values - (c1 + 0.5) / c1_bins
    c2 = np.clip(
        ((residual * c1_bins + 0.5) * c2_bins).astype(np.int32), 0, c2_bins - 1
    )
    residual2 = residual - (c2 + 0.5) / (c1_bins * c2_bins)
    c3 = np.clip(
        ((residual2 * c1_bins * c2_bins + 0.5) * c3_bins).astype(np.int32),
        0,
        c3_bins - 1,
    )
    return c1, c2, c3


def _hierarchical_dequantize(
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    c1_bins: int = CODEBOOK_SIZE_C1,
    c2_bins: int = CODEBOOK_SIZE_C2,
    c3_bins: int = CODEBOOK_SIZE_C3,
) -> np.ndarray:
    """Reconstruct values from three-level codes."""
    v = (c1 + 0.5) / c1_bins
    v = v + (c2 + 0.5) / (c1_bins * c2_bins) - 0.5 / (c1_bins * c2_bins)
    v = (
        v
        + (c3 + 0.5) / (c1_bins * c2_bins * c3_bins)
        - 0.5 / (c1_bins * c2_bins * c3_bins)
    )
    return v


def _encode_coordinate(
    coord: float,
    prev_c1: int | None = None,
) -> list[int]:
    """Encode a single coordinate with prefix sharing.

    Returns
    -------
    tokens : list of 1 or 3 ints.
        If ``prev_c1`` matches the new c1, only (c2, c3) are emitted (2 tokens).
        Otherwise (c1, c2, c3) are emitted (3 tokens).
    """
    c1_arr, c2_arr, c3_arr = _hierarchical_quantize(np.array([coord]))
    c1, c2, c3 = int(c1_arr[0]), int(c2_arr[0]), int(c3_arr[0])

    if prev_c1 is not None and c1 == prev_c1:
        return [c2, c3]
    return [c1, c2, c3]


def _decode_coordinates(
    tokens: list[int],
    expected_len: int,
) -> tuple[np.ndarray, list[int]]:
    """Decode a sequence of hierarchical coordinates.

    Parameters
    ----------
    tokens : flat list of c1/c2/c3 codes.
    expected_len : number of coordinates to decode.

    Returns
    -------
    values : (expected_len,) float array.
    consumed : number of tokens consumed.
    """
    values = []
    idx = 0
    while len(values) < expected_len and idx < len(tokens):
        c1 = tokens[idx]
        if idx + 2 < len(tokens):
            c2 = tokens[idx + 1]
            c3 = tokens[idx + 2]
            values.append(float(_hierarchical_dequantize(np.array([c1]), np.array([c2]), np.array([c3]))[0]))
            idx += 3
        elif idx + 1 < len(tokens):
            c2 = tokens[idx + 1]
            c3 = tokens[idx + 2] if idx + 2 < len(tokens) else 0
            values.append(float(_hierarchical_dequantize(np.array([c1]), np.array([c2]), np.array([c3]))[0]))
            idx += 3
        else:
            idx += 1
    return np.array(values[:expected_len]), idx


# ---------------------------------------------------------------------------
# Mesh normalization
# ---------------------------------------------------------------------------


def _normalize_vertices(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize vertices to [0, 1] and return (normalized, mins, scale)."""
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    scale = float((maxs - mins).max())
    if scale < 1e-12:
        scale = 1.0
    return (vertices - mins) / scale, mins, scale


def _denormalize_vertices(
    normalized: np.ndarray, mins: np.ndarray, scale: float
) -> np.ndarray:
    return normalized * scale + mins


# ---------------------------------------------------------------------------
# Strip traversal for triangles
# ---------------------------------------------------------------------------


def _build_adjacency(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Build edge → face mapping for triangle meshes."""
    edge_map: dict[tuple[int, int], list[int]] = {}
    for fi, f in enumerate(faces):
        for j in range(3):
            e = tuple(sorted((int(f[j]), int(f[(j + 1) % 3]))))
            edge_map.setdefault(e, []).append(fi)
    return edge_map


def _triangle_strip_traversal(
    faces: np.ndarray, max_strip_faces: int = 20
) -> list[list[int]]:
    """Greedy strip traversal.  Each strip is a list of face indices.

    A new face is appended to the current strip if it shares an edge with the
    last face of the strip.  When no such face exists (or the strip is full)
    a new strip is started.
    """
    n_faces = len(faces)
    visited = np.zeros(n_faces, dtype=bool)
    edge_map = _build_adjacency(faces)

    strips: list[list[int]] = []
    face_to_strip: dict[int, int] = {}

    for seed in range(n_faces):
        if visited[seed]:
            continue
        strip: list[int] = [seed]
        visited[seed] = True
        face_to_strip[seed] = len(strips)
        current_face = seed

        while len(strip) < max_strip_faces:
            # Find an unvisited neighbour sharing an edge
            found = False
            for j in range(3):
                fi = int(faces[current_face][j])
                fj = int(faces[current_face][(j + 1) % 3])
                e = tuple(sorted((fi, fj)))
                for nbr in edge_map.get(e, []):
                    if not visited[nbr]:
                        strip.append(nbr)
                        visited[nbr] = True
                        face_to_strip[nbr] = len(strips)
                        current_face = nbr
                        found = True
                        break
                if found:
                    break
            if not found:
                break

        strips.append(strip)

    return strips


def serialize(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_strip_faces: int = 20,
) -> list[int]:
    """Serialize a triangle mesh to a strip-based token sequence.

    Parameters
    ----------
    vertices : (V, 3) float array.
    faces : (F, 3) int array of triangle indices.
    max_strip_faces : maximum faces per strip.

    Returns
    -------
    tokens : list of int
        Token sequence encoding vertex positions, strip transitions, and mesh
        metadata.  Decodable with :func:`deserialize`.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    norm_verts, mins, scale = _normalize_vertices(vertices)

    strips = _triangle_strip_traversal(faces, max_strip_faces)

    # Flatten face ordering according to strips.
    ordered_faces = [fi for strip in strips for fi in strip]

    tokens: list[int] = []

    # Header: number of vertices, number of faces, number of strips, mins, scale
    tokens.append(len(vertices))
    tokens.append(len(faces))
    tokens.append(len(strips))

    # Encode mins and scale (3 + 1 floats → hierarchical codes)
    for coord in mins:
        tokens.extend(_encode_coordinate(coord))
    tokens.extend(_encode_coordinate(scale))

    prev_c1_per_axis: list[int | None] = [None, None, None]

    for strip_idx, strip in enumerate(strips):
        if strip_idx > 0:
            # Strip transition token: encode the strip index as the c1* value
            tokens.append(STRIP_TRANSITION_BASE + strip_idx)

            # Reset prefix context
            prev_c1_per_axis = [None, None, None]

        for face_pos, fi in enumerate(strip):
            if face_pos == 0:
                # First face: emit all 3 vertices (9 coordinates)
                for vi in faces[fi]:
                    for axis in range(3):
                        toks = _encode_coordinate(
                            norm_verts[vi, axis], prev_c1_per_axis[axis]
                        )
                        tokens.extend(toks)
                        c1_arr, _, _ = _hierarchical_quantize(
                            np.array([norm_verts[vi, axis]])
                        )
                        prev_c1_per_axis[axis] = int(c1_arr[0])
            else:
                # Subsequent face in strip: emit 1 new vertex (3 coords)
                new_vi = int(faces[fi][2])
                for axis in range(3):
                    toks = _encode_coordinate(
                        norm_verts[new_vi, axis], prev_c1_per_axis[axis]
                    )
                    tokens.extend(toks)
                    c1_arr, _, _ = _hierarchical_quantize(
                        np.array([norm_verts[new_vi, axis]])
                    )
                    prev_c1_per_axis[axis] = int(c1_arr[0])

    return tokens


# ---------------------------------------------------------------------------
# Strip traversal for quads
# ---------------------------------------------------------------------------


def _build_adjacency_quad(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Build edge → face mapping for quad meshes (4 indices per face)."""
    edge_map: dict[tuple[int, int], list[int]] = {}
    for fi, f in enumerate(faces):
        n_verts_face = 4
        for j in range(n_verts_face):
            e = tuple(sorted((int(f[j]), int(f[(j + 1) % n_verts_face]))))
            edge_map.setdefault(e, []).append(fi)
    return edge_map


def _quad_strip_traversal(
    faces: np.ndarray, max_strip_faces: int = 20
) -> list[list[int]]:
    """Greedy strip traversal for quad meshes.

    Each strip emits the first two vertices of the seed face, then one new
    quad (two new vertices) per step.
    """
    n_faces = len(faces)
    visited = np.zeros(n_faces, dtype=bool)
    edge_map = _build_adjacency_quad(faces)

    strips: list[list[int]] = []

    for seed in range(n_faces):
        if visited[seed]:
            continue
        strip: list[int] = [seed]
        visited[seed] = True
        current_face = seed

        while len(strip) < max_strip_faces:
            found = False
            for j in range(4):
                fi = int(faces[current_face][j])
                fj = int(faces[current_face][(j + 1) % 4])
                e = tuple(sorted((fi, fj)))
                for nbr in edge_map.get(e, []):
                    if not visited[nbr]:
                        strip.append(nbr)
                        visited[nbr] = True
                        current_face = nbr
                        found = True
                        break
                if found:
                    break
            if not found:
                break

        strips.append(strip)

    return strips


def serialize_quad(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_strip_faces: int = 20,
) -> list[int]:
    """Serialize a quad mesh to a strip-based token sequence.

    Supports mixed triangle/quad meshes by padding triangles with ``-1``.

    Parameters
    ----------
    vertices : (V, 3) float array.
    faces : (F, 4) int array.  Triangles should have ``faces[i, -1] == -1``.
    max_strip_faces : maximum faces per strip.

    Returns
    -------
    tokens : list of int
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    norm_verts, mins, scale = _normalize_vertices(vertices)

    # Detect whether we have any triangle faces (padded with -1).
    is_triangle = faces[:, -1] == -1
    has_mixed = bool(is_triangle.any())

    strips = _quad_strip_traversal(faces, max_strip_faces)

    tokens: list[int] = []

    # Header
    tokens.append(len(vertices))
    tokens.append(len(faces))
    tokens.append(len(strips))
    tokens.append(1 if has_mixed else 0)

    for coord in mins:
        tokens.extend(_encode_coordinate(coord))
    tokens.extend(_encode_coordinate(scale))

    prev_c1_per_axis: list[int | None] = [None, None, None]

    for strip_idx, strip in enumerate(strips):
        if strip_idx > 0:
            tokens.append(STRIP_TRANSITION_BASE + strip_idx)
            prev_c1_per_axis = [None, None, None]

        for face_pos, fi in enumerate(strip):
            face_verts = [v for v in faces[fi] if v >= 0]

            if face_pos == 0:
                # Emit all vertices of the first face.
                for vi in face_verts:
                    for axis in range(3):
                        toks = _encode_coordinate(
                            norm_verts[vi, axis], prev_c1_per_axis[axis]
                        )
                        tokens.extend(toks)
                        c1_arr, _, _ = _hierarchical_quantize(
                            np.array([norm_verts[vi, axis]])
                        )
                        prev_c1_per_axis[axis] = int(c1_arr[0])
            else:
                # Emit only the new vertices added by this face.
                new_vis = face_verts[2:] if len(face_verts) == 4 else face_verts[1:]
                for vi in new_vis:
                    for axis in range(3):
                        toks = _encode_coordinate(
                            norm_verts[vi, axis], prev_c1_per_axis[axis]
                        )
                        tokens.extend(toks)
                        c1_arr, _, _ = _hierarchical_quantize(
                            np.array([norm_verts[vi, axis]])
                        )
                        prev_c1_per_axis[axis] = int(c1_arr[0])

    return tokens


# ---------------------------------------------------------------------------
# Deserialization
# ---------------------------------------------------------------------------


def deserialize(
    tokens: list[int],
    return_strips: bool = False,
) -> dict:
    """Deserialize a token sequence back into vertices and faces.

    Parameters
    ----------
    tokens : token list produced by :func:`serialize` or :func:`serialize_quad`.
    return_strips : if ``True``, include ``strip_boundaries`` in the result.

    Returns
    -------
    dict with keys:
        ``vertices`` : (V, 3) float array.
        ``faces``    : (F, ...) int array (triangles or quads).
        ``is_quad``  : bool – whether the mesh is quad-only.
        ``strip_boundaries`` : (optional) list of face indices where strips start.
    """
    tokens = list(tokens)
    idx = 0

    n_verts = tokens[idx]; idx += 1
    n_faces = tokens[idx]; idx += 1
    n_strips = tokens[idx]; idx += 1
    has_mixed = False
    if idx < len(tokens) and tokens[idx] in (0, 1):
        has_mixed = bool(tokens[idx]); idx += 1

    # Decode mins (3 coords)
    mins_vals: list[float] = []
    for _ in range(3):
        c1 = tokens[idx]; c2 = tokens[idx + 1]; c3 = tokens[idx + 2]
        mins_vals.append(float(_hierarchical_dequantize(np.array([c1]), np.array([c2]), np.array([c3]))[0]))
        idx += 3

    # Decode scale
    c1 = tokens[idx]; c2 = tokens[idx + 1]; c3 = tokens[idx + 2]
    scale = float(_hierarchical_dequantize(np.array([c1]), np.array([c2]), np.array([c3]))[0])
    idx += 3

    mins = np.array(mins_vals, dtype=np.float64)

    # Parse remaining tokens into coordinate groups.
    vertices_list: list[list[float]] = []
    strip_boundaries: list[int] = []
    face_vertex_counts: list[int] = []  # 3 for triangle, 4 for quad
    current_face_verts: list[int] = []

    def _flush_face() -> None:
        nonlocal current_face_verts
        if current_face_verts:
            face_vertex_counts.append(len(current_face_verts))
            current_face_verts = []

    prev_c1_per_axis: list[int | None] = [None, None, None]

    while idx < len(tokens):
        tok = tokens[idx]

        # Strip transition
        if tok >= STRIP_TRANSITION_BASE:
            _flush_face()
            idx += 1
            prev_c1_per_axis = [None, None, None]
            continue

        # Try to decode 3 coordinates (one vertex).
        coords_decoded = 0
        trial_idx = idx
        decoded_coords: list[float] = []

        for axis in range(3):
            if trial_idx >= len(tokens):
                break

            # Peek at whether this could be a full c1,c2,c3 or a prefix-shared c2,c3.
            # Heuristic: if the value is < CODEBOOK_SIZE_C1 and we haven't set prev,
            # treat as full.  This works because prefix-shared tokens only appear
            # when prev is set, and the first vertex of a strip always emits full.
            val_token = tokens[trial_idx]

            if prev_c1_per_axis[axis] is not None and val_token < CODEBOOK_SIZE_C2:
                # Possibly prefix-shared – but we need two more tokens for c2, c3.
                if trial_idx + 2 < len(tokens):
                    c2_tok = tokens[trial_idx + 1]
                    c3_tok = tokens[trial_idx + 2]
                    val = float(
                        _hierarchical_dequantize(
                            np.array([prev_c1_per_axis[axis]]),
                            np.array([c2_tok]),
                            np.array([c3_tok]),
                        )[0]
                    )
                    decoded_coords.append(val)
                    # Update c1 (stays the same for prefix sharing)
                    trial_idx += 2
                    coords_decoded += 1
                else:
                    break
            else:
                # Full c1, c2, c3
                if trial_idx + 2 < len(tokens):
                    c1_tok = tokens[trial_idx]
                    c2_tok = tokens[trial_idx + 1]
                    c3_tok = tokens[trial_idx + 2]
                    val = float(
                        _hierarchical_dequantize(
                            np.array([c1_tok]),
                            np.array([c2_tok]),
                            np.array([c3_tok]),
                        )[0]
                    )
                    decoded_coords.append(val)
                    prev_c1_per_axis[axis] = c1_tok
                    trial_idx += 3
                    coords_decoded += 1
                else:
                    break

        if coords_decoded == 3:
            vertices_list.append(decoded_coords)
            vi = len(vertices_list) - 1
            current_face_verts.append(vi)
            idx = trial_idx

            # Determine how many vertices this face needs.
            needed = 4 if has_mixed else 3
            # For mixed: triangles have 3 verts. We can't know ahead of time
            # without the original data, so we use a heuristic:
            # If the strip just started and we have 3 verts, check next token
            # for a strip transition or face end.
            # In practice, serialize always emits 3 verts for tri faces.
            if len(current_face_verts) >= needed:
                _flush_face()
        else:
            idx += 1

    _flush_face()

    verts_np = _denormalize_vertices(np.array(vertices_list, dtype=np.float64), mins, scale)

    # Determine face layout.
    is_quad = not has_mixed and all(c == 4 for c in face_vertex_counts)
    if is_quad or has_mixed:
        max_face_size = 4
        faces_arr = np.full((len(face_vertex_counts), max_face_size), -1, dtype=np.int64)
        for i, count in enumerate(face_vertex_counts):
            start = sum(face_vertex_counts[:i])
            for j in range(count):
                faces_arr[i, j] = j
        # Remap face vertex indices to the actual vertex indices in ordered_faces.
        # We stored vertex order during deserialization; rebuild faces.
        ordered_vi = 0
        faces_list: list[list[int]] = []
        vi_counter = 0
        for count in face_vertex_counts:
            faces_list.append(list(range(vi_counter, vi_counter + count)))
            vi_counter += count
        faces_np = np.array(
            [f + [-1] * (4 - len(f)) for f in faces_list], dtype=np.int64
        )
    else:
        faces_list_t: list[list[int]] = []
        vi_counter = 0
        for count in face_vertex_counts:
            faces_list_t.append(list(range(vi_counter, vi_counter + count)))
            vi_counter += count
        faces_np = np.array(faces_list_t, dtype=np.int64)

    result: dict = {
        "vertices": verts_np,
        "faces": faces_np,
        "is_quad": is_quad,
    }

    if return_strips:
        result["strip_boundaries"] = strip_boundaries

    return result


# ---------------------------------------------------------------------------
# UV island decomposition
# ---------------------------------------------------------------------------


class UVIslandDecomposer:
    """Decompose a mesh into connected UV islands."""

    @staticmethod
    def decompose_by_uv(
        vertices: np.ndarray,
        faces: np.ndarray,
        uv_coords: np.ndarray,
    ) -> list[np.ndarray]:
        """Find connected UV islands via flood-fill on shared UV positions.

        Parameters
        ----------
        vertices : (V, 3) unused structurally but required for interface.
        faces : (F, K) int array of face indices.
        uv_coords : (V, 2) float array of UV positions.

        Returns
        -------
        islands : list of (N_i,) int arrays, one per UV island, containing
                  face indices belonging to that island.
        """
        faces = np.asarray(faces, dtype=np.int64)
        uv_coords = np.asarray(uv_coords, dtype=np.float64)
        n_faces = len(faces)

        # Two faces share a UV edge when their corresponding UV positions
        # (after rounding for floating-point) coincide on two vertices.
        precision = 6
        uv_rounded = np.round(uv_coords, precision)

        # Build edge → face map using UV vertex IDs.
        edge_map: dict[tuple[int, int], list[int]] = {}
        face_uv_keys: list[list[tuple[float, float]]] = []
        for fi, f in enumerate(faces):
            face_uv_keys.append(
                [tuple(uv_rounded[vi]) for vi in f if vi >= 0]
            )

        visited = np.zeros(n_faces, dtype=bool)
        islands: list[np.ndarray] = []

        for seed in range(n_faces):
            if visited[seed]:
                continue
            stack = [seed]
            visited[seed] = True
            component: list[int] = [seed]

            while stack:
                current = stack.pop()
                # Find neighbours sharing a UV edge.
                current_uvs = set(face_uv_keys[current])
                for other in range(n_faces):
                    if visited[other]:
                        continue
                    other_uvs = set(face_uv_keys[other])
                    shared = current_uvs & other_uvs
                    # Two shared UV positions indicate a shared edge.
                    if len(shared) >= 2:
                        visited[other] = True
                        stack.append(other)
                        component.append(other)

            islands.append(np.array(sorted(component), dtype=np.int64))

        return islands

    @staticmethod
    def decompose_by_seams(
        vertices: np.ndarray,
        faces: np.ndarray,
        seam_edges: np.ndarray,
    ) -> list[np.ndarray]:
        """Decompose mesh into charts defined by explicit seam edges.

        Parameters
        ----------
        vertices : (V, 3) unused structurally.
        faces : (F, K) int array.
        seam_edges : (E, 2) int array of edge vertex pairs that are seams.

        Returns
        -------
        charts : list of (N_i,) int arrays of face indices per chart.
        """
        faces = np.asarray(faces, dtype=np.int64)
        seam_edges = np.asarray(seam_edges, dtype=np.int64)
        n_faces = len(faces)

        # Build edge → face map.
        edge_map: dict[tuple[int, int], list[int]] = {}
        for fi, f in enumerate(faces):
            k = len(f)
            for j in range(k):
                e = tuple(sorted((int(f[j]), int(f[(j + 1) % k]))))
                edge_map.setdefault(e, []).append(fi)

        # Mark seam edges.
        seam_set: set[tuple[int, int]] = set()
        for se in seam_edges:
            seam_set.add(tuple(sorted((int(se[0]), int(se[1])))))

        visited = np.zeros(n_faces, dtype=bool)
        charts: list[np.ndarray] = []

        for seed in range(n_faces):
            if visited[seed]:
                continue
            stack = [seed]
            visited[seed] = True
            component: list[int] = [seed]

            while stack:
                current = stack.pop()
                for j in range(len(faces[current])):
                    e = tuple(
                        sorted(
                            (
                                int(faces[current][j]),
                                int(faces[current][(j + 1) % len(faces[current])]),
                            )
                        )
                    )
                    if e in seam_set:
                        continue
                    for nbr in edge_map.get(e, []):
                        if not visited[nbr]:
                            visited[nbr] = True
                            stack.append(nbr)
                            component.append(nbr)

            charts.append(np.array(sorted(component), dtype=np.int64))

        return charts


# ---------------------------------------------------------------------------
# UV-segmented serialization
# ---------------------------------------------------------------------------


def uv_segmentation_serialize(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_islands: list[np.ndarray],
    max_strip_faces: int = 20,
) -> tuple[list[int], np.ndarray]:
    """Serialize a mesh with UV chart boundary tokens.

    Faces are traversed island-by-island.  Within each island a strip traversal
    is performed.  UV transition tokens mark island boundaries.

    Parameters
    ----------
    vertices : (V, 3) float array.
    faces : (F, 3) int array.
    uv_islands : list of (N_i,) int arrays – face indices per UV island.
    max_strip_faces : maximum faces per strip.

    Returns
    -------
    tokens : list of int
    island_labels : (F,) int array mapping each face to its island index.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    norm_verts, mins, scale = _normalize_vertices(vertices)

    n_faces_total = len(faces)
    island_labels = np.full(n_faces_total, -1, dtype=np.int64)
    for island_idx, island_faces in enumerate(uv_islands):
        island_labels[island_faces] = island_idx

    # Flatten island ordering.
    ordered_faces: list[int] = [int(fi) for isl in uv_islands for fi in isl]

    # Build strips over the full face set (re-ordered by islands).
    # We build strips independently per island.
    all_strips: list[list[int]] = []
    island_strip_ranges: list[tuple[int, int]] = []  # (start, end) in all_strips

    for isl_idx, isl_faces in enumerate(uv_islands):
        isl_faces_arr = faces[isl_faces]
        # Remap face indices to a local 0..n-1 range for the sub-mesh.
        local_strips = _triangle_strip_traversal(isl_faces_arr, max_strip_faces)
        start = len(all_strips)
        for strip in local_strips:
            all_strips.append([isl_faces[s] for s in strip])
        island_strip_ranges.append((start, len(all_strips)))

    tokens: list[int] = []

    # Header
    tokens.append(len(vertices))
    tokens.append(n_faces_total)
    tokens.append(len(all_strips))
    tokens.append(len(uv_islands))

    for coord in mins:
        tokens.extend(_encode_coordinate(coord))
    tokens.extend(_encode_coordinate(scale))

    prev_c1_per_axis: list[int | None] = [None, None, None]
    current_island = -1

    for strip_idx, strip in enumerate(all_strips):
        # Detect island transitions.
        for isl_idx, (s, e) in enumerate(island_strip_ranges):
            if s == strip_idx and isl_idx != current_island:
                if current_island >= 0:
                    # UV transition token
                    tokens.append(UV_TRANSITION_BASE + isl_idx)
                current_island = isl_idx
                break

        if strip_idx > 0:
            tokens.append(STRIP_TRANSITION_BASE + strip_idx)
            prev_c1_per_axis = [None, None, None]

        for face_pos, fi in enumerate(strip):
            if face_pos == 0:
                for vi in faces[fi]:
                    for axis in range(3):
                        toks = _encode_coordinate(
                            norm_verts[vi, axis], prev_c1_per_axis[axis]
                        )
                        tokens.extend(toks)
                        c1_arr, _, _ = _hierarchical_quantize(
                            np.array([norm_verts[vi, axis]])
                        )
                        prev_c1_per_axis[axis] = int(c1_arr[0])
            else:
                new_vi = int(faces[fi][2])
                for axis in range(3):
                    toks = _encode_coordinate(
                        norm_verts[new_vi, axis], prev_c1_per_axis[axis]
                    )
                    tokens.extend(toks)
                    c1_arr, _, _ = _hierarchical_quantize(
                        np.array([norm_verts[new_vi, axis]])
                    )
                    prev_c1_per_axis[axis] = int(c1_arr[0])

    return tokens, island_labels
