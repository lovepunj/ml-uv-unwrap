from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(v: Tensor, eps: float = 1e-8) -> Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def _dihedral_angle(n1: Tensor, n2: Tensor) -> Tensor:
    """Signed dihedral angle between two face normals."""
    cos = F.cosine_similarity(n1, n2, dim=-1).clamp(-1, 1)
    return torch.acos(cos)


def _face_area(vertices: Tensor, faces: Tensor) -> Tensor:
    """Compute per-face areas using cross product."""
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)


def _face_normals(vertices: Tensor, faces: Tensor) -> Tensor:
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return _normalize(torch.cross(v1 - v0, v2 - v0, dim=-1))


def _vertex_normals(vertices: Tensor, faces: Tensor, num_verts: int) -> Tensor:
    """Area-weighted vertex normals."""
    fn = _face_normals(vertices, faces)
    fa = _face_area(vertices, faces)
    vn = torch.zeros(num_verts, 3, device=vertices.device, dtype=vertices.dtype)
    for i in range(3):
        vn.index_add_(0, faces[:, i], fn * fa.unsqueeze(-1))
    return _normalize(vn)


def _vertex_curvature(vertices: Tensor, faces: Tensor, num_verts: int) -> Tensor:
    """Simple discrete mean curvature approximation via Laplacian."""
    lap = torch.zeros_like(vertices)
    num_faces = faces.shape[0]
    ones = torch.ones(num_faces, 1, device=vertices.device, dtype=vertices.dtype)
    count = torch.zeros(num_verts, 1, device=vertices.device, dtype=vertices.dtype)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            lap.index_add_(0, faces[:, i], vertices[faces[:, j]])
            count.index_add_(0, faces[:, i], ones)
    lap = lap / count.clamp(min=1)
    return (vertices - lap).norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Graph data structure
# ---------------------------------------------------------------------------

@dataclass
class MeshGraph:
    """Graph representation of a triangle mesh."""
    num_nodes: int
    num_edges: int
    edge_index: Tensor          # (2, E) – undirected edges stored twice
    node_features: Tensor       # (N, F_n)
    edge_features: Tensor       # (E, F_e)
    neighbors: list[list[int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SeamTokenizer – utility conversions between mesh, graph and charts
# ---------------------------------------------------------------------------

class SeamTokenizer:
    """Converts between mesh, graph and chart representations.

    Key insight from the paper: operating directly on the mesh graph
    eliminates projection artifacts.  Sequence length is 3.4×–5.1× shorter
    than coordinate-based methods.
    """

    @staticmethod
    def mesh_to_graph(vertices: Tensor, faces: Tensor) -> MeshGraph:
        """Convert a triangle mesh to a graph representation.

        Args:
            vertices: (N, 3) vertex positions.
            faces:    (F, 3) triangle face indices.

        Returns:
            MeshGraph with per-node features (position, normal, curvature,
            degree) and per-edge features (length, dihedral angle).
        """
        num_verts = vertices.shape[0]

        # --- build edge set (each undirected edge stored once) ---
        edges_set: set[tuple[int, int]] = set()
        edge_to_faces: dict[tuple[int, int], list[int]] = {}
        for fi in range(faces.shape[0]):
            tri = faces[fi].tolist()
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = (min(a, b), max(a, b))
                edges_set.add(key)
                edge_to_faces.setdefault(key, []).append(fi)

        sorted_edges = sorted(edges_set)
        E = len(sorted_edges)
        edge_index = torch.tensor(sorted_edges, dtype=torch.long).t()  # (2, E)

        # --- node features ---
        normals = _vertex_normals(vertices, faces, num_verts)
        curvature = _vertex_curvature(vertices, faces, num_verts)
        deg = torch.zeros(num_verts, 1, dtype=vertices.dtype)
        for a, b in sorted_edges:
            deg[a] += 1
            deg[b] += 1
        node_features = torch.cat([vertices, normals, curvature, deg], dim=-1)

        # --- edge features ---
        lengths = (vertices[edge_index[0]] - vertices[edge_index[1]]).norm(dim=-1, keepdim=True)

        face_normals = _face_normals(vertices, faces)
        dihedral = torch.zeros(E, 1, device=vertices.device, dtype=vertices.dtype)
        for ei, (a, b) in enumerate(sorted_edges):
            adj_faces = edge_to_faces[(a, b)]
            if len(adj_faces) == 2:
                dihedral[ei] = _dihedral_angle(
                    face_normals[adj_faces[0]], face_normals[adj_faces[1]]
                ).unsqueeze(0)
        edge_features = torch.cat([lengths, dihedral], dim=-1)

        # --- adjacency lists (edge-to-edge via shared vertices) ---
        neighbors: list[list[int]] = [[] for _ in range(E)]
        for ei in range(E):
            a, b = int(edge_index[0, ei]), int(edge_index[1, ei])
            # Two edges are adjacent if they share a vertex
            for ej in range(E):
                if ei == ej:
                    continue
                ca, cb = int(edge_index[0, ej]), int(edge_index[1, ej])
                if a == ca or a == cb or b == ca or b == cb:
                    neighbors[ei].append(ej)

        return MeshGraph(
            num_nodes=num_verts,
            num_edges=E,
            edge_index=edge_index,
            node_features=node_features,
            edge_features=edge_features,
            neighbors=neighbors,
        )

    @staticmethod
    def seams_to_charts(seams: set[tuple[int, int]], faces: Tensor) -> Tensor:
        """Convert seam edges to per-face chart labels via flood fill.

        Args:
            seams:  set of undirected seam edges {(min_idx, max_idx), ...}.
            faces:  (F, 3) face indices.

        Returns:
            (F,) int tensor of chart labels.
        """
        F_count = faces.shape[0]
        face_adj: list[list[int]] = [[] for _ in range(F_count)]
        for fi in range(F_count):
            for fj in range(fi + 1, F_count):
                shared = set(faces[fi].tolist()) & set(faces[fj].tolist())
                if len(shared) == 2:
                    edge = tuple(sorted(shared))
                    if edge not in seams:
                        face_adj[fi].append(fj)
                        face_adj[fj].append(fi)

        labels = torch.full((F_count,), -1, dtype=torch.long)
        cid = 0
        for start in range(F_count):
            if labels[start] != -1:
                continue
            queue = deque([start])
            labels[start] = cid
            while queue:
                cur = queue.popleft()
                for nb in face_adj[cur]:
                    if labels[nb] == -1:
                        labels[nb] = cid
                        queue.append(nb)
            cid += 1
        return labels

    @staticmethod
    def charts_to_seams(chart_labels: Tensor, faces: Tensor) -> set[tuple[int, int]]:
        """Convert chart labels back to seam edges.

        Args:
            chart_labels: (F,) int tensor of chart labels.
            faces:        (F, 3) face indices.

        Returns:
            Set of undirected seam edges.
        """
        seams: set[tuple[int, int]] = set()
        F_count = faces.shape[0]
        for fi in range(F_count):
            for fj in range(fi + 1, F_count):
                if chart_labels[fi] != chart_labels[fj]:
                    shared = set(faces[fi].tolist()) & set(faces[fj].tolist())
                    if len(shared) == 2:
                        seams.add(tuple(sorted(shared)))
        return seams


# ---------------------------------------------------------------------------
# MeshGraphEncoder – GraphSAGE-based vertex encoder
# ---------------------------------------------------------------------------

class _SAGELayer(nn.Module):
    """Single GraphSAGE layer (mean aggregation)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim + in_dim, out_dim)

    def forward(
        self, x: Tensor, edge_index: Tensor, node_feat: Tensor
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        # aggregate neighbours
        msg = x[src]
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg)
        num_edges = dst.shape[0]
        deg = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        ones = torch.ones(num_edges, 1, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, ones)
        agg = agg / deg.clamp(min=1)
        out = self.linear(torch.cat([x, agg], dim=-1))
        return F.relu(out)


class MeshGraphEncoder(nn.Module):
    """Graph neural-network encoder for triangle meshes.

    Builds a graph from mesh vertices and edges where:
    - Each node = mesh vertex with features (position, normal, curvature, degree).
    - Each edge = mesh edge with features (length, dihedral angle).

    Uses 3 layers of GraphSAGE message passing to encode local geometry.
    Output: per-vertex embeddings of dimension 256.
    """

    NODE_IN: int = 8   # pos(3) + normal(3) + curvature(1) + degree(1)

    def __init__(self, hidden: int = 256, n_layers: int = 3) -> None:
        super().__init__()
        self.node_embed = nn.Linear(self.NODE_IN, hidden)
        self.layers = nn.ModuleList(
            [SAGELayer(hidden, hidden) for SAGELayer in
             [_SAGELayer] * n_layers]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def forward(self, graph: MeshGraph) -> Tensor:
        """Encode all vertices.

        Returns:
            (N, 256) vertex embeddings.
        """
        x = F.relu(self.node_embed(graph.node_features))
        for layer, norm in zip(self.layers, self.norms):
            x = norm(layer(x, graph.edge_index, x) + x)
        return x


# ---------------------------------------------------------------------------
# PointerDecoder – GPT-style causal transformer over edges
# ---------------------------------------------------------------------------

class _EdgePositionalEncoding(nn.Module):
    """Learnable positional encoding for edge sequences."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, pos: Tensor) -> Tensor:
        return self.pe(pos)


class PointerDecoder(nn.Module):
    """GPT-style causal transformer that predicts seam paths over mesh edges.

    At each step, predicts the next edge in the seam sequence from
    neighbouring edges using a pointer network mechanism.  By construction
    predictions are edge-aligned – no coordinate snapping is required.

    Input:  current seam path + graph embeddings.
    Output: probability distribution over neighbouring edges.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # edge encoder: project edge features + endpoints into d_model
        self.edge_in_proj = nn.Linear(256 * 2 + 2, d_model)
        self.edge_pos_enc = _EdgePositionalEncoding(d_model)

        # causal transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # pointer: query from decoder → score over candidate edges
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

        self.norm = nn.LayerNorm(d_model)

    def _encode_edge_sequence(
        self, edge_embs: Tensor, seq_pos: Tensor
    ) -> Tensor:
        """Inject positional encoding into edge embedding sequence."""
        return edge_embs + self.edge_pos_enc(seq_pos)

    def forward(
        self,
        path_embeddings: Tensor,
        candidate_embeddings: Tensor,
        path_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute pointer logits for the next edge.

        Args:
            path_embeddings:      (B, T, d) embeddings of the current path.
            candidate_embeddings: (B, C, d) embeddings of all candidate edges.
            path_mask:            (B, T) boolean mask (True = ignore).

        Returns:
            (B, C) logits over candidates.
        """
        B, T, _ = path_embeddings.shape
        C = candidate_embeddings.shape[1]

        # causal mask (triangular) + optional padding mask
        causal = nn.Transformer.generate_square_subsequent_mask(
            T, device=path_embeddings.device
        )
        if path_mask is not None:
            # expand mask: (B, T) -> (B, T, T) diagonal padding mask
            pad = path_mask.unsqueeze(1).expand(B, T, T)
            causal = causal.unsqueeze(0).expand(B, T, T).clone()
            causal.masked_fill_(pad, float("-inf"))

        decoded = self.transformer(
            path_embeddings,
            path_embeddings,
            tgt_mask=causal,
        )

        query = self.query_proj(decoded[:, -1])        # (B, d)
        keys = self.key_proj(candidate_embeddings)      # (B, C, d)
        logits = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)  # (B, C)
        return logits

    def embed_edges(
        self,
        edge_indices: Tensor,
        vertex_embs: Tensor,
        edge_features: Tensor,
    ) -> Tensor:
        """Embed a set of edges by concatenating endpoint vertex embeddings.

        Args:
            edge_indices:  (2, E) edge_index.
            vertex_embs:   (N, d) vertex embeddings.
            edge_features: (E, 2) edge features.

        Returns:
            (E, 2*d+2) edge embeddings.
        """
        src = vertex_embs[edge_indices[0]]
        dst = vertex_embs[edge_indices[1]]
        return torch.cat([src, dst, edge_features], dim=-1)


# ---------------------------------------------------------------------------
# MeshTailorModel – full seam generation model
# ---------------------------------------------------------------------------

@dataclass
class BeamState:
    """State of a single beam during beam search."""
    path: list[int]
    score: float
    visited: set[int] = field(default_factory=set)


class MeshTailorModel(nn.Module):
    """MeshTailor: Cutting Seams via Generative Mesh Traversal (March 2026).

    Combines a GraphSAGE mesh encoder with a GPT-style pointer decoder to
    autoregressively generate UV seam paths directly on the mesh graph.

    Key advantages over coordinate-based methods:
    - Predictions are naturally edge-aligned (no projection artifacts).
    - Sequence length is 3.4×–5.1× shorter.
    - Fragmentation and distortion are jointly optimised.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        encoder_layers: int = 3,
        decoder_layers: int = 6,
        dim_ff: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = MeshGraphEncoder(hidden=d_model, n_layers=encoder_layers)
        self.decoder = PointerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=decoder_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        self.d_model = d_model

        # start-token embedding (learned)
        self.start_token = nn.Parameter(torch.randn(d_model))

    # ------------------------------------------------------------------
    # seam quality metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_seam_quality(
        seams: set[tuple[int, int]],
        vertices: Tensor,
        faces: Tensor,
    ) -> dict[str, Tensor]:
        """Compute distortion and fragmentation metrics for a set of seams.

        Args:
            seams:   set of undirected seam edges.
            vertices: (N, 3) vertex positions.
            faces:    (F, 3) face indices.

        Returns:
            Dict with 'distortion' (scalar) and 'fragmentation' (scalar).
        """
        chart_labels = SeamTokenizer.seams_to_charts(seams, faces)
        num_charts = int(chart_labels.max().item()) + 1 if chart_labels.numel() > 0 else 0

        # fragmentation = number of charts (lower is better)
        frag = torch.tensor(float(num_charts), device=vertices.device)

        # distortion = average stretch across seam edges
        if num_charts == 0 or len(seams) == 0:
            return {"distortion": torch.tensor(0.0, device=vertices.device),
                    "fragmentation": frag}

        total_stretch = torch.tensor(0.0, device=vertices.device)
        for a, b in seams:
            edge_len = (vertices[a] - vertices[b]).norm()
            total_stretch = total_stretch + edge_len
        avg_stretch = total_stretch / len(seams)

        return {"distortion": avg_stretch, "fragmentation": frag}

    # ------------------------------------------------------------------
    # greedy decoding
    # ------------------------------------------------------------------

    def _greedy_step(
        self,
        path: list[int],
        graph: MeshGraph,
        vertex_embs: Tensor,
        edge_embs: Tensor,
        visited: set[int],
        edge_features: Tensor,
    ) -> int:
        """Pick the best next edge via greedy pointer decoding."""
        device = vertex_embs.device
        path_len = len(path)

        # build path embedding sequence
        path_t = torch.tensor(path, dtype=torch.long, device=device)  # (T,)
        path_emb = edge_embs[path_t].unsqueeze(0)  # (1, T, 2d+2)
        path_proj = self.decoder.edge_in_proj(path_emb)
        path_proj = self.decoder._encode_edge_sequence(
            path_proj, torch.arange(path_len, device=device).unsqueeze(0)
        )

        # collect candidate edges: edges adjacent to the head of the path
        head = path[-1]
        candidates = [
            ei for ei in graph.neighbors[head]
            if ei not in visited
        ]
        if not candidates:
            return -1  # dead end

        cand_idx = torch.tensor(candidates, dtype=torch.long, device=device)
        cand_emb = edge_embs[cand_idx].unsqueeze(0)  # (1, C, 2d+2)
        cand_proj = self.decoder.edge_in_proj(cand_emb)

        logits = self.decoder.forward(path_proj, cand_proj)
        best = int(cand_idx[logits.argmax(dim=-1).item()].item())
        return best

    @torch.no_grad()
    def generate_seams(
        self,
        vertices: Tensor,
        faces: Tensor,
        max_length: int = 200,
        beam_width: int = 1,
        temperature: float = 1.0,
    ) -> list[set[tuple[int, int]]]:
        """Autoregressively generate seam paths.

        Args:
            vertices:    (N, 3) vertex positions.
            faces:       (F, 3) face indices.
            max_length:  maximum seam path length in edges.
            beam_width:  number of beams (1 = greedy).
            temperature: sampling temperature (ignored when beam_width=1).

        Returns:
            List of seam edge-sets, one per beam.
        """
        device = vertices.device
        graph = SeamTokenizer.mesh_to_graph(vertices, faces)

        if graph.num_edges == 0:
            return [set()]

        vertex_embs = self.encoder(graph)                # (N, d)
        raw_edge_embs = self.decoder.embed_edges(        # (E, 2d+2)
            graph.edge_index, vertex_embs, graph.edge_features
        )

        # --- greedy path (single beam) ---
        if beam_width <= 1:
            # start from the edge with highest degree endpoint
            start_edge = 0
            path = [start_edge]
            visited = {start_edge}
            for _ in range(max_length - 1):
                next_ei = self._greedy_step(
                    path, graph, vertex_embs, raw_edge_embs,
                    visited, graph.edge_features,
                )
                if next_ei == -1:
                    break
                path.append(next_ei)
                visited.add(next_ei)

            seams = set()
            ei_map = {i: (int(graph.edge_index[0, i]), int(graph.edge_index[1, i]))
                      for i in range(graph.num_edges)}
            for idx in path:
                seams.add(ei_map[idx])
            return [seams]

        # --- beam search ---
        beams: list[BeamState] = []
        edge_map = {i: (int(graph.edge_index[0, i]), int(graph.edge_index[1, i]))
                    for i in range(graph.num_edges)}
        for e0 in range(min(graph.num_edges, beam_width)):
            beams.append(BeamState(
                path=[e0],
                score=0.0,
                visited={e0},
            ))

        for _step in range(max_length - 1):
            new_beams: list[BeamState] = []
            for beam in beams:
                if beam.path[-1] == -1:
                    new_beams.append(beam)
                    continue
                head = beam.path[-1]
                candidates = [
                    ei for ei in graph.neighbors[head]
                    if ei not in beam.visited
                ]
                if not candidates:
                    beam.path.append(-1)
                    new_beams.append(beam)
                    continue

                device = vertex_embs.device
                path_t = torch.tensor(beam.path, dtype=torch.long, device=device)
                path_emb = raw_edge_embs[path_t].unsqueeze(0)
                path_proj = self.decoder.edge_in_proj(path_emb)
                path_proj = self.decoder._encode_edge_sequence(
                    path_proj,
                    torch.arange(len(beam.path), device=device).unsqueeze(0),
                )

                cand_idx = torch.tensor(candidates, dtype=torch.long, device=device)
                cand_emb = raw_edge_embs[cand_idx].unsqueeze(0)
                cand_proj = self.decoder.edge_in_proj(cand_emb)

                logits = self.decoder.forward(path_proj, cand_proj).squeeze(0)
                if temperature != 1.0:
                    logits = logits / temperature
                probs = F.softmax(logits, dim=-1)

                topk = min(beam_width, len(candidates))
                topk_vals, topk_idx = probs.topk(topk)
                for ki in range(topk):
                    ei = int(cand_idx[topk_idx[ki]].item())
                    new_path = beam.path + [ei]
                    new_visited = beam.visited | {ei}
                    new_score = beam.score + float(math.log(topk_vals[ki].item() + 1e-12))
                    new_beams.append(BeamState(new_path, new_score, new_visited))

            # keep top beams
            new_beams.sort(key=lambda b: b.score, reverse=True)
            beams = new_beams[:beam_width]

        results: list[set[tuple[int, int]]] = []
        for beam in beams:
            seams = set()
            for idx in beam.path:
                if idx == -1:
                    continue
                seams.add(edge_map[idx])
            results.append(seams)
        return results
