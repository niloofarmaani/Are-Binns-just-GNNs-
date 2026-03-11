from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class LayerStep:
    """Precomputed tensors for one propagation step.

    The step corresponds to all edges whose destination nodes should be updated
    at this step.

    Shapes:
      src: [E]
      dst: [E]
      dst_unique: [U]
      dst_pos: [E] (maps each edge's dst to an index in dst_unique)
      eid: [E] global edge ids
    """

    src: torch.LongTensor
    dst: torch.LongTensor
    dst_unique: torch.LongTensor
    dst_pos: torch.LongTensor
    eid: torch.LongTensor


@dataclass(frozen=True)
class LayeredSchedule:
    """Schedule for a padded, strictly layer-to-layer graph.

    This matches the classic BINN feedforward semantics where every edge goes
    from layer ℓ to layer ℓ+1 (after inserting copy nodes).
    """

    N: int
    E: int
    L: int
    steps: List[LayerStep]

    gene_ids: torch.LongTensor
    root_ids: torch.LongTensor

    # Edge metadata
    is_padding_edge: torch.BoolTensor


@dataclass(frozen=True)
class DepthSchedule:
    """Schedule for a DAG without padding, grouped by node depth.

    Depth is defined as the longest-path distance from source nodes.
    Edges may connect non-consecutive depths; that's fine.
    """

    N: int
    E: int
    D: int
    steps: List[LayerStep]

    gene_ids: torch.LongTensor
    root_ids: torch.LongTensor

    depth: torch.LongTensor  # [N]


def _as_long(x: np.ndarray | torch.Tensor) -> torch.LongTensor:
    if isinstance(x, torch.Tensor):
        return x.long()
    return torch.from_numpy(x.astype(np.int64))


def build_layered_schedule(
    *,
    edge_index: torch.LongTensor,
    node_layer: Sequence[int] | np.ndarray,
    edge_type: Optional[Sequence[str]] = None,
    gene_ids: Sequence[int] | np.ndarray | torch.Tensor,
    root_ids: Sequence[int] | np.ndarray | torch.Tensor,
) -> LayeredSchedule:
    """Build a per-layer message-passing schedule from a layered (padded) graph.

    Preconditions:
      - edge_index is [2, E]
      - all edges go from layer ℓ to ℓ+1

    The schedule groups edges by destination layer (1..L) and precomputes
    dst_unique and dst_pos for index_add aggregation.
    """

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2,E], got {tuple(edge_index.shape)}")

    node_layer = np.asarray(node_layer, dtype=np.int64)
    if node_layer.ndim != 1:
        raise ValueError("node_layer must be 1D")

    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)

    N = int(node_layer.shape[0])
    E = int(edge_index.size(1))

    dst_layer = node_layer[dst]
    src_layer = node_layer[src]
    deltas = dst_layer - src_layer
    if not np.all(deltas == 1):
        bad = np.where(deltas != 1)[0][:10]
        raise ValueError(
            "Layered graph invariant violated: found edges not going from ℓ to ℓ+1. "
            f"Example edge indices: {bad.tolist()}"
        )

    L = int(node_layer.max())
    # Edge ids are just the saved order
    edge_id = np.arange(E, dtype=np.int64)

    steps: List[LayerStep] = []
    for l in range(1, L + 1):
        m = dst_layer == l
        if not np.any(m):
            # Allow empty layers, but keep a step so depths match L.
            ei = np.zeros((2, 0), dtype=np.int64)
            eids = np.zeros((0,), dtype=np.int64)
        else:
            ei = np.vstack([src[m], dst[m]])
            eids = edge_id[m]

        # Precompute dst_unique and inverse indices
        dst_t = torch.from_numpy(ei[1]).long()
        if dst_t.numel() == 0:
            dst_unique = dst_t
            inv = dst_t
        else:
            dst_unique, inv = torch.unique(dst_t, sorted=True, return_inverse=True)

        step = LayerStep(
            src=torch.from_numpy(ei[0]).long(),
            dst=dst_t,
            dst_unique=dst_unique,
            dst_pos=inv,
            eid=torch.from_numpy(eids).long(),
        )
        steps.append(step)

    # Edge metadata
    if edge_type is None:
        is_padding_edge = torch.zeros((E,), dtype=torch.bool)
    else:
        if len(edge_type) != E:
            raise ValueError("edge_type length must match number of edges")
        is_padding_edge = torch.tensor([t == "padding_identity" for t in edge_type], dtype=torch.bool)

    return LayeredSchedule(
        N=N,
        E=E,
        L=L,
        steps=steps,
        gene_ids=_as_long(np.asarray(gene_ids)),
        root_ids=_as_long(np.asarray(root_ids)),
        is_padding_edge=is_padding_edge,
    )


def _topological_order(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Kahn topological sort for a directed graph.

    Returns an array of node ids in topological order.

    Raises ValueError if the graph contains a cycle.
    """

    src = edge_index[0]
    dst = edge_index[1]
    indeg = np.zeros((num_nodes,), dtype=np.int64)
    np.add.at(indeg, dst, 1)

    q = list(np.where(indeg == 0)[0])
    out = []

    # Build adjacency list
    adj: Dict[int, List[int]] = {}
    for s, d in zip(src.tolist(), dst.tolist()):
        adj.setdefault(int(s), []).append(int(d))

    head = 0
    while head < len(q):
        u = int(q[head])
        head += 1
        out.append(u)
        for v in adj.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(out) != num_nodes:
        raise ValueError("Graph is not a DAG (topological sort failed).")
    return np.asarray(out, dtype=np.int64)


def build_depth_schedule(
    *,
    edge_index: torch.LongTensor,
    gene_ids: Sequence[int] | np.ndarray | torch.Tensor,
    root_ids: Sequence[int] | np.ndarray | torch.Tensor,
) -> DepthSchedule:
    """Build a message-passing schedule on a raw DAG, grouped by node depth.

    Depth(v) = max_{u->v} depth(u)+1 with depth(source)=0.

    This corresponds to a single feedforward pass over the DAG without inserting
    copy nodes. It is the minimal change that removes the padding artifact.
    """

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2,E], got {tuple(edge_index.shape)}")

    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)

    N = int(max(src.max(initial=0), dst.max(initial=0)) + 1)
    E = int(edge_index.size(1))

    topo = _topological_order(np.vstack([src, dst]), N)

    depth = np.zeros((N,), dtype=np.int64)
    # DP along topo order
    # Build incoming edges list for fast iteration
    incoming: Dict[int, List[int]] = {}
    for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
        incoming.setdefault(int(d), []).append(i)

    # We can compute depth by scanning edges in topo order
    for v in topo:
        for ei in incoming.get(int(v), []):
            s = src[ei]
            depth[v] = max(depth[v], depth[s] + 1)

    D = int(depth.max(initial=0))

    edge_id = np.arange(E, dtype=np.int64)

    steps: List[LayerStep] = []
    for d in range(1, D + 1):
        m = depth[dst] == d
        if not np.any(m):
            ei = np.zeros((2, 0), dtype=np.int64)
            eids = np.zeros((0,), dtype=np.int64)
        else:
            ei = np.vstack([src[m], dst[m]])
            eids = edge_id[m]

        dst_t = torch.from_numpy(ei[1]).long()
        if dst_t.numel() == 0:
            dst_unique = dst_t
            inv = dst_t
        else:
            dst_unique, inv = torch.unique(dst_t, sorted=True, return_inverse=True)

        steps.append(
            LayerStep(
                src=torch.from_numpy(ei[0]).long(),
                dst=dst_t,
                dst_unique=dst_unique,
                dst_pos=inv,
                eid=torch.from_numpy(eids).long(),
            )
        )

    return DepthSchedule(
        N=N,
        E=E,
        D=D,
        steps=steps,
        gene_ids=_as_long(np.asarray(gene_ids)),
        root_ids=_as_long(np.asarray(root_ids)),
        depth=torch.from_numpy(depth).long(),
    )
