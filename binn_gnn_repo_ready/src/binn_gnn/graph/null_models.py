from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


def directed_double_edge_swap(
    edge_index: torch.LongTensor,
    num_nodes: int,
    n_swap: int,
    seed: int = 0,
    max_tries: Optional[int] = None,
    allow_self_loops: bool = False,
) -> torch.LongTensor:
    """Degree-preserving randomization for a directed simple graph.

    Performs Maslov-Sneppen style endpoint swaps:
      (a->b, c->d) becomes (a->d, c->b)

    Preserves in-degree and out-degree sequences exactly.

    Notes:
    - This is implemented on CPU using numpy for simplicity and determinism.
    - It assumes edge_index represents a *simple* directed graph (no multi-edges).

    If you need to preserve bipartite structure (genes->pathways) or edge types,
    call this separately on each edge-type subgraph.
    """

    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must be [2,E], got {tuple(edge_index.shape)}")

    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)
    E = src.shape[0]

    if max_tries is None:
        max_tries = n_swap * 20

    rng = np.random.default_rng(seed)

    # Use a python set for O(1) membership
    edges = set(zip(src.tolist(), dst.tolist()))

    swaps = 0
    tries = 0
    while swaps < n_swap and tries < max_tries:
        tries += 1
        i, j = rng.integers(0, E, size=2)
        if i == j:
            continue

        a, b = int(src[i]), int(dst[i])
        c, d = int(src[j]), int(dst[j])

        # Avoid trivial cases
        if a == c or b == d:
            continue

        new1 = (a, d)
        new2 = (c, b)

        if not allow_self_loops and (new1[0] == new1[1] or new2[0] == new2[1]):
            continue

        # Avoid multi-edges
        if new1 in edges or new2 in edges:
            continue

        # Apply swap
        edges.remove((a, b))
        edges.remove((c, d))
        edges.add(new1)
        edges.add(new2)

        src[i], dst[i] = new1
        src[j], dst[j] = new2

        swaps += 1

    if swaps < n_swap:
        raise RuntimeError(f"Could only perform {swaps}/{n_swap} swaps (tries={tries}).")

    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)


def degree_sequences(edge_index: torch.LongTensor, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    src = edge_index[0].cpu().numpy().astype(np.int64)
    dst = edge_index[1].cpu().numpy().astype(np.int64)
    out_deg = np.bincount(src, minlength=num_nodes)
    in_deg = np.bincount(dst, minlength=num_nodes)
    return in_deg, out_deg
