from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .schedule import LayeredSchedule, build_layered_schedule


@dataclass(frozen=True)
class LayeredGraph:
    node_table: pd.DataFrame
    edge_table: pd.DataFrame
    edge_index: torch.LongTensor

    layer_of_node: np.ndarray  # [N_layered]

    gene_layered_ids: torch.LongTensor
    root_layered_ids: torch.LongTensor

    schedule: LayeredSchedule

    layer_info: Dict


def layerize_reactome_like(raw_node_table: pd.DataFrame, raw_edge_table: pd.DataFrame) -> LayeredGraph:
    """Create a padded, strictly feedforward layered graph from a raw Reactome DAG.

    This mirrors the logic in `01_graph_preparation.ipynb`:

    - genes are layer 0
    - pathway layers are computed from longest-path distance to leaves
    - ragged branches are padded by duplicating pathways across layers
    - edges are rewritten so every edge goes from layer ℓ to ℓ+1
    - padding edges connect duplicate pathway copies across layers

    Note: gene->pathway edges are restricted to *leaf pathways* to keep genes
    exclusively in layer 0 (this matches the original notebook implementation).
    """

    node_table = raw_node_table.copy()
    edge_table = raw_edge_table.copy()

    num_nodes = int(node_table.shape[0])
    num_genes = int((node_table["node_type"] == "gene").sum())

    # Pathway hierarchy edges (child -> parent)
    pw_edges = edge_table.loc[edge_table["edge_type"].eq("pathway_to_parent"), ["src", "dst"]].astype(int)
    pw_src = pw_edges["src"].to_numpy(dtype=np.int64)
    pw_dst = pw_edges["dst"].to_numpy(dtype=np.int64)

    root_candidates = set(pw_dst.tolist()) - set(pw_src.tolist())
    root_pathways = sorted(int(x) for x in root_candidates)

    # Identify ids
    gene_ids = node_table.loc[node_table["node_type"].eq("gene"), "node_id"].astype(int).to_numpy()
    pathway_ids = node_table.loc[node_table["node_type"].eq("pathway"), "node_id"].astype(int).to_numpy()

    # indegree in pathway-only graph: number of children
    indeg = np.zeros(num_nodes, dtype=np.int32)
    np.add.at(indeg, pw_dst, 1)
    indeg0 = indeg.copy()

    parents: Dict[int, List[int]] = {}
    for s, d in zip(pw_src.tolist(), pw_dst.tolist()):
        parents.setdefault(int(s), []).append(int(d))

    # Kahn + DP for longest-path levels from leaves
    from collections import deque

    level_base = np.zeros(num_nodes, dtype=np.int32)
    q = deque([int(p) for p in pathway_ids.tolist() if indeg[int(p)] == 0])

    seen = 0
    while q:
        u = int(q.popleft())
        seen += 1
        for p in parents.get(u, []):
            if level_base[p] < level_base[u] + 1:
                level_base[p] = level_base[u] + 1
            indeg[p] -= 1
            if indeg[p] == 0:
                q.append(p)

    if seen != len(pathway_ids):
        raise ValueError(
            "Pathway graph appears cyclic or disconnected (unexpected for Reactome). "
            f"Topological pass saw {seen} of {len(pathway_ids)} pathway nodes."
        )

    pathway_layer = level_base + 1  # reserve layer 0 for genes
    L = int(pathway_layer[pathway_ids].max(initial=0))

    leaf_pathways = set(int(p) for p in pathway_ids.tolist() if indeg0[int(p)] == 0)

    # Compute upward duplication requirements
    max_required_layer = pathway_layer.copy()
    for s, d in zip(pw_src.tolist(), pw_dst.tolist()):
        req = int(pathway_layer[int(d)] - 1)
        if req > max_required_layer[int(s)]:
            max_required_layer[int(s)] = req

    # Pad roots to global max layer
    for p in root_pathways:
        max_required_layer[int(p)] = L

    name_by_id = dict(zip(node_table["node_id"].astype(int), node_table["node_name"].astype(str)))

    layered_lookup: Dict[Tuple[int, int], int] = {}
    layered_records: List[Dict] = []

    next_id = 0
    # Genes, layer 0
    for gid in gene_ids.tolist():
        gid = int(gid)
        layered_lookup[(gid, 0)] = next_id
        layered_records.append(
            {
                "layered_id": next_id,
                "orig_id": gid,
                "orig_name": name_by_id[gid],
                "node_type": "gene",
                "layer": 0,
                "is_duplicate": False,
            }
        )
        next_id += 1

    pathway_ids_sorted = np.sort(pathway_ids.astype(int))
    for layer in range(1, L + 1):
        for pid in pathway_ids_sorted.tolist():
            pid = int(pid)
            if pathway_layer[pid] <= layer <= max_required_layer[pid]:
                layered_lookup[(pid, layer)] = next_id
                layered_records.append(
                    {
                        "layered_id": next_id,
                        "orig_id": pid,
                        "orig_name": name_by_id[pid],
                        "node_type": "pathway",
                        "layer": layer,
                        "is_duplicate": bool(layer > pathway_layer[pid]),
                    }
                )
                next_id += 1

    node_table_layered = pd.DataFrame(layered_records)

    # Build layered edges (delta=1)
    src_list: List[int] = []
    dst_list: List[int] = []
    etype_list: List[str] = []

    # Gene -> leaf pathways only (0->1)
    g2p = edge_table.loc[edge_table["edge_type"].eq("gene_to_pathway"), ["src", "dst"]].astype(int)
    g2p = g2p[g2p["dst"].isin(leaf_pathways)].copy()

    for s, d in g2p.itertuples(index=False):
        src_list.append(layered_lookup[(int(s), 0)])
        dst_list.append(layered_lookup[(int(d), 1)])
        etype_list.append("gene_to_leaf_pathway")

    # Pathway child -> parent aligned to parent's layer
    for s, d in zip(pw_src.tolist(), pw_dst.tolist()):
        s = int(s)
        d = int(d)
        parent_layer = int(pathway_layer[d])
        src_layer = parent_layer - 1
        dst_layer = parent_layer

        src_list.append(layered_lookup[(s, src_layer)])
        dst_list.append(layered_lookup[(d, dst_layer)])
        etype_list.append("pathway_child_to_parent")

    # Padding edges for duplicates
    for pid in pathway_ids_sorted.tolist():
        pid = int(pid)
        start = int(pathway_layer[pid])
        end = int(max_required_layer[pid])
        for layer in range(start, end):
            src_list.append(layered_lookup[(pid, layer)])
            dst_list.append(layered_lookup[(pid, layer + 1)])
            etype_list.append("padding_identity")

    src_arr = np.asarray(src_list, dtype=np.int64)
    dst_arr = np.asarray(dst_list, dtype=np.int64)

    edge_table_layered = pd.DataFrame({"src": src_arr, "dst": dst_arr, "edge_type": etype_list})
    edge_index_layered = torch.tensor(np.vstack([src_arr, dst_arr]), dtype=torch.long)

    # Validate deltas
    layer_by_id = node_table_layered.sort_values("layered_id")["layer"].to_numpy(dtype=np.int64)
    deltas = layer_by_id[dst_arr] - layer_by_id[src_arr]
    if not np.all(deltas == 1):
        bad = np.where(deltas != 1)[0][:10]
        raise AssertionError(f"Found edges not going from ℓ to ℓ+1, examples: {bad.tolist()}")

    root_layered_ids = torch.tensor([layered_lookup[(int(p), L)] for p in root_pathways], dtype=torch.long)
    gene_layered_ids = torch.arange(len(gene_ids), dtype=torch.long)

    layer_counts = node_table_layered["layer"].value_counts().sort_index()
    edge_counts_by_layer = pd.Series(layer_by_id[dst_arr]).value_counts().sort_index()

    layer_info = {
        "L": int(L),
        "num_layered_nodes": int(node_table_layered.shape[0]),
        "num_layered_edges": int(edge_table_layered.shape[0]),
        "nodes_per_layer": {int(k): int(v) for k, v in layer_counts.to_dict().items()},
        "edges_per_dst_layer": {int(k): int(v) for k, v in edge_counts_by_layer.to_dict().items()},
        "num_leaf_pathways": int(len(leaf_pathways)),
        "num_root_pathways": int(len(root_pathways)),
        "root_pathways_orig_ids": [int(p) for p in root_pathways],
        "root_pathways_layered_ids": [int(i) for i in root_layered_ids.tolist()],
    }

    schedule = build_layered_schedule(
        edge_index=edge_index_layered,
        node_layer=layer_by_id,
        edge_type=edge_table_layered["edge_type"].astype(str).tolist(),
        gene_ids=gene_layered_ids,
        root_ids=root_layered_ids,
    )

    return LayeredGraph(
        node_table=node_table_layered,
        edge_table=edge_table_layered,
        edge_index=edge_index_layered,
        layer_of_node=layer_by_id,
        gene_layered_ids=gene_layered_ids,
        root_layered_ids=root_layered_ids,
        schedule=schedule,
        layer_info=layer_info,
    )


def save_layered_graph(layered: LayeredGraph, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    layered.node_table.to_csv(out_dir / "node_table_layered.csv", index=False)
    layered.edge_table.to_csv(out_dir / "edge_table_layered.csv", index=False)
    torch.save(layered.edge_index, out_dir / "edge_index_layered.pt")

    import json

    with open(out_dir / "layer_info.json", "w") as f:
        json.dump(layered.layer_info, f, indent=2)

    # Core tensors used in training
    torch.save(layered.root_layered_ids, out_dir / "root_pathway_idx.pt")
    torch.save(layered.gene_layered_ids, out_dir / "gene_layered_ids.pt")

    # Schedule artifacts
    torch.save([s.src for s in layered.schedule.steps], out_dir / "src_by_layer.pt")
    torch.save([s.dst for s in layered.schedule.steps], out_dir / "dst_by_layer.pt")
    torch.save([s.dst_unique for s in layered.schedule.steps], out_dir / "dst_unique_by_layer.pt")
    torch.save([s.dst_pos for s in layered.schedule.steps], out_dir / "dst_pos_by_layer.pt")
    torch.save([s.eid for s in layered.schedule.steps], out_dir / "edge_id_by_layer.pt")

    # For backwards compatibility with existing notebooks
    torch.save([torch.stack([s.src, s.dst], dim=0) for s in layered.schedule.steps], out_dir / "edge_index_by_layer.pt")
    torch.save([s.eid for s in layered.schedule.steps], out_dir / "edge_id_by_layer.pt")
    torch.save([s.dst_unique for s in layered.schedule.steps], out_dir / "dst_nodes_by_layer.pt")
