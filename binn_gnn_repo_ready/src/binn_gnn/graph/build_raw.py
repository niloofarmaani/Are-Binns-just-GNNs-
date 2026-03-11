from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class RawGraph:
    node_table: pd.DataFrame
    edge_table: pd.DataFrame
    edge_index: torch.LongTensor
    is_gene: torch.BoolTensor
    node2id: Dict[str, int]

    num_genes: int


def build_raw_reactome_graph(
    *,
    expr_gene_ids: List[str],
    g2p: pd.DataFrame,
    rel: pd.DataFrame,
) -> RawGraph:
    """Build the raw (unlayered) Reactome graph used for all later steps.

    This follows the logic in the original notebook:
      - nodes = genes (from expr index) + closure of pathway ancestors
      - edges = gene->pathway membership + pathway(child)->pathway(parent)

    Expected columns:
      g2p: [ensembl_or_id, pathway_id]
      rel: [child_pathway, parent_pathway]
    """

    gene_nodes = sorted([str(g) for g in expr_gene_ids])
    genes_in_expr = set(gene_nodes)

    g2p_f = g2p[g2p["ensembl_or_id"].astype(str).isin(genes_in_expr)].copy()
    g2p_f = g2p_f.drop_duplicates()

    parents_map = rel.groupby("child_pathway")["parent_pathway"].apply(list).to_dict()
    direct_pathways = set(g2p_f["pathway_id"].astype(str))

    pathways_all = set(direct_pathways)
    stack = list(direct_pathways)
    while stack:
        child = stack.pop()
        for parent in parents_map.get(child, []):
            if parent not in pathways_all:
                pathways_all.add(parent)
                stack.append(parent)

    pathway_nodes = sorted(pathways_all)

    all_nodes = gene_nodes + pathway_nodes
    node2id = {n: i for i, n in enumerate(all_nodes)}

    num_genes = len(gene_nodes)
    num_nodes = len(all_nodes)

    # Edge tables
    g2p_edges = (
        g2p_f[g2p_f["pathway_id"].isin(pathways_all)][["ensembl_or_id", "pathway_id"]]
        .drop_duplicates()
        .sort_values(["ensembl_or_id", "pathway_id"])
    )
    rel_f = (
        rel[rel["child_pathway"].isin(pathways_all) & rel["parent_pathway"].isin(pathways_all)]
        .drop_duplicates()
        .sort_values(["child_pathway", "parent_pathway"])
    )

    src_gene = g2p_edges["ensembl_or_id"].map(node2id).to_numpy(dtype=np.int64)
    dst_gene = g2p_edges["pathway_id"].map(node2id).to_numpy(dtype=np.int64)

    src_pw = rel_f["child_pathway"].map(node2id).to_numpy(dtype=np.int64)
    dst_pw = rel_f["parent_pathway"].map(node2id).to_numpy(dtype=np.int64)

    src = np.concatenate([src_gene, src_pw])
    dst = np.concatenate([dst_gene, dst_pw])

    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    is_gene = torch.zeros((num_nodes,), dtype=torch.bool)
    is_gene[:num_genes] = True

    node_table = pd.DataFrame(
        {
            "node_id": np.arange(num_nodes, dtype=int),
            "node_name": all_nodes,
            "node_type": ["gene"] * num_genes + ["pathway"] * (num_nodes - num_genes),
        }
    )

    edge_types = (["gene_to_pathway"] * len(src_gene)) + (["pathway_to_parent"] * len(src_pw))
    edge_table = pd.DataFrame(
        {
            "src": src,
            "dst": dst,
            "src_name": [all_nodes[i] for i in src],
            "dst_name": [all_nodes[i] for i in dst],
            "edge_type": edge_types,
        }
    )

    return RawGraph(
        node_table=node_table,
        edge_table=edge_table,
        edge_index=edge_index,
        is_gene=is_gene,
        node2id=node2id,
        num_genes=num_genes,
    )


def save_raw_graph(raw: RawGraph, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw.node_table.to_csv(out_dir / "node_table.csv", index=False)
    raw.edge_table.to_csv(out_dir / "edge_table.csv", index=False)
    torch.save(raw.edge_index, out_dir / "edge_index.pt")
    torch.save(raw.is_gene, out_dir / "is_gene.pt")

    import json

    with open(out_dir / "node2id.json", "w") as f:
        json.dump(raw.node2id, f)
