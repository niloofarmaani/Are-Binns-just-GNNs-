from __future__ import annotations

import torch
import torch.nn as nn

from binn_gnn.graph.schedule import DepthSchedule


class DAGBINNExactD1(nn.Module):
    """Native DAG propagation baseline (no padding/copy nodes).

    This model does a single feedforward pass over the raw DAG, updating nodes
    grouped by depth. Each node is activated exactly once.

    It uses a unique scalar weight per edge (BINN-like), but differs from the
    padded layered BINN in that ragged path lengths are *not* equalized by
    adding extra nonlinearities.
    """

    def __init__(
        self,
        *,
        schedule: DepthSchedule,
        gene_map: torch.LongTensor,
        dropout: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__()

        self.schedule = schedule
        self.gene_map = gene_map.clone().long()
        self.dropout = nn.Dropout(dropout)

        self.edge_weight = nn.Parameter(torch.empty(schedule.E))
        nn.init.normal_(self.edge_weight, mean=0.0, std=0.01)

        self.node_bias = nn.Parameter(torch.zeros(schedule.N))

        self.head = nn.Linear(int(schedule.root_ids.numel()), int(num_classes))

    def forward(self, X_gene_batch: torch.Tensor) -> torch.Tensor:
        B = X_gene_batch.size(0)
        h = torch.zeros((B, self.schedule.N), device=X_gene_batch.device, dtype=X_gene_batch.dtype)
        h[:, self.gene_map.to(h.device)] = X_gene_batch

        for step in self.schedule.steps:
            if step.eid.numel() == 0:
                continue

            src = step.src.to(h.device)
            dst_unique = step.dst_unique.to(h.device)
            dst_pos = step.dst_pos.to(h.device)
            eid = step.eid.to(h.device)

            w = self.edge_weight.index_select(0, eid)
            msg = h.index_select(1, src) * w.view(1, -1)

            agg = torch.zeros((B, dst_unique.numel()), device=h.device, dtype=h.dtype)
            agg.index_add_(1, dst_pos, msg)

            h_dst = torch.tanh(agg + self.node_bias.index_select(0, dst_unique).view(1, -1))
            h_dst = self.dropout(h_dst)
            h[:, dst_unique] = h_dst

        roots = h.index_select(1, self.schedule.root_ids.to(h.device))
        return self.head(roots)
