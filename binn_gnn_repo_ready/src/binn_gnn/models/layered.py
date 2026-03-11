from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from binn_gnn.graph.schedule import LayeredSchedule


class LayeredMPNNBase(nn.Module):
    def __init__(
        self,
        *,
        schedule: LayeredSchedule,
        gene_map: torch.LongTensor,
        dropout: float = 0.0,
        num_classes: int = 2,
        readout: str = "roots_concat",
        layer_of_node: Optional[torch.LongTensor] = None,
    ):
        super().__init__()

        self.schedule = schedule
        self.gene_map = gene_map.clone().long()
        self.dropout = nn.Dropout(dropout)
        self.num_classes = int(num_classes)
        self.readout = readout

        self.layer_of_node = layer_of_node  # optional, for deep supervision

    def _init_state_scalar(self, X_gene_batch: torch.Tensor) -> torch.Tensor:
        # h: [B, N]
        B = X_gene_batch.size(0)
        h = torch.zeros((B, self.schedule.N), device=X_gene_batch.device, dtype=X_gene_batch.dtype)
        h[:, self.gene_map.to(X_gene_batch.device)] = X_gene_batch
        return h

    def _init_state_vector(self, X_gene_batch: torch.Tensor, d: int, in_proj: nn.Module) -> torch.Tensor:
        # X_gene_batch: [B, G]
        B = X_gene_batch.size(0)
        h = torch.zeros((B, self.schedule.N, d), device=X_gene_batch.device, dtype=X_gene_batch.dtype)
        gene_vals = X_gene_batch.unsqueeze(-1)  # [B,G,1]
        gene_emb = in_proj(gene_vals)  # [B,G,d]
        h[:, self.gene_map.to(X_gene_batch.device), :] = gene_emb
        return h

    def _readout_roots_concat(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, N] or [B, N, d]
        roots = h.index_select(1, self.schedule.root_ids.to(h.device))
        return roots.reshape(roots.size(0), -1)


class BINNExactD1(LayeredMPNNBase):
    """Layered BINN-as-message-passing with unique scalar weight per edge.

    This matches the masked sparse FFNN computation when the graph is the padded
    layered graph and padding edges are fixed to 1.
    """

    def __init__(
        self,
        *,
        schedule: LayeredSchedule,
        gene_map: torch.LongTensor,
        dropout: float = 0.0,
        learn_padding: bool = False,
        num_classes: int = 2,
    ):
        super().__init__(
            schedule=schedule,
            gene_map=gene_map,
            dropout=dropout,
            num_classes=num_classes,
            readout="roots_concat",
        )

        self.learn_padding = bool(learn_padding)

        self.edge_weight = nn.Parameter(torch.empty(schedule.E))
        nn.init.normal_(self.edge_weight, mean=0.0, std=0.01)

        self.node_bias = nn.Parameter(torch.zeros(schedule.N))

        self.head = nn.Linear(int(schedule.root_ids.numel()), int(num_classes))

    def forward(self, X_gene_batch: torch.Tensor) -> torch.Tensor:
        h = self._init_state_scalar(X_gene_batch)

        for step in self.schedule.steps:
            if step.eid.numel() == 0:
                continue

            src = step.src.to(h.device)
            dst_unique = step.dst_unique.to(h.device)
            dst_pos = step.dst_pos.to(h.device)
            eid = step.eid.to(h.device)

            w = self.edge_weight.index_select(0, eid)  # [E_step]

            if not self.learn_padding:
                pad_mask = self.schedule.is_padding_edge.index_select(0, eid).to(h.device)
                if pad_mask.any():
                    w = w.clone()
                    w[pad_mask] = 1.0

            msg = h.index_select(1, src) * w.view(1, -1)

            agg = torch.zeros((h.size(0), dst_unique.numel()), device=h.device, dtype=h.dtype)
            agg.index_add_(1, dst_pos, msg)

            h_dst = torch.tanh(agg + self.node_bias.index_select(0, dst_unique).view(1, -1))
            h_dst = self.dropout(h_dst)
            h[:, dst_unique] = h_dst

        feat = self._readout_roots_concat(h)
        return self.head(feat)


class GCNSharedD1(LayeredMPNNBase):
    """Layered message passing with a single scalar weight per layer (GCN-like sharing)."""

    def __init__(
        self,
        *,
        schedule: LayeredSchedule,
        gene_map: torch.LongTensor,
        dropout: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__(schedule=schedule, gene_map=gene_map, dropout=dropout, num_classes=num_classes)

        self.layer_weight = nn.Parameter(torch.ones(schedule.L) * 0.01)
        self.node_bias = nn.Parameter(torch.zeros(schedule.N))
        self.head = nn.Linear(int(schedule.root_ids.numel()), int(num_classes))

    def forward(self, X_gene_batch: torch.Tensor) -> torch.Tensor:
        h = self._init_state_scalar(X_gene_batch)

        for li, step in enumerate(self.schedule.steps):
            if step.eid.numel() == 0:
                continue

            src = step.src.to(h.device)
            dst_unique = step.dst_unique.to(h.device)
            dst_pos = step.dst_pos.to(h.device)

            w = self.layer_weight[li]
            msg = h.index_select(1, src) * w

            agg = torch.zeros((h.size(0), dst_unique.numel()), device=h.device, dtype=h.dtype)
            agg.index_add_(1, dst_pos, msg)

            h_dst = torch.tanh(agg + self.node_bias.index_select(0, dst_unique).view(1, -1))
            h_dst = self.dropout(h_dst)
            h[:, dst_unique] = h_dst

        feat = self._readout_roots_concat(h)
        return self.head(feat)


class GATGateD1(LayeredMPNNBase):
    """Layered message passing with a simple attention-like gate per edge."""

    def __init__(
        self,
        *,
        schedule: LayeredSchedule,
        gene_map: torch.LongTensor,
        dropout: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__(schedule=schedule, gene_map=gene_map, dropout=dropout, num_classes=num_classes)

        self.layer_weight = nn.Parameter(torch.ones(schedule.L) * 0.01)
        self.a_src = nn.Parameter(torch.zeros(schedule.L))
        self.a_dst = nn.Parameter(torch.zeros(schedule.L))
        self.node_bias = nn.Parameter(torch.zeros(schedule.N))

        self.head = nn.Linear(int(schedule.root_ids.numel()), int(num_classes))

    def forward(self, X_gene_batch: torch.Tensor) -> torch.Tensor:
        h = self._init_state_scalar(X_gene_batch)

        for li, step in enumerate(self.schedule.steps):
            if step.eid.numel() == 0:
                continue

            src = step.src.to(h.device)
            dst = step.dst.to(h.device)
            dst_unique = step.dst_unique.to(h.device)
            dst_pos = step.dst_pos.to(h.device)

            w = self.layer_weight[li]
            h_src = h.index_select(1, src)
            h_dst_prev = h.index_select(1, dst)

            gate = torch.sigmoid(self.a_src[li] * h_src + self.a_dst[li] * h_dst_prev)
            msg = h_src * (w * gate)

            agg = torch.zeros((h.size(0), dst_unique.numel()), device=h.device, dtype=h.dtype)
            agg.index_add_(1, dst_pos, msg)

            h_dst = torch.tanh(agg + self.node_bias.index_select(0, dst_unique).view(1, -1))
            h_dst = self.dropout(h_dst)
            h[:, dst_unique] = h_dst

        feat = self._readout_roots_concat(h)
        return self.head(feat)


class VectorSharedD(LayeredMPNNBase):
    """Layered propagation with d-dimensional node embeddings and per-layer shared linear transforms."""

    def __init__(
        self,
        *,
        schedule: LayeredSchedule,
        gene_map: torch.LongTensor,
        d: int = 16,
        dropout: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__(schedule=schedule, gene_map=gene_map, dropout=dropout, num_classes=num_classes)

        self.d = int(d)
        self.in_proj = nn.Linear(1, self.d)

        self.W = nn.Parameter(torch.empty(schedule.L, self.d, self.d))
        nn.init.xavier_uniform_(self.W)

        self.bias = nn.Parameter(torch.zeros(schedule.N, self.d))
        self.head = nn.Linear(int(schedule.root_ids.numel()) * self.d, int(num_classes))

    def forward(self, X_gene_batch: torch.Tensor) -> torch.Tensor:
        h = self._init_state_vector(X_gene_batch, self.d, self.in_proj)

        for li, step in enumerate(self.schedule.steps):
            if step.eid.numel() == 0:
                continue

            src = step.src.to(h.device)
            dst_unique = step.dst_unique.to(h.device)
            dst_pos = step.dst_pos.to(h.device)

            h_src = h.index_select(1, src)  # [B,E,d]
            msg = torch.matmul(h_src, self.W[li])

            agg = torch.zeros((h.size(0), dst_unique.numel(), self.d), device=h.device, dtype=h.dtype)
            agg.index_add_(1, dst_pos, msg)

            h_dst = torch.tanh(agg + self.bias.index_select(0, dst_unique).unsqueeze(0))
            h_dst = self.dropout(h_dst)
            h[:, dst_unique, :] = h_dst

        feat = self._readout_roots_concat(h)
        return self.head(feat)


class DenseMLPBaseline(nn.Module):
    def __init__(self, in_dim: int, dropout: float = 0.2, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, X_gene_batch: torch.Tensor) -> torch.Tensor:
        return self.net(X_gene_batch)


