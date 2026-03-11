import numpy as np
import torch

from binn_gnn.graph.schedule import build_layered_schedule


def layered_forward_index_add(
    *,
    h0: torch.Tensor,  # [B,N]
    schedule,
    edge_weight: torch.Tensor,  # [E]
    node_bias: torch.Tensor,  # [N]
    learn_padding: bool = False,
):
    h = h0.clone()
    for step in schedule.steps:
        if step.eid.numel() == 0:
            continue
        src = step.src
        dst_unique = step.dst_unique
        dst_pos = step.dst_pos
        eid = step.eid

        w = edge_weight.index_select(0, eid)
        if not learn_padding:
            pad_mask = schedule.is_padding_edge.index_select(0, eid)
            if pad_mask.any():
                w = w.clone()
                w[pad_mask] = 1.0

        msg = h.index_select(1, src) * w.view(1, -1)
        agg = torch.zeros((h.size(0), dst_unique.numel()), dtype=h.dtype)
        agg.index_add_(1, dst_pos, msg)

        h[:, dst_unique] = torch.tanh(agg + node_bias.index_select(0, dst_unique).view(1, -1))
    return h


def masked_ffnn_forward_sparsemm(
    *,
    h0: torch.Tensor,  # [B,N]
    schedule,
    edge_weight: torch.Tensor,  # [E]
    node_bias: torch.Tensor,  # [N]
    learn_padding: bool = False,
):
    # Build a sparse matrix per layer and do sparse mm.
    h = h0.clone().t().contiguous()  # [N,B]
    N = h.size(0)

    for step in schedule.steps:
        if step.eid.numel() == 0:
            continue

        src = step.src
        dst = step.dst
        dst_unique = step.dst_unique
        eid = step.eid

        w = edge_weight.index_select(0, eid)
        if not learn_padding:
            pad_mask = schedule.is_padding_edge.index_select(0, eid)
            if pad_mask.any():
                w = w.clone()
                w[pad_mask] = 1.0

        # W[dst, src] = w
        idx = torch.stack([dst, src], dim=0)
        W = torch.sparse_coo_tensor(idx, w, size=(N, N), dtype=h.dtype)
        agg = torch.sparse.mm(W, h)  # [N,B]

        # update only dst nodes
        upd = torch.tanh(agg.index_select(0, dst_unique) + node_bias.index_select(0, dst_unique).view(-1, 1))
        h.index_copy_(0, dst_unique, upd)

    return h.t().contiguous()  # [B,N]


def test_layered_equivalence_toy_graph():
    # Toy layered graph with one padding edge
    # nodes: 0 gene, 1 pathway, 2 pathway_copy, 3 root
    node_layer = np.array([0, 1, 2, 3], dtype=np.int64)

    # edges: 0->1 (gene), 1->2 (padding), 2->3 (hierarchy)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_type = ["gene_to_leaf_pathway", "padding_identity", "pathway_child_to_parent"]

    schedule = build_layered_schedule(
        edge_index=edge_index,
        node_layer=node_layer,
        edge_type=edge_type,
        gene_ids=[0],
        root_ids=[3],
    )

    torch.manual_seed(0)

    B = 5
    N = 4
    E = edge_index.size(1)

    h0 = torch.randn((B, N), dtype=torch.float64)
    edge_weight = torch.randn((E,), dtype=torch.float64)
    node_bias = torch.randn((N,), dtype=torch.float64)

    h_a = layered_forward_index_add(
        h0=h0,
        schedule=schedule,
        edge_weight=edge_weight,
        node_bias=node_bias,
        learn_padding=False,
    )

    h_b = masked_ffnn_forward_sparsemm(
        h0=h0,
        schedule=schedule,
        edge_weight=edge_weight,
        node_bias=node_bias,
        learn_padding=False,
    )

    assert torch.allclose(h_a, h_b, atol=1e-10, rtol=1e-10)
