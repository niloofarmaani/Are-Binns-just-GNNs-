import numpy as np
import torch

from binn_gnn.graph.schedule import build_depth_schedule


def test_depth_schedule_properties():
    # DAG: 0->2, 1->2, 2->3, 1->3 (skip)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 1],
            [2, 2, 3, 3],
        ],
        dtype=torch.long,
    )

    sched = build_depth_schedule(edge_index=edge_index, gene_ids=[0, 1], root_ids=[3])

    # Depth must increase along edges
    src = edge_index[0]
    dst = edge_index[1]
    assert torch.all(sched.depth[dst] > sched.depth[src])

    # Root should be at max depth
    assert int(sched.depth[3].item()) == int(sched.depth.max().item())

    # There should be at least one step
    assert sched.D >= 1
    assert len(sched.steps) == sched.D
