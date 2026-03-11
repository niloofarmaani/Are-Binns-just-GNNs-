import numpy as np
import torch

from binn_gnn.graph.null_models import degree_sequences, directed_double_edge_swap


def test_directed_swap_preserves_degrees():
    # Simple directed graph (no self loops)
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2, 3, 4],
            [1, 2, 2, 3, 4, 0],
        ],
        dtype=torch.long,
    )
    num_nodes = 5

    in0, out0 = degree_sequences(edge_index, num_nodes)

    rewired = directed_double_edge_swap(edge_index, num_nodes, n_swap=10, seed=0)

    in1, out1 = degree_sequences(rewired, num_nodes)

    assert np.array_equal(in0, in1)
    assert np.array_equal(out0, out1)
