# BINNs as GNNs (refactor scaffold)

This folder is a proposed refactor of the original notebook-only repository into a small Python package plus thin notebooks.

Goals:

- Make the *exact* BINN equivalence claim testable (unit tests).
- Provide a single experimental harness where architectural choices are explicit toggles.
- Add a second, *native DAG* implementation that removes dummy-node padding (the main engineering artifact discussed in the paper).

The original notebooks are copied into `notebooks/` unchanged so you can diff against them.

## Quick layout

- `src/binn_gnn/graph/` graph building, layerization (padded) and depth scheduling (unpadded DAG)
- `src/binn_gnn/models/` layered BINN-as-MPNN baselines and a native DAG propagation baseline
- `src/binn_gnn/experiments/` training loops and cross-validation helpers
- `tests/` correctness tests (equivalence + invariants)
- `scripts/` command-line entrypoints (optional)

