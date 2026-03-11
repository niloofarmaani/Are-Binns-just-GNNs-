"""BINNs as (special cases of) GNNs.

This package is intentionally small:
- `binn_gnn.graph` builds/loads graphs and schedules
- `binn_gnn.models` contains reference models used for ablations
- `binn_gnn.experiments` runs training and evaluation

The design goal is scientific comparability: the only difference between
models should be an explicit config flag (conv type, readout, topology mode, etc.).
"""

from .config import ExperimentConfig, ModelConfig, TrainingConfig
