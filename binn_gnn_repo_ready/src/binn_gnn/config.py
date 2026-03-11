from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


TopologyMode = Literal["layered_padded", "dag_depth"]
ConvType = Literal["binn_edge", "gcn_shared", "gat_gate", "vector_shared"]
ReadoutType = Literal["roots_concat", "pseudo_node", "deep_supervision"]


@dataclass(frozen=True)
class ModelConfig:
    topology: TopologyMode = "layered_padded"
    conv: ConvType = "binn_edge"
    readout: ReadoutType = "roots_concat"

    # Representation size per node (d=1 is classic BINN/P-Net style)
    d: int = 1

    # Padding behavior (only relevant for layered_padded)
    learn_padding: bool = False

    # Deep supervision (only relevant when readout="deep_supervision")
    deep_supervision_weight: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    dropout: float = 0.2

    # Evaluation
    early_stopping_patience: int = 10

    # Repeated CV
    n_splits: int = 5
    n_repeats: int = 3


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = ModelConfig()
    train: TrainingConfig = TrainingConfig()

    # Paths (left optional so notebooks/scripts can set them)
    base_dir: Optional[str] = None
