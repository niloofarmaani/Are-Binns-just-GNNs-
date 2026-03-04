# BINN as a GNN on TCGA gene expression (tumor vs normal)

This repository contains a small, reproducible pipeline that reformulates a BINN/VNN-style masked feedforward pathway network as a layered message passing model on a Reactome pathway graph, using TCGA gene expression.

The work is organized as three notebooks that run in order:

- `01_graph_preparation.ipynb`: download/prepare TCGA + Reactome, build a layered (feedforward) graph.
- `02_binn_exact_equivalence.ipynb`: implement a BINN-exact layered MPNN forward pass and verify numerical equivalence with a masked feedforward computation.
- `03_training_and_comparison.ipynb`: train quick baselines and compare strict BINN-like variants to relaxed graph models.

## Main idea

Reactome gives (1) gene to pathway membership and (2) pathway hierarchy (a directed acyclic graph). A classic BINN/VNN uses masked linear layers, one per hierarchy level. We build an equivalent layered graph where every edge goes from layer ℓ to layer ℓ+1 (by duplicating pathway nodes when needed), then run sequential message passing to match the masked feedforward schedule.

## Data and knowledge sources
data is located at: https://drive.google.com/drive/folders/158aqRXnJK84neihJr1e-EWGHF5jYEsfq?usp=drive_link
See `DATASET.md` for details.

Important: large raw datasets are not committed to GitHub. The notebooks download public Reactome files and expect TCGA expression/phenotype files to exist locally (or in Google Drive) depending on your workflow.

## Expected outputs

The notebooks write artifacts under `outputs/` (ignored by git), including:

- cached expression and labels (Reactome-mapped)
- raw and layered graph artifacts (`node_table`, `edge_table`, `edge_index`, layer metadata)
- per-layer edge lists for the BINN-exact forward schedule
- trained model checkpoints and comparison plots (quick runs)

## Recommended run environment

These notebooks were designed for Google Colab with Google Drive mounted.

High level steps:

1. Open the notebook in Colab.
2. Mount Drive (the notebooks include a cell for this).
3. Run notebooks in order from `01_...` to `03_...`.

If you run locally, install requirements and update paths as needed.

## Installation

Create an environment and install:

```bash
pip install -r requirements.txt
```

For PyTorch Geometric, follow the official installation guide for your CUDA / torch version.

## Collaboration workflow

- Put notebooks under version control.
- Keep generated artifacts out of git (`outputs/` is ignored).
- Use small commits with clear messages (for example, "Fix Reactome relation direction" or "Add equivalence visualization").

## Notes

- The current task is pan-cancer TCGA (all cancer types combined): tumor vs normal.
- Class imbalance is substantial (tumor is the majority). Metrics like balanced accuracy and PR curves for the minority class are more informative than accuracy.
