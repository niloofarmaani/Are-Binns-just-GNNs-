# Dataset and knowledge resources

This project uses public gene expression and pathway resources.

## TCGA gene expression

Input matrix:
- TCGA gene expression (Toil pipeline), RSEM gene TPM
- Ensembl gene identifiers (ENSG...)

Common file name in the workflow:
- `tcga_RSEM_gene_tpm.gz`

Phenotype/labels:
- TCGA phenotype file with sample annotations used to derive labels for the binary task:
  - Primary Tumor (label 1)
  - Solid Tissue Normal (label 0)

Common file name in the workflow:
- `TcgaTargetGTEX_phenotype.txt.gz`

Task:
- Pan-cancer tumor vs normal classification (no restriction to a single TCGA project).

## Reactome pathway knowledge

Downloaded from the Reactome "current" release:

- `Ensembl2Reactome.txt`
  - Ensembl gene to Reactome pathway mapping.
  - Contains multiple species; the notebooks filter to Homo sapiens and Reactome pathways with IDs starting `R-HSA-`.

- `ReactomePathwaysRelation.txt`
  - Pathway hierarchy relations.
  - Stored as parent pathway then child pathway.
  - The notebook converts this into directed child to parent edges for upward propagation.

## What is saved locally

The notebooks cache intermediate artifacts under `outputs/` (not committed), including:

- Expression matrix aligned to Reactome-mapped genes (about 11k genes).
- Labels aligned to expression samples.
- A raw gene + pathway graph and a layered (padded) feedforward graph.
- Model checkpoints and evaluation plots for quick comparisons.

## Data usage note

Do not commit large raw matrices or derived tensors to git. Keep them in Drive or a local data directory, and regenerate caches by running the notebooks.
