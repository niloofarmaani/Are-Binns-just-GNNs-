# Contributing

## Notebook conventions

- Keep notebooks linear and runnable top to bottom.
- Avoid hard-coded absolute paths where possible. Prefer a single `BASE_DIR` constant.
- Do not commit `outputs/` or large data artifacts.

## Suggested workflow

1. Create a branch for each change.
2. Run the relevant notebook(s) end to end.
3. Commit with a short message and a longer description if needed.
4. Open a pull request and request review.

## Reproducibility

If you change dataset filters, graph construction, or model settings, update:
- README.md "Notes"
- DATASET.md if the source or label definition changes
