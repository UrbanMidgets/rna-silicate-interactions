# GitHub Pages Viewer

This folder contains a static proof-of-concept molecular viewer for the curated thesis dataset.

## Files

- `index.html`: app shell.
- `styles.css`: layout and styling.
- `app.js`: filter logic and 3Dmol rendering.
- `data_index.json`: generated index consumed by the app.

## Regenerate Index

Run from repository root:

```bash
python3 scripts/build_pages_index.py
```

The app expects dataset files under `../data/...` relative to `docs/`.
