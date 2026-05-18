# Curated Thesis Data

This directory contains a curated copy of the RNA nucleotide/silicate calculations. The raw source archive is `/data/Seafile`, which was left untouched during migration.

## How To Navigate

Start with `MANIFEST.tsv` for a complete file index. For the main thesis systems, use `primary_calculations/`, then choose the surface type and nucleotide. Each nucleotide/surface folder keeps its docking provenance beside the selected frames so the path from docking to optimisation is visible.

## Layout

- `primary_calculations/canonical_surface/`: neutral AMP, CMP, GMP, and UMP on the canonical silicate surface.
- `primary_calculations/aluminium_surface/`: neutral AMP, CMP, GMP, and UMP on the aluminium-substituted surface.
- `docking/`: docking inputs/outputs retained with the nucleotide/surface they generated.
- `selected_frames/`: frames selected from docking and then optimised.
- `checks/`: non-primary calculations such as unconstrained tests, R2SCAN checks, MD checks, or accidental reruns.
- `supporting_calculations/`: protonated systems, dimer data, and magnesium docking organized by surface type.
- `reference_structures/`: standalone reference surface calculations.
- `scripts/`: scripts copied from the source archive for provenance.

## Frame Folders

A typical selected frame is organised as follows:

```text
frameXX/
  initial/
  solvated/
  dry/
```

`solvated/` contains the CPCM(water) geometry optimisation. `dry/` contains the subsequent dry optimisation started from the solvated result where available.

## Included Files

Included file types are ORCA inputs/outputs, final and trajectory XYZ files, optimisation metadata, bibliography files, density metadata, CPCM correction files, and small CSV/SVG files for check calculations.

Excluded file types include large restart/scratch/binary files such as `.gbw`, `.densities`, `.tmp`, `.log`, and media/rendering files that are not needed for navigating the thesis dataset. Docking swarm files (`*.all.swarm.xyz`) are excluded by default to keep the GitHub dataset smaller. Restore them with `python3 scripts/migrate_seafile_data.py --include-swarms` if the full docking swarms are needed.

## Manifest Columns

`MANIFEST.tsv` contains `system`, `surface`, `frame`, `state`, `role`, `status`, `source_path`, `repo_path`, `size_bytes`, and `notes`. The `status` column is populated for ORCA `.out` files when normal termination could be detected.
