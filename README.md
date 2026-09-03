# RNA Silicate Interactions

Supporting data for a master's thesis on RNA nucleotide interactions with silicate surfaces.

The repository contains a curated, GitHub-friendly subset of the raw calculation archive. It is organised for thesis examiners and supervisors to navigate the workflow from docking, to selected candidate frames, to solvated and dry geometry optimisations.

## Contents

- `data/`: curated thesis data and manifest.
- `data/primary_calculations/`: neutral nucleotide calculations on canonical and aluminium-substituted silicate surfaces.
- `data/supporting_calculations/`: protonated systems, dimer calculations, magnesium docking, and exploratory checks.
- `data/reference_structures/`: standalone reference surface calculations.
- `scripts/migrate_seafile_data.py`: reproducible migration script used to build `data/` from the local raw archive.
- `scripts/thesis/`: curated thesis workflow, DFT/ORCA, ESP, frequency-analysis, and visualization scripts copied from the local work archive.

## Data Layout

Primary calculations are grouped by surface and nucleotide:

```text
data/primary_calculations/
  canonical_surface/
    amp/
    cmp/
    gmp/
    ump/
  aluminium_surface/
    amp/
    cmp/
    gmp/
    ump/
```

Each nucleotide/surface folder keeps the docking provenance beside the selected frames:

```text
docking/
selected_frames/
candidate_frames/
checks/
```

Within `selected_frames/`, each frame is split into `initial/`, `solvated/`, and `dry/` where available.

## Manifest

`data/MANIFEST.tsv` records every copied file, including source path, repository path, system, surface, frame, calculation state, role, ORCA termination status where applicable, and size.

## Raw Data Provenance

The migration source was `/data/Seafile`. That source directory was treated as read-only and is not modified by the migration script.

Large scratch/restart/binary files are intentionally excluded from this repository, including `.gbw`, `.densities`, `.tmp`, `.log`, and related ORCA scratch files. Docking swarm files (`*.all.swarm.xyz`) are also excluded by default to keep the GitHub dataset below roughly 1 GB.

To regenerate the curated dataset from the local raw archive:

```bash
python3 scripts/migrate_seafile_data.py
```

To include the full docking swarm XYZ files as well:

```bash
python3 scripts/migrate_seafile_data.py --include-swarms
```
