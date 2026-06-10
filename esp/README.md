# ESP Source Calculation Files

This folder contains local copies of optimized ORCA calculation files intended for electrostatic potential map generation.

Included targets:

- `amp_protonated/` from `/data/Seafile/amp/protonated/geom/`
- `cmp_protonated/` from `/data/Seafile/cmp/protonated/geom/`
- `ump/` from `/data/Seafile/ump/geom/`
- `aluminium_surface/` from `/data/Seafile/al_surf/`

Each folder includes the final `.xyz`, `.inp`, `.out`, `.gbw`, and supporting ORCA metadata needed to inspect the calculation and generate ESP maps with ORCA tooling.

Large `.densities` files were not copied; they are not needed for `orca_plot` ESP cube generation from `.gbw` files.
