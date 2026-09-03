# Thesis Scripts

Curated scripts copied from the local thesis work archive for GitHub access alongside the calculation data.

The source locations were checked on 2026-09-03:

- `/data/Seafile`: primary source archive.
- `~` (`/data/scratch/bic1`): contained only duplicates of the ESP script under `Downloads/esp` plus unrelated/non-DFT MD helper scripts.
- `~/bin`: not present in this environment.

Mirrored duplicates under `/data/Seafile/SeafileData` and installer files such as `Miniconda3-latest-Linux-x86_64.sh` were intentionally excluded.

## Layout

- `workflow/extract_frames.py`: extracts selected ORCA DOCKER structures into `frameXX` folders.
- `workflow/orca_queue.py`: simple local ORCA job queue runner.
- `workflow/orca_jobs.queue`: queue file template.
- `esp/generate_esp_cubes.sh`: batch generation of electron-density and ESP cube files using `orca_plot`.
- `esp/esp_probe.py`: samples ESP above silicate slab sites from cubes or `orca_vpot` point evaluations.
- `frequency/oh_diff.py`: plots O-H-region differences from ORCA `.out.ir.stk` spectra.
- `visualization/render_energy_frames_amp_frame15.py`: renders AMP frame 15 optimization energy frames.
- `visualization/render_energy_frames_ump_frame14_dry.py`: renders UMP frame 14 dry optimization energy frames.

## Source Map

| Repository path | Source path |
| --- | --- |
| `workflow/extract_frames.py` | `/data/Seafile/extract_frames.py` |
| `workflow/orca_queue.py` | `/data/Seafile/orca_queue.py` |
| `workflow/orca_jobs.queue` | `/data/Seafile/orca_jobs.queue` |
| `esp/generate_esp_cubes.sh` | `/data/Seafile/esp/generate_esp_cubes.sh` |
| `esp/esp_probe.py` | `/data/Seafile/esp/Dry/esp_probe.py` |
| `frequency/oh_diff.py` | `/data/Seafile/freq_calculations/oh_diff.py` |
| `visualization/render_energy_frames_amp_frame15.py` | `/data/Seafile/amp/frames/frame15/opt_anim/render_energy_frames.py` |
| `visualization/render_energy_frames_ump_frame14_dry.py` | `/data/Seafile/ump/frames/frame14/dry/opt_anim/render_energy_frames.py` |
