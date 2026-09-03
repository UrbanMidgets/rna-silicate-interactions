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
- `analysis/mayer_valence_check.py`: scans ORCA outputs for Mayer-valence evidence of P-O-Si linkages.
- `analysis/track_posi.py`: tracks phosphate oxygen to surface silicon distances through XYZ trajectories.
- `frequency/oh_diff.py`: plots O-H-region differences from ORCA `.out.ir.stk` spectra.
- `frequency/plot_freq_zoom.py`: re-broadens ORCA stick spectra over a narrow frequency window.
- `frequency/plot_freqs.py`: plots intensity versus wavenumber from ORCA/orca_mapspc spectra.
- `visualization/make_ch2_figures.py`: generates Chapter 2 schematic figures.
- `visualization/make_figure_2_3.py`: composes the Chapter 2 continuum-solvation figure.
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
| `analysis/mayer_valence_check.py` | `/mnt/c/Users/UrbanMidgets/Downloads/mayer_valence_check.py` |
| `analysis/track_posi.py` | `/mnt/c/Users/UrbanMidgets/Downloads/track_posi.py` |
| `frequency/oh_diff.py` | `/data/Seafile/freq_calculations/oh_diff.py` |
| `frequency/plot_freq_zoom.py` | `/mnt/c/Users/UrbanMidgets/Downloads/plot_freq_zoom.py` |
| `frequency/plot_freqs.py` | `/mnt/c/Users/UrbanMidgets/Downloads/plot_freqs.py` |
| `visualization/make_ch2_figures.py` | `/mnt/c/Users/UrbanMidgets/Downloads/files_script/make_ch2_figures.py` |
| `visualization/make_figure_2_3.py` | `/mnt/c/Users/UrbanMidgets/Downloads/files_2.3/make_figure_2_3.py` |
| `visualization/render_energy_frames_amp_frame15.py` | `/data/Seafile/amp/frames/frame15/opt_anim/render_energy_frames.py` |
| `visualization/render_energy_frames_ump_frame14_dry.py` | `/data/Seafile/ump/frames/frame14/dry/opt_anim/render_energy_frames.py` |
