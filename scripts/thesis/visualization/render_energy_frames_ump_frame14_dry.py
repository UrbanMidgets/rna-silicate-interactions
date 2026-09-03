#!/usr/bin/env python3
import os
import re
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

ENERGY_FILE = "ump_frame14_dry_energies.txt"   # your extracted energies file
OUTDIR = "frames_energy"          # output folder for plot frames
DPI = 150

# --- read energies ---
energies = []
pattern = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+(?:[Ee][+-]?\d+)?)")

with open(ENERGY_FILE, "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            energies.append(float(m.group(1)))

if not energies:
    raise SystemExit(f"No energies found in {ENERGY_FILE}. Expected lines like 'FINAL SINGLE POINT ENERGY   -123.45'")

energies = np.array(energies)
steps = np.arange(1, len(energies) + 1)

# --- convert to relative energy (kJ/mol) for readability ---
E_ref = energies[-1]
relE = (energies - E_ref) * 2625.499638  # Eh -> kJ/mol

# --- ensure output dir ---
os.makedirs(OUTDIR, exist_ok=True)

# --- fixed axis limits so the plot doesn't jump ---
pad = 0.05 * (relE.max() - relE.min() + 1e-12)
ymin, ymax = relE.min() - pad, relE.max() + pad

# --- render frames ---
for i in range(len(relE)):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=DPI)

    ax.plot(steps, relE, lw=2)
    ax.plot(steps[i], relE[i], "o")          # moving dot
    ax.axvline(steps[i], lw=1, alpha=0.5)    # optional vertical indicator

    ax.set_xlim(1, len(relE))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("ΔE relative to final (kJ/mol)")
    ax.set_title("Energy convergence")
    ax.grid(True)
    fig.tight_layout()

    fig.savefig(f"{OUTDIR}/energy_{i+1:04d}.png")
    plt.close(fig)

print(f"[OK] Wrote {len(relE)} frames to {OUTDIR}/ (energy_0001.png ...)")
