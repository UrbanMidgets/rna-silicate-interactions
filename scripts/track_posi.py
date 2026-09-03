#!/usr/bin/env python3
"""
Track phosphate-oxygen to surface-silicon distances through an ORCA MD trajectory.

Usage:
    python track_posi.py trajectory.xyz
    python track_posi.py trajectory.xyz --timestep 1.0 --cutoff 2.0 --out posi.png

Identifies the phosphorus atom, the oxygens bonded to it in frame 0, and all
silicon atoms. For every frame it reports the closest Si to each phosphate
oxygen, flags the frame at which any of them first falls below the covalent
cutoff, and reports whether the contact persists thereafter.
"""

import argparse
import sys
import numpy as np

# Covalent-radius sums used only for identifying P-O bonds in frame 0
PO_BOND_MAX = 1.80  # Angstrom


def read_xyz(path):
    """Read a multi-frame XYZ file. Returns (symbols, coords) with coords
    of shape (nframes, natoms, 3)."""
    frames = []
    symbols = None
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                natoms = int(line.split()[0])
            except (ValueError, IndexError):
                raise ValueError(f"Expected atom count, got: {line!r}")
            fh.readline()  # comment line
            syms, xyz = [], []
            for _ in range(natoms):
                parts = fh.readline().split()
                syms.append(parts[0])
                xyz.append([float(v) for v in parts[1:4]])
            if symbols is None:
                symbols = syms
            elif syms != symbols:
                raise ValueError("Atom ordering changes between frames")
            frames.append(xyz)
    if not frames:
        raise ValueError("No frames found")
    return symbols, np.asarray(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory")
    ap.add_argument("--timestep", type=float, default=1.0,
                    help="fs per frame (default 1.0)")
    ap.add_argument("--cutoff", type=float, default=2.0,
                    help="O-Si distance defining a linkage, Angstrom (default 2.0)")
    ap.add_argument("--out", default="posi_distances.png")
    ap.add_argument("--csv", default=None, help="optional CSV output")
    args = ap.parse_args()

    symbols, coords = read_xyz(args.trajectory)
    nframes, natoms = coords.shape[0], coords.shape[1]
    print(f"{nframes} frames, {natoms} atoms")

    p_idx = [i for i, s in enumerate(symbols) if s.upper() == "P"]
    si_idx = [i for i, s in enumerate(symbols) if s.upper() == "SI"]
    o_idx = [i for i, s in enumerate(symbols) if s.upper() == "O"]
    if len(p_idx) != 1:
        sys.exit(f"Expected exactly one P, found {len(p_idx)}")
    P = p_idx[0]
    print(f"P at index {P}; {len(si_idx)} Si; {len(o_idx)} O")

    # Phosphate oxygens: bonded to P in the first frame
    d0 = np.linalg.norm(coords[0][o_idx] - coords[0][P], axis=1)
    phos_o = [o_idx[k] for k in range(len(o_idx)) if d0[k] < PO_BOND_MAX]
    print(f"Phosphate oxygens (P-O < {PO_BOND_MAX} A in frame 0): {phos_o}")
    labels = {}
    for oi in phos_o:
        # classify by what else the oxygen is bonded to in frame 0
        others = []
        for j, s in enumerate(symbols):
            if j == oi or j == P:
                continue
            d = np.linalg.norm(coords[0][oi] - coords[0][j])
            if s.upper() == "H" and d < 1.20:
                others.append("H")
            elif s.upper() == "C" and d < 1.60:
                others.append("C")
        if "C" in others:
            kind = "ester (P-O-C)"
        elif "H" in others:
            kind = "hydroxyl (P-OH)"
        else:
            kind = "free"
        labels[oi] = kind
        print(f"   O{oi}: d(P-O) = "
              f"{np.linalg.norm(coords[0][oi]-coords[0][P]):.3f} A  [{kind}]")

    si_coords = coords[:, si_idx, :]                      # (nf, nSi, 3)
    times = np.arange(nframes) * args.timestep

    # min distance from each phosphate O to any Si, per frame
    tracks = {}
    partners = {}
    for oi in phos_o:
        oc = coords[:, oi, :][:, None, :]                 # (nf, 1, 3)
        d = np.linalg.norm(si_coords - oc, axis=2)        # (nf, nSi)
        tracks[oi] = d.min(axis=1)
        partners[oi] = [si_idx[j] for j in d.argmin(axis=1)]

    # formation detection
    print(f"\nLinkage threshold: {args.cutoff} A")
    for oi in phos_o:
        t = tracks[oi]
        below = np.where(t < args.cutoff)[0]
        if below.size == 0:
            print(f"  O{oi} [{labels[oi]}]: never below cutoff (min {t.min():.3f} A "
                  f"at frame {t.argmin()})")
            continue
        first = below[0]
        after = t[first:]
        frac = (after < args.cutoff).mean()
        broke = np.where(after >= args.cutoff)[0]
        msg = (f"  O{oi} [{labels[oi]}]: first below at frame {first} "
               f"({times[first]:.0f} fs), partner Si{partners[oi][first]}, "
               f"d = {t[first]:.3f} A")
        print(msg)
        print(f"        stays below for {100*frac:.1f}% of remaining frames; "
              f"final d = {t[-1]:.3f} A")
        if broke.size:
            print(f"        first excursion above cutoff at frame "
                  f"{first + broke[0]} -- linkage is not continuously formed")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not available; distances printed above")

    fig, ax = plt.subplots(figsize=(9, 5))
    for oi in phos_o:
        ax.plot(times, tracks[oi], lw=1.0, label=f"O{oi} ({labels[oi]})")
    ax.axhline(args.cutoff, color="k", ls="--", lw=0.8,
               label=f"{args.cutoff} A cutoff")
    ax.set_xlabel("Time (fs)")
    ax.set_ylabel("Closest O$\\cdots$Si distance (\u00c5)")
    ax.set_title("Phosphate oxygen to surface silicon")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"\nWrote {args.out}")

    if args.csv:
        cols = ["time_fs"] + [f"O{oi}" for oi in phos_o]
        data = np.column_stack([times] + [tracks[oi] for oi in phos_o])
        np.savetxt(args.csv, data, delimiter=",", header=",".join(cols),
                   comments="", fmt="%.4f")
        print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
