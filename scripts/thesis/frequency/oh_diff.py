#!/usr/bin/env python3
"""
O-H region difference spectrum from two ORCA .out.ir.stk files.

    python oh_difference.py unbonded.stk bonded.stk

Produces a two-panel figure: broadened spectra of both structures over the
O-H stretching region, and their difference (bonded minus unbonded) with the
integration windows shaded and the assigned modes marked.

Defaults are set for the AMP frame 11 / frame 15 comparison; adjust --windows
and --marks for other systems.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_stk(path):
    """ORCA .out.ir.stk: two columns, wavenumber and intensity."""
    w, i = [], []
    for line in open(path):
        p = line.split()
        if len(p) < 2:
            continue
        try:
            w.append(float(p[0])); i.append(float(p[1]))
        except ValueError:
            continue
    return np.asarray(w), np.asarray(i)


def broaden(w, inten, grid, fwhm, shape="lorentz"):
    out = np.zeros_like(grid)
    if shape == "lorentz":
        g = fwhm / 2.0
        for wi, ii in zip(w, inten):
            out += ii * (g**2) / ((grid - wi)**2 + g**2)
    else:
        s = fwhm / (2*np.sqrt(2*np.log(2)))
        for wi, ii in zip(w, inten):
            out += ii * np.exp(-0.5*((grid-wi)/s)**2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unbonded"); ap.add_argument("bonded")
    ap.add_argument("--lo", type=float, default=3000)
    ap.add_argument("--hi", type=float, default=4000)
    ap.add_argument("--fwhm", type=float, default=20,
                    help="O-H bands are broad experimentally; 20-40 cm^-1 is "
                         "more representative than the 4 cm^-1 used for the "
                         "phosphate region")
    ap.add_argument("--windows", default="3000:3400,3400:3650,3650:3950",
                    help="integration windows, colon-separated pairs")
    ap.add_argument("--marks", default="3282.2,3648.0,3881.0",
                    help="assigned mode positions to annotate")
    ap.add_argument("--labels", default="unbonded (Fig. 1c),bonded (Fig. 1b)")
    ap.add_argument("--out", default="oh_difference.png")
    args = ap.parse_args()

    wu, iu = read_stk(args.unbonded)
    wb, ib = read_stk(args.bonded)
    grid = np.linspace(args.lo, args.hi, 4000)
    su = broaden(wu, iu, grid, args.fwhm)
    sb = broaden(wb, ib, grid, args.fwhm)
    diff = sb - su
    lab_u, lab_b = args.labels.split(",")

    wins = []
    for w in args.windows.split(","):
        a, b = w.split(":"); wins.append((float(a), float(b)))
    marks = [float(v) for v in args.marks.split(",")]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})

    ax1.plot(grid, su, color="#2471a3", lw=1.2, label=lab_u)
    ax1.plot(grid, sb, color="#c0392b", lw=1.2, label=lab_b)
    ax1.set_ylabel("Intensity (km mol$^{-1}$)")
    ax1.legend(frameon=False, fontsize=9)
    ax1.set_title(f"O-H stretching region (FWHM {args.fwhm:.0f} cm$^{{-1}}$)")

    ax2.axhline(0, color="k", lw=0.6)
    ax2.plot(grid, diff, color="#7d3c98", lw=1.2)
    ax2.fill_between(grid, 0, diff, where=diff > 0, color="#7d3c98", alpha=0.18)
    ax2.fill_between(grid, 0, diff, where=diff < 0, color="#7d3c98", alpha=0.18)
    ax2.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax2.set_ylabel("Difference (bonded − unbonded)")

    print(f"{'window':>16} {'integral':>12}")
    for a, b in wins:
        m = (grid >= a) & (grid <= b)
        integ = np.trapezoid(diff[m], grid[m]) if hasattr(np, "trapezoid") \
                else np.trapz(diff[m], grid[m])
        print(f"{a:6.0f}-{b:<6.0f} {integ:12.1f}")
        for ax in (ax1, ax2):
            ax.axvspan(a, b, color="grey", alpha=0.07)
        ax2.text((a+b)/2, ax2.get_ylim()[1]*0.85, f"{integ:+.0f}",
                 ha="center", fontsize=8, color="#555")

    for m in marks:
        for ax in (ax1, ax2):
            ax.axvline(m, color="k", ls=":", lw=0.7)
        ax1.text(m, ax1.get_ylim()[1]*0.95, f"{m:.0f}", rotation=90,
                 va="top", ha="right", fontsize=7)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
