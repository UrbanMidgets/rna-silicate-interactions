#!/usr/bin/env python3
"""Zoomed-in IR band comparison for ORCA frequency calculations.

The broadened *.out.ir.dat spectra written by orca_mapspc are sampled at a
fixed ~3.6 cm^-1 step across the whole 300-4000 cm^-1 range, which smears
neighboring bands together in a narrow window. This script instead re-
broadens the raw stick spectrum (*.out.ir.stk: wavenumber, intensity in
km/mol) with a Lorentzian lineshape at fine resolution, restricted to a
chosen wavenumber interval, so closely spaced peaks stay resolved.

Defaults to comparing amp_frame11 and amp_frame15 in the 1300-1350 cm^-1
window, but any set of .stk files and any interval can be passed on the
command line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_FILES = [
    Path("/data/Seafile/freq_calculations/amp_frame11_dry_freq/amp_frame11_dry_freq.out.ir.stk"),
    Path("/data/Seafile/freq_calculations/amp_frame15_dry_freq/amp_frame15_dry_freq.out.ir.stk"),
]


def read_sticks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    freqs: list[float] = []
    ints: list[float] = []
    with path.open(errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "!", ";")):
                continue
            parts = stripped.split()
            freqs.append(float(parts[0]))
            ints.append(float(parts[1]))
    return np.array(freqs), np.array(ints)


def label_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".out.ir.stk", ".out.ir.dat"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def lorentzian_broaden(freqs: np.ndarray, ints: np.ndarray, x: np.ndarray, fwhm: float) -> np.ndarray:
    gamma = fwhm / 2.0
    # (n_sticks, n_points) via broadcasting, summed over sticks.
    diff = x[None, :] - freqs[:, None]
    lineshape = (gamma**2) / (diff**2 + gamma**2)
    return (ints[:, None] * lineshape).sum(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare zoomed-in IR bands from ORCA stick spectra (.out.ir.stk)."
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        default=DEFAULT_FILES,
        help="stick spectrum file(s) (.out.ir.stk); default is amp_frame11 and amp_frame15",
    )
    parser.add_argument("--labels", nargs="*", help="legend labels, one per file (default: derived from filename)")
    parser.add_argument("--xmin", type=float, default=1300.0, help="lower bound of the wavenumber window")
    parser.add_argument("--xmax", type=float, default=1350.0, help="upper bound of the wavenumber window")
    parser.add_argument(
        "--fwhm",
        type=float,
        default=4.0,
        help="Lorentzian FWHM in cm^-1 used to broaden each stick (default: 4.0)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=30.0,
        help="include sticks within this many cm^-1 outside the window so their tails still contribute",
    )
    parser.add_argument("--points", type=int, default=2000, help="number of samples across the window")
    parser.add_argument(
        "--no-sticks",
        action="store_true",
        help="don't overlay the raw stick positions/intensities as vertical markers (ignored with --stick-only)",
    )
    parser.add_argument(
        "--stick-only",
        action="store_true",
        help="skip Lorentzian broadening entirely and just plot the raw stick spectrum as vertical lines",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="save one plot per input file instead of overlaying them together",
    )
    parser.add_argument("--normalize", action="store_true", help="scale each spectrum so its max in-window intensity is 1")
    parser.add_argument(
        "--annotate",
        nargs=2,
        action="append",
        metavar=("X", "TEXT"),
        help="add an arrow annotation at wavenumber X with label TEXT; can be repeated",
    )
    parser.add_argument("-o", "--output", help="output image path (combined mode) or directory (--separate)")
    parser.add_argument("--title", help="plot title (default generated from the window)")
    parser.add_argument("--dpi", type=int, default=300, help="output image DPI")
    args = parser.parse_args()

    if args.xmin >= args.xmax:
        raise SystemExit("--xmin must be less than --xmax")

    labels = args.labels if args.labels else [label_from_path(f) for f in args.files]
    if len(labels) != len(args.files):
        raise SystemExit("--labels must have exactly one entry per input file")

    x = np.linspace(args.xmin, args.xmax, args.points)

    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def render(ax, path, label, color):
        freqs, ints = read_sticks(path)
        margin = 0.0 if args.stick_only else args.margin
        in_window = (freqs >= args.xmin - margin) & (freqs <= args.xmax + margin)
        freqs, ints = freqs[in_window], ints[in_window]

        if freqs.size == 0:
            print(f"{label}: no peaks within {args.xmin - margin:.1f}-{args.xmax + margin:.1f} cm^-1")
            return

        peaks_in_view = (freqs >= args.xmin) & (freqs <= args.xmax)
        stick_freqs = freqs[peaks_in_view]
        stick_ints = ints[peaks_in_view]
        annotation_xy = None

        if args.stick_only:
            scale = stick_ints.max() if args.normalize and stick_ints.size and stick_ints.max() > 0 else 1.0
            y_plot = stick_ints / scale
            ax.vlines(stick_freqs, 0, y_plot, color=color, linewidth=1.4, label=label)
            annotation_xy = (stick_freqs, y_plot)
        else:
            y = lorentzian_broaden(freqs, ints, x, args.fwhm)
            scale = y.max() if args.normalize and y.max() > 0 else 1.0
            y_plot = y / scale
            ax.plot(x, y_plot, color=color, linewidth=1.6, label=label)
            annotation_xy = (x, y_plot)
            if not args.no_sticks and peaks_in_view.any():
                ax.vlines(stick_freqs, 0, stick_ints / scale, color=color, linewidth=1, alpha=0.5, linestyle="--")

        print(f"{label}: peaks in {args.xmin:.1f}-{args.xmax:.1f} cm^-1:")
        for f0, inten in zip(stick_freqs, stick_ints):
            print(f"    {f0:9.2f} cm^-1   {inten:.6f} km/mol")
        return annotation_xy

    def finish_axes(ax):
        ax.set_xlim(args.xmin, args.xmax)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel("Intensity (normalized)" if args.normalize else "Intensity (km/mol)")
        ax.grid(alpha=0.25)

    if args.separate:
        out_dir = Path(args.output) if args.output else Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (path, label) in enumerate(zip(args.files, labels)):
            fig, ax = plt.subplots(figsize=(10, 5.5))
            render(ax, path, label, colors[i % len(colors)])
            finish_axes(ax)
            kind = "sticks" if args.stick_only else "spectrum"
            ax.set_title(args.title or f"{label}, {args.xmin:.0f}-{args.xmax:.0f} cm$^{{-1}}$")
            fig.tight_layout()
            output_path = out_dir / f"{label}_{kind}_{args.xmin:.0f}-{args.xmax:.0f}.png"
            fig.savefig(output_path, dpi=args.dpi)
            plt.close(fig)
            print(f"Wrote plot to {output_path}")
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        annotation_series = []
        for i, (path, label) in enumerate(zip(args.files, labels)):
            annotation_xy = render(ax, path, label, colors[i % len(colors)])
            if annotation_xy is not None:
                annotation_series.append(annotation_xy)
        finish_axes(ax)
        for annotation in args.annotate or []:
            x0, text = annotation
            x0 = float(x0)
            candidates = []
            for xs, ys in annotation_series:
                if xs.size:
                    candidates.append(float(np.interp(x0, xs, ys)))
            if candidates:
                y0 = max(candidates)
                ax.annotate(
                    text,
                    xy=(x0, y0),
                    xytext=(x0 + 2.0, y0 * 1.15),
                    arrowprops={"arrowstyle": "->", "linewidth": 0.8},
                    fontsize=9,
                )
        kind = "raw sticks" if args.stick_only else f"FWHM={args.fwhm:g} cm$^{{-1}}$"
        ax.set_title(args.title or f"IR spectrum, {args.xmin:.0f}-{args.xmax:.0f} cm$^{{-1}}$ ({kind})")
        ax.legend()
        fig.tight_layout()
        output_path = Path(args.output) if args.output else Path(f"freq_zoom_{args.xmin:.0f}-{args.xmax:.0f}.png")
        fig.savefig(output_path, dpi=args.dpi)
        plt.close(fig)
        print(f"Wrote plot to {output_path}")


if __name__ == "__main__":
    main()
