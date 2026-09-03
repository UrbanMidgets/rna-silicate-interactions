#!/usr/bin/env python3
"""Plot intensity vs wavenumber from ORCA/orca_mapspc spectrum files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_numeric_table(path: Path, x_col: int, y_col: int) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []

    with path.open(errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "!", ";")):
                continue

            parts = next(csv.reader([stripped], skipinitialspace=True)) if "," in stripped else stripped.split()
            if len(parts) <= max(x_col, y_col):
                continue

            try:
                x_values.append(float(parts[x_col]))
                y_values.append(float(parts[y_col]))
            except ValueError:
                # Allow one or more header/comment lines in otherwise numeric files.
                if x_values:
                    raise ValueError(f"Non-numeric data at {path}:{line_number}: {line.rstrip()}")

    if not x_values:
        raise ValueError(f"No numeric data found in {path}")

    return x_values, y_values


def default_plot_kind(path: Path) -> str:
    return "stick" if path.suffix.lower() == ".stk" else "line"


def default_output_path(path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return path.with_suffix("").with_name(f"{path.with_suffix('').name}_intensity.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot intensity vs wavenumber from two-column spectrum data."
    )
    parser.add_argument("input", help="two-column .dat, .stk, .txt, or .csv file")
    parser.add_argument("-o", "--output", help="output image path; default is INPUT_intensity.png")
    parser.add_argument(
        "--kind",
        choices=("auto", "line", "stick"),
        default="auto",
        help="plot type; auto uses stick for .stk and line otherwise",
    )
    parser.add_argument("--x-col", type=int, default=1, help="1-based wavenumber column")
    parser.add_argument("--y-col", type=int, default=2, help="1-based intensity column")
    parser.add_argument(
        "--invert",
        action="store_true",
        help="convert transmittance-like dips to upward intensity peaks",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        help="baseline for --invert; default uses the maximum y value",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="scale intensity so the maximum is 1",
    )
    parser.add_argument(
        "--reverse-x",
        action="store_true",
        help="plot high wavenumber on the left, common for IR spectra",
    )
    parser.add_argument("--title", default="IR Spectrum", help="plot title")
    parser.add_argument(
        "--xlabel", default="Wavenumber (cm$^{-1}$)", help="x-axis label"
    )
    parser.add_argument("--ylabel", default="Intensity", help="y-axis label")
    parser.add_argument("--dpi", type=int, default=300, help="output image DPI")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = default_output_path(input_path, args.output)
    x_col = args.x_col - 1
    y_col = args.y_col - 1
    if x_col < 0 or y_col < 0:
        raise SystemExit("--x-col and --y-col must be 1 or greater")

    wavenumbers, intensities = parse_numeric_table(input_path, x_col, y_col)

    if args.invert:
        baseline = args.baseline if args.baseline is not None else max(intensities)
        intensities = [baseline - value for value in intensities]

    if args.normalize:
        max_intensity = max(abs(value) for value in intensities)
        if max_intensity:
            intensities = [value / max_intensity for value in intensities]

    kind = default_plot_kind(input_path) if args.kind == "auto" else args.kind

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    if kind == "stick":
        ax.vlines(wavenumbers, 0, intensities, color="tab:blue", linewidth=1.2)
        ax.set_ylim(0, max(intensities) * 1.08 if max(intensities) > 0 else 1)
    else:
        ax.plot(wavenumbers, intensities, color="tab:blue", linewidth=1.4)

    ax.set_title(args.title)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_xlim(min(wavenumbers), max(wavenumbers))
    if args.reverse_x:
        ax.invert_xaxis()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)

    print(f"Read {len(wavenumbers)} points from {input_path}")
    print(f"Wavenumber range: {min(wavenumbers):.2f} to {max(wavenumbers):.2f} cm^-1")
    print(f"Intensity range: {min(intensities):.6g} to {max(intensities):.6g}")
    print(f"Wrote {kind} plot to {output_path}")


if __name__ == "__main__":
    main()
