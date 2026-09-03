#!/usr/bin/env python3
"""
Generates schematic figures 2.1, 2.2 and 2.4 for Chapter 2.

Outputs both PDF (vector, for LaTeX) and PNG at 600 dpi (for Docs/Word).

Usage:  python3 make_ch2_figures.py [outdir]
"""

import sys
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle, Arc, Wedge

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- style ----
INK = "#1a1a1a"      # curves, text
MUTED = "#8a8a8a"    # guides, annotation leaders
ACCENT = "#B2432F"   # highlight (UCPH-ish red)
FILL = "#dfe4ea"     # panel fills
FILL2 = "#c3cbd6"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.linewidth": 0.9,
    "axes.edgecolor": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), dpi=600,
                    transparent=False, facecolor="white")
    plt.close(fig)
    print(f"wrote {name}.pdf / {name}.png")


# ==========================================================================
# Figure 2.1 - potential energy surface slice
# ==========================================================================
def figure_2_1():
    fig, ax = plt.subplots(figsize=(5.6, 3.3))

    # Asymmetric double well: local minimum on the left, global on the right.
    x = np.linspace(-2.35, 2.5, 1200)
    y = 0.55 * (x**2 - 1.55) ** 2 - 0.34 * x + 1.05

    ax.plot(x, y, color=INK, lw=1.7, zorder=3, solid_capstyle="round")

    def loc(a, b):
        """Return (x, y) of the curve minimum on the interval [a, b]."""
        m = (x >= a) & (x <= b)
        i = np.argmin(y[m])
        return x[m][i], y[m][i]

    def peak(a, b):
        m = (x >= a) & (x <= b)
        i = np.argmax(y[m])
        return x[m][i], y[m][i]

    xl, yl = loc(-2.0, -0.5)   # local minimum
    xg, yg = loc(0.5, 2.0)     # global minimum
    xs, ys = peak(-0.6, 0.6)   # saddle point / barrier top

    for px, py in [(xl, yl), (xg, yg), (xs, ys)]:
        ax.plot(px, py, "o", ms=6.5, mfc="white", mec=INK, mew=1.5, zorder=5)

    # Starting structure, uphill on the left flank.
    x0 = -1.95
    y0 = np.interp(x0, x, y)
    ax.plot(x0, y0, "o", ms=6.5, color=ACCENT, zorder=6)

    # Descent arrows from the start point down into the local minimum.
    path_x = np.linspace(x0, xl, 60)
    path_y = np.interp(path_x, x, y)
    for i in (14, 34, 50):
        ax.annotate("", xy=(path_x[i + 6], path_y[i + 6] + 0.05),
                    xytext=(path_x[i], path_y[i] + 0.05),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT,
                                    lw=1.4, mutation_scale=11), zorder=6)

    # Barrier height, not determined by an optimisation.
    ax.annotate("", xy=(xs, ys), xytext=(xs, yl),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0,
                                mutation_scale=9), zorder=2)
    ax.plot([xl, xs], [yl, yl], ls=(0, (2, 2)), color=MUTED, lw=0.8, zorder=1)
    ax.plot([xs, xs], [yl - 0.14, yl], color=MUTED, lw=0.8, zorder=2)
    ax.text(xs, yl - 0.16, "barrier (not located)", fontsize=7.6,
            color=MUTED, ha="center", va="top", zorder=6)

    # Energy difference between the two minima.
    ax.plot([xg - 0.42, 2.42], [yg, yg], ls=(0, (2, 2)), color=MUTED, lw=0.8, zorder=1)
    ax.plot([xl, 2.42], [yl, yl], ls=(0, (2, 2)), color=MUTED, lw=0.8, zorder=1)
    ax.annotate("", xy=(2.32, yl), xytext=(2.32, yg),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0,
                                mutation_scale=9), zorder=2)
    ax.text(2.26, (yl + yg) / 2, r"$\Delta E$", fontsize=8, color=MUTED,
            ha="right", va="center")

    ax.annotate("starting structure", xy=(x0, y0), xytext=(x0 - 0.06, y0 + 0.72),
                fontsize=8, color=ACCENT, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8,
                                shrinkA=1, shrinkB=4))
    ax.annotate("local minimum", xy=(xl, yl), xytext=(xl - 0.18, yl - 0.70),
                fontsize=8, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=1, shrinkB=4))
    ax.annotate("saddle point", xy=(xs, ys), xytext=(xs - 0.55, ys + 0.52),
                fontsize=8, ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=1, shrinkB=4))
    ax.annotate("global minimum", xy=(xg, yg), xytext=(xg - 0.05, yg - 0.70),
                fontsize=8, ha="center", va="top",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8,
                                shrinkA=1, shrinkB=4))

    ax.set_xlabel("nuclear coordinate", fontsize=9)
    ax.set_ylabel("potential energy", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-2.55, 2.62)
    ax.set_ylim(min(y) - 1.15, max(y[(x > -2.2) & (x < 2.2)]) + 0.55)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    save(fig, "figure_2_1_pes")


# ==========================================================================
# Figure 2.2 - Jacob's ladder
# ==========================================================================
def figure_2_2():
    fig, ax = plt.subplots(figsize=(6.3, 3.4))

    rungs = [
        ("Local density approximation (LDA)", r"$\rho$"),
        ("Generalised gradient approximation (GGA)", r"$\rho,\ \nabla\rho$"),
        ("Meta-GGA", r"$\rho,\ \nabla\rho,\ \tau$"),
        ("Hybrid functionals", r"$+\ $exact exchange"),
        ("Double hybrids", r"$+\ $PT2 correlation"),
    ]
    used = {2: (r"r$^2$SCAN", "trialled"), 3: ("PBE0", "used here")}

    h, w, x0 = 0.62, 7.3, 0.55
    for i, (name, dep) in enumerate(rungs):
        y = i * 1.0
        hot = i in used
        ax.add_patch(Rectangle((x0, y), w, h,
                               facecolor=ACCENT if hot else FILL,
                               edgecolor=ACCENT if hot else "#9aa3ad",
                               lw=1.1, alpha=1.0 if hot else 1.0, zorder=3))
        ax.text(x0 + 0.22, y + h / 2, name, fontsize=8.3, va="center",
                ha="left", color="white" if hot else INK,
                fontweight="bold" if hot else "normal", zorder=4)
        ax.text(x0 + w - 0.22, y + h / 2, dep, fontsize=8, va="center",
                ha="right", color="#f2dcd7" if hot else "#5c6570", zorder=4)
        ax.text(x0 - 0.20, y + h / 2, str(i + 1), fontsize=8.6, va="center",
                ha="right", color=MUTED, zorder=4)
        if hot:
            label, note = used[i]
            ax.annotate(f"{label}\n{note}", xy=(x0 + w, y + h / 2),
                        xytext=(x0 + w + 0.62, y + h / 2),
                        fontsize=8, color=ACCENT, va="center", ha="left",
                        linespacing=1.3,
                        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8,
                                        shrinkA=2, shrinkB=2))

    # Side rails of the ladder.
    for xr in (x0 - 0.42, x0 + w + 0.20):
        ax.plot([xr, xr], [-0.18, 4 + h + 0.18], color="#9aa3ad", lw=1.0, zorder=1)

    ax.add_patch(FancyArrowPatch((x0 - 0.80, -0.10), (x0 - 0.80, 4 + h + 0.10),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=INK, lw=1.2, zorder=3))
    ax.text(x0 - 0.98, (4 + h) / 2, "increasing accuracy and cost",
            rotation=90, va="center", ha="center", fontsize=8.4)

    ax.set_xlim(-0.85, x0 + w + 2.25)
    ax.set_ylim(-0.45, 4 + h + 0.75)
    ax.axis("off")
    save(fig, "figure_2_2_jacobs_ladder")


# ==========================================================================
# Figure 2.4 - the terms of a classical force field
# ==========================================================================
def figure_2_4():
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.1))

    def atom(ax, xy, r=0.15, c=FILL2):
        ax.add_patch(Circle(xy, r, facecolor=c, edgecolor=INK, lw=1.1, zorder=5))

    def stick(ax, a, b, lw=1.4, c=INK, z=3, ls="-"):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=c, lw=lw, zorder=z, ls=ls)

    def spring(ax, a, b, coils=6, amp=0.11):
        a, b = np.array(a, float), np.array(b, float)
        d = b - a
        L = np.linalg.norm(d)
        u = d / L
        n = np.array([-u[1], u[0]])
        t = np.linspace(0, 1, 260)
        pad = 0.20
        tt = np.clip((t - pad) / (1 - 2 * pad), 0, 1)
        off = amp * np.sin(2 * np.pi * coils * tt) * ((t > pad) & (t < 1 - pad))
        pts = a + np.outer(t, d) + np.outer(off, n)
        ax.plot(pts[:, 0], pts[:, 1], color=INK, lw=1.3, zorder=3)

    def frame(ax, title, eq):
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.92, 1.08)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0, 1.00, title, ha="center", va="top", fontsize=8.8,
                fontweight="bold")
        ax.text(0, -0.90, eq, ha="center", va="bottom", fontsize=8.2,
                color="#404040")

    # --- bond stretching ---
    ax = axes[0, 0]
    spring(ax, (-0.62, 0.15), (0.62, 0.15))
    atom(ax, (-0.62, 0.15)); atom(ax, (0.62, 0.15))
    ax.annotate("", xy=(0.92, 0.15), xytext=(0.68, 0.15),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3,
                                mutation_scale=10))
    ax.annotate("", xy=(-0.92, 0.15), xytext=(-0.68, 0.15),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3,
                                mutation_scale=10))
    ax.text(0, -0.14, "$r$", ha="center", fontsize=8.6, color=ACCENT)
    frame(ax, "Bond stretching", r"$\frac{1}{2}k_b\,(r-r_0)^2$")

    # --- angle bending ---
    ax = axes[0, 1]
    v, a1, a2 = (0.0, -0.28), (-0.72, 0.42), (0.72, 0.42)
    stick(ax, v, a1); stick(ax, v, a2)
    ax.add_patch(Arc(v, 0.74, 0.74, theta1=44, theta2=136,
                     edgecolor=ACCENT, lw=1.3, zorder=4))
    ax.annotate("", xy=(0.30, 0.34), xytext=(0.42, 0.16),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2,
                                mutation_scale=9,
                                connectionstyle="arc3,rad=0.35"))
    atom(ax, v); atom(ax, a1); atom(ax, a2)
    ax.text(0, 0.20, r"$\theta$", ha="center", fontsize=9, color=ACCENT)
    frame(ax, "Angle bending", r"$\frac{1}{2}k_\theta\,(\theta-\theta_0)^2$")

    # --- proper dihedral ---
    ax = axes[0, 2]
    p1, p2, p3, p4 = (-0.86, 0.10), (-0.30, -0.20), (0.30, -0.20), (0.80, 0.52)
    stick(ax, p1, p2); stick(ax, p2, p3, lw=2.0); stick(ax, p3, p4)
    ax.add_patch(Arc((0.05, -0.20), 1.05, 0.50, theta1=185, theta2=352,
                     edgecolor=ACCENT, lw=1.2, ls=(0, (2, 2)), zorder=2))
    ax.annotate("", xy=(0.60, -0.16), xytext=(0.44, -0.36),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.2,
                                mutation_scale=9,
                                connectionstyle="arc3,rad=0.45"))
    for p in (p1, p2, p3, p4):
        atom(ax, p)
    ax.text(0.10, -0.56, r"$\phi$", ha="center", fontsize=9, color=ACCENT)
    frame(ax, "Proper dihedral",
          r"$\frac{V_n}{2}\left[1+\cos(n\phi-\gamma)\right]$")

    # --- improper dihedral ---
    ax = axes[1, 0]
    c = (0.0, 0.02)
    s1, s2, s3 = (-0.72, -0.24), (0.72, -0.24), (0.0, 0.60)
    for s in (s1, s2, s3):
        stick(ax, c, s)
    ax.plot([-0.95, 0.95], [-0.24, -0.24], color=MUTED, lw=0.9,
            ls=(0, (3, 2)), zorder=1)
    ax.annotate("", xy=(0.0, 0.44), xytext=(0.0, -0.20),
                arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.3,
                                mutation_scale=10))
    for s in (s1, s2, s3):
        atom(ax, s)
    atom(ax, c, c="white")
    ax.text(0.13, 0.24, r"$\xi$", fontsize=9, color=ACCENT)
    frame(ax, "Improper dihedral", r"$\frac{1}{2}k_\xi\,(\xi-\xi_0)^2$")

    # --- electrostatics ---
    ax = axes[1, 1]
    for xy, lab, col in [((-0.58, 0.15), "$+$", "#c9d6e8"),
                         ((0.58, 0.15), "$-$", "#e8d3cd")]:
        ax.add_patch(Circle(xy, 0.24, facecolor=col, edgecolor=INK,
                            lw=1.1, zorder=5))
        ax.text(xy[0], xy[1], lab, ha="center", va="center",
                fontsize=11, zorder=6)
    for rr, al in [(0.34, 0.55), (0.44, 0.35), (0.54, 0.20)]:
        ax.add_patch(Arc((-0.58, 0.15), rr * 2, rr * 2, theta1=-58, theta2=58,
                         edgecolor=MUTED, lw=0.8, alpha=al, zorder=2))
        ax.add_patch(Arc((0.58, 0.15), rr * 2, rr * 2, theta1=122, theta2=238,
                         edgecolor=MUTED, lw=0.8, alpha=al, zorder=2))
    ax.annotate("", xy=(0.28, 0.15), xytext=(-0.28, 0.15),
                arrowprops=dict(arrowstyle="<|-|>", color=ACCENT, lw=1.2,
                                mutation_scale=9))
    ax.text(0.0, -0.06, "$r_{ij}$", ha="center", fontsize=8.6, color=ACCENT)
    frame(ax, "Electrostatics",
          r"$\frac{1}{4\pi\varepsilon_0}\frac{q_iq_j}{r_{ij}}$")

    # --- Lennard-Jones ---
    ax = axes[1, 2]
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.92, 1.08)
    r = np.linspace(0.99, 3.2, 500)
    e = 4 * 1.0 * ((1 / r) ** 12 - (1 / r) ** 6)
    rx = -0.92 + (r - 0.99) * (1.90 / (3.2 - 0.99))
    ry = 0.24 + e * 0.45
    ax.plot(rx, ry, color=INK, lw=1.5, zorder=4)
    ax.axhline(0.24, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1,
               xmin=0.04, xmax=0.96)
    i = np.argmin(ry)
    ax.plot([rx[i], rx[i]], [0.24, ry[i]], color=ACCENT, lw=1.0,
            ls=(0, (2, 2)), zorder=3)
    ax.plot(rx[i], ry[i], "o", ms=4.5, color=ACCENT, zorder=5)
    ax.text(rx[i] + 0.26, (0.24 + ry[i]) / 2 - 0.02, r"$\varepsilon_{ij}$",
            fontsize=8.6, color=ACCENT, va="center")
    j = np.argmin(np.abs(ry[:i] - 0.24))
    ax.plot(rx[j], 0.24, "o", ms=4.5, mfc="white", mec=INK, mew=1.2, zorder=5)
    ax.text(rx[j] - 0.04, 0.33, r"$\sigma_{ij}$", fontsize=8.6,
            ha="right", color=INK)
    ax.text(-0.86, 0.66, r"$r^{-12}$ repulsion", fontsize=7.4, color=MUTED)
    ax.text(1.02, 0.09, r"$r^{-6}$ dispersion", fontsize=7.4,
            color=MUTED, ha="right")
    ax.axis("off")
    ax.text(0, 1.00, "Lennard-Jones", ha="center", va="top",
            fontsize=8.8, fontweight="bold")
    ax.text(0, -0.90,
            r"$4\varepsilon_{ij}\left[(\sigma_{ij}/r_{ij})^{12}"
            r"-(\sigma_{ij}/r_{ij})^{6}\right]$",
            ha="center", va="bottom", fontsize=8.2, color="#404040")

    fig.subplots_adjust(wspace=0.05, hspace=0.30)
    save(fig, "figure_2_4_force_field_terms")


if __name__ == "__main__":
    figure_2_1()
    figure_2_2()
    figure_2_4()
