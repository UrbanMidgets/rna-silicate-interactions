#!/usr/bin/env python3
"""
Sample the electrostatic potential at defined heights above the dorsal face of a
silicate slab, at lateral positions above Si centres and above bridging oxygens.

Two modes:

  1. Interpolate an existing ESP cube file (no new ORCA runs needed):
         python esp_probe.py slab.xyz --cube slab_esp.cube

  2. Emit a point list for orca_vpot, then read its output back:
         python esp_probe.py slab.xyz --emit-points points.inp
         orca_vpot slab.gbw slab.scfp points.inp points.out
         python esp_probe.py slab.xyz --read-vpot points.out

Sampling is done on a plane parallel to the slab at a fixed height above the
highest dorsal atom, so that Si and O sites are probed at equal distance from
the surface rather than at equal distance from their own nucleus. Peripheral
atoms are excluded to avoid edge artefacts.
"""

import argparse
import sys
import numpy as np

BOHR = 0.529177210903          # Angstrom per Bohr
HARTREE_KCAL = 627.5094740631  # kcal/mol per Hartree

COV = {"H": 0.31, "O": 0.66, "SI": 1.11, "AL": 1.21, "P": 1.07,
       "C": 0.76, "N": 0.71, "MG": 1.41}


def read_xyz(path):
    with open(path) as fh:
        n = int(fh.readline().split()[0])
        fh.readline()
        syms, xyz = [], []
        for _ in range(n):
            p = fh.readline().split()
            syms.append(p[0])
            xyz.append([float(v) for v in p[1:4]])
    return syms, np.asarray(xyz)


def read_cube(path):
    """Gaussian cube. Returns origin(A), voxel matrix(A), data grid."""
    with open(path) as fh:
        fh.readline(); fh.readline()
        parts = fh.readline().split()
        natoms = int(parts[0])
        origin = np.array([float(v) for v in parts[1:4]])
        vox, ns = [], []
        for _ in range(3):
            p = fh.readline().split()
            ns.append(int(p[0]))
            vox.append([float(v) for v in p[1:4]])
        vox = np.asarray(vox)
        for _ in range(abs(natoms)):
            fh.readline()
        vals = []
        for line in fh:
            vals.extend(float(v) for v in line.split())
    data = np.asarray(vals[:ns[0]*ns[1]*ns[2]]).reshape(ns)
    # cube is in Bohr when natoms > 0
    if natoms > 0:
        origin = origin * BOHR
        vox = vox * BOHR
    return origin, vox, data


def interp_cube(origin, vox, data, pts):
    """Trilinear interpolation at Cartesian points (Angstrom)."""
    inv = np.linalg.inv(vox.T)
    frac = (pts - origin) @ inv.T
    out = np.full(len(pts), np.nan)
    for i, f in enumerate(frac):
        if np.any(f < 0) or np.any(f > np.array(data.shape) - 1):
            continue
        i0 = np.floor(f).astype(int)
        d = f - i0
        acc = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((1-d[0]) if dx == 0 else d[0]) * \
                        ((1-d[1]) if dy == 0 else d[1]) * \
                        ((1-d[2]) if dz == 0 else d[2])
                    acc += w * data[i0[0]+dx, i0[1]+dy, i0[2]+dz]
        out[i] = acc
    return out


def classify(syms, xyz, edge_frac):
    """Identify dorsal Si/Al centres and dorsal bridging oxygens."""
    n = len(syms)
    up = np.array([s.upper() for s in syms])

    # bond list
    nb = [[] for _ in range(n)]
    for i in range(n):
        ri = COV.get(up[i], 0.8)
        for j in range(i+1, n):
            rj = COV.get(up[j], 0.8)
            if np.linalg.norm(xyz[i]-xyz[j]) < 1.25*(ri+rj):
                nb[i].append(j); nb[j].append(i)

    # slab normal from the inertia tensor: normal = axis of largest moment
    c = xyz - xyz.mean(0)
    w, v = np.linalg.eigh(c.T @ c)
    normal = v[:, 0]                      # smallest spread = thinnest direction
    h = c @ normal

    # orient so that the face with fewer hydrogens is positive (dorsal)
    hyd = up == "H"
    if hyd.sum():
        if h[hyd].mean() > 0:
            normal, h = -normal, -h

    # lateral radius for edge exclusion
    lat = c - np.outer(h, normal)
    rad = np.linalg.norm(lat, axis=1)
    rmax = rad.max()
    keep_r = edge_frac * rmax

    tet = [i for i in range(n) if up[i] in ("SI", "AL")]
    if not tet:
        sys.exit("no Si or Al found")
    dorsal_h = np.median([h[i] for i in tet])

    si_sites, ob_sites = [], []
    for i in tet:
        if h[i] > dorsal_h - 0.5 and rad[i] < keep_r:
            si_sites.append(i)
    for i in range(n):
        if up[i] != "O":
            continue
        heavy = [j for j in nb[i] if up[j] in ("SI", "AL")]
        has_h = any(up[j] == "H" for j in nb[i])
        if len(heavy) >= 2 and not has_h and rad[i] < keep_r and h[i] > dorsal_h - 0.5:
            ob_sites.append(i)
    return normal, h, rad, si_sites, ob_sites, up


def build_points(xyz, normal, h, sites, heights):
    """Probe points on planes at fixed height above the highest dorsal atom."""
    top = h.max()
    pts, meta = [], []
    for z in heights:
        plane = top + z
        for i in sites:
            lateral = xyz[i] - normal*h[i]      # foot of atom on mid-plane
            pts.append(lateral + normal*plane)
            meta.append((i, z))
    return np.asarray(pts), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xyz")
    ap.add_argument("--cube")
    ap.add_argument("--emit-points")
    ap.add_argument("--read-vpot")
    ap.add_argument("--heights", default="2.0,3.0,4.0",
                    help="probe heights above highest dorsal atom, Angstrom")
    ap.add_argument("--edge-frac", type=float, default=0.6,
                    help="keep sites within this fraction of max lateral radius")
    ap.add_argument("--kcal", action="store_true", help="also report kcal/mol")
    args = ap.parse_args()

    syms, xyz = read_xyz(args.xyz)
    heights = [float(v) for v in args.heights.split(",")]
    normal, h, rad, si, ob, up = classify(syms, xyz, args.edge_frac)

    print(f"{len(syms)} atoms; slab normal {np.round(normal,3)}")
    print(f"dorsal Si/Al sites retained: {len(si)}   "
          f"bridging O sites retained: {len(ob)}")
    print(f"(edge exclusion: lateral radius < {args.edge_frac:.2f} x max)")

    pts, meta = build_points(xyz, normal, h, si + ob, heights)

    if args.emit_points:
        with open(args.emit_points, "w") as fh:
            fh.write(f"{len(pts)}\n")
            for p in pts / BOHR:
                fh.write(f"  {p[0]:14.8f} {p[1]:14.8f} {p[2]:14.8f}\n")
        np.save(args.emit_points + ".meta.npy",
                np.array([(i, z) for i, z in meta]))
        print(f"\nwrote {len(pts)} points to {args.emit_points} (Bohr)")
        print("run:  orca_vpot <base>.gbw <base>.scfp "
              f"{args.emit_points} {args.emit_points}.out")
        print(f"then: python {sys.argv[0]} {args.xyz} "
              f"--read-vpot {args.emit_points}.out")
        return

    if args.cube:
        origin, vox, data = read_cube(args.cube)
        V = interp_cube(origin, vox, data, pts)
    elif args.read_vpot:
        rows = []
        with open(args.read_vpot) as fh:
            for line in fh:
                p = line.split()
                if len(p) >= 4:
                    try:
                        rows.append(float(p[3]))
                    except ValueError:
                        continue
        V = np.asarray(rows)
        if len(V) != len(pts):
            sys.exit(f"expected {len(pts)} values, got {len(V)}")
    else:
        sys.exit("give one of --cube, --emit-points, --read-vpot")

    nsi = len(si)
    print(f"\n{'height':>8} {'V(Si) mean':>12} {'V(O) mean':>12} "
          f"{'delta':>10} {'range':>10}")
    print("-"*58)
    for z in heights:
        m = np.array([mz == z for _, mz in meta])
        vals = V[m]
        vsi = vals[:nsi]; vob = vals[nsi:]
        vsi = vsi[~np.isnan(vsi)]; vob = vob[~np.isnan(vob)]
        if not len(vsi) or not len(vob):
            print(f"{z:8.1f}   (outside cube bounds)")
            continue
        d = vsi.mean() - vob.mean()
        rng = np.nanmax(vals) - np.nanmin(vals)
        print(f"{z:8.1f} {vsi.mean():12.5f} {vob.mean():12.5f} "
              f"{d:10.5f} {rng:10.5f}")
        if args.kcal:
            print(f"{'':8} {vsi.mean()*HARTREE_KCAL:12.2f} "
                  f"{vob.mean()*HARTREE_KCAL:12.2f} "
                  f"{d*HARTREE_KCAL:10.2f} {rng*HARTREE_KCAL:10.2f}  kcal/mol")
        print(f"{'':8} sd {vsi.std():9.5f} {vob.std():12.5f}")
    print("\ndelta = V(above Si) - V(above bridging O), a within-surface")
    print("difference and therefore free of any net-charge offset.")


if __name__ == "__main__":
    main()
