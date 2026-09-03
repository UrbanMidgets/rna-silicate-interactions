#!/usr/bin/env python3
"""
Run from the root of a clone of UrbanMidgets/rna-silicate-interactions:

    python3 mayer_valence_check.py > mayer_valence_results.tsv

Scans every ORCA .out file under data/primary_calculations for a converged
Mayer population analysis. For each file, checks whether the P atom has a
Mayer bond order > 0.5 to any of its own oxygens' silicon neighbours (i.e.
a P-O-Si linkage). Where a linkage is found, reports the coordinating
silicon's total Mayer valence (VA) against the mean/min/max of every OTHER
silicon's VA in the SAME structure -- an apples-to-apples, within-structure
comparison, not a cross-structure aggregate.

Prints one TSV row per linked structure found, plus a final summary block.
Also flags files where no MAYER POPULATION ANALYSIS block is found (not yet
converged / not a relevant job type) so nothing is silently skipped.
"""
import re
import sys
from pathlib import Path

ROOT = Path("data/primary_calculations")
BO_THRESHOLD = 0.5  # Mayer bond order above which we call it a real bond


def parse_last_mayer_block(text):
    """Return (atoms: {idx: (element, VA)}, bonds: {(i,j): BO}) from the
    LAST Mayer population analysis + bond order listing in the file."""
    starts = [m.start() for m in re.finditer(r'MAYER POPULATION ANALYSIS', text)]
    if not starts:
        return None, None
    block = text[starts[-1]:]

    atoms = {}
    for line in block.splitlines():
        m = re.match(r'\s*(\d+)\s+([A-Za-z]+)\s+[\d.]+\s+[\d.]+\s+-?[\d.]+\s+'
                     r'(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$', line)
        if m:
            idx, el, va, bva, fa = m.groups()
            atoms[int(idx)] = (el, float(va))
        elif atoms and line.strip() == '':
            break  # end of the atom table

    bonds = {}
    for m in re.finditer(r'B\(\s*(\d+)-(\w+)\s*,\s*(\d+)-(\w+)\s*\)\s*:\s*(-?[\d.]+)',
                         block):
        i, ei, j, ej, bo = m.groups()
        bonds[(int(i), int(j))] = (ei, ej, float(bo))

    return atoms, bonds


def analyse(path):
    text = path.read_text(errors='ignore')
    atoms, bonds = parse_last_mayer_block(text)
    if atoms is None:
        return ('no_mayer_block', None)

    p_idx = [i for i, (el, va) in atoms.items() if el == 'P']
    if not p_idx:
        return ('no_phosphorus', None)

    # oxygens directly bonded to P (the phosphate's own 4 oxygens)
    p_oxygens = set()
    for (i, j), (ei, ej, bo) in bonds.items():
        if ei == 'P' and i in p_idx and bo > BO_THRESHOLD:
            p_oxygens.add(j)
        if ej == 'P' and j in p_idx and bo > BO_THRESHOLD:
            p_oxygens.add(i)

    # does any of those oxygens also bond to a silicon?
    coord_si = None
    for (i, j), (ei, ej, bo) in bonds.items():
        if bo <= BO_THRESHOLD:
            continue
        if ei == 'Si' and j in p_oxygens:
            coord_si = i
        elif ej == 'Si' and i in p_oxygens:
            coord_si = j
    if coord_si is None:
        return ('unlinked', None)

    coord_va = atoms[coord_si][1]
    other_si = [va for idx, (el, va) in atoms.items()
                if el == 'Si' and idx != coord_si]
    if not other_si:
        return ('no_framework_si', None)

    fw_mean = sum(other_si) / len(other_si)
    return ('linked', dict(
        coord_si_idx=coord_si, coord_va=coord_va,
        fw_mean=fw_mean, fw_min=min(other_si), fw_max=max(other_si),
        fw_n=len(other_si), diff=coord_va - fw_mean,
    ))


def main():
    if not ROOT.exists():
        print(f"ERROR: {ROOT} not found -- run this from the repo root",
              file=sys.stderr)
        sys.exit(1)

    out_files = sorted(ROOT.rglob("*.out"))
    print(f"# scanned {len(out_files)} .out files under {ROOT}",
          file=sys.stderr)

    print("file\tcoord_si_idx\tcoord_VA\tframework_mean\tframework_min\t"
          "framework_max\tframework_n\tdiff")
    linked_diffs = []
    status_counts = {}
    for f in out_files:
        status, data = analyse(f)
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == 'linked':
            linked_diffs.append(data['diff'])
            print(f"{f.relative_to(ROOT.parent)}\t{data['coord_si_idx']}\t"
                  f"{data['coord_va']:.4f}\t{data['fw_mean']:.4f}\t"
                  f"{data['fw_min']:.4f}\t{data['fw_max']:.4f}\t"
                  f"{data['fw_n']}\t{data['diff']:+.4f}")

    print("\n# --- summary ---", file=sys.stderr)
    for status, n in sorted(status_counts.items()):
        print(f"#   {status}: {n}", file=sys.stderr)
    if linked_diffs:
        mean_diff = sum(linked_diffs) / len(linked_diffs)
        pos = sum(1 for d in linked_diffs if d > 0)
        neg = sum(1 for d in linked_diffs if d < 0)
        print(f"# linked structures found: {len(linked_diffs)}", file=sys.stderr)
        print(f"# mean (coord_VA - framework_mean): {mean_diff:+.4f}",
              file=sys.stderr)
        print(f"# positive: {pos}, negative: {neg}", file=sys.stderr)
        print(f"# individual diffs: {[round(d,4) for d in linked_diffs]}",
              file=sys.stderr)


if __name__ == '__main__':
    main()
