#!/usr/bin/env python3
"""Create a curated thesis data layout from /data/Seafile.

The source tree is treated as read-only. The script only writes inside the
repository's data/ directory.
"""

from __future__ import annotations

import csv
import os
import argparse
import shutil
from pathlib import Path


SOURCE_ROOT = Path("/data/Seafile")
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

NUCLEOTIDES = ("amp", "cmp", "gmp", "ump")

CALC_SUFFIXES = (
    ".inp",
    ".out",
    ".xyz",
    ".engrad",
    ".opt",
    ".bibtex",
    ".densitiesinfo",
    ".cpcm_corr",
    ".cpcm",
    ".property.txt",
)

CHECK_SUFFIXES = CALC_SUFFIXES + (
    ".csv",
    ".svg",
)

MD_SUFFIXES = CHECK_SUFFIXES + (
    ".md.log",
    ".mdinput",
)

MD_RUNS = (
    ("amp", "frame11", "dry_md", SOURCE_ROOT / "amp" / "frames" / "frame11" / "dry" / "dry_md"),
    ("amp", "frame11", "dry_md_fixed", SOURCE_ROOT / "amp" / "frames" / "frame11" / "dry" / "dry_md_fixed"),
    ("amp", "frame12", "dry_md", SOURCE_ROOT / "amp" / "frames" / "frame12" / "dry" / "dry_md"),
    ("ump", "frame15", "dry_md", SOURCE_ROOT / "ump" / "frames" / "frame15" / "dry" / "dry_md"),
    ("ump", "frame15", "1000deg", SOURCE_ROOT / "ump" / "frames" / "frame15" / "dry" / "1000deg"),
)

DRY_CHECKS_TO_INCLUDE = {
    ("amp", "canonical", "frame12", "dry_md"),
    ("ump", "canonical", "frame15", "1000deg"),
}

DRY_CHECK_TRAJECTORY_NAMES = {
    ("amp", "canonical", "frame12", "dry_md"): "amp_frame12_dry_md_fixed_trj.xyz",
    ("ump", "canonical", "frame15", "1000deg"): "ump_frame15_dry2wet_1000deg_trj.xyz",
}

DOCKING_KEEP_NAMES = {
    "docking.inp",
    "docking.out",
    "flower_ring_al.xyz",
    "flower_ring.xyz",
}

DOCKING_KEEP_ENDINGS = (
    ".all.swarm.xyz",
    ".inp",
    ".out",
)

DOCKING_SWARM_ENDING = ".all.swarm.xyz"

EXCLUDED_DIRS = {
    ".seafile-data",
    "SeafileData",
    "__pycache__",
}

manifest: list[dict[str, str]] = []
include_swarms = False


def normal_status(path: Path) -> str:
    if path.suffix.lower() != ".out":
        return ""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 20000))
            tail = handle.read().decode("utf-8", "ignore")
    except OSError:
        return "unknown"
    if "ORCA TERMINATED NORMALLY" in tail or "NORMAL TERMINATION" in tail:
        return "normal"
    return "not_normal"


def copy_file(src: Path, dst: Path, *, role: str, system: str, surface: str, frame: str, state: str, notes: str = "") -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest.append(
        {
            "system": system,
            "surface": surface,
            "frame": frame,
            "state": state,
            "role": role,
            "status": normal_status(src),
            "source_path": str(src),
            "repo_path": str(dst.relative_to(REPO_ROOT)),
            "size_bytes": str(dst.stat().st_size),
            "notes": notes,
        }
    )


def should_copy_calc_file(path: Path, suffixes: tuple[str, ...]) -> bool:
    name = path.name.lower()
    if name.endswith(DOCKING_SWARM_ENDING) and not include_swarms:
        return False
    return any(name.endswith(suffix) for suffix in suffixes)


def copy_flat_calc_dir(src_dir: Path, dst_dir: Path, *, role: str, system: str, surface: str, frame: str, state: str, suffixes: tuple[str, ...] = CALC_SUFFIXES, notes: str = "") -> None:
    if not src_dir.is_dir():
        return
    for src in sorted(src_dir.iterdir()):
        if src.is_file() and should_copy_calc_file(src, suffixes):
            copy_file(src, dst_dir / src.name, role=role, system=system, surface=surface, frame=frame, state=state, notes=notes)


def copy_recursive_filtered(src_dir: Path, dst_dir: Path, *, role: str, system: str, surface: str, frame: str, state: str, suffixes: tuple[str, ...], notes: str = "", name_overrides: dict[str, str] | None = None) -> None:
    if not src_dir.is_dir():
        return
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in sorted(dirs) if d not in EXCLUDED_DIRS]
        root_path = Path(root)
        rel = root_path.relative_to(src_dir)
        for name in sorted(files):
            src = root_path / name
            if should_copy_calc_file(src, suffixes):
                dst_name = name_overrides.get(name, name) if name_overrides else name
                copy_file(src, dst_dir / rel / dst_name, role=role, system=system, surface=surface, frame=frame, state=state, notes=notes)


def has_outputs(frame_dir: Path) -> bool:
    return bool(list(frame_dir.glob("*.out")) or list((frame_dir / "dry").glob("*.out")))


def copy_frame_set(src_frames: Path, dst_system: Path, *, system: str, surface: str, role: str) -> None:
    if not src_frames.is_dir():
        return
    selected_root = dst_system / "selected_frames"
    checks_root = dst_system / "checks"

    for frame_dir in sorted(p for p in src_frames.iterdir() if p.is_dir() and p.name.startswith("frame")):
        frame = frame_dir.name
        initial_xyz = frame_dir / f"{frame}.xyz"

        if not has_outputs(frame_dir):
            continue

        if initial_xyz.is_file():
            copy_file(initial_xyz, selected_root / frame / "initial" / initial_xyz.name, role=role, system=system, surface=surface, frame=frame, state="initial")

        copy_flat_calc_dir(frame_dir, selected_root / frame / "solvated", role=role, system=system, surface=surface, frame=frame, state="solvated")
        copy_flat_calc_dir(frame_dir / "dry", selected_root / frame / "dry", role=role, system=system, surface=surface, frame=frame, state="dry")

        dry_dir = frame_dir / "dry"
        if dry_dir.is_dir():
            for extra in sorted(p for p in dry_dir.iterdir() if p.is_dir()):
                dry_check = (system, surface, frame, extra.name)
                if dry_check not in DRY_CHECKS_TO_INCLUDE:
                    continue
                name_overrides = {"trajectory.xyz": DRY_CHECK_TRAJECTORY_NAMES[dry_check]}
                copy_recursive_filtered(extra, checks_root / frame / "dry" / extra.name, role="check", system=system, surface=surface, frame=frame, state=f"dry/{extra.name}", suffixes=CHECK_SUFFIXES, notes="non-primary dry check or exploratory calculation", name_overrides=name_overrides)

        for extra in sorted(p for p in frame_dir.iterdir() if p.is_dir() and p.name != "dry"):
            copy_recursive_filtered(extra, checks_root / frame / extra.name, role="check", system=system, surface=surface, frame=frame, state=extra.name, suffixes=CHECK_SUFFIXES, notes="non-primary check or exploratory calculation")


def copy_docking(src_dir: Path, dst_system: Path, *, system: str, surface: str, role: str, label: str = "raw") -> None:
    if not src_dir.is_dir():
        return
    dst = dst_system / "docking" / label
    for src in sorted(src_dir.iterdir()):
        if not src.is_file():
            continue
        name = src.name.lower()
        if name.endswith(DOCKING_SWARM_ENDING) and not include_swarms:
            continue
        keep = src.name in DOCKING_KEEP_NAMES or any(name.endswith(ending) for ending in DOCKING_KEEP_ENDINGS)
        if keep:
            copy_file(src, dst / src.name, role=role, system=system, surface=surface, frame="", state="docking", notes="docking provenance used for frame selection")


def write_readmes() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "README.md").write_text(
        "# Curated Thesis Data\n\n"
        "This directory contains a curated copy of the RNA nucleotide/silicate calculations. "
        "The raw source archive is `/data/Seafile`, which was left untouched during migration.\n\n"
        "## How To Navigate\n\n"
        "Start with `MANIFEST.tsv` for a complete file index. For the main thesis systems, use "
        "`primary_calculations/`, then choose the surface type and nucleotide. Each nucleotide/surface folder "
        "keeps its docking provenance beside the selected frames so the path from docking to optimisation is visible.\n\n"
        "## Layout\n\n"
        "- `primary_calculations/canonical_surface/`: neutral AMP, CMP, GMP, and UMP on the canonical silicate surface.\n"
        "- `primary_calculations/aluminium_surface/`: neutral AMP, CMP, GMP, and UMP on the aluminium-substituted surface.\n"
        "- `docking/`: docking inputs/outputs retained with the nucleotide/surface they generated.\n"
        "- `selected_frames/`: frames selected from docking and then optimised.\n"
        "- `checks/`: non-primary calculations such as unconstrained tests, R2SCAN checks, MD checks, or accidental reruns.\n"
        "- `supporting_calculations/`: protonated systems, dimer data, and magnesium docking organized by surface type.\n"
        "- `reference_structures/`: standalone reference surface calculations.\n"
        "- `scripts/`: scripts copied from the source archive for provenance.\n\n"
        "## Frame Folders\n\n"
        "A typical selected frame is organised as follows:\n\n"
        "```text\n"
        "frameXX/\n"
        "  initial/\n"
        "  solvated/\n"
        "  dry/\n"
        "```\n\n"
        "`solvated/` contains the CPCM(water) geometry optimisation. `dry/` contains the subsequent dry optimisation "
        "started from the solvated result where available.\n\n"
        "## Included Files\n\n"
        "Included file types are ORCA inputs/outputs, final and trajectory XYZ files, optimisation metadata, "
        "bibliography files, density metadata, CPCM correction files, and small CSV/SVG files for check calculations.\n\n"
        "Excluded file types include large restart/scratch/binary files such as `.gbw`, `.densities`, `.tmp`, `.log`, "
        "and media/rendering files that are not needed for navigating the thesis dataset. Docking swarm files "
        "(`*.all.swarm.xyz`) are excluded by default to keep the GitHub dataset smaller. Restore them with "
        "`python3 scripts/migrate_seafile_data.py --include-swarms` if the full docking swarms are needed.\n\n"
        "## Manifest Columns\n\n"
        "`MANIFEST.tsv` contains `system`, `surface`, `frame`, `state`, `role`, `status`, `source_path`, "
        "`repo_path`, `size_bytes`, and `notes`. The `status` column is populated for ORCA `.out` files when "
        "normal termination could be detected.\n",
        encoding="utf-8",
    )

    for rel, text in {
        "primary_calculations/README.md": "# Primary Calculations\n\nPrimary neutral nucleotide calculations are split into `canonical_surface/` and `aluminium_surface/`. Within each surface, AMP, CMP, GMP, and UMP each contain docking provenance and selected optimisation frames.\n\nUse `selected_frames/` for the thesis-primary solvated and dry optimisations. `checks/` contains non-primary calculations.\n",
        "supporting_calculations/README.md": "# Supporting Calculations\n\nSupporting data includes protonated nucleotide systems, AMP dimer calculations, and magnesium docking. These are organized by surface type (`canonical_surface` or `aluminium_surface`) and system name to match the primary calculation layout.\n",
        "reference_structures/README.md": "# Reference Structures\n\nStandalone reference structures used by the calculation set, including the aluminium-substituted silicate surface reference.\n",
        "md/README.md": "# Molecular Dynamics Data\n\nCurated MD runs copied from `/data/Seafile`. This folder keeps trajectories, MD inputs/logs, energy and constraint CSVs, ORCA input/output, density metadata, and plots. Large restart, scratch, binary, and basis files remain only in the raw archive.\n\nIncluded systems are AMP frame11 dry MD, AMP frame11 fixed dry MD, AMP frame12 fixed dry MD, UMP frame15 dry-to-wet MD, and UMP frame15 1000 K dry-to-wet MD. No AMP dimer MD run was found under `/data/Seafile/dimer`; AMP dimer optimisation/docking data remains in `data/supporting_calculations/canonical_surface/amp_dimer/`.\n",
    }.items():
        path = DATA_ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def write_manifest() -> None:
    fields = ["system", "surface", "frame", "state", "role", "status", "source_path", "repo_path", "size_bytes", "notes"]
    path = DATA_ROOT / "MANIFEST.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(manifest)


def copy_primary() -> None:
    for nt in NUCLEOTIDES:
        canonical_dst = DATA_ROOT / "primary_calculations" / "canonical_surface" / nt
        copy_docking(SOURCE_ROOT / nt / "docking", canonical_dst, system=nt, surface="canonical", role="primary")
        copy_frame_set(SOURCE_ROOT / nt / "frames", canonical_dst, system=nt, surface="canonical", role="primary")

        al_dst = DATA_ROOT / "primary_calculations" / "aluminium_surface" / nt
        copy_docking(SOURCE_ROOT / nt / "al_surf" / "al_docking", al_dst, system=nt, surface="aluminium", role="primary")
        # GMP has several aluminium docking attempts; retain their top-level provenance.
        if nt == "gmp":
            for name in ("al_docking_complete", "al_docking_complete_nopt_100", "al_docking_complete_nopt_50", "al_docking_no_viable", "al_docking_nopt"):
                src = SOURCE_ROOT / nt / "al_surf" / name
                if src.is_dir():
                    copy_docking(src, al_dst, system=nt, surface="aluminium", role="primary", label=name)
        copy_frame_set(SOURCE_ROOT / nt / "al_surf" / "frames", al_dst, system=nt, surface="aluminium", role="primary")


def copy_supporting() -> None:
    for nt in ("amp", "cmp"):
        sys_name = f"{nt}_protonated"
        canonical_dst = DATA_ROOT / "supporting_calculations" / "canonical_surface" / sys_name
        copy_docking(SOURCE_ROOT / nt / "protonated" / "docking", canonical_dst, system=sys_name, surface="canonical", role="supporting")
        copy_frame_set(SOURCE_ROOT / nt / "protonated" / "frames", canonical_dst, system=sys_name, surface="canonical", role="supporting")

        al_dst = DATA_ROOT / "supporting_calculations" / "aluminium_surface" / sys_name
        copy_docking(SOURCE_ROOT / nt / "al_surf" / "protonated_al_surf" / "docking", al_dst, system=sys_name, surface="aluminium", role="supporting")
        copy_frame_set(SOURCE_ROOT / nt / "al_surf" / "protonated_al_surf" / "frames", al_dst, system=sys_name, surface="aluminium", role="supporting")

    # Magnesium docking on canonical surface
    mg_dst = DATA_ROOT / "supporting_calculations" / "canonical_surface" / "mg"
    copy_docking(SOURCE_ROOT / "mg_docking", mg_dst, system="mg", surface="canonical", role="supporting")

    # AMP Dimer on canonical surface
    dimer_dst = DATA_ROOT / "supporting_calculations" / "canonical_surface" / "amp_dimer"
    copy_docking(SOURCE_ROOT / "dimer" / "docking", dimer_dst, system="amp_dimer", surface="canonical", role="supporting")
    copy_frame_set(SOURCE_ROOT / "dimer" / "frames_gfn" / "frames", dimer_dst, system="amp_dimer", surface="canonical", role="supporting")
    copy_recursive_filtered(SOURCE_ROOT / "dimer" / "geom_opt", dimer_dst / "checks" / "geom_opt", role="supporting", system="amp_dimer", surface="canonical", frame="", state="supporting", suffixes=CHECK_SUFFIXES, notes="dimer geometry optimization checks")
    # Small XYZ references in dimer root
    for name in ("amp2.xyz", "amp2_c.xyz", "amp2_h.xyz", "amp3.xyz"):
        src = SOURCE_ROOT / "dimer" / name
        if src.is_file():
            copy_file(src, dimer_dst / name, role="supporting", system="amp_dimer", surface="canonical", frame="", state="supporting", notes="dimer reference fragment")


def copy_references_and_scripts() -> None:
    copy_flat_calc_dir(SOURCE_ROOT / "al_surf", DATA_ROOT / "reference_structures" / "aluminium_surface", role="reference", system="surface", surface="aluminium", frame="", state="reference")
    scripts_dst = DATA_ROOT / "scripts"
    for name in ("extract_frames.py", "orca_queue.py", "orca_jobs.queue"):
        src = SOURCE_ROOT / name
        if src.is_file():
            copy_file(src, scripts_dst / name, role="script", system="", surface="", frame="", state="script")


def copy_md_data() -> None:
    for system, frame, run_name, src_dir in MD_RUNS:
        dst_dir = DATA_ROOT / "md" / system / frame / run_name
        name_overrides = {"trajectory.xyz": f"{system}_{frame}_{run_name}_trajectory.xyz"}
        copy_recursive_filtered(
            src_dir,
            dst_dir,
            role="md",
            system=system,
            surface="canonical",
            frame=frame,
            state=run_name,
            suffixes=MD_SUFFIXES,
            notes="curated molecular dynamics run; raw scratch and restart files excluded",
            name_overrides=name_overrides,
        )


def main() -> None:
    global include_swarms
    parser = argparse.ArgumentParser(description="Create a curated thesis data layout from /data/Seafile.")
    parser.add_argument("--include-swarms", action="store_true", help="include large docking *.all.swarm.xyz files")
    args = parser.parse_args()
    include_swarms = args.include_swarms

    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"Source directory not found: {SOURCE_ROOT}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    copy_primary()
    copy_supporting()
    copy_references_and_scripts()
    copy_md_data()
    write_readmes()
    write_manifest()
    print(f"Copied {len(manifest)} files into {DATA_ROOT}")


if __name__ == "__main__":
    main()
