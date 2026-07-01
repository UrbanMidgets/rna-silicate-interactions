#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"
MODE="status"
FORCE=0
ORCA_PLOT="${ORCA_PLOT:-orca_plot}"
RESOLUTION="80 80 80"
ONLY=""

usage() {
  cat <<'EOF'
Usage:
  ./generate_esp_cubes.sh status
  ./generate_esp_cubes.sh run [--resolution "NX NY NZ"] [--only NAME] [--force]

Purpose:
  Batch-generate electron-density and electrostatic-potential cube files from
  completed ORCA geometry optimisations.

Notes:
  - This script uses the optimisation .gbw directly; no extra single-point job.
  - It only processes folders containing *_geom.inp.
  - It requires the matching *_geom.out to contain ORCA TERMINATED NORMALLY.
  - It requires the matching *_geom.gbw file to exist.
  - It runs orca_plot from inside each calculation directory because ORCA also
    needs local sidecar files such as *.densitiesinfo.
  - For ESP plots it selects plot type 43 and the matching <stem>.scfp density.
  - For density plots it selects plot type 2 and the matching <stem>.scfp density.
  - The ORCA 6.1 ESP tutorial uses 100 100 100 as a publication-quality
    example. This script defaults to 80 80 80 for batch generation; pass
    --resolution "100 100 100" to regenerate selected systems at that setting.
  - ORCA reports Grid3d/Cube as the default output format in this workflow, so
    the script does not explicitly select menu option 5 -> 7.

Examples:
  ./generate_esp_cubes.sh status
  ./generate_esp_cubes.sh run
  ./generate_esp_cubes.sh run --only amp
  ./generate_esp_cubes.sh run --resolution "100 100 100" --force
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    status|run)
      MODE="$1"
      shift
      ;;
    --resolution)
      RESOLUTION="${2:-}"
      shift 2
      ;;
    --only)
      ONLY="${2:-}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v "$ORCA_PLOT" >/dev/null 2>&1; then
  echo "ERROR: orca_plot not found. Set ORCA_PLOT=/path/to/orca_plot if needed." >&2
  exit 1
fi

if [[ ! "$RESOLUTION" =~ ^[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+$ ]]; then
  echo "ERROR: resolution must be three integers, for example: 80 80 80" >&2
  exit 2
fi

shopt -s nullglob

found=0
for inp in "$ROOT"/*/*_geom.inp; do
  found=1
  dir="$(dirname "$inp")"
  stem="$(basename "$inp" .inp)"
  name="$(basename "$dir")"

  if [[ -n "$ONLY" && "$name" != "$ONLY" && "$stem" != "$ONLY" ]]; then
    continue
  fi

  out="$dir/$stem.out"
  gbw="$dir/$stem.gbw"
  density_cube="$dir/${stem}_density.cube"
  esp_cube="$dir/${stem}_esp.cube"
  density_name="$stem.scfp"
  raw_density_cube="$dir/${stem}.eldens.cube"
  raw_esp_cube="$dir/${density_name}.esp.cube"

  if [[ ! -f "$out" ]]; then
    printf '%-24s %s\n' "$name" "WAITING: no $stem.out"
    continue
  fi

  if ! grep -q "ORCA TERMINATED NORMALLY" "$out"; then
    printf '%-24s %s\n' "$name" "WAITING: optimisation not normally terminated"
    continue
  fi

  if [[ ! -f "$gbw" ]]; then
    printf '%-24s %s\n' "$name" "ERROR: missing $stem.gbw"
    continue
  fi

  if [[ "$MODE" == "status" ]]; then
    if [[ -f "$density_cube" && -f "$esp_cube" ]]; then
      printf '%-24s %s\n' "$name" "DONE: cubes present"
    else
      printf '%-24s %s\n' "$name" "READY: $stem.gbw available"
    fi
    continue
  fi

  if [[ $FORCE -eq 0 && -f "$density_cube" && -f "$esp_cube" ]]; then
    printf '%-24s %s\n' "$name" "SKIP: cubes already present"
    continue
  fi

  echo "Generating density cube for $stem"
  (
    cd "$dir"
    printf '1\n2\ny\n4\n%s\n11\n12\n' "$RESOLUTION" | "$ORCA_PLOT" "$stem.gbw" -i
  )

  if [[ ! -f "$raw_density_cube" ]]; then
    echo "ERROR: expected density cube not produced: $raw_density_cube" >&2
    exit 1
  fi
  mv -f "$raw_density_cube" "$density_cube"

  echo "Generating ESP cube for $stem"
  (
    cd "$dir"
    printf '1\n43\n%s\n4\n%s\n11\n12\n' "$density_name" "$RESOLUTION" | "$ORCA_PLOT" "$stem.gbw" -i
  )

  if [[ ! -f "$raw_esp_cube" ]]; then
    echo "ERROR: expected ESP cube not produced: $raw_esp_cube" >&2
    exit 1
  fi
  mv -f "$raw_esp_cube" "$esp_cube"

  printf '%-24s %s\n' "$name" "DONE: cubes generated"
done

if [[ $found -eq 0 ]]; then
  echo "No *_geom.inp files found under $ROOT/*/." >&2
  exit 1
fi
