#!/usr/bin/env bash
# run_full_export.sh — export every recording that has no CSV yet, memory-safely.
#
#   ./scripts/v6/run_full_export.sh CORPUS_ROOT OUT_DIR [SMALL_GB=1.2] [SMALL_LANES=2]
#
#   PRIORITY_FROM=labels.csv ./scripts/v6/run_full_export.sh ...
#       export the recordings named in labels.csv FIRST, then everything else.
#
# ONLY THE MISSING ONES. batch_export.py skips any recording whose CSV already
# exists, so this always picks up where the last run stopped. Nothing is redone.
#
# WHY RAM, NOT CORES, SETS THE PARALLELISM. Export memory is ~4.25x the .paudio
# size (5.6 GB peak measured for a 1.32 GB file). The 16-09-2026 run used six
# parallel shards and macOS jetsam silently killed four of them. So each phase is
# two passes:
#
#   pass 1  files <= SMALL_GB in SMALL_LANES parallel lanes
#           (2 lanes x ~5.1 GB for a 1.2 GB file ≈ 10 GB — fits a 16 GB Mac)
#   pass 2  files >  SMALL_GB one at a time, largest first
#           (a 2.2 GB file peaks near 9.4 GB; it cannot share the machine)
#
# More lanes than that does not go faster — it gets processes killed.
#
# caffeinate -i keeps the Mac awake (the exporter's own wake lock fails under
# nohup); -m keeps the DISK awake — on 16-09-2026 the external drive dropped out
# at ~21:01 mid-run, and idle disk sleep is the likeliest cause.
# The export mode is `unfiltered` on purpose: every other mode drops
# fit_valid == 0 before Stage 2 — a measured 9.0 % of clicks. See batch_export.py.
set -euo pipefail

ROOT="${1:?usage: run_full_export.sh CORPUS_ROOT OUT_DIR [SMALL_GB] [SMALL_LANES]}"
OUT="${2:?usage: run_full_export.sh CORPUS_ROOT OUT_DIR [SMALL_GB] [SMALL_LANES]}"
SMALL_GB="${3:-1.2}"
LANES="${4:-2}"
PRIORITY_FROM="${PRIORITY_FROM:-}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$REPO/.venv/bin/python"
EXP="$REPO/scripts/v6/batch_export.py"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
[ -d "$ROOT" ] || { echo "ERROR: cannot read $ROOT" >&2; exit 2; }
if [ -n "$PRIORITY_FROM" ] && [ ! -f "$PRIORITY_FROM" ]; then
  echo "ERROR: PRIORITY_FROM file not found: $PRIORITY_FROM" >&2; exit 2
fi

STAMP="$(date +%Y%m%d_%H%M)"
DRIVER="$LOGS/driver_${STAMP}.sh"
DRIVER_LOG="$LOGS/run_${STAMP}.log"

# The driver is written out as a plain script (not an inline bash -c) so it can be
# read, and so its quoting is ordinary quoting. Values are fixed at launch time.
cat > "$DRIVER" <<EOF
#!/usr/bin/env bash
set -u
export QT_QPA_PLATFORM=offscreen
ROOT=$(printf '%q' "$ROOT")
OUT=$(printf '%q' "$OUT")
LOGS=$(printf '%q' "$LOGS")
PY=$(printf '%q' "$PY")
EXP=$(printf '%q' "$EXP")

phase() {   # phase NAME [extra batch_export args...]
  local name="\$1"; shift
  echo "[\$(date +%H:%M)] \$name / pass 1: <= $SMALL_GB GB, $LANES lanes"
  for i in \$(seq 1 $LANES); do
    "\$PY" "\$EXP" "\$ROOT" -o "\$OUT" --export-mode unfiltered \\
      --max-gb $SMALL_GB --shard "\$i/$LANES" "\$@" \\
      > "\$LOGS/${STAMP}_\${name}_pass1_lane\$i.log" 2>&1 &
  done
  wait
  echo "[\$(date +%H:%M)] \$name / pass 2: > $SMALL_GB GB, one at a time, largest first"
  "\$PY" "\$EXP" "\$ROOT" -o "\$OUT" --export-mode unfiltered \\
    --min-gb $SMALL_GB --order largest-first "\$@" \\
    > "\$LOGS/${STAMP}_\${name}_pass2.log" 2>&1
  echo "[\$(date +%H:%M)] \$name done"
}

EOF
if [ -n "$PRIORITY_FROM" ]; then
  printf 'phase priority --only-from %q\n' "$PRIORITY_FROM" >> "$DRIVER"
fi
cat >> "$DRIVER" <<'EOF'
phase rest
echo "[$(date +%H:%M)] finished. Still missing (should be 0):"
"$PY" "$EXP" "$ROOT" -o "$OUT" --dry-run 2>&1 | grep -E "to export"
EOF
chmod +x "$DRIVER"

nohup caffeinate -im "$DRIVER" > "$DRIVER_LOG" 2>&1 &

cat <<EOF
Launched (pid $!) — caffeinated and detached; you can close the terminal.

  corpus   : $ROOT
  output   : $OUT
  priority : ${PRIORITY_FROM:-none}
  pass 1   : files <= $SMALL_GB GB, $LANES lanes in parallel
  pass 2   : larger files, one at a time, largest first

  overview : cat "$DRIVER_LOG"
  progress : tail -f "$LOGS"/${STAMP}_*.log | grep -E 'rows|ETA'
  count    : find "$OUT/CSVs" -name '*_candidates.csv' | wc -l
  stop     : pkill -f "$DRIVER"; pkill -f batch_export.py    # loses only the file in progress
  resume   : run this exact command again
EOF
