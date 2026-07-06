#!/usr/bin/env bash
# Rigor-upgrade run: expanded slip study, Phase B seed repeats,
# 5-seed/7-bisect shape+material tables, sweep variance audit.
# Total worker budget: 10 (studies run sequentially; step 3 splits 5+5).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=results/rigor_run.log

echo "=== rigor run started $(date) ===" >> "$LOG"

# Preserve the published 2-seed shape/material numbers before overwriting.
mkdir -p results/archive_2seed
cp -n results/shape_*.json results/material_*.json results/archive_2seed/ 2>/dev/null

step () {
  echo "--- [$(date +%H:%M:%S)] $1 ---" >> "$LOG"
}

step "1/4 slip detector, expanded (90 episodes)"
nice -n 10 $PY scripts/eval_slip_big.py --workers 10 >> "$LOG" 2>&1 \
  || echo "STEP-FAILED eval_slip_big" >> "$LOG"

step "2/4 Phase B seed repeats (50 runs)"
nice -n 10 $PY scripts/phaseB_seeds.py --workers 10 >> "$LOG" 2>&1 \
  || echo "STEP-FAILED phaseB_seeds" >> "$LOG"

step "3/4 shapes + materials, 5 seeds / 7 bisect (concurrent 5+5)"
nice -n 10 $PY scripts/eval_shapes.py --shape all --seeds 5 --n-bisect 7 \
  --workers 5 >> "$LOG" 2>&1 &
SHAPES_PID=$!
nice -n 10 $PY scripts/eval_materials.py --level all --seeds 5 --n-bisect 7 \
  --workers 5 >> "$LOG" 2>&1 &
MATS_PID=$!
wait $SHAPES_PID || echo "STEP-FAILED eval_shapes" >> "$LOG"
wait $MATS_PID   || echo "STEP-FAILED eval_materials" >> "$LOG"

step "4/4 sweep variance audit (60 bisections)"
nice -n 10 $PY scripts/run_audit.py --workers 10 >> "$LOG" 2>&1 \
  || echo "STEP-FAILED run_audit" >> "$LOG"

echo "=== rigor run finished $(date) ===" >> "$LOG"
