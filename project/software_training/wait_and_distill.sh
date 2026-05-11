#!/usr/bin/env bash
# wait_and_distill.sh
# Polls until night Optuna finishes, then runs the full distillation pipeline:
#   day teacher → day student → night teacher → night student
#
# Usage:
#   bash project/software_training/wait_and_distill.sh
#
# Checkpoints produced:
#   bnn_teacher_day.pth      bnn_distilled_day.pth
#   bnn_teacher_night.pth    bnn_distilled_night.pth

set -euo pipefail

NIGHT_JSON="project/software_training/optuna_night.json"
DAY_JSON="project/software_training/optuna_day.json"
PYTHON=".venv/bin/python3"

echo "=== wait_and_distill.sh ==="
echo "Waiting for night Optuna to finish ($NIGHT_JSON)..."

while true; do
    if [[ -f "$NIGHT_JSON" ]] && python3 -c "
import json, sys
d = json.load(open('$NIGHT_JSON'))
sys.exit(0 if 'best_params' in d else 1)
" 2>/dev/null; then
        echo "Night Optuna complete."
        break
    fi
    echo "  $(date '+%H:%M:%S')  still waiting..."
    sleep 60
done

# ── Extract best params ───────────────────────────────────────────────────────
echo ""
echo "Extracting best params..."

read DAY_LR DAY_WD DAY_BW DAY_GC < <(python3 - <<'EOF'
import json
d = json.load(open("project/software_training/optuna_day.json"))["best_params"]
print(d["lr"], d["weight_decay"], d["blank_weight"], d["grad_clip"])
EOF
)

read NIGHT_LR NIGHT_WD NIGHT_BW NIGHT_GC < <(python3 - <<'EOF'
import json
d = json.load(open("project/software_training/optuna_night.json"))["best_params"]
print(d["lr"], d["weight_decay"], d["blank_weight"], d["grad_clip"])
EOF
)

echo "Day   params: lr=$DAY_LR  wd=$DAY_WD  bw=$DAY_BW  gc=$DAY_GC"
echo "Night params: lr=$NIGHT_LR  wd=$NIGHT_WD  bw=$NIGHT_BW  gc=$NIGHT_GC"
echo ""

# ── Day: teacher then student ─────────────────────────────────────────────────
echo "=========================================="
echo "DAY — Phase 1: Train teacher"
echo "=========================================="
$PYTHON project/software_training/distill.py teacher \
    --data-root project/data_20k_day \
    --epochs 30 \
    --teacher-checkpoint bnn_teacher_day.pth \
    --blank-weight "$DAY_BW"

echo ""
echo "=========================================="
echo "DAY — Phase 2: Distill student"
echo "=========================================="
$PYTHON project/software_training/distill.py student \
    --data-root project/data_20k_day \
    --epochs 50 \
    --teacher-checkpoint bnn_teacher_day.pth \
    --student-checkpoint bnn_distilled_day.pth \
    --blank-weight "$DAY_BW" \
    --lr "$DAY_LR" \
    --weight-decay "$DAY_WD" \
    --grad-clip "$DAY_GC"

# ── Night: teacher then student ───────────────────────────────────────────────
echo ""
echo "=========================================="
echo "NIGHT — Phase 1: Train teacher"
echo "=========================================="
$PYTHON project/software_training/distill.py teacher \
    --data-root project/data_20k_night \
    --epochs 30 \
    --teacher-checkpoint bnn_teacher_night.pth \
    --blank-weight "$NIGHT_BW"

echo ""
echo "=========================================="
echo "NIGHT — Phase 2: Distill student"
echo "=========================================="
$PYTHON project/software_training/distill.py student \
    --data-root project/data_20k_night \
    --epochs 50 \
    --teacher-checkpoint bnn_teacher_night.pth \
    --student-checkpoint bnn_distilled_night.pth \
    --blank-weight "$NIGHT_BW" \
    --lr "$NIGHT_LR" \
    --weight-decay "$NIGHT_WD" \
    --grad-clip "$NIGHT_GC"

echo ""
echo "=== All done ==="
echo "Checkpoints: bnn_distilled_day.pth  bnn_distilled_night.pth"
