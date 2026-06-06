#!/usr/bin/env bash
# Extract timing, area, and power summary from a completed OpenLane 2 run.
# Usage: ./extract_results.sh [run_dir]
# If run_dir is omitted, uses the most recent RUN_* directory.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="$SCRIPT_DIR/runs"

if [[ $# -ge 1 ]]; then
    RUN_DIR="$1"
else
    RUN_DIR="$(ls -dt "$RUNS_DIR"/RUN_* 2>/dev/null | head -1)"
    if [[ -z "$RUN_DIR" ]]; then
        echo "No RUN_* directories found in $RUNS_DIR" >&2
        exit 1
    fi
fi

echo "=== OpenLane Results: $(basename "$RUN_DIR") ==="
echo ""

# ── Synthesis (Yosys) ─────────────────────────────────────────────────────────
SYNTH_LOG="$(ls "$RUN_DIR"/06-yosys-synthesis/*.log 2>/dev/null | head -1)"
if [[ -n "$SYNTH_LOG" && -f "$SYNTH_LOG" ]]; then
    echo "── Synthesis ──"
    grep -E "Number of (cells|wires|wire bits|memories|memory bits|processes|flip-flops)|Chip area" "$SYNTH_LOG" 2>/dev/null || true
    echo ""
fi

# ── Floorplan / die area ──────────────────────────────────────────────────────
FP_LOG="$(ls "$RUN_DIR"/13-openroad-floorplan/*.log 2>/dev/null | head -1)"
if [[ -n "$FP_LOG" && -f "$FP_LOG" ]]; then
    echo "── Floorplan ──"
    grep -E "Die area|Core area|Utilization|IFP-" "$FP_LOG" 2>/dev/null | head -10 || true
    echo ""
fi

# ── Global placement overflow ─────────────────────────────────────────────────
GP_LOG="$(ls "$RUN_DIR"/27-openroad-globalplacement/*.log 2>/dev/null | head -1)"
if [[ -n "$GP_LOG" && -f "$GP_LOG" ]]; then
    echo "── Global Placement ──"
    grep -E "Iter.*overflow|TotalRouteOverflow|GPL-0301" "$GP_LOG" 2>/dev/null | tail -5 || true
    echo ""
fi

# ── Static timing (worst slack, WNS, TNS) ─────────────────────────────────────
# Look in final STA step (highest-numbered openroad-sta* dir)
STA_DIR="$(ls -d "$RUN_DIR"/*openroad-sta* 2>/dev/null | sort | tail -1)"
if [[ -n "$STA_DIR" ]]; then
    STA_LOG="$(ls "$STA_DIR"/nom_tt_025C_1v80/sta.log 2>/dev/null | head -1)"
    if [[ -n "$STA_LOG" && -f "$STA_LOG" ]]; then
        echo "── Timing (TT 1.8V 25°C) ──"
        grep -E "^(wns|tns|worst|slack|startpoint|endpoint)" "$STA_LOG" 2>/dev/null | head -20 || true
        echo ""
    fi
fi

# ── Routing (DRC violations) ──────────────────────────────────────────────────
DRC_LOG="$(ls "$RUN_DIR"/*magic-drc*/*.log 2>/dev/null | head -1)"
if [[ -n "$DRC_LOG" && -f "$DRC_LOG" ]]; then
    echo "── DRC ──"
    grep -E "Total DRC|violations" "$DRC_LOG" 2>/dev/null | head -5 || true
    echo ""
fi

# ── Power (OpenROAD power analysis) ──────────────────────────────────────────
# Look in any step whose dir contains "power" or final CTS/routing STA
POWER_LOG="$(ls "$RUN_DIR"/*openroad-*power*/*.log 2>/dev/null | head -1)"
if [[ -z "$POWER_LOG" ]]; then
    # Fall back: look for PSM/power lines in the last STA log
    POWER_LOG="$STA_LOG"
fi
if [[ -n "$POWER_LOG" && -f "$POWER_LOG" ]]; then
    echo "── Power ──"
    grep -E "(Total|Dynamic|Leakage|Internal|Switching) [Pp]ower|mW|uW" "$POWER_LOG" 2>/dev/null | head -10 || true
    echo ""
fi

# ── Final metrics JSON (OpenLane 2 summary) ───────────────────────────────────
METRICS_JSON="$(ls "$RUN_DIR"/final_summary_report.json \
                   "$RUN_DIR"/metrics.json \
                   "$RUN_DIR"/resolved.json 2>/dev/null | head -1)"
if [[ -n "$METRICS_JSON" && -f "$METRICS_JSON" ]]; then
    echo "── Key Metrics (from $(basename "$METRICS_JSON")) ──"
    # Extract the most useful fields if python3 available
    python3 - <<'PYEOF' "$METRICS_JSON" 2>/dev/null || cat "$METRICS_JSON" | head -60
import json, sys
with open(sys.argv[1]) as f:
    m = json.load(f)
keys = [
    "design__instance__count", "design__instance__count__stdcell",
    "design__core__area", "design__die__area",
    "clock__skew__worst_hold", "clock__skew__worst_setup",
    "timing__hold__ws", "timing__setup__ws",
    "timing__hold__tns", "timing__setup__tns",
    "power__total", "power__internal", "power__switching", "power__leakage",
    "route__wirelength__estimated", "antenna__violating__nets",
    "magic__drc_errors",
]
for k in keys:
    if k in m:
        print(f"  {k}: {m[k]}")
PYEOF
    echo ""
fi

echo "=== Done ==="
