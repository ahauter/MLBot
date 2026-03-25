#!/usr/bin/env bash
# Run 10 independent baseline seeds (0-9).
#
# Usage:
#   # Run all seeds in parallel:
#   bash training/run_baseline_seeds.sh
#
#   # With Optuna-tuned params:
#   bash training/run_baseline_seeds.sh --params-from optuna_baseline.db
#
#   # Run specific seeds:
#   bash training/run_baseline_seeds.sh --seeds "0 1 2"

set -euo pipefail

EXTRA_ARGS=""
SEEDS="0 1 2 3 4 5 6 7 8 9"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "=== Baseline Experiment: 10-Seed Run ==="
echo "Seeds: $SEEDS"
echo "Extra args: $EXTRA_ARGS"
echo ""

PIDS=()

for seed in $SEEDS; do
    echo "Launching seed $seed..."
    python training/train.py \
        --seed "$seed" \
        --model-dir "models/baseline" \
        $EXTRA_ARGS \
        > "models/baseline/seed_${seed}.log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} seeds launched. PIDs: ${PIDS[*]}"
echo "Logs: models/baseline/seed_*.log"
echo ""
echo "To monitor: tail -f models/baseline/seed_0.log"
echo "To stop all: kill ${PIDS[*]}"

# Wait for all to complete
wait "${PIDS[@]}"
echo "All seeds complete."
