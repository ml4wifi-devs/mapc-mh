#!/usr/bin/env bash
# Full F-Optimal pipeline: tune → evaluate → baseline → report → plot
#
# Method: F-CG only.
# Repetitions:
#   F-CG       : 5 topology seeds × 30 method reps per config (configs up to 3×3)
#   F-Optimal  : 5 topology seeds per config, 1 rep, ≤ 3×3 only
#
# Usage:
#   bash run_pipeline_f.sh           # full run with tuning
#   bash run_pipeline_f.sh --no-tune # skip tuning, use existing configs
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# ── Parameters ────────────────────────────────────────────────────────────────
N_SCENARIO_SEEDS=5  # topology seeds (scenario realizations) per config
N_REPS_METHODS=30   # method repetitions per scenario: F-CG
N_STEPS=300         # outer CG iterations (budget knob)
SEED=42
F_OPTIMAL_MAX_APS=9 # restrict to ≤ 3×3 = 9 stations
SKIP_TUNE=0

for arg in "$@"; do
    [[ "$arg" == "--no-tune" ]] && SKIP_TUNE=1
done

mkdir -p results_f

# ── Step 1: Hyperparameter tuning ─────────────────────────────────────────────
if [[ $SKIP_TUNE -eq 0 ]]; then
    echo ""
    echo "=========================================================================="
    echo " Step 1: Tune hyperparameters (F-CG)"
    echo "=========================================================================="

    python -m mapc_mh.tune \
        --method f_cg --n_trials 100 --n_steps "$N_STEPS"
else
    echo ""
    echo "Skipping tuning — using existing configs in mapc_mh/methods/fairness/configs/"
fi

# ── Step 2: Evaluate F-CG ─────────────────────────────────────────────────────
echo ""
echo "=========================================================================="
echo " Step 2: Evaluate F-CG (${N_SCENARIO_SEEDS} topology seeds × ${N_REPS_METHODS} reps)"
echo "=========================================================================="

python -m mapc_mh.evaluate \
    --skip_t \
    --methods f_cg \
    --n_seeds "$N_SCENARIO_SEEDS" \
    --n_reps  "$N_REPS_METHODS" \
    --n_steps "$N_STEPS" \
    --seed    "$SEED" \
    --output  results_f/evaluation.json

# ── Step 3: F-Optimal baseline (3×3 and smaller only) ─────────────────────────
echo ""
echo "=========================================================================="
echo " Step 3: F-Optimal baseline (1 rep, ${N_SCENARIO_SEEDS} seeds, ≤ 3×3)"
echo "=========================================================================="

python -m mapc_mh.baselines \
    --agents            f_optimal \
    --n_seeds           "$N_SCENARIO_SEEDS" \
    --n_reps            1 \
    --n_steps           "$N_STEPS" \
    --seed              "$SEED" \
    --f_optimal_max_aps "$F_OPTIMAL_MAX_APS" \
    --output            results_f/baselines_f_optimal.json

# ── Step 4: Statistical comparison report ─────────────────────────────────────
echo ""
echo "=========================================================================="
echo " Step 4: Statistical comparison report"
echo "=========================================================================="

python -m mapc_mh.report \
    --input results_f/evaluation.json \
            results_f/baselines_f_optimal.json \
    --alpha 0.05

# ── Step 5: Plots ─────────────────────────────────────────────────────────────
echo ""
echo "=========================================================================="
echo " Step 5: Generate convergence plots"
echo "=========================================================================="

python -m mapc_mh.plot \
    --input  results_f/evaluation.json \
             results_f/baselines_f_optimal.json \
    --output results_f/convergence

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================================================="
echo " F-Optimal pipeline complete!"
echo "=========================================================================="
echo " results_f/evaluation.json             F-CG              (5 seeds × 30 reps)"
echo " results_f/baselines_f_optimal.json    F-Optimal         (5 seeds, ≤3×3)"
echo " results_f/convergence.{pdf,png,csv}   convergence plots"
