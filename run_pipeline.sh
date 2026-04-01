#!/usr/bin/env bash
# Full pipeline: tune → evaluate → baselines → report → plot
#
# Repetitions:
#   SA / RRHC / Tabu : 5 topology seeds × 30 method reps per config (all 9 configs)
#   H-MAB            : 1 topology seed of 2×2 only, 30 method reps
#   DCF              : 1 topology seed of 2×2 only,  5 method reps
#   T-Optimal        : 5 topology seeds per config, 1 rep, ≤ 4×4 only
#
# Usage:
#   bash run_pipeline.sh           # full run with tuning
#   bash run_pipeline.sh --no-tune # skip tuning, use existing configs
set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate

# ── Parameters ────────────────────────────────────────────────────────────────
N_SCENARIO_SEEDS=5  # topology seeds (scenario realizations) per config
N_REPS_METHODS=30   # method repetitions per scenario: SA, RRHC, Tabu, H-MAB
N_REPS_DCF=5        # method repetitions per scenario: DCF
N_STEPS=2000
SEED=42
SKIP_TUNE=0

for arg in "$@"; do
    [[ "$arg" == "--no-tune" ]] && SKIP_TUNE=1
done

mkdir -p results

# ── Step 1: Hyperparameter tuning ─────────────────────────────────────────────
if [[ $SKIP_TUNE -eq 0 ]]; then
    echo ""
    echo "======================================================"
    echo " Step 1: Tune hyperparameters"
    echo "======================================================"

    python -m mapc_mh.tune \
        --method sa --n_trials 100 --n_steps "$N_STEPS"

    python -m mapc_mh.tune \
        --method rrhc --n_trials 100 --n_steps "$N_STEPS"

    python -m mapc_mh.tune \
        --method tabu --n_trials 100 --n_steps "$N_STEPS"
else
    echo ""
    echo "Skipping tuning — using existing configs in mapc_mh/methods/configs/"
fi

# ── Step 2: Evaluate SA / RRHC / Tabu ────────────────────────────────────────
echo ""
echo "======================================================"
echo " Step 2: Evaluate methods (${N_SCENARIO_SEEDS} topology seeds × ${N_REPS_METHODS} reps, 9 configs)"
echo "======================================================"

python -m mapc_mh.evaluate \
    --n_seeds "$N_SCENARIO_SEEDS" \
    --n_reps  "$N_REPS_METHODS" \
    --n_steps "$N_STEPS" \
    --seed    "$SEED" \
    --output  results/evaluation.json

# ── Step 3a: H-MAB baseline ───────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Step 3a: H-MAB baseline (${N_REPS_METHODS} reps, 2×2 only)"
echo "======================================================"

python -m mapc_mh.baselines \
    --agents  h_mab \
    --n_seeds 1 \
    --n_reps  "$N_REPS_METHODS" \
    --n_steps "$N_STEPS" \
    --seed    "$SEED" \
    --only_first_2x2 \
    --output  results/baselines_h_mab.json

# ── Step 3b: DCF baseline ─────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Step 3b: DCF baseline (${N_REPS_DCF} reps, 2×2 only)"
echo "======================================================"

python -m mapc_mh.baselines \
    --agents  dcf \
    --n_seeds 1 \
    --n_reps  "$N_REPS_DCF" \
    --n_steps "$N_STEPS" \
    --seed    "$SEED" \
    --only_first_2x2 \
    --output  results/baselines_dcf.json

# ── Step 3c: T-Optimal baseline (4×4 and smaller only) ───────────────────────
echo ""
echo "======================================================"
echo " Step 3c: T-Optimal baseline (1 rep, ${N_SCENARIO_SEEDS} seeds, ≤ 4×4)"
echo "======================================================"

python -m mapc_mh.baselines \
    --agents            t_optimal \
    --n_seeds           "$N_SCENARIO_SEEDS" \
    --n_steps           "$N_STEPS" \
    --seed              "$SEED" \
    --t_optimal_max_aps 16 \
    --output            results/baselines_t_optimal.json

# ── Step 4: Statistical comparison report ────────────────────────────────────
echo ""
echo "======================================================"
echo " Step 4: Statistical comparison report"
echo "======================================================"

python -m mapc_mh.report \
    --input results/evaluation.json \
            results/baselines_h_mab.json \
            results/baselines_dcf.json \
            results/baselines_t_optimal.json \
    --alpha 0.05

# ── Step 5: Plots ─────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Step 5: Generate convergence plots"
echo "======================================================"

python -m mapc_mh.plot \
    --input results/evaluation.json \
            results/baselines_h_mab.json \
            results/baselines_dcf.json \
            results/baselines_t_optimal.json \
    --output results/convergence

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo " Pipeline complete!"
echo "======================================================"
echo " results/evaluation.json           SA / RRHC / Tabu  (5 seeds × 30 reps)"
echo " results/baselines_h_mab.json      H-MAB             (30 reps, 2×2 only)"
echo " results/baselines_dcf.json        DCF               ( 5 reps, 2×2 only)"
echo " results/baselines_t_optimal.json  T-Optimal         (5 seeds, ≤4×4)"
echo " results/convergence.{pdf,png,csv} convergence plots"
