# Cooling, Climbing, and Forgetting: Trajectory-Based Metaheuristics for Coordinated Spatial Reuse

We propose using **trajectory-based metaheuristics** — Simulated Annealing (SA), Random Restart Hill Climbing (RRHC), and Tabu Search — to optimize coordinated spatial reuse (Co-SR) scheduling in IEEE 802.11bn (Wi-Fi 8) networks. Starting from a random configuration, each method iteratively proposes and evaluates neighboring configurations, converging to high-throughput solutions without a surrogate model or offline training.

## How It Works

1. **Initialization** — Sample a random Co-SR configuration (active APs, STA selection, MCS, transmit power) and evaluate its throughput directly in the simulator
2. **Neighbor proposal** — Perturb the current configuration by mutating one parameter of one AP
3. **Acceptance** — Each method applies its own acceptance criterion (Metropolis for SA, strict improvement for RRHC, tabu list for Tabu Search)
4. **Top-N tracking** — Maintain the best *N* configurations seen across all steps for round-robin deployment

## Installation

Requires Python >= 3.12. We recommend using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/ml4wifi-devs/mapc-mh.git
cd mapc-mh
uv venv
source .venv/bin/activate
uv sync
```

## Usage

### Hyperparameter Tuning

Tune hyperparameters for a single method using Optuna (TPE sampler):

```bash
python -m mapc_mh.tune --method t_sa   --n_trials 100
python -m mapc_mh.tune --method t_rrhc --n_trials 50
python -m mapc_mh.tune --method t_tabu --n_trials 100
```

By default tunes over 1 scenario seed per config (9 scenarios total) and saves results to `mapc_mh/methods/configs/best_params_{method}.json`.

### Evaluation

Evaluate all methods on all scenarios using tuned (or default) hyperparameters:

```bash
python -m mapc_mh.evaluate --n_seeds 5 --n_reps 30
```

`--n_seeds` controls the number of topology realizations per config; `--n_reps` controls how many times each realization is repeated with different method seeds (to measure algorithm variance). Results are saved to `results/evaluation.json`.

### Baselines

Run baselines (each saves to a separate file):

```bash
# H-MAB: 30 reps on one 2×2 realization
python -m mapc_mh.baselines --agents h_mab --n_seeds 1 --n_reps 30 --only_first_2x2 \
    --output results/baselines_h_mab.json

# DCF: 5 reps on one 2×2 realization
python -m mapc_mh.baselines --agents dcf --n_seeds 1 --n_reps 5 --only_first_2x2 \
    --output results/baselines_dcf.json

# T-Optimal: 1 rep per topology seed, configs ≤ 4×4 only
python -m mapc_mh.baselines --agents t_optimal --n_seeds 5 \
    --output results/baselines_t_optimal.json
```

### Statistical Report

Print a comparison table (mean ± std) and pairwise Mann-Whitney U significance tests from saved results:

```bash
python -m mapc_mh.report --input results/evaluation.json \
    results/baselines_h_mab.json results/baselines_dcf.json results/baselines_t_optimal.json
```

### Plotting

Plot convergence curves (best throughput vs. step) with mean ± 95% CI:

```bash
python -m mapc_mh.plot --input results/evaluation.json \
    results/baselines_h_mab.json results/baselines_dcf.json results/baselines_t_optimal.json
```

Produces a PDF, PNG, and CSV compatible with TikZ/pgfplots.

## Project Structure

```
mapc_mh/
├── env.py           # JAX CPU environment setup (imported first by all modules)
├── config.py        # NetworkConfig, ScenarioInfo, array conversion helpers
├── scenarios.py     # Scenario definitions and build_scenarios(n_seeds)
├── evaluate.py      # Run all methods on all scenarios, save histories
├── baselines.py     # H-MAB, DCF, and T-Optimal baselines
├── report.py        # Statistical comparison from saved results
├── plot.py          # Convergence plots with CI bands and CSV export
├── tune.py          # Optuna hyperparameter search
└── methods/
    ├── core.py      # Shared logic: neighbor generation, top-N buffer, Result
    ├── sa.py        # Simulated Annealing
    ├── rrhc.py      # Random Restart Hill Climbing
    ├── tabu.py      # Tabu Search
    └── configs/     # Default hyperparameter configs (tracked by git)
```

## Citation

```bibtex
@article{wojnar2026cooling,
  title={Cooling, Climbing, and Forgetting: Trajectory-Based Metaheuristics for Coordinated Spatial Reuse},
  author={Wojnar, Maksymilian},
  year={2026}
}
```
