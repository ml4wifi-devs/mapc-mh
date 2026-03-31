# Trajectory-Based Metaheuristics for MAPC

We propose using **trajectory-based metaheuristics** — Simulated Annealing (SA), Random Restart Hill Climbing (RRHC), and Tabu Search — to optimize coordinated spatial reuse (Co-SR) scheduling in IEEE 802.11bn (Wi-Fi 8) networks. Starting from a random configuration, each method iteratively proposes and evaluates neighboring configurations, converging to high-throughput solutions without a surrogate model or offline training.

## How It Works

1. **Initialization** — Sample a random Co-SR configuration (active APs, STA selection, MCS, transmit power) and evaluate its throughput directly in the simulator
2. **Neighbor proposal** — Perturb the current configuration by mutating one parameter of one AP
3. **Acceptance** — Each method applies its own acceptance criterion (Metropolis for SA, strict improvement for RRHC, tabu list for Tabu Search)
4. **Top-N tracking** — Maintain the best *N* configurations seen across all steps for round-robin deployment

## Installation

Requires Python >= 3.12. We recommend using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/ml4wifi-devs/mapc-sa.git
cd mapc-sa
uv venv
source .venv/bin/activate
uv sync
```

## Usage

### Hyperparameter Tuning

Tune hyperparameters for a single method using Optuna (TPE sampler):

```bash
python -m mapc_sa.tune --method sa   --n_trials 100
python -m mapc_sa.tune --method rrhc --n_trials 50
python -m mapc_sa.tune --method tabu --n_trials 100
```

By default tunes over 1 scenario seed per config (9 scenarios total) and saves results to `mapc_sa/methods/configs/best_params_{method}.json`.

### Evaluation

Evaluate all methods on all scenarios using tuned (or default) hyperparameters:

```bash
python -m mapc_sa.evaluate
```

With custom configs:

```bash
python -m mapc_sa.evaluate \
    --params_sa   mapc_sa/methods/configs/best_params_sa.json \
    --params_rrhc mapc_sa/methods/configs/best_params_rrhc.json \
    --params_tabu mapc_sa/methods/configs/best_params_tabu.json
```

Results are saved to `results/evaluation.json`.

### Baselines

Run H-MAB, DCF, and T-Optimal (SUM) baselines:

```bash
python -m mapc_sa.baselines
python -m mapc_sa.baselines --agents t_optimal
```

Results are saved to `results/baselines.json`.

### Statistical Report

Print a comparison table (mean ± std) and pairwise Mann-Whitney U significance tests from saved results:

```bash
python -m mapc_sa.report --input results/evaluation.json
python -m mapc_sa.report --input results/evaluation.json results/baselines.json
```

### Plotting

Plot convergence curves (best throughput vs. step) with mean ± 95% CI:

```bash
python -m mapc_sa.plot --input results/evaluation.json
```

Produces a PDF, PNG, and CSV compatible with TikZ/pgfplots.

## Project Structure

```
mapc_sa/
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
@article{wojnar2026sa,
  title={Trajectory-Based Metaheuristics for MAPC},
  author={Wojnar, Maksymilian},
  year={2026}
}
```
