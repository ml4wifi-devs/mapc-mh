# Simulated Annealing for Multi-AP Coordination

We propose using **simulated annealing (SA)** to optimize coordinated spatial reuse (Co-SR) scheduling in IEEE 802.11bn (Wi-Fi 8) networks. Starting from a random configuration, SA iteratively proposes and accepts/rejects neighboring configurations guided by a temperature schedule — converging to high-throughput solutions without a surrogate model or offline training.

## How It Works

1. **Initialization** — Sample a random Co-SR configuration (active APs, STA selection, MCS, transmit power) and evaluate its throughput directly in the simulator
2. **Neighbor proposal** — Perturb the current configuration by mutating one parameter of one AP
3. **Metropolis acceptance** — Accept improvements always; accept degradations with probability exp(Δ/T), where T decays over time
4. **Top-N tracking** — Maintain the best *N* configurations seen across all steps for round-robin deployment

## SA Versions

| Version | Selection mutation | MCS / tx_power mutation |
|---------|-------------------|------------------------|
| **V1** — Pure SA | Toggle AP on/off, random STA | Uniform random draw |
| **V2** — Ordinal-aware | Toggle AP on/off, random STA | ±1 step only |
| **V3** — Temperature-dependent jumps | Toggle AP on/off, random STA | ±*k* steps, *k* shrinks with temperature |
| **V4** — Concurrency control | Add/remove AP via sigmoid(target − n_active) | ±*k* steps, *k* shrinks with temperature |

All versions are implemented as **pure JAX functions** and compiled end-to-end with `jax.lax.scan`, enabling fast execution on CPU.

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

Tune SA hyperparameters using Optuna (TPE sampler) over 160 randomly generated scenarios:

```bash
python -m mapc_sa.tuning --version 4 --n_trials 100 --n_scenario_jobs 24
```

Run all 4 versions sequentially:

```bash
python -m mapc_sa.tuning --n_trials 100 --n_scenario_jobs 24
```

Results are saved to `results/best_params_v{1,2,3,4}.json`.

### Evaluation

Evaluate a specific version on the residential scenario using tuned hyperparameters:

```bash
python -m mapc_sa.evaluate --version 4 --params results/best_params_v4.json \
    --n_steps 10000 --n_runs 8 --n_jobs 8
```

Evaluate all 4 versions at once (use `{}` as the version placeholder):

```bash
python -m mapc_sa.evaluate --params results/best_params_v{}.json \
    --n_steps 10000 --n_runs 8 --n_jobs 8
```

Results are saved to `results/eval_v{1,2,3,4}.json`.

### Plotting

Plot throughput vs. step with mean ± 95% CI across runs:

```bash
python -m mapc_sa.plot --input results/eval_v1.json results/eval_v4.json \
    --labels V1 V4 --output results/plots/compare
```

Produces a PDF, PNG, and a CSV compatible with TikZ/pgfplots.

## Project Structure

```
mapc_sa/
├── _env.py          # JAX CPU environment setup (imported first by all modules)
├── config.py        # NetworkConfig (JAX NamedTuple), ScenarioInfo, array conversion
├── neighbor.py      # Neighbor functions for V1–V4 (pure JAX, JIT-compatible)
├── annealing.py     # JIT-compiled SA loop (jax.lax.scan), T₀ calibration, top-N buffer
├── versions.py      # Version runners: run_sa_v1 … run_sa_v4
├── scenarios.py     # Residential and random scenario definitions
├── tuning.py        # Optuna hyperparameter search (CLI entry point)
├── evaluate.py      # Final evaluation on residential scenario (CLI entry point)
└── plot.py          # Throughput-vs-step plots with CI bands and CSV export
```

## Citation

```bibtex
@article{wojnar2026sa,
  title={Simulated Annealing for Multi-AP Coordination},
  author={Wojnar, Maksymilian},
  year={2026}
}
```
