# Flights200K MC, CVaR, and CR-DRO experiments

## Run

Use Python 3.12. In the extracted folder, install the dependencies and run the experiment.

```bash
python -m pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 scikit-learn==1.8.0 matplotlib==3.10.8 joblib==1.5.3 threadpoolctl==3.6.0
```

```bash
python run_experiment.py
```

The script downloads Flights200K when needed and generates all four figures in `results`, each in PDF and PNG format.

| Figure | Experiment |
| --- | --- |
| `Gap_and_bound.pdf` | Multicalibration |
| `CVaR_alpha_sensitivity.pdf` | CVaR levels 0.2, 0.5, 0.8, 0.95 |
| `CR_DRO_c_sensitivity.pdf` | CR-DRO shapes 1.25, 1.5, 2, 4 at radius 0.1 |
| `CR_DRO_epsilon_sensitivity.pdf` | CR-DRO radii 0, 0.01, 0.1, 1 at shape 2 |

## Settings and estimates

The defaults use 20 independent complete experiments, sample sizes 512 through 32,768 in powers of two, Gaussian noise levels 0.05 and 0.10, 32 contexts per setting, 16 Gaussian draws per context, and 16 selected pair positions. The master seed is 0. Up to eight workers run with one numerical-library thread each.

Each context generates a fresh 20% held-out split, 10% reference sample, candidate pairs, and selection bits. Seven-input logistic regression uses `C=1.0`, `solver="newton-cholesky"`, `tol=1e-10`, and `max_iter=100`. Independent Gaussian noise perturbs all eight fitted parameters, including the intercept.

MC uses nine reference-defined groups and five equal-width score bins. CVaR and CR-DRO evaluate Brier loss `(p - y)^2`. All three objectives use the same sampled data and perturbed models within each context. CR-DRO uses one-dimensional bisection for the threshold search. [Reference](https://doi.org/10.1214/20-AOS2004)

For each independent experiment, signed validation-minus-training differences are averaged over contexts and Gaussian draws before taking their absolute value. Conditional Gaussian MI uses 80-point Gauss-Hermite quadrature. MC uses the single-index bound in Corollary 1, Eq. (12), averaging MI over contexts before each pair's square root. CVaR uses Corollary 5. Positive-radius CR-DRO uses Corollary 6, Eq. (17), and zero-radius CR-DRO uses Theorem 2 with sensitivity `1/n`.

For both the gap and bound, the plotted points and error bars show the mean and sample standard deviation (`ddof=1`) across the 20 independent experiments. The generalization gap uses dashed lines and the upper bound uses solid lines. Each figure places noise level 0.05 on the left and 0.10 on the right. Nonpositive lower error-bar endpoints are marked with downward arrows on the logarithmic axes.

## Saved results and reruns

MC estimates are saved in `mc_independent_runs.csv` and `mc_estimates.csv`. CVaR uses `cvar_independent_runs.csv` and `cvar_estimates.csv`. CR-DRO uses `independent_runs.csv` and `estimates.csv`. `context_estimates.npz` and `context_checkpoints` store the context-level calculations.

Ordinary runs recompute all three experiments. For the first run of this updated code, use the command above. To continue an interrupted run with the same code, data, and settings, or regenerate all figures from its completed checkpoints, run

```bash
python run_experiment.py --resume
```

Use `--jobs 4` to change the worker count, `--seed 1` for a different random seed, `--output-dir results_new` for a different destination, or `--data-path path/to/flights-200k.json` for an existing dataset. [Dataset](https://cdn.jsdelivr.net/npm/vega-datasets@3.2.1/data/flights-200k.json)
