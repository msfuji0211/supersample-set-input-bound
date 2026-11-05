#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run retrain eval for Phi=worst and Phi=CVaR with identical settings."
    )
    parser.add_argument("--datasets", default="ACSIncome,BankMarketing,CreditDefault,MEPS,HMDA")
    parser.add_argument("--models", default="LogisticRegression,NaiveBayes,DecisionTree,RandomForest,SVM")
    parser.add_argument("--supersample-seeds", default="15,25,35")
    parser.add_argument("--model-seeds", default="0,1,2")
    parser.add_argument("--sample-sizes", default=None)
    parser.add_argument("--sample-size-percents", default=None)
    parser.add_argument("--k-neighbors", type=int, default=1)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--groups-collection", default="default")
    parser.add_argument("--input-dir", default="results/supersample_retrain_worst_train")
    parser.add_argument(
        "--output-dir-base",
        default="results/supersample_retrain_phi_eval",
        help="Base directory. Results are written to <base>/worst and <base>/cvar_aXX.",
    )
    parser.add_argument(
        "--cvar-alpha",
        type=float,
        default=0.2,
        help="CVaR alpha for the cvar run (recommended in [0.1, 0.3]).",
    )
    args = parser.parse_args()

    if args.sample_sizes is not None and args.sample_size_percents is not None:
        raise ValueError("Use either --sample-sizes or --sample-size-percents, not both.")
    if not (0.0 <= args.cvar_alpha <= 0.5):
        raise ValueError("--cvar-alpha must satisfy 0 <= alpha <= 0.5.")

    eval_script = Path(SCRIPT_DIR) / "run_supersample_retrain_worst_eval.py"
    cvar_tag = f"cvar_a{str(args.cvar_alpha).replace('.', 'p')}"
    output_worst = str(Path(args.output_dir_base) / "worst")
    output_cvar = str(Path(args.output_dir_base) / cvar_tag)

    common_cmd = [
        sys.executable,
        str(eval_script),
        "--datasets",
        args.datasets,
        "--models",
        args.models,
        "--supersample-seeds",
        args.supersample_seeds,
        "--model-seeds",
        args.model_seeds,
        "--k-neighbors",
        str(args.k_neighbors),
        "--n-bins",
        str(args.n_bins),
        "--groups-collection",
        args.groups_collection,
        "--input-dir",
        args.input_dir,
    ]
    if args.sample_sizes is not None:
        common_cmd.extend(["--sample-sizes", args.sample_sizes])
    if args.sample_size_percents is not None:
        common_cmd.extend(["--sample-size-percents", args.sample_size_percents])

    worst_cmd = common_cmd + [
        "--phi-objective",
        "worst",
        "--output-dir",
        output_worst,
    ]
    cvar_cmd = common_cmd + [
        "--phi-objective",
        "cvar",
        "--cvar-alpha",
        str(args.cvar_alpha),
        "--output-dir",
        output_cvar,
    ]

    _run(worst_cmd)
    _run(cvar_cmd)


if __name__ == "__main__":
    main()
