#!/usr/bin/env python3
"""Plot bound / observed-gap curves from supersample retrain eval summaries.

Follows the style of ``plot_supersample_single_index_bound.py``, but reads
``summary.csv`` produced by ``run_supersample_retrain_worst_eval.py``
(e.g. ``results/supersample_retrain_phi_eval/worst/summary.csv``).

Optionally pass a CVaR summary to make a paper-style side-by-side figure.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(REPO_ROOT) / path


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"summary CSV not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"summary CSV is empty: {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"summary CSV has no columns to parse: {path}") from exc
    if df.empty:
        raise ValueError(f"summary CSV has no rows: {path}")
    return df


def _panel_title(df: pd.DataFrame, dataset_name: str, model_name: str) -> str:
    phi = "worst"
    if "phi_objective" in df.columns and df["phi_objective"].notna().any():
        phi = str(df["phi_objective"].iloc[0]).lower()
    if phi == "cvar":
        label = "CVaR"
        if "cvar_alpha" in df.columns and df["cvar_alpha"].notna().any():
            label = f"CVaR (α={float(df['cvar_alpha'].iloc[0]):g})"
    else:
        label = "multicalibration"
    return f"{dataset_name} / {model_name} ({label})"


def _bound_columns(df: pd.DataFrame) -> tuple[str, str | None]:
    if "local_bound_proxy_seed_mean" in df.columns:
        mean_col = "local_bound_proxy_seed_mean"
        std_col = "local_bound_proxy_seed_std" if "local_bound_proxy_seed_std" in df.columns else None
        return mean_col, std_col
    if "eq33_subsample_proxy_seed_mean" in df.columns:
        mean_col = "eq33_subsample_proxy_seed_mean"
        std_col = "eq33_subsample_proxy_seed_std" if "eq33_subsample_proxy_seed_std" in df.columns else None
        return mean_col, std_col
    if "local_bound_proxy" in df.columns:
        return "local_bound_proxy", None
    if "mc_local_bound_proxy" in df.columns:
        return "mc_local_bound_proxy", None
    raise ValueError(
        "summary CSV must contain one of: local_bound_proxy_seed_mean, "
        "eq33_subsample_proxy_seed_mean, local_bound_proxy, mc_local_bound_proxy"
    )


def _plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    *,
    log_y: bool,
    show_gap: bool,
) -> None:
    subset = df[(df["dataset"] == dataset_name) & (df["model"] == model_name)].copy()
    if subset.empty:
        ax.set_visible(False)
        return
    subset = subset.sort_values("n_total_points")
    xs = subset["n_total_points"].to_numpy(dtype=float)

    bound_mean_col, bound_std_col = _bound_columns(subset)
    ys_bound = subset[bound_mean_col].to_numpy(dtype=float)
    line_bound = ax.plot(
        xs,
        ys_bound,
        marker="o",
        linewidth=2,
        color="C0",
        label="local_bound_proxy",
    )[0]
    if bound_std_col is not None and bound_std_col in subset.columns:
        errs = subset[bound_std_col].to_numpy(dtype=float)
        lower = np.maximum(ys_bound - errs, 1e-12) if log_y else (ys_bound - errs)
        upper = ys_bound + errs
        ax.fill_between(xs, lower, upper, alpha=0.2, color=line_bound.get_color())

    if show_gap and "observed_abs_vt_gap_mean" in subset.columns:
        ys_gap = subset["observed_abs_vt_gap_mean"].to_numpy(dtype=float)
        ax.plot(
            xs,
            ys_gap,
            marker="o",
            linewidth=2,
            linestyle="--",
            color="black",
            label="observed_abs_vt_gap_mean",
        )

    ax.set_xlabel("total n")
    ax.set_ylabel("value")
    ax.set_title(_panel_title(subset, dataset_name, model_name))
    ax.grid(True, alpha=0.3)
    ax.legend()
    if log_y:
        ax.set_yscale("log")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot supersample retrain bound/gap curves from eval summary.csv."
    )
    parser.add_argument(
        "--summary-csv",
        required=True,
        help="Path to worst (or any) retrain eval summary.csv / summary_provisional.csv.",
    )
    parser.add_argument(
        "--cvar-summary-csv",
        default=None,
        help="Optional CVaR summary.csv for a paper-style side-by-side figure.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for plots (default: <summary parent>/plots).",
    )
    parser.add_argument("--datasets", default=None, help="Optional comma-separated dataset filter.")
    parser.add_argument("--models", default=None, help="Optional comma-separated model filter.")
    parser.add_argument(
        "--no-gap",
        action="store_true",
        help="Plot only the bound proxy (no observed_abs_vt_gap_mean).",
    )
    parser.add_argument(
        "--no-log-y",
        action="store_true",
        help="Use a linear y-axis instead of log scale.",
    )
    args = parser.parse_args()

    summary_path = _resolve_path(args.summary_csv)
    df_main = _load_summary(summary_path)
    df_cvar = None
    if args.cvar_summary_csv is not None:
        df_cvar = _load_summary(_resolve_path(args.cvar_summary_csv))

    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir is not None
        else summary_path.parent / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = _csv(args.datasets)
    models = _csv(args.models)
    if datasets:
        df_main = df_main[df_main["dataset"].isin(datasets)]
        if df_cvar is not None:
            df_cvar = df_cvar[df_cvar["dataset"].isin(datasets)]
    if models:
        df_main = df_main[df_main["model"].isin(models)]
        if df_cvar is not None:
            df_cvar = df_cvar[df_cvar["model"].isin(models)]
    if df_main.empty:
        raise ValueError("No rows remain after filtering; nothing to plot.")

    show_gap = not args.no_gap
    log_y = not args.no_log_y
    pairs = sorted(
        {
            (str(r.dataset), str(r.model))
            for r in df_main[["dataset", "model"]].drop_duplicates().itertuples(index=False)
        }
    )

    for dataset_name, model_name in pairs:
        if df_cvar is None:
            fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5))
            _plot_panel(
                ax,
                df_main,
                dataset_name,
                model_name,
                log_y=log_y,
                show_gap=show_gap,
            )
            stem = f"{dataset_name}_{model_name}_bound_vs_n"
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
            _plot_panel(
                axes[0],
                df_main,
                dataset_name,
                model_name,
                log_y=log_y,
                show_gap=show_gap,
            )
            _plot_panel(
                axes[1],
                df_cvar,
                dataset_name,
                model_name,
                log_y=log_y,
                show_gap=show_gap,
            )
            stem = f"{dataset_name}_{model_name}_bound_vs_n_side_by_side"

        fig.tight_layout()
        fig.savefig(output_dir / f"{stem}.png", dpi=200)
        fig.savefig(output_dir / f"{stem}.pdf")
        plt.close(fig)
        print(f"Saved {output_dir / (stem + '.png')}")

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
