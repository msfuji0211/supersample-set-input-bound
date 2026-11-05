#!/usr/bin/env python3
"""Rebuild a provisional summary.csv from partial eval outputs.

Useful while a long ``run_supersample_retrain_worst_eval`` / ``*_cvar_eval`` job is
still running: ``local_rows_*.csv`` (and optionally ``context_mi_rows_*.csv``) are
written per ``n``, but ``summary.csv`` is only written at the end of the job.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_supersample_retrain_worst_eval import _context_mi_rows, _summarize  # noqa: E402
from supersample_retrain_worst_common import _csv  # noqa: E402

LOCAL_RE = re.compile(r"local_rows_(?P<dataset>.+)_(?P<model>.+)_n(?P<n>\d+)\.csv$")


def _infer_beta(phi_objective: str, cvar_alpha: float) -> float:
    if phi_objective == "cvar":
        return 1.0 / (1.0 - cvar_alpha)
    return 2.0


def _drop_incomplete_seeds(local_df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """Keep only supersample seeds whose unique index_t count matches the max."""
    if local_df.empty or "supersample_seed" not in local_df.columns or "index_t" not in local_df.columns:
        return local_df, []
    counts = local_df.groupby("supersample_seed")["index_t"].nunique()
    if counts.empty:
        return local_df, []
    expected = int(counts.max())
    keep = counts[counts >= expected].index
    dropped = sorted(int(s) for s in counts.index if s not in keep)
    if not dropped:
        return local_df, []
    return local_df[local_df["supersample_seed"].isin(keep)].copy(), dropped


def _rebuild_one(
    local_path: Path,
    *,
    k_neighbors: int,
    phi_objective: str | None,
    cvar_alpha: float | None,
    prefer_existing_context: bool,
    drop_incomplete_seeds: bool,
) -> pd.DataFrame | None:
    match = LOCAL_RE.match(local_path.name)
    if not match:
        return None

    if local_path.stat().st_size == 0:
        print(f"[skip] empty local rows: {local_path.name}")
        return None
    try:
        local_df = pd.read_csv(local_path)
    except pd.errors.EmptyDataError:
        print(f"[skip] empty local rows: {local_path.name}")
        return None
    if local_df.empty:
        print(f"[skip] empty local rows: {local_path.name}")
        return None

    dataset_name = str(match.group("dataset"))
    model_name = str(match.group("model"))
    total_points = int(match.group("n"))
    n_pairs = total_points // 2

    if "n_total_points" in local_df.columns and local_df["n_total_points"].notna().any():
        total_points = int(local_df["n_total_points"].iloc[0])
        n_pairs = total_points // 2

    dropped_seeds: list[int] = []
    if drop_incomplete_seeds:
        local_df, dropped_seeds = _drop_incomplete_seeds(local_df)
        if dropped_seeds:
            print(
                f"[drop] {local_path.name}: incomplete supersample_seed(s) "
                f"{dropped_seeds} (fewer index_t than complete seeds)"
            )
        if local_df.empty:
            print(f"[skip] no complete seeds left in {local_path.name}")
            return None

    row_phi = None
    row_alpha = None
    row_beta = None
    if "phi_objective" in local_df.columns and local_df["phi_objective"].notna().any():
        row_phi = str(local_df["phi_objective"].iloc[0])
    if "cvar_alpha" in local_df.columns and local_df["cvar_alpha"].notna().any():
        row_alpha = float(local_df["cvar_alpha"].iloc[0])
    if "beta" in local_df.columns and local_df["beta"].notna().any():
        row_beta = float(local_df["beta"].iloc[0])

    resolved_phi = phi_objective or row_phi or "worst"
    resolved_alpha = float(cvar_alpha if cvar_alpha is not None else (row_alpha if row_alpha is not None else 0.2))
    beta_value = float(row_beta) if row_beta is not None else _infer_beta(resolved_phi, resolved_alpha)

    context_path = local_path.with_name(
        f"context_mi_rows_{dataset_name}_{model_name}_n{total_points}.csv"
    )
    # If seeds were dropped, always recompute context MI from the filtered local rows.
    context_mi_df: pd.DataFrame
    if (
        prefer_existing_context
        and not dropped_seeds
        and context_path.exists()
        and context_path.stat().st_size > 0
    ):
        try:
            context_mi_df = pd.read_csv(context_path)
        except pd.errors.EmptyDataError:
            context_mi_df = pd.DataFrame()
        if context_mi_df.empty:
            print(f"[warn] empty context file, recomputing: {context_path.name}")
            context_mi_df = _context_mi_rows(local_df, k_neighbors)
    else:
        if prefer_existing_context and context_path.exists() and not dropped_seeds:
            print(f"[warn] empty context file, recomputing: {context_path.name}")
        context_mi_df = _context_mi_rows(local_df, k_neighbors)

    if context_mi_df.empty:
        print(f"[skip] no contexts for {local_path.name}")
        return None

    m_selected = int(local_df["index_t"].nunique()) if "index_t" in local_df.columns else int(len(context_mi_df))
    summary_df = _summarize(context_mi_df, local_df, m_selected, beta_value, n_pairs)
    if summary_df.empty:
        return None

    summary_df.insert(0, "dataset", dataset_name)
    summary_df.insert(1, "model", model_name)
    summary_df.insert(2, "n_total_points", total_points)
    summary_df.insert(3, "phi_objective", resolved_phi)
    summary_df.insert(4, "beta", float(beta_value))
    summary_df.insert(5, "cvar_alpha", float(resolved_alpha))
    return summary_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild provisional summary.csv from existing local_rows_*.csv files."
    )
    parser.add_argument(
        "--eval-dir",
        required=True,
        help="Eval output directory containing local_rows_*.csv (e.g. results/.../worst).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Primary output path (default: <eval-dir>/summary_provisional.csv).",
    )
    parser.add_argument(
        "--fill-summary",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also write <eval-dir>/summary.csv when it is missing or empty. "
        "Default is false to avoid conflicting with an in-progress eval job; "
        "prefer pointing analyze_*.py at summary_provisional.csv via --summary-csv.",
    )

    parser.add_argument(
        "--datasets",
        default=None,
        help="Optional comma-separated dataset filter (e.g., CelebA,HMDA).",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Optional comma-separated model filter (e.g., ViT,LogisticRegression).",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=1,
        help="Neighbors used when (re)computing context MI.",
    )
    parser.add_argument(
        "--phi-objective",
        default=None,
        choices=["worst", "cvar"],
        help="Override Phi objective (default: infer from local_rows if present).",
    )
    parser.add_argument(
        "--cvar-alpha",
        type=float,
        default=None,
        help="Override CVaR alpha when --phi-objective=cvar (or when rows lack cvar_alpha).",
    )
    parser.add_argument(
        "--recompute-context-mi",
        action="store_true",
        help="Ignore existing context_mi_rows_*.csv and recompute context MI from local_rows.",
    )
    parser.add_argument(
        "--drop-incomplete-seeds",
        action="store_true",
        help="Drop supersample seeds whose unique index_t count is below the max "
        "in that local_rows file (useful for partially finished n).",
    )
    args = parser.parse_args()

    eval_dir = Path(REPO_ROOT) / args.eval_dir
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    output_csv = (
        Path(REPO_ROOT) / args.output_csv
        if args.output_csv is not None
        else eval_dir / "summary_provisional.csv"
    )

    datasets = set(_csv(args.datasets))
    models = set(_csv(args.models))
    prefer_existing_context = not args.recompute_context_mi

    frames: list[pd.DataFrame] = []
    for local_path in sorted(eval_dir.glob("local_rows_*.csv")):
        match = LOCAL_RE.match(local_path.name)
        if not match:
            continue
        dataset_name = match.group("dataset")
        model_name = match.group("model")
        if datasets and dataset_name not in datasets:
            continue
        if models and model_name not in models:
            continue
        summary_df = _rebuild_one(
            local_path,
            k_neighbors=args.k_neighbors,
            phi_objective=args.phi_objective,
            cvar_alpha=args.cvar_alpha,
            prefer_existing_context=prefer_existing_context,
            drop_incomplete_seeds=args.drop_incomplete_seeds,
        )
        if summary_df is not None:
            frames.append(summary_df)
            print(
                f"[ok] {local_path.name} -> n={int(summary_df['n_total_points'].iloc[0])} "
                f"proxy={float(summary_df['eq33_subsample_proxy'].iloc[0]):.6g}"
            )

    if not frames:
        raise RuntimeError(f"No usable local_rows_*.csv found under {eval_dir}")

    final_summary = pd.concat(frames, ignore_index=True)
    final_summary = final_summary.sort_values(
        ["dataset", "model", "phi_objective", "n_total_points"]
    ).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_summary.to_csv(output_csv, index=False)
    print(f"Wrote provisional summary ({len(final_summary)} rows) to {output_csv}")

    summary_csv = eval_dir / "summary.csv"
    summary_is_empty = (not summary_csv.exists()) or summary_csv.stat().st_size == 0
    if args.fill_summary and summary_is_empty:
        # Avoid clobbering a finished non-empty summary from a completed eval job.
        final_summary.to_csv(summary_csv, index=False)
        print(
            f"Filled empty/missing {summary_csv} so analyze_*.py can read it. "
            "A still-running eval job will overwrite this again when it finishes."
        )
    elif args.fill_summary and summary_csv.resolve() != output_csv.resolve():
        print(
            f"Left existing non-empty {summary_csv} unchanged "
            f"(use --output-csv {summary_csv} to force overwrite)."
        )


if __name__ == "__main__":
    main()
