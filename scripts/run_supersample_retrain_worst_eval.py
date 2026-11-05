#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from supersample_retrain_worst_common import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    IMAGE_MODELS,
    REPO_ROOT,
    SIMPLE_MODELS,
    SUPER_SPLIT,
    _branch_indices,
    _csv,
    _csv_ints,
    _estimate_mi,
    _evaluate_phi_from_bundle,
    _load_pickle,
    _metadata_path,
    _model_bundle_path,
    _sample_sizes_from_percentages,
)


def _mixed_path_indices(
    paired_idxs: np.ndarray,
    base_u: np.ndarray,
    t: int,
    branch: int,
) -> np.ndarray:
    """
    Proposition-style mixed path S(t, branch):
      i < t uses U_i, i = t uses branch, i > t uses 1 - U_i.
    """
    selector = 1 - base_u.copy()
    selector[:t] = base_u[:t]
    selector[t] = branch
    return np.sort(paired_idxs[np.arange(len(paired_idxs)), selector])


def _context_mi_rows(local_df: pd.DataFrame, k_neighbors: int) -> pd.DataFrame:
    if local_df.empty:
        return pd.DataFrame()

    # Eq.33 uses D_t = Phi(S^(t,0)) - Phi(S^(t,1)).
    # local_increment is kept as a backwards-compatible alias below.
    mi_value_col = "D_t" if "D_t" in local_df.columns else "local_increment"

    context_rows = []
    grouped = local_df.groupby(["supersample_seed", "index_t"], sort=True)
    for (supersample_seed, index_t), idx_df in grouped:
        local_mi = _estimate_mi(
            idx_df[mi_value_col].to_numpy(dtype=float),
            idx_df["u_t"].to_numpy(dtype=int),
            k_neighbors=k_neighbors,
        )
        context_rows.append(
            {
                "supersample_seed": int(supersample_seed),
                "index_t": int(index_t),
                "n_samples": int(len(idx_df)),
                "n_u0": int((idx_df["u_t"] == 0).sum()),
                "n_u1": int((idx_df["u_t"] == 1).sum()),
                "mi_value_col": mi_value_col,
                "local_mi": float(local_mi),
            }
        )
    return pd.DataFrame(context_rows)

def _summarize(
    context_mi_df: pd.DataFrame,
    local_df: pd.DataFrame,
    m_selected: int,
    beta_value: float,
    n_pairs: int,
) -> pd.DataFrame:
    if context_mi_df.empty:
        return pd.DataFrame()

    local_mis_arr = context_mi_df["local_mi"].to_numpy(dtype=float)
    sqrt_terms = np.sqrt(2.0 * np.maximum(local_mis_arr, 0.0))
    seed_summary_series = (
        context_mi_df.assign(sqrt_2_local_mi=sqrt_terms)
        .groupby("supersample_seed", sort=True)["sqrt_2_local_mi"]
        .mean()
    )
    seed_summaries_arr = seed_summary_series.to_numpy(dtype=float)

    # Eq.33 has coefficient B_n = beta/n. For MC beta=2; for CVaR beta=1/(1-alpha).
    # With uniform subsampling of m indices, (beta/n) * sum_t sqrt(2 I_t)
    # is estimated by beta * mean_{t in sampled T} sqrt(2 I_t).
    eq33_coef = float(beta_value)
    mean_sqrt = float(sqrt_terms.mean()) if len(sqrt_terms) else 0.0
    seed_mean = float(seed_summaries_arr.mean()) if len(seed_summaries_arr) else 0.0
    seed_std = float(seed_summaries_arr.std(ddof=0)) if len(seed_summaries_arr) else 0.0
    eq33_proxy = float(eq33_coef * mean_sqrt)
    sampled_sum_only = float(eq33_coef * sqrt_terms.sum() / n_pairs) if n_pairs and len(sqrt_terms) else 0.0
    full_sum_if_all_indices = sampled_sum_only if n_pairs and m_selected == n_pairs else np.nan

    vt_gap_mean = float(local_df["vt_gap"].mean()) if "vt_gap" in local_df.columns and not local_df.empty else 0.0
    abs_vt_gap_mean = (
        float(local_df["abs_vt_gap"].mean()) if "abs_vt_gap" in local_df.columns and not local_df.empty else 0.0
    )
    return pd.DataFrame(
        [
            {
                "n_pairs": int(n_pairs),
                "m_selected_indices": int(m_selected),
                "all_indices_evaluated": bool(n_pairs > 0 and m_selected == n_pairs),
                "n_contexts": int(len(context_mi_df)),
                "local_mi_mean": float(local_mis_arr.mean()) if len(local_mis_arr) else 0.0,
                "local_mi_std": float(local_mis_arr.std(ddof=0)) if len(local_mis_arr) else 0.0,
                "local_mi_max": float(local_mis_arr.max()) if len(local_mis_arr) else 0.0,
                "sum_sqrt_2_local_mi": float(sqrt_terms.sum()),
                "mean_sqrt_2_local_mi": mean_sqrt,
                "subsample_mean_sqrt_2_local_mi": mean_sqrt,
                "eq33_coefficient_beta": eq33_coef,
                "eq33_subsample_proxy": eq33_proxy,
                "mc_wang_eq33_proxy": eq33_proxy,
                "eq33_sampled_sum_only": sampled_sum_only,
                "eq33_full_sum_if_all_indices": full_sum_if_all_indices,
                # Backwards-compatible aliases. These are now Eq.33-scaled, not the old 2*beta scale.
                "mc_local_bound_proxy": eq33_proxy,
                "local_bound_proxy": eq33_proxy,
                "legacy_2beta_local_bound_proxy": float(2.0 * eq33_proxy),
                "subsample_mean_sqrt_2_local_mi_seed_mean": seed_mean,
                "subsample_mean_sqrt_2_local_mi_seed_std": seed_std,
                "eq33_subsample_proxy_seed_mean": float(eq33_coef * seed_mean),
                "eq33_subsample_proxy_seed_std": float(eq33_coef * seed_std),
                "mc_wang_eq33_proxy_seed_mean": float(eq33_coef * seed_mean),
                "mc_wang_eq33_proxy_seed_std": float(eq33_coef * seed_std),
                "mc_local_bound_proxy_seed_mean": float(eq33_coef * seed_mean),
                "mc_local_bound_proxy_seed_std": float(eq33_coef * seed_std),
                "local_bound_proxy_seed_mean": float(eq33_coef * seed_mean),
                "local_bound_proxy_seed_std": float(eq33_coef * seed_std),
                "observed_vt_gap_mean": vt_gap_mean,
                "observed_abs_vt_gap_mean": abs_vt_gap_mean,
            }
        ]
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved retrained worst-calibration models and estimate MI.")
    parser.add_argument("--datasets", default=DEFAULT_DATASETS, help="Comma-separated dataset names.")
    parser.add_argument("--models", default=DEFAULT_MODELS, help="Comma-separated model names.")
    parser.add_argument("--supersample-seeds", default="15,25,35", help="Comma-separated supersample seeds.")
    parser.add_argument("--model-seeds", default="0,1,2", help="Comma-separated model initialization seeds.")
    parser.add_argument("--sample-sizes", default=None, help="Comma-separated total non-test point counts.")
    parser.add_argument(
        "--sample-size-percents",
        default=None,
        help="Comma-separated percentages of non-test points, e.g. 100,55 for 100%% and 55%%.",
    )
    parser.add_argument("--k-neighbors", type=int, default=1, help="Neighbors for MI estimator.")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of bins for worst calibration error.")
    parser.add_argument(
        "--phi-objective",
        default="worst",
        choices=["worst", "cvar"],
        help="Set-input objective Phi used in L_t.",
    )
    parser.add_argument(
        "--cvar-alpha",
        type=float,
        default=0.2,
        help="CVaR alpha in [0, 0.5]. Used only when --phi-objective=cvar.",
    )
    parser.add_argument("--groups-collection", default="default", help="Group collection name.")
    parser.add_argument("--input-dir", default="results/supersample_retrain_worst_train", help="Training output dir.")
    parser.add_argument("--output-dir", default="results/supersample_retrain_worst_eval", help="Evaluation output dir.")
    args = parser.parse_args()
    if args.phi_objective == "cvar" and not (0.0 <= args.cvar_alpha <= 0.5):
        raise ValueError("--cvar-alpha must satisfy 0 <= alpha <= 0.5 for the current setup.")
    beta_value = 1.0 / (1.0 - args.cvar_alpha) if args.phi_objective == "cvar" else 2.0

    datasets = _csv(args.datasets)
    models = _csv(args.models)
    supersample_seeds = _csv_ints(args.supersample_seeds)
    model_seeds = _csv_ints(args.model_seeds)
    input_dir = Path(REPO_ROOT) / args.input_dir
    output_dir = Path(REPO_ROOT) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    unsupported_models = sorted(set(models) - (SIMPLE_MODELS | IMAGE_MODELS))
    if unsupported_models:
        raise ValueError(
            f"Unsupported models for this script: {unsupported_models}. Supported: {sorted(SIMPLE_MODELS | IMAGE_MODELS)}"
        )

    summary_frames: list[pd.DataFrame] = []
    for dataset_name in datasets:
        from Dataset import Dataset

        dataset_obj = Dataset(
            dataset_name,
            split=SUPER_SPLIT,
            groups=args.groups_collection,
            scale=False,
            verbose=True,
        )
        candidate_size = len(dataset_obj.train_idxs) + len(dataset_obj.val_idxs)
        min_total_points = 20

        for model_name in models:
            if args.sample_sizes is not None and args.sample_size_percents is not None:
                raise ValueError("Use either --sample-sizes or --sample-size-percents, not both.")
            sample_sizes = _csv_ints(args.sample_sizes) if args.sample_sizes else []
            if args.sample_size_percents is not None:
                sample_sizes = _sample_sizes_from_percentages(
                    candidate_size,
                    min_total_points,
                    [float(x) for x in _csv(args.sample_size_percents)],
                )
            elif not sample_sizes:
                sample_size_root = input_dir / "metadata" / dataset_name / model_name
                sample_sizes = sorted(
                    int(path.name.split("=")[1]) for path in sample_size_root.iterdir() if path.name.startswith("n=")
                )

            for total_points in sample_sizes:
                local_rows = []
                m_selected_global = None
                for supersample_seed in supersample_seeds:
                    meta_path = _metadata_path(input_dir, dataset_name, model_name, total_points, supersample_seed)
                    if not meta_path.exists():
                        continue
                    metadata = np.load(meta_path)
                    paired_idxs = metadata["paired_idxs"]
                    base_u = metadata["base_u"]
                    selected_indices = metadata["selected_indices"]
                    m_selected_global = int(metadata["n_selected_indices"])
                    n_pairs = int(len(paired_idxs))

                    for index_t in selected_indices.tolist():
                        branch0_train_idxs, branch0_val_idxs = _branch_indices(paired_idxs, base_u, int(index_t), 0)
                        branch1_train_idxs, branch1_val_idxs = _branch_indices(paired_idxs, base_u, int(index_t), 1)
                        eval0_idxs = _mixed_path_indices(paired_idxs, base_u, int(index_t), 0)
                        eval1_idxs = _mixed_path_indices(paired_idxs, base_u, int(index_t), 1)
                        base_u_t = int(base_u[int(index_t)])
                        # Eq.33 estimates I(D_t; U_t | Z_tilde, U_{-t}) with the raw
                        # one-point replacement increment D_t = Phi(S^(t,0)) - Phi(S^(t,1)).
                        scale = 1.0
                        for model_seed in model_seeds:
                            for train_branch in (0, 1):
                                bundle_path = _model_bundle_path(
                                    input_dir,
                                    dataset_name,
                                    model_name,
                                    total_points,
                                    supersample_seed,
                                    int(index_t),
                                    model_seed,
                                    train_branch,
                                )
                                if not bundle_path.exists():
                                    continue
                                bundle = _load_pickle(bundle_path)

                                phi_on_0 = _evaluate_phi_from_bundle(
                                    bundle=bundle,
                                    dataset_obj=dataset_obj,
                                    eval_idxs=eval0_idxs,
                                    objective=args.phi_objective,
                                    n_bins=args.n_bins,
                                    cvar_alpha=args.cvar_alpha,
                                )
                                phi_on_1 = _evaluate_phi_from_bundle(
                                    bundle=bundle,
                                    dataset_obj=dataset_obj,
                                    eval_idxs=eval1_idxs,
                                    objective=args.phi_objective,
                                    n_bins=args.n_bins,
                                    cvar_alpha=args.cvar_alpha,
                                )
                                phi_t0 = float(phi_on_0["phi_value"])
                                phi_t1 = float(phi_on_1["phi_value"])
                                raw_diff = float(phi_t0 - phi_t1)
                                D_t = raw_diff
                                epsilon_t = int(2 * train_branch - 1)
                                epsilon_t_D_t = float(epsilon_t * D_t)
                                if train_branch == 0:
                                    train_eval_idxs = branch0_train_idxs
                                    val_eval_idxs = branch0_val_idxs
                                else:
                                    train_eval_idxs = branch1_train_idxs
                                    val_eval_idxs = branch1_val_idxs
                                phi_train = float(
                                    _evaluate_phi_from_bundle(
                                        bundle=bundle,
                                        dataset_obj=dataset_obj,
                                        eval_idxs=train_eval_idxs,
                                        objective=args.phi_objective,
                                        n_bins=args.n_bins,
                                        cvar_alpha=args.cvar_alpha,
                                    )["phi_value"]
                                )
                                phi_val = float(
                                    _evaluate_phi_from_bundle(
                                        bundle=bundle,
                                        dataset_obj=dataset_obj,
                                        eval_idxs=val_eval_idxs,
                                        objective=args.phi_objective,
                                        n_bins=args.n_bins,
                                        cvar_alpha=args.cvar_alpha,
                                    )["phi_value"]
                                )
                                vt_gap = float(phi_val - phi_train)
                                abs_vt_gap = float(abs(vt_gap))

                                local_rows.append(
                                    {
                                        "dataset": dataset_name,
                                        "model": model_name,
                                        "n_total_points": total_points,
                                        "supersample_seed": supersample_seed,
                                        "model_seed": model_seed,
                                        "index_t": int(index_t),
                                        "u_t": int(train_branch),
                                        "train_branch": int(train_branch),
                                        "base_u_t": base_u_t,
                                        "phi_objective": args.phi_objective,
                                        "beta": float(beta_value),
                                        "cvar_alpha": float(args.cvar_alpha),
                                        "branch0_train_size": int(len(branch0_train_idxs)),
                                        "branch1_train_size": int(len(branch1_train_idxs)),
                                        "branch0_val_size": int(len(branch0_val_idxs)),
                                        "branch1_val_size": int(len(branch1_val_idxs)),
                                        "eval0_size": int(len(eval0_idxs)),
                                        "eval1_size": int(len(eval1_idxs)),
                                        "phi_t0": phi_t0,
                                        "phi_t1": phi_t1,
                                        "raw_diff": raw_diff,
                                        "D_t": D_t,
                                        "epsilon_t": epsilon_t,
                                        "epsilon_t_D_t": epsilon_t_D_t,
                                        "scale_multiplier": float(scale),
                                        "local_increment": float(D_t),
                                        "phi_train": phi_train,
                                        "phi_val": phi_val,
                                        "vt_gap": vt_gap,
                                        "abs_vt_gap": abs_vt_gap,
                                        "worst0_bin": int(phi_on_0.get("worst_bin", -1)),
                                        "worst0_group_idx": int(phi_on_0.get("worst_group_idx", -1)),
                                        "worst0_group_name": str(phi_on_0.get("worst_group_name", "NA")),
                                        "worst0_class": int(phi_on_0.get("worst_class", -1)),
                                        "worst0_moment_abs_cal_error": float(phi_on_0.get("worst_moment_abs_cal_error", phi_t0)),
                                        "worst0_conditional_abs_cal_error_at_moment_witness": float(phi_on_0.get("worst_conditional_abs_cal_error_at_moment_witness", np.nan)),
                                        "worst0_cell_count": int(phi_on_0.get("worst_cell_count", -1)),
                                        "worst0_cell_frac": float(phi_on_0.get("worst_cell_frac", np.nan)),
                                        "worst0_conditional_worst_abs_cal_error": float(phi_on_0.get("conditional_worst_abs_cal_error", np.nan)),
                                        "worst0_conditional_worst_cell_count": int(phi_on_0.get("conditional_worst_cell_count", -1)),
                                        "worst0_conditional_worst_cell_frac": float(phi_on_0.get("conditional_worst_cell_frac", np.nan)),
                                        "worst0_conditional_worst_bin": int(phi_on_0.get("conditional_worst_bin", -1)),
                                        "worst0_conditional_worst_group_idx": int(phi_on_0.get("conditional_worst_group_idx", -1)),
                                        "worst0_conditional_worst_group_name": str(phi_on_0.get("conditional_worst_group_name", "NA")),
                                        "worst0_conditional_worst_class": int(phi_on_0.get("conditional_worst_class", -1)),
                                        "worst1_bin": int(phi_on_1.get("worst_bin", -1)),
                                        "worst1_group_idx": int(phi_on_1.get("worst_group_idx", -1)),
                                        "worst1_group_name": str(phi_on_1.get("worst_group_name", "NA")),
                                        "worst1_class": int(phi_on_1.get("worst_class", -1)),
                                        "worst1_moment_abs_cal_error": float(phi_on_1.get("worst_moment_abs_cal_error", phi_t1)),
                                        "worst1_conditional_abs_cal_error_at_moment_witness": float(phi_on_1.get("worst_conditional_abs_cal_error_at_moment_witness", np.nan)),
                                        "worst1_cell_count": int(phi_on_1.get("worst_cell_count", -1)),
                                        "worst1_cell_frac": float(phi_on_1.get("worst_cell_frac", np.nan)),
                                        "worst1_conditional_worst_abs_cal_error": float(phi_on_1.get("conditional_worst_abs_cal_error", np.nan)),
                                        "worst1_conditional_worst_cell_count": int(phi_on_1.get("conditional_worst_cell_count", -1)),
                                        "worst1_conditional_worst_cell_frac": float(phi_on_1.get("conditional_worst_cell_frac", np.nan)),
                                        "worst1_conditional_worst_bin": int(phi_on_1.get("conditional_worst_bin", -1)),
                                        "worst1_conditional_worst_group_idx": int(phi_on_1.get("conditional_worst_group_idx", -1)),
                                        "worst1_conditional_worst_group_name": str(phi_on_1.get("conditional_worst_group_name", "NA")),
                                        "worst1_conditional_worst_class": int(phi_on_1.get("conditional_worst_class", -1)),
                                    }
                                )

                local_df = pd.DataFrame(local_rows)
                detail_path = output_dir / f"local_rows_{dataset_name}_{model_name}_n{total_points}.csv"
                local_df.to_csv(detail_path, index=False)
                context_mi_df = _context_mi_rows(local_df, args.k_neighbors)
                context_path = output_dir / f"context_mi_rows_{dataset_name}_{model_name}_n{total_points}.csv"
                context_mi_df.to_csv(context_path, index=False)
                summary_df = _summarize(context_mi_df, local_df, m_selected_global or 0, beta_value, total_points // 2)
                if not summary_df.empty:
                    summary_df.insert(0, "dataset", dataset_name)
                    summary_df.insert(1, "model", model_name)
                    summary_df.insert(2, "n_total_points", total_points)
                    summary_df.insert(3, "phi_objective", args.phi_objective)
                    summary_df.insert(4, "beta", float(beta_value))
                    summary_df.insert(5, "cvar_alpha", float(args.cvar_alpha))
                    summary_frames.append(summary_df)

    final_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary_path = output_dir / "summary.csv"
    final_summary.to_csv(summary_path, index=False)
    print(f"Saved evaluation summary to {summary_path}")
    if not final_summary.empty:
        print(final_summary.to_string(index=False))


if __name__ == "__main__":
    main()
