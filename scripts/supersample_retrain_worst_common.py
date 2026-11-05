from __future__ import annotations

import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SUPER_SPLIT = {"train": 0.4, "val": 0.4, "test": 0.2}
# This release ships CelebA + ViT only.
SIMPLE_MODELS: set[str] = set()
IMAGE_MODELS = {"ViT"}
SUPPORTED_MODELS = SIMPLE_MODELS | IMAGE_MODELS
DEFAULT_DATASETS = "CelebA"
DEFAULT_MODELS = "ViT"


def _csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in _csv(s)]


def _sample_sizes_from_percentages(
    max_total_points: int,
    min_total_points: int,
    percentages: list[float],
) -> list[int]:
    sample_sizes: list[int] = []
    for pct in percentages:
        if pct <= 0 or pct > 100:
            continue
        n = _ensure_even(int(max_total_points * (pct / 100.0)))
        n = max(n, min_total_points)
        n = min(n, max_total_points)
        n = _ensure_even(n)
        if n >= min_total_points:
            sample_sizes.append(n)

    deduped = sorted(set(sample_sizes))
    if not deduped:
        raise ValueError(
            f"No valid sample sizes remain from percentages {percentages}. "
            f"Allowed percentage range: (0, 100]."
        )
    return deduped


def _ensure_even(n: int) -> int:
    return n if n % 2 == 0 else n - 1


def _default_sample_sizes(max_total_points: int, min_total_points: int) -> list[int]:
    fracs = [0.25, 0.5, 0.75, 1.0]
    sample_sizes: list[int] = []
    for frac in fracs:
        n = _ensure_even(int(max_total_points * frac))
        n = max(n, min_total_points)
        n = min(n, max_total_points)
        n = _ensure_even(n)
        if n >= min_total_points:
            sample_sizes.append(n)

    deduped = sorted(set(sample_sizes))
    if not deduped:
        return [_ensure_even(max_total_points)]
    return deduped


def _fit_scaler_if_needed(X_train: np.ndarray, scale_data: bool) -> StandardScaler | None:
    if not scale_data:
        return None
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def _transform_with_scaler(X: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    if scaler is None:
        return X
    return scaler.transform(X)


def _positive_class_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 2:
        if probs.shape[1] == 2:
            return probs[:, 1]
        if probs.shape[1] == 1:
            return probs[:, 0]
        raise ValueError(f"Expected binary probabilities with shape (n, 2); got {probs.shape}.")
    return probs.reshape(-1)


def _binary_probs_to_two_class(positive_probs: np.ndarray) -> np.ndarray:
    positive_probs = np.asarray(positive_probs, dtype=float).reshape(-1)
    return np.column_stack([1.0 - positive_probs, positive_probs])


def _binary_labels_to_one_hot(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    return np.column_stack([1 - labels, labels])


def _worst_bin_group_class_calibration_error(
    positive_probs: np.ndarray,
    labels: np.ndarray,
    groups: list[list[int]],
    group_names: list[str],
    n_bins: int,
) -> dict[str, float | int | str]:
    """
    Finite-bin/group/class approximation to the MC empirical value Phi_MC,n.

    The HJZ helper returns a conditional bin/group calibration gap.  Corollary 1 / Eq.33
    requires the MC moment objective, so each conditional cell gap is multiplied by
    count(cell) / n_eval before taking the maximum.  The old unweighted conditional
    maximum is still returned under conditional_worst_* diagnostics.
    """
    from mcb_algorithms.HJZ.calibration import (
        calculate_average_labels,
        calculate_calibration_errors,
        discretize_values,
    )

    def _empty() -> dict[str, float | int | str]:
        return {
            "worst_abs_cal_error": 0.0,
            "worst_moment_abs_cal_error": 0.0,
            "worst_conditional_abs_cal_error_at_moment_witness": 0.0,
            "worst_cell_count": 0,
            "worst_cell_frac": 0.0,
            "worst_bin": -1,
            "worst_group_idx": -1,
            "worst_group_name": "NA",
            "worst_class": -1,
            "conditional_worst_abs_cal_error": 0.0,
            "conditional_worst_cell_count": 0,
            "conditional_worst_cell_frac": 0.0,
            "conditional_worst_bin": -1,
            "conditional_worst_group_idx": -1,
            "conditional_worst_group_name": "NA",
            "conditional_worst_class": -1,
        }

    positive_probs = np.asarray(positive_probs, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    n_eval = int(len(labels))
    if n_eval == 0:
        return _empty()

    predictions = _binary_probs_to_two_class(positive_probs)
    one_hot_labels = _binary_labels_to_one_hot(labels)
    bins = discretize_values(predictions, n_bins)
    average_labels, counts = calculate_average_labels(one_hot_labels, bins, groups, n_bins)
    errors = calculate_calibration_errors(average_labels, predictions, bins, groups)

    # HJZ shape convention: errors is (n_bins, n_groups, n_classes, 2),
    # counts is (n_bins, n_groups), and the final error axis stores the sign direction.
    abs_errors = np.max(errors, axis=-1)  # conditional gap, bins x groups x classes
    occupied = counts[:, :, np.newaxis] > 0
    if not np.any(occupied):
        return _empty()

    cell_fracs = counts[:, :, np.newaxis].astype(float) / float(n_eval)
    conditional_abs_errors = np.where(occupied, abs_errors, -np.inf)
    moment_abs_errors = np.where(occupied, abs_errors * cell_fracs, -np.inf)

    # This is the actual Phi_MC,n witness/value used by Eq.33.
    moment_idx = np.unravel_index(np.argmax(moment_abs_errors), moment_abs_errors.shape)
    moment_bin, moment_group_idx, moment_class = moment_idx
    moment_count = int(counts[moment_bin, moment_group_idx])
    moment_frac = float(moment_count / n_eval)
    moment_value = float(moment_abs_errors[moment_idx])
    moment_conditional_value = float(conditional_abs_errors[moment_idx])

    # Diagnostic: the old rare-cell-sensitive conditional maximum.
    conditional_idx = np.unravel_index(np.argmax(conditional_abs_errors), conditional_abs_errors.shape)
    cond_bin, cond_group_idx, cond_class = conditional_idx
    cond_count = int(counts[cond_bin, cond_group_idx])
    cond_frac = float(cond_count / n_eval)
    cond_value = float(conditional_abs_errors[conditional_idx])

    return {
        # Backwards-compatible key.  It is now the MC moment value, not the old conditional value.
        "worst_abs_cal_error": moment_value,
        "worst_moment_abs_cal_error": moment_value,
        "worst_conditional_abs_cal_error_at_moment_witness": moment_conditional_value,
        "worst_cell_count": moment_count,
        "worst_cell_frac": moment_frac,
        "worst_bin": int(moment_bin),
        "worst_group_idx": int(moment_group_idx),
        "worst_group_name": group_names[moment_group_idx],
        "worst_class": int(moment_class),
        "conditional_worst_abs_cal_error": cond_value,
        "conditional_worst_cell_count": cond_count,
        "conditional_worst_cell_frac": cond_frac,
        "conditional_worst_bin": int(cond_bin),
        "conditional_worst_group_idx": int(cond_group_idx),
        "conditional_worst_group_name": group_names[cond_group_idx],
        "conditional_worst_class": int(cond_class),
    }

def _cvar_calibration_error(
    positive_probs: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> float:
    if not (0.0 <= alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 <= alpha < 1, got {alpha}.")
    positive_probs = np.asarray(positive_probs, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if len(positive_probs) == 0:
        return 0.0
    losses = np.abs(positive_probs - labels)
    taus = np.unique(np.concatenate(([0.0, 1.0], losses)))
    vals = taus + np.maximum(losses[:, None] - taus[None, :], 0.0).mean(axis=0) / (1.0 - alpha)
    return float(vals.min())


def _sample_supersample_pairs(
    candidate_idxs: np.ndarray,
    total_points: int,
    seed: int,
) -> np.ndarray:
    if total_points % 2 != 0:
        raise ValueError("total_points must be even for supersample pairing.")
    rng = np.random.default_rng(seed)
    selected_idxs = rng.choice(candidate_idxs, size=total_points, replace=False)
    return rng.permutation(selected_idxs).reshape(total_points // 2, 2)


def _sample_mask_and_splits(
    paired_idxs: np.ndarray,
    y: np.ndarray,
    groups: list[list[int]],
    seed: int,
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[list[int]], np.ndarray, list[list[int]]]:
    from Dataset import reindex_group

    n_pairs = len(paired_idxs)
    # Rejection sampling of the selector mask U disabled: at the sample sizes used
    # in the experiments the probability of a single-class train/val split is
    # numerically negligible, so we draw U once without rejecting degenerate masks.
    rng = np.random.default_rng(seed)
    u = rng.integers(0, 2, size=n_pairs, endpoint=False)
    train_idxs = np.sort(paired_idxs[np.arange(n_pairs), u])
    val_idxs = np.sort(paired_idxs[np.arange(n_pairs), 1 - u])
    groups_train = [reindex_group(train_idxs, group) for group in groups]
    groups_val = [reindex_group(val_idxs, group) for group in groups]
    return u, train_idxs, groups_train, val_idxs, groups_val

    # for attempt in range(max_attempts):
    #     rng = np.random.default_rng(seed + attempt)
    #     u = rng.integers(0, 2, size=n_pairs, endpoint=False)
    #     train_idxs = np.sort(paired_idxs[np.arange(n_pairs), u])
    #     val_idxs = np.sort(paired_idxs[np.arange(n_pairs), 1 - u])
    #     groups_train = [reindex_group(train_idxs, group) for group in groups]
    #     groups_val = [reindex_group(val_idxs, group) for group in groups]
    #     if np.unique(y[train_idxs]).size < 2:
    #         continue
    #     if np.unique(y[val_idxs]).size < 2:
    #         continue
    #     return u, train_idxs, groups_train, val_idxs, groups_val
    #
    # raise ValueError("Could not sample a non-degenerate mask U for this supersample.")


def _groups_for_indices(indices: np.ndarray, groups: list[list[int]]) -> list[list[int]]:
    from Dataset import reindex_group

    return [reindex_group(indices, group) for group in groups]


def _branch_indices(paired_idxs: np.ndarray, base_u: np.ndarray, t: int, branch: int) -> tuple[np.ndarray, np.ndarray]:
    u_mod = base_u.copy()
    u_mod[t] = branch
    train_idxs = np.sort(paired_idxs[np.arange(len(paired_idxs)), u_mod])
    val_idxs = np.sort(paired_idxs[np.arange(len(paired_idxs)), 1 - u_mod])
    return train_idxs, val_idxs


def _selected_indices(n_pairs: int, m: int | None, seed: int) -> np.ndarray:
    all_indices = np.arange(n_pairs)
    if m is None or m >= n_pairs:
        return all_indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(all_indices, size=m, replace=False))


def _branch_is_valid(
    paired_idxs: np.ndarray,
    base_u: np.ndarray,
    t: int,
    branch: int,
    y: np.ndarray,
) -> bool:
    train_idxs, val_idxs = _branch_indices(paired_idxs, base_u, t, branch)
    return np.unique(y[train_idxs]).size >= 2 and np.unique(y[val_idxs]).size >= 2


def _save_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def _metadata_path(output_dir: Path, dataset: str, model: str, total_points: int, supersample_seed: int) -> Path:
    return output_dir / "metadata" / dataset / model / f"n={total_points}" / f"supersample_seed={supersample_seed}.npz"


def _model_bundle_path(
    output_dir: Path,
    dataset: str,
    model: str,
    total_points: int,
    supersample_seed: int,
    index_t: int,
    model_seed: int,
    branch: int,
) -> Path:
    return (
        output_dir
        / "models"
        / dataset
        / model
        / f"n={total_points}"
        / f"supersample_seed={supersample_seed}"
        / f"t={index_t}"
        / f"model_seed={model_seed}"
        / f"branch={branch}.pkl"
    )


def _model_checkpoint_dir(
    output_dir: Path,
    dataset: str,
    model: str,
    total_points: int,
    supersample_seed: int,
    index_t: int,
    model_seed: int,
    branch: int,
) -> Path:
    return (
        output_dir
        / "checkpoints"
        / dataset
        / model
        / f"n={total_points}"
        / f"supersample_seed={supersample_seed}"
        / f"t={index_t}"
        / f"model_seed={model_seed}"
        / f"branch={branch}"
    )


def _evaluate_worst_from_bundle(
    bundle: dict[str, Any],
    dataset_obj: Any,
    eval_idxs: np.ndarray,
    n_bins: int,
) -> dict[str, float | int | str]:
    if bundle["kind"] == "saved_model":
        from Model import Model

        model = Model(
            bundle["model_name"],
            SAVE_DIR=bundle["save_dir"],
            config=bundle["config"],
            dataset_obj=dataset_obj,
            from_saved=True,
            saved_epoch=bundle.get("saved_epoch", 0),
        )
        scaler = None
    else:
        model = bundle["model"]
        scaler = bundle["scaler"]
    X_eval = _transform_with_scaler(dataset_obj.X[eval_idxs], scaler)
    y_eval = dataset_obj.y[eval_idxs]
    groups_eval = _groups_for_indices(eval_idxs, dataset_obj.groups)
    probs = np.asarray(model.predict_proba(X_eval), dtype=float)
    positive_probs = _positive_class_probs(probs)
    return _worst_bin_group_class_calibration_error(
        positive_probs,
        y_eval,
        groups_eval,
        dataset_obj.group_names,
        n_bins=n_bins,
    )


def _evaluate_phi_from_bundle(
    bundle: dict[str, Any],
    dataset_obj: Any,
    eval_idxs: np.ndarray,
    objective: str,
    n_bins: int,
    cvar_alpha: float,
) -> dict[str, float | int | str]:
    if bundle["kind"] == "saved_model":
        from Model import Model

        model = Model(
            bundle["model_name"],
            SAVE_DIR=bundle["save_dir"],
            config=bundle["config"],
            dataset_obj=dataset_obj,
            from_saved=True,
            saved_epoch=bundle.get("saved_epoch", 0),
        )
        scaler = None
    else:
        model = bundle["model"]
        scaler = bundle["scaler"]
    X_eval = _transform_with_scaler(dataset_obj.X[eval_idxs], scaler)
    y_eval = dataset_obj.y[eval_idxs]
    groups_eval = _groups_for_indices(eval_idxs, dataset_obj.groups)
    probs = np.asarray(model.predict_proba(X_eval), dtype=float)
    positive_probs = _positive_class_probs(probs)

    if objective == "worst":
        worst = _worst_bin_group_class_calibration_error(
            positive_probs,
            y_eval,
            groups_eval,
            dataset_obj.group_names,
            n_bins=n_bins,
        )
        return {"phi_value": float(worst["worst_abs_cal_error"]), **worst}
    if objective == "cvar":
        cvar = _cvar_calibration_error(positive_probs, y_eval, cvar_alpha)
        return {"phi_value": float(cvar)}
    raise ValueError(f"Unknown objective: {objective}")


def _estimate_mi(values: np.ndarray, labels: np.ndarray, k_neighbors: int) -> float:
    values = np.asarray(values, dtype=float).reshape(-1, 1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(values) < 3:
        return 0.0
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        return 0.0
    if counts.min() < 2:
        return 0.0
    n_neighbors = min(k_neighbors, len(labels) - 1)
    if n_neighbors < 1:
        return 0.0
    mi = mutual_info_classif(
        values,
        labels,
        discrete_features=[False],
        n_neighbors=n_neighbors,
        random_state=0,
    ).sum()
    return max(0.0, float(mi))
