#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from supersample_retrain_worst_common import (
    DEFAULT_DATASETS,
    DEFAULT_MODELS,
    IMAGE_MODELS,
    REPO_ROOT,
    SIMPLE_MODELS,
    SUPER_SPLIT,
    _branch_indices,
    _branch_is_valid,
    _csv,
    _csv_ints,
    _default_sample_sizes,
    _ensure_even,
    _fit_scaler_if_needed,
    _metadata_path,
    _model_bundle_path,
    _model_checkpoint_dir,
    _sample_mask_and_splits,
    _sample_supersample_pairs,
    _sample_sizes_from_percentages,
    _save_pickle,
    _selected_indices,
    _set_global_seed,
    _transform_with_scaler,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/save retrained worst-calibration supersample models.")
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
    parser.add_argument(
        "--n-selected-indices",
        type=int,
        default=3,
        help="Number m of t-indices subsampled per supersample. Use <=0 for all.",
    )
    parser.add_argument("--index-seed", type=int, default=0, help="Seed for subsampling t-indices.")
    parser.add_argument("--groups-collection", default="default", help="Group collection name.")
    parser.add_argument(
        "--allow-complement-validation-selection",
        action="store_true",
        help=(
            "Use the complementary branch as the validation/checkpoint-selection set. "
            "Default is False, so W depends only on the training branch S."
        ),
    )
    parser.add_argument("--output-dir", default="results/supersample_retrain_worst_train", help="Output directory.")
    parser.add_argument(
        "--gpu",
        default=None,
        help=(
            "GPU id(s) to expose via CUDA_VISIBLE_DEVICES, e.g. 0 or 1. "
            "After this mapping, the model uses cuda:0 inside PyTorch."
        ),
    )
    parser.add_argument(
        "--train-log-interval",
        type=int,
        default=5000,
        help=(
            "Compute and print training accuracy/ECE every this many mini-batches. "
            "Larger values reduce overhead; use <=0 to disable intra-epoch train metrics."
        ),
    )
    parser.add_argument(
        "--loss-log-interval",
        type=int,
        default=100,
        help=(
            "Print cheap loss-only progress every this many mini-batches. "
            "This does not compute accuracy/ECE. Use <=0 to disable."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader num_workers for image/text models. On Windows, 0 is often more reliable; "
            "increase to 2 or 4 only after confirming it does not hang."
        ),
    )
    args = parser.parse_args()

    if args.gpu is not None and str(args.gpu).strip() != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[device] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (PyTorch will see this as cuda:0)")

    datasets = _csv(args.datasets)
    models = _csv(args.models)
    supersample_seeds = _csv_ints(args.supersample_seeds)
    model_seeds = _csv_ints(args.model_seeds)
    output_dir = Path(REPO_ROOT) / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    unsupported_models = sorted(set(models) - (SIMPLE_MODELS | IMAGE_MODELS))
    if unsupported_models:
        raise ValueError(
            f"Unsupported models for this script: {unsupported_models}. Supported: {sorted(SIMPLE_MODELS | IMAGE_MODELS)}"
        )

    for dataset_name in datasets:
        from Dataset import Dataset
        from Model import Model
        from configs.hyperparameters import get_hyperparameters

        dataset_obj = Dataset(
            dataset_name,
            split=SUPER_SPLIT,
            groups=args.groups_collection,
            scale=False,
            verbose=True,
        )
        candidate_size = len(dataset_obj.train_idxs) + len(dataset_obj.val_idxs)
        min_total_points = 20
        candidate_idxs = np.sort(np.concatenate([dataset_obj.train_idxs, dataset_obj.val_idxs]))

        if args.sample_sizes is not None and args.sample_size_percents is not None:
            raise ValueError("Use either --sample-sizes or --sample-size-percents, not both.")
        if args.sample_size_percents is not None:
            sample_sizes = _sample_sizes_from_percentages(
                candidate_size,
                min_total_points,
                [float(x) for x in _csv(args.sample_size_percents)],
            )
        elif args.sample_sizes is None:
            sample_sizes = _default_sample_sizes(candidate_size, min_total_points)
        else:
            sample_sizes = sorted({_ensure_even(int(n)) for n in _csv(args.sample_sizes)})
            sample_sizes = [n for n in sample_sizes if min_total_points <= n <= candidate_size]
            if not sample_sizes:
                raise ValueError(
                    f"No valid sample sizes remain for {dataset_name}. "
                    f"Allowed range: [{min_total_points}, {candidate_size}] and sizes must be even."
                )

        print(
            f"========== dataset={dataset_name} candidate_size={candidate_size} sample_sizes={sample_sizes} =========="
        )

        for model_name in models:
            hp = get_hyperparameters(model_name, dataset_name, 0)
            scale_data = bool(hp.get("scale_data", False))

            for total_points in sample_sizes:
                n_pairs = total_points // 2
                selected_indices = _selected_indices(
                    n_pairs,
                    None if args.n_selected_indices <= 0 else args.n_selected_indices,
                    args.index_seed,
                )
                print(
                    f"[retrain-train] dataset={dataset_name} model={model_name} "
                    f"supersample_seeds={supersample_seeds} n={total_points} m={len(selected_indices)}"
                )

                for supersample_seed in supersample_seeds:
                    paired_idxs = _sample_supersample_pairs(candidate_idxs, total_points, supersample_seed)
                    base_u, _, _, _, _ = _sample_mask_and_splits(
                        paired_idxs=paired_idxs,
                        y=dataset_obj.y,
                        groups=dataset_obj.groups,
                        seed=supersample_seed + 7919,
                    )
                    valid_indices = [
                        int(t)
                        for t in selected_indices
                        if _branch_is_valid(paired_idxs, base_u, int(t), 0, dataset_obj.y)
                        and _branch_is_valid(paired_idxs, base_u, int(t), 1, dataset_obj.y)
                    ]
                    metadata = {
                        "paired_idxs": paired_idxs,
                        "base_u": base_u,
                        "selected_indices": np.asarray(valid_indices, dtype=int),
                        "n_selected_indices": len(valid_indices),
                    }
                    meta_path = _metadata_path(output_dir, dataset_name, model_name, total_points, supersample_seed)
                    meta_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(meta_path, **metadata)

                    for index_t in valid_indices:
                        for model_seed in model_seeds:
                            for branch in (0, 1):
                                train_idxs, val_idxs = _branch_indices(paired_idxs, base_u, index_t, branch)
                                X_train_raw = dataset_obj.X[train_idxs]
                                y_train = dataset_obj.y[train_idxs]
                                X_val_raw = dataset_obj.X[val_idxs]
                                y_val = dataset_obj.y[val_idxs]
                                from Dataset import reindex_group

                                branch_groups_train = [reindex_group(train_idxs, group) for group in dataset_obj.groups]
                                branch_groups_val = [reindex_group(val_idxs, group) for group in dataset_obj.groups]

                                # For the supersample theory, W should be A(S, R), not A(S, S', R).
                                # By default, neural checkpoint callbacks therefore do not see the
                                # complementary branch.  To reproduce the old behavior, pass
                                # --allow-complement-validation-selection.
                                if args.allow_complement_validation_selection:
                                    selection_source = "complement"
                                    X_selection_raw = X_val_raw
                                    y_selection = y_val
                                    branch_groups_selection = branch_groups_val
                                else:
                                    selection_source = "train"
                                    X_selection_raw = X_train_raw
                                    y_selection = y_train
                                    branch_groups_selection = branch_groups_train

                                scaler = _fit_scaler_if_needed(X_train_raw, scale_data)
                                X_train = _transform_with_scaler(X_train_raw, scaler)
                                X_selection = _transform_with_scaler(X_selection_raw, scaler)
                                config = {
                                    "model": model_name,
                                    "dataset": dataset_name,
                                    "groups_collection": args.groups_collection,
                                    "calib_frac": 0,
                                    "val_split_seed": supersample_seed,
                                    "split": SUPER_SPLIT,
                                    "random_state": model_seed,
                                    "val_save_epoch": 0,
                                    "val_eval_epoch": 1,
                                    "allow_complement_validation_selection": bool(args.allow_complement_validation_selection),
                                    "selection_source": selection_source,
                                    "checkpoint_selection": "best-val-acc" if args.allow_complement_validation_selection else "final",
                                    "train_log_interval": int(args.train_log_interval),
                                    "loss_log_interval": int(args.loss_log_interval),
                                    "num_workers": int(args.num_workers),
                                    **hp,
                                }
                                _set_global_seed(model_seed)
                                save_dir = None
                                if model_name in IMAGE_MODELS:
                                    save_dir = str(
                                        _model_checkpoint_dir(
                                            output_dir,
                                            dataset_name,
                                            model_name,
                                            total_points,
                                            supersample_seed,
                                            index_t,
                                            model_seed,
                                            branch,
                                        )
                                    ) + "/"
                                save_scheme = "best-val-acc" if args.allow_complement_validation_selection else "final"
                                model = Model(
                                    model_name,
                                    config=config,
                                    dataset_obj=dataset_obj,
                                    SAVE_DIR=save_dir,
                                    save_scheme=save_scheme if model_name == "ViT" else "best-val-acc",
                                )
                                model.train(
                                    X_train,
                                    y_train,
                                    branch_groups_train,
                                    X_selection,
                                    y_selection,
                                    branch_groups_selection,
                                )
                                if model_name in IMAGE_MODELS:
                                    bundle = {
                                        "kind": "saved_model",
                                        "model_name": model_name,
                                        "save_dir": save_dir,
                                        "config": config,
                                        "saved_epoch": int(config["epochs"] - 1),
                                        "dataset": dataset_name,
                                        "base_u_t": int(base_u[index_t]),
                                        "index_t": int(index_t),
                                        "branch": int(branch),
                                        "model_seed": int(model_seed),
                                        "supersample_seed": int(supersample_seed),
                                        "selection_source": selection_source,
                                        "allow_complement_validation_selection": bool(args.allow_complement_validation_selection),
                                        "checkpoint_selection": config.get("checkpoint_selection", "best-val-acc"),
                                    }
                                else:
                                    bundle = {
                                        "kind": "pickle_model",
                                        "model": model,
                                        "scaler": scaler,
                                        "dataset": dataset_name,
                                        "base_u_t": int(base_u[index_t]),
                                        "index_t": int(index_t),
                                        "branch": int(branch),
                                        "model_seed": int(model_seed),
                                        "supersample_seed": int(supersample_seed),
                                        "selection_source": selection_source,
                                        "allow_complement_validation_selection": bool(args.allow_complement_validation_selection),
                                        "checkpoint_selection": config.get("checkpoint_selection", "best-val-acc"),
                                    }
                                bundle_path = _model_bundle_path(
                                    output_dir,
                                    dataset_name,
                                    model_name,
                                    total_points,
                                    supersample_seed,
                                    index_t,
                                    model_seed,
                                    branch,
                                )
                                _save_pickle(bundle_path, bundle)

    print(f"Saved model bundles under {output_dir}")


if __name__ == "__main__":
    main()
