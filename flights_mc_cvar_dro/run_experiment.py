#!/usr/bin/env python3
"""Run the Flights200K MC, CVaR, and CR-DRO experiments and all four figures."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from math import comb
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory
import time
from urllib.request import urlopen
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    FixedFormatter, FixedLocator, LogFormatterSciNotation, NullFormatter, NullLocator,
)
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


DATA_URL = "https://cdn.jsdelivr.net/npm/vega-datasets@3.2.1/data/flights-200k.json"
SAMPLE_SIZES = (512, 1024, 2048, 4096, 8192, 16384, 32768)
SIGMAS = (0.05, 0.10)
PARAMETER_SETTINGS = (
    (1.25, 0.1),
    (1.5, 0.1),
    (2.0, 0.1),
    (4.0, 0.1),
    (2.0, 0.0),
    (2.0, 0.01),
    (2.0, 1.0),
)
CVAR_ALPHAS = (0.2, 0.5, 0.8, 0.95)
N_BINS = 5
QUADRATURE_ORDER = 80
PROTOCOL_VERSION = 3
COLORS = ("#155778", "#C26C21", "#297A5C", "#855BA1")
MARKERS = ("o", "s", "^", "D")
METRICS = (
    "train_value", "validation_value", "signed_gap", "dro_gap",
    "gaussian_l2_bound", "information_sum",
)
CVAR_METRICS = (
    "train_value", "validation_value", "signed_gap", "cvar_gap",
    "gaussian_cvar_bound", "information_sum",
)


def obtain_data(path: Path) -> Path:
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Flights200K from vega-datasets 3.2.1 ...", flush=True)
    # A failed or interrupted download must not become the cached dataset.
    with TemporaryDirectory(prefix="flights-download-", dir=path.parent) as folder:
        temporary_path = Path(folder) / "flights-200k.json"
        try:
            with urlopen(DATA_URL, timeout=90) as response:
                with temporary_path.open("wb") as destination:
                    shutil.copyfileobj(response, destination)
        except OSError as error:
            raise RuntimeError(
                "Flights200K download failed. Retry this command, or download "
                f"{DATA_URL} and pass its path with --data-path."
            ) from error
        temporary_path.replace(path)
    return path


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Seven fixed, label-independent features and a binary delay label."""
    frame = pd.read_json(path)
    if set(frame.columns) != {"delay", "distance", "time"} or len(frame) != 200000:
        raise ValueError("Expected the 200000-row Flights200K delay/distance/time table.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Flights200K contains non-finite values.")

    distance = frame["distance"].to_numpy(dtype=float)
    time_hours = frame["time"].to_numpy(dtype=float)
    if np.any(distance < 0) or np.any((time_hours < 0) | (time_hours >= 24)):
        raise ValueError("Flights200K distance or time is outside its expected range.")
    log_distance = np.log1p(distance) / np.log1p(5000.0)
    phase = 2.0 * np.pi * time_hours / 24.0
    time_sin = np.sin(phase)
    time_cos = np.cos(phase)
    features = np.column_stack(
        [
            log_distance,
            time_sin,
            time_cos,
            log_distance * time_sin,
            log_distance * time_cos,
            (time_hours < 6.0).astype(float),
            (time_hours >= 18.0).astype(float),
        ]
    ).astype(np.float64, copy=False)
    labels = (frame["delay"].to_numpy(dtype=float) > 15.0).astype(int)
    return features, labels


def fit_center(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return seven logistic coefficients followed by the intercept."""
    if np.unique(labels).size < 2:
        positive_rate = (np.sum(labels) + 1.0) / (len(labels) + 2.0)
        center = np.zeros(features.shape[1] + 1, dtype=float)
        center[-1] = np.log(positive_rate / (1.0 - positive_rate))
        return center
    model = LogisticRegression(
        C=1.0, solver="newton-cholesky", tol=1e-10, max_iter=100
    )
    model.fit(features, labels)
    return np.concatenate([model.coef_.reshape(-1), model.intercept_.reshape(-1)])


def gaussian_information(standardized_distances: np.ndarray) -> np.ndarray:
    """Binary equal-covariance Gaussian-channel MI in nats, using quadrature."""
    nodes, weights = np.polynomial.hermite.hermgauss(QUADRATURE_ORDER)
    normal_nodes = np.sqrt(2.0) * nodes
    distances = np.asarray(standardized_distances, dtype=float).reshape(-1, 1)
    log_ratio = 0.5 * distances**2 + distances * normal_nodes[None, :]
    conditional_entropy = np.logaddexp(0.0, -log_ratio) @ weights / np.sqrt(np.pi)
    return np.clip(np.log(2.0) - conditional_entropy, 0.0, np.log(2.0))


def make_context(
    features: np.ndarray, n_train: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw fresh reference data, candidate pairs, and independent fair bits."""
    split_seeds = rng.integers(0, 2**32, size=2, dtype=np.uint32)
    remaining, _ = train_test_split(
        np.arange(len(features)), test_size=0.20,
        random_state=int(split_seeds[0]), shuffle=True,
    )
    candidates, reference = train_test_split(
        remaining,
        test_size=0.10 / (1.0 - 0.20),
        random_state=int(split_seeds[1]),
        shuffle=True,
    )
    thresholds = np.median(features[reference][:, [0, 1, 2]], axis=0)
    selected = rng.choice(candidates, size=2 * n_train, replace=False)
    pairs = rng.permutation(selected).reshape(n_train, 2)
    selector = rng.integers(0, 2, size=n_train, endpoint=False)
    return pairs, selector, thresholds


def group_masks(features: np.ndarray, thresholds: np.ndarray) -> list[np.ndarray]:
    masks = [np.ones(len(features), dtype=bool)]
    for feature_index, threshold in enumerate(thresholds):
        masks.extend(
            [features[:, feature_index] <= threshold, features[:, feature_index] > threshold]
        )
    first_high = features[:, 0] > thresholds[0]
    second_high = features[:, 1] > thresholds[1]
    masks.extend([first_high & second_high, (~first_high) & (~second_high)])
    return masks


def multicalibration(
    parameters: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    masks: list[np.ndarray],
) -> float:
    logits = np.clip(features @ parameters[:-1] + parameters[-1], -40.0, 40.0)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    bin_indices = np.minimum((probabilities * N_BINS).astype(int), N_BINS - 1)
    residuals = labels - probabilities
    value = 0.0
    for mask in masks:
        for bin_index in range(N_BINS):
            selected = mask & (bin_indices == bin_index)
            # The denominator is the full sample size, including points outside the group.
            moment = float(np.mean(selected.astype(float) * residuals))
            value = max(value, abs(moment))
    return value

def _bracketed_roots(evaluate, upper, log_budget):
    """Solve increasing log-moment equations by bisection."""
    lower = np.zeros_like(upper)
    upper = upper.copy()
    roots = np.empty_like(upper)
    pending = np.arange(upper.size)

    for _ in range(90):
        current = (lower[pending] + upper[pending]) / 2
        values = evaluate(current, pending)
        residual = values - log_budget
        above = residual > 0
        upper[pending[above]] = current[above]
        lower[pending[~above]] = current[~above]
        widths = upper[pending] - lower[pending]
        finished = widths <= 2e-13 * np.maximum(1.0, np.abs(current))
        roots[pending[finished]] = current[finished]
        if np.all(finished):
            return roots
        pending = pending[~finished]

    raise RuntimeError("Cressie--Read moment equation did not converge")


def _polynomial_moment(moments, degree, points):
    """Return E[(1 + points * centered_loss)^degree] and its derivative."""
    value = moments[degree].copy()
    derivative = np.zeros_like(points)
    for order in range(degree - 1, -1, -1):
        derivative = derivative * points + value
        value = value * points + comb(degree, order) * moments[order]
    return value, derivative


def _log_polynomial_moment(moments, degree, points):
    """Keep the centered moment's small excess over one before taking its log."""
    excess = moments[degree].copy()
    for order in range(degree - 1, 1, -1):
        excess = excess * points + comb(degree, order) * moments[order]
    return np.log1p(excess * points * points)


def _integer_full_support_values(centered, means, variances, exponent, c, log_budget):
    """Solve the all-positive-weight cases for integer exponents 2 and 4.

    Centered moments make each root evaluation independent of the sample size.
    NaN marks a column whose optimum requires truncating some weights to zero.
    """
    degree = exponent + 1
    moments = np.empty((degree + 1, centered.shape[1]))
    moments[0] = 1.0
    moments[1] = 0.0
    moments[2] = variances
    powers = centered * centered
    for order in range(3, degree + 1):
        powers *= centered
        moments[order] = np.mean(powers, axis=0)

    upper = 1.0 / means
    lower_moment, _ = _polynomial_moment(moments, exponent, upper)
    higher_moment, _ = _polynomial_moment(moments, degree, upper)
    boundary = np.log(higher_moment) - c * np.log(lower_moment)
    full_support = boundary >= log_budget
    result = np.full_like(means, np.nan)
    if not np.any(full_support):
        return result

    selected_moments = moments[:, full_support]
    selected_means = means[full_support]

    def evaluate(points, indices):
        current_moments = selected_moments[:, indices]
        if log_budget < 1e-6:
            lower_log = _log_polynomial_moment(current_moments, exponent, points)
            higher_log = _log_polynomial_moment(current_moments, degree, points)
            return higher_log - c * lower_log
        lower_value, _ = _polynomial_moment(current_moments, exponent, points)
        higher_value, _ = _polynomial_moment(current_moments, degree, points)
        return np.log(higher_value) - c * np.log(lower_value)

    roots = _bracketed_roots(evaluate, upper[full_support], log_budget)
    denominator, _ = _polynomial_moment(selected_moments, exponent, roots)
    _, numerator_derivative = _polynomial_moment(selected_moments, degree, roots)
    result[full_support] = selected_means + numerator_derivative / (degree * denominator)
    return result


def _powers(values, exponent):
    if exponent == 1.0:
        return values
    if exponent == 2.0:
        return values * values
    if exponent == 4.0:
        squared = values * values
        return squared * squared
    if exponent == 1.0 / 3.0:
        return np.cbrt(values)
    return values ** exponent


def _general_values(losses, c, log_budget):
    """Solve the truncated power-weight moment equation for each column."""
    exponent = 1.0 / (c - 1.0)
    distances = np.asfortranarray(1.0 - losses)
    count = losses.shape[1]

    def moment_values(points, indices):
        selected = distances if indices.size == count else distances[:, indices]
        positive = np.maximum(1.0 - selected * points, 0.0)
        weights = _powers(positive, exponent)
        moment = np.mean(weights, axis=0)
        higher_moment = np.mean(weights * positive, axis=0)
        log_ratio = np.log(higher_moment) - c * np.log(moment)
        if log_budget < 1e-6:
            near_uniform = (exponent + 1.0) * points < 0.01
            if np.any(near_uniform):
                # Avoid rounding the nearly uniform power moments to one.
                log_positive = np.log1p(
                    -selected[:, near_uniform] * points[near_uniform]
                )
                lower_log = np.log1p(
                    np.mean(np.expm1(exponent * log_positive), axis=0)
                )
                higher_log = np.log1p(
                    np.mean(np.expm1((exponent + 1.0) * log_positive), axis=0)
                )
                log_ratio[near_uniform] = higher_log - c * lower_log
        return log_ratio

    upper = np.ones(count)
    pending = np.arange(count)
    for _ in range(64):
        values = moment_values(upper[pending], pending)
        pending = pending[values < log_budget]
        if pending.size == 0:
            break
        upper[pending] *= 2.0
    else:
        raise RuntimeError("Could not bracket the Cressie--Read moment equation")

    roots = _bracketed_roots(moment_values, upper, log_budget)
    positive = np.maximum(1.0 - distances * roots, 0.0)
    weights = _powers(positive, exponent)
    result = np.sum(weights * losses, axis=0) / np.sum(weights, axis=0)
    moment = np.mean(weights, axis=0)
    higher_moment = np.mean(weights * positive, axis=0)
    residual = np.log(higher_moment) - c * np.log(moment) - log_budget
    unresolved = np.abs(residual) > 2e-12 * max(1.0, abs(log_budget))
    if np.any(unresolved):
        # Near a changing support boundary, a tiny threshold interval can still
        # produce noticeably different weights when c > 2. The dual objective
        # is smooth there and gives an accurate value even in that interval.
        log_scale = log_budget / c + np.log(higher_moment[unresolved]) / (exponent + 1.0)
        result[unresolved] = 1.0 + np.expm1(log_scale) / roots[unresolved]
    return result


def cr_dro_values(losses, c, epsilon):
    """Return one CR-DRO value per column of losses in [0, 1].

    ``losses`` is an n-by-k array, or a one-dimensional array for one column.
    The output always has shape (k,). The divergence generator is
    (t**c - c*t + c - 1) / (c*(c - 1)), with c > 1 and epsilon >= 0.
    Numerical threshold searches use bisection, following Duchi and Namkoong
    (2021), Section 3, https://doi.org/10.1214/20-AOS2004.
    """
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("losses must be a nonempty one- or two-dimensional array")
    if not np.isfinite(c) or c <= 1:
        raise ValueError("c must be finite and greater than one")
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    if values.shape[1] == 0:
        return np.empty(0, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("losses must contain finite values in [0, 1]")
    if epsilon == 0:
        return np.mean(values, axis=0)

    minimum = np.min(values, axis=0)
    maximum = np.max(values, axis=0)
    ranges = maximum - minimum
    result = minimum.copy()
    active = np.flatnonzero(ranges > 0)
    if active.size == 0:
        return result

    log_budget = np.log1p(c * (c - 1.0) * epsilon)
    maxima_count = np.sum(values[:, active] == maximum[active], axis=0)
    log_saturation = (c - 1.0) * np.log(values.shape[0] / maxima_count)
    saturation_tolerance = 32 * np.finfo(float).eps * np.maximum(1.0, log_saturation)
    saturated = log_budget >= log_saturation - saturation_tolerance
    result[active[saturated]] = maximum[active[saturated]]
    active = active[~saturated]
    if active.size == 0:
        return result

    normalized = np.asfortranarray(
        (values[:, active] - minimum[active]) / ranges[active]
    )
    means = np.mean(normalized, axis=0)
    centered = normalized - means
    variances = np.mean(centered * centered, axis=0)
    normalized_result = np.full(active.size, np.nan)

    if c == 2.0:
        full_support = variances >= 2.0 * epsilon * means * means
        normalized_result[full_support] = means[full_support] + np.sqrt(
            2.0 * epsilon * variances[full_support]
        )
    else:
        exponent = 1.0 / (c - 1.0)
        if exponent in (2.0, 4.0):
            normalized_result = _integer_full_support_values(
                centered, means, variances, int(exponent), c, log_budget
            )

    remaining = np.flatnonzero(np.isnan(normalized_result))
    if remaining.size:
        normalized_result[remaining] = _general_values(
            normalized[:, remaining], c, log_budget,
        )
    result[active] = minimum[active] + ranges[active] * np.clip(
        normalized_result, means, 1.0
    )
    return result


def cvar_values(losses: np.ndarray, alphas=CVAR_ALPHAS) -> np.ndarray:
    """Return upper-tail CVaR values with shape (number of alphas, columns).

    Alpha is a confidence level, so the averaged tail has mass 1 - alpha.
    A single descending sort serves every alpha. Fractional observations at
    the tail boundary receive their exact remaining empirical mass.
    """
    values = np.asarray(losses, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("losses must be a nonempty one- or two-dimensional array")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("losses must contain finite values in [0, 1]")
    alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
    if (
        alphas.ndim != 1 or not alphas.size
        or not np.isfinite(alphas).all()
        or np.any((alphas < 0) | (alphas >= 1))
    ):
        raise ValueError("alphas must be confidence levels in [0, 1)")

    descending = np.sort(values, axis=0)[::-1]
    cumulative = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(descending, axis=0)])
    tail_counts = values.shape[0] * (1.0 - alphas)
    complete_counts = np.floor(tail_counts).astype(int)
    fractional_counts = tail_counts - complete_counts
    next_indices = np.minimum(complete_counts, values.shape[0] - 1)
    tail_sums = (
        cumulative[complete_counts]
        + fractional_counts[:, None] * descending[next_indices]
    )
    return tail_sums / tail_counts[:, None]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, allow_nan=False, sort_keys=True)
            stream.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def file_metadata(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}


def prepare_checkpoints(args: argparse.Namespace) -> tuple[Path, str, dict]:
    checkpoint_directory = args.output_dir / "context_checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_directory / "manifest.json"
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "sample_sizes": list(SAMPLE_SIZES),
        "sigmas": list(SIGMAS),
        "parameter_settings": [list(item) for item in PARAMETER_SETTINGS],
        "cvar_alphas": list(CVAR_ALPHAS),
        "mc_bins": N_BINS,
        "seed": args.seed,
        "independent_runs": args.independent_runs,
        "outer_repeats": args.outer_repeats,
        "noise_draws": args.noise_draws,
        "selected_pairs": args.selected_pairs,
        "quadrature_order": QUADRATURE_ORDER,
        "loss": "Brier loss of the Gaussian-perturbed logistic probability",
        "data": file_metadata(args.data_path),
        "code": file_metadata(Path(__file__).resolve()),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "joblib", "threadpoolctl"]
        },
    }
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol") != protocol:
            raise ValueError(
                "The checkpoints use different data, code, package versions, or settings. "
                "Run without --resume to recompute the experiment."
            )
        return checkpoint_directory, manifest["run_token"], protocol

    run_token = uuid.uuid4().hex
    atomic_json(manifest_path, {"protocol": protocol, "run_token": run_token})
    return checkpoint_directory, run_token, protocol


def read_checkpoint(
    path: Path, key: tuple[int, int, int, int], run_token: str, selected_pairs: int,
) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("run_token") != run_token:
        return None
    if payload.get("key") != list(key):
        raise ValueError(f"Checkpoint settings do not match the requested context: {path}")
    for name, count in [
        ("train_values", len(PARAMETER_SETTINGS)),
        ("validation_values", len(PARAMETER_SETTINGS)),
        ("cvar_train_values", len(CVAR_ALPHAS)),
        ("cvar_validation_values", len(CVAR_ALPHAS)),
    ]:
        values = np.asarray(payload[name], dtype=float)
        if (
            values.shape != (count,)
            or not np.isfinite(values).all()
            or np.any((values < -1e-10) | (values > 1 + 1e-10))
        ):
            raise ValueError(f"Invalid objective estimates in checkpoint: {path}")
    mc_signed_gap = np.asarray(payload["mc_signed_gap"], dtype=float)
    if (
        mc_signed_gap.shape != ()
        or not np.isfinite(mc_signed_gap)
        or abs(mc_signed_gap) > 1 + 1e-10
    ):
        raise ValueError(f"Invalid MC gap estimate in checkpoint: {path}")
    information = np.asarray(payload["parameter_information"], dtype=float)
    if (
        information.shape != (selected_pairs,)
        or not np.isfinite(information).all()
        or np.any((information < 0) | (information > np.log(2.0) + 1e-12))
    ):
        raise ValueError(f"Invalid information estimates in checkpoint: {path}")
    return payload


def run_context(
    features: np.ndarray,
    labels: np.ndarray,
    key: tuple[int, int, int, int],
    context_seed: np.random.SeedSequence,
    pair_indices: np.ndarray,
    noise_draws: int,
    checkpoint_path: Path,
    run_token: str,
) -> tuple[tuple[int, int, int, int], dict]:
    """Compute one context, including all objective settings and Gaussian MI."""
    _, size_index, sigma_index, _ = key
    n_train = SAMPLE_SIZES[size_index]
    sigma = SIGMAS[sigma_index]
    data_seed, noise_seed = context_seed.spawn(2)
    pairs, selector, thresholds = make_context(
        features, n_train, np.random.default_rng(data_seed)
    )
    positions = np.arange(n_train)
    train_indices = pairs[positions, selector]
    validation_indices = pairs[positions, 1 - selector]
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    validation_features = features[validation_indices]
    validation_labels = labels[validation_indices]
    train_masks = group_masks(train_features, thresholds)
    validation_masks = group_masks(validation_features, thresholds)

    with threadpool_limits(limits=1):
        center = fit_center(train_features, train_labels)
        squared_distances = []
        for pair_index in pair_indices:
            other_indices = train_indices.copy()
            other_indices[pair_index] = pairs[pair_index, 1 - selector[pair_index]]
            other_center = fit_center(features[other_indices], labels[other_indices])
            difference = center - other_center
            squared_distances.append(float(difference @ difference))
        information = gaussian_information(np.sqrt(squared_distances) / sigma)

        noise_rng = np.random.default_rng(noise_seed)
        noise = noise_rng.normal(size=(noise_draws, len(center)))
        parameters = center[None, :] + sigma * noise
        train_logits = np.clip(
            train_features @ parameters[:, :-1].T + parameters[:, -1], -40.0, 40.0
        )
        validation_logits = np.clip(
            features[validation_indices] @ parameters[:, :-1].T + parameters[:, -1],
            -40.0, 40.0,
        )
        train_probabilities = 1.0 / (1.0 + np.exp(-train_logits))
        validation_probabilities = 1.0 / (1.0 + np.exp(-validation_logits))
        train_losses = (train_probabilities - train_labels[:, None])**2
        validation_losses = (
            validation_probabilities - labels[validation_indices, None]
        )**2
        train_values = []
        validation_values = []
        for c, epsilon in PARAMETER_SETTINGS:
            train = cr_dro_values(train_losses, c, epsilon)
            validation = cr_dro_values(validation_losses, c, epsilon)
            values = np.concatenate([train, validation])
            if (
                train.shape != (noise_draws,)
                or validation.shape != train.shape
                or not np.isfinite(values).all()
                or np.any((values < -1e-10) | (values > 1 + 1e-10))
            ):
                raise RuntimeError("Invalid bounded-loss DRO estimates.")
            train_values.append(float(train.mean()))
            validation_values.append(float(validation.mean()))
        cvar_train_values = cvar_values(train_losses).mean(axis=1)
        cvar_validation_values = cvar_values(validation_losses).mean(axis=1)

        mc_signed_gaps = []
        for perturbed_parameters in parameters:
            train_mc = multicalibration(
                perturbed_parameters, train_features, train_labels, train_masks
            )
            validation_mc = multicalibration(
                perturbed_parameters, validation_features, validation_labels, validation_masks
            )
            mc_signed_gaps.append(validation_mc - train_mc)

    payload = {
        "run_token": run_token,
        "key": list(key),
        "train_values": train_values,
        "validation_values": validation_values,
        "cvar_train_values": cvar_train_values.tolist(),
        "cvar_validation_values": cvar_validation_values.tolist(),
        "mc_signed_gap": float(np.mean(mc_signed_gaps)),
        "parameter_information": information.tolist(),
    }
    atomic_json(checkpoint_path, payload)
    return key, payload


def summarize_runs(
    train_values: np.ndarray,
    validation_values: np.ndarray,
    information: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Average contexts before taking the gap's absolute value or MI roots."""
    rows = []
    for run_index in range(args.independent_runs):
        for size_index, n_train in enumerate(SAMPLE_SIZES):
            for sigma_index, sigma in enumerate(SIGMAS):
                key = (run_index, size_index, sigma_index)
                train_means = train_values[key].mean(axis=0)
                validation_means = validation_values[key].mean(axis=0)
                signed_gaps = (validation_values[key] - train_values[key]).mean(axis=0)
                mean_information = information[key].mean(axis=0)
                sum_information = float(n_train * mean_information.mean())
                for parameter_index, (c, epsilon) in enumerate(PARAMETER_SETTINGS):
                    kappa = float(np.exp(np.log1p(c * (c - 1.0) * epsilon) / c))
                    if epsilon == 0:
                        l2_bound = float(np.sqrt(2.0 * sum_information / n_train))
                    else:
                        exponent = min(0.5, (c - 1.0) / c)
                        l2_bound = float(
                            2.0 * kappa * n_train**(-exponent) * np.sqrt(2.0 * sum_information)
                        )
                    rows.append({
                        "run_id": run_index + 1,
                        "n_train": n_train,
                        "sigma": sigma,
                        "c": c,
                        "epsilon": epsilon,
                        "train_value": float(train_means[parameter_index]),
                        "validation_value": float(validation_means[parameter_index]),
                        "signed_gap": float(signed_gaps[parameter_index]),
                        "dro_gap": float(abs(signed_gaps[parameter_index])),
                        "gaussian_l2_bound": l2_bound,
                        "information_sum": sum_information,
                        "outer_repeats": args.outer_repeats,
                        "noise_draws": args.noise_draws,
                        "selected_pairs": args.selected_pairs,
                        "seed": args.seed,
                    })
    return pd.DataFrame(rows)


def summarize_cvar_runs(
    train_values: np.ndarray,
    validation_values: np.ndarray,
    information: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Use Corollary 5 with the same estimate of total information as CR DRO."""
    rows = []
    for run_index in range(args.independent_runs):
        for size_index, n_train in enumerate(SAMPLE_SIZES):
            for sigma_index, sigma in enumerate(SIGMAS):
                key = (run_index, size_index, sigma_index)
                train_means = train_values[key].mean(axis=0)
                validation_means = validation_values[key].mean(axis=0)
                signed_gaps = (validation_values[key] - train_values[key]).mean(axis=0)
                mean_information = information[key].mean(axis=0)
                sum_information = float(n_train * mean_information.mean())
                for alpha_index, alpha in enumerate(CVAR_ALPHAS):
                    tail_mass = 1.0 - alpha
                    coefficient = min(
                        1.0 / (tail_mass * np.sqrt(n_train)),
                        2.0 / np.sqrt(tail_mass * n_train),
                    )
                    bound = coefficient * np.sqrt(2.0 * sum_information)
                    rows.append({
                        "run_id": run_index + 1,
                        "n_train": n_train,
                        "sigma": sigma,
                        "alpha": alpha,
                        "tail_mass": tail_mass,
                        "train_value": float(train_means[alpha_index]),
                        "validation_value": float(validation_means[alpha_index]),
                        "signed_gap": float(signed_gaps[alpha_index]),
                        "cvar_gap": float(abs(signed_gaps[alpha_index])),
                        "gaussian_cvar_bound": float(bound),
                        "information_sum": sum_information,
                        "outer_repeats": args.outer_repeats,
                        "noise_draws": args.noise_draws,
                        "selected_pairs": args.selected_pairs,
                        "seed": args.seed,
                    })
    return pd.DataFrame(rows)


def summarize(
    independent_runs: pd.DataFrame,
    grouping=("n_train", "sigma", "c", "epsilon"),
    metrics=METRICS,
) -> pd.DataFrame:
    """The plotted SD is the sample SD of complete independent estimates."""
    rows = []
    grouping = list(grouping)
    if independent_runs.duplicated(["run_id"] + grouping).any():
        raise ValueError("Duplicate independent-run rows were supplied.")
    for keys, subset in independent_runs.groupby(grouping, sort=True):
        if len(subset) < 2:
            raise ValueError("At least two independent runs are required to report a sample SD.")
        row = dict(zip(grouping, keys))
        row["independent_runs"] = len(subset)
        for metric in metrics:
            values = subset[metric].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {metric} values in independent runs.")
            mean = float(values.mean())
            standard_deviation = float(values.std(ddof=1))
            row[metric] = mean
            row[metric + "_std"] = standard_deviation
            row[metric + "_sd_lower"] = mean - standard_deviation
            row[metric + "_sd_upper"] = mean + standard_deviation
        row["negative_signed_runs"] = int((subset["signed_gap"] < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_mc_runs(
    signed_gaps: np.ndarray,
    information: np.ndarray,
    noise_draws: int,
    seed: int,
) -> pd.DataFrame:
    """Keep the MI context expectation inside the square root in every run."""
    rows = []
    for run_index in range(signed_gaps.shape[0]):
        for size_index, n_train in enumerate(SAMPLE_SIZES):
            for sigma_index, sigma in enumerate(SIGMAS):
                signed_gap = float(signed_gaps[run_index, size_index, sigma_index].mean())
                mi_matrix = information[run_index, size_index, sigma_index]
                bound = 2.0 * np.mean(np.sqrt(2.0 * mi_matrix.mean(axis=0)))
                rows.append(
                    {
                        "run_id": run_index + 1,
                        "n_train": n_train,
                        "sigma": sigma,
                        "signed_gap": signed_gap,
                        "mc_gap": abs(signed_gap),
                        "gaussian_bound": float(bound),
                        "outer_repeats": signed_gaps.shape[-1],
                        "noise_draws": noise_draws,
                        "selected_pairs": information.shape[-1],
                        "seed": seed,
                    }
                )
    return pd.DataFrame(rows)


def summarize_mc(independent_runs: pd.DataFrame) -> pd.DataFrame:
    """Report variability across independently recomputed experiment estimates."""
    rows = []
    for n_train in SAMPLE_SIZES:
        for sigma in SIGMAS:
            subset = independent_runs[
                (independent_runs["n_train"] == n_train)
                & (independent_runs["sigma"] == sigma)
            ]
            signed_mean = float(subset["signed_gap"].mean())
            signed_std = float(subset["signed_gap"].std(ddof=1))
            gap_mean = float(subset["mc_gap"].mean())
            gap_std = float(subset["mc_gap"].std(ddof=1))
            bound_mean = float(subset["gaussian_bound"].mean())
            bound_std = float(subset["gaussian_bound"].std(ddof=1))
            # Both curves summarize the per-run estimates by their mean and SD.
            # Each run's gap estimate already includes its absolute value.
            settings = subset.iloc[0]
            rows.append(
                {
                    "n_train": n_train,
                    "sigma": sigma,
                    "mc_gap": gap_mean,
                    "mc_gap_std": gap_std,
                    "mc_gap_variance": gap_std**2,
                    "mc_gap_sd_lower": gap_mean - gap_std,
                    "mc_gap_sd_upper": gap_mean + gap_std,
                    "signed_gap_mean": signed_mean,
                    "signed_gap_std": signed_std,
                    "signed_gap_variance": signed_std**2,
                    "signed_gap_sd_lower": signed_mean - signed_std,
                    "signed_gap_sd_upper": signed_mean + signed_std,
                    "gaussian_bound": bound_mean,
                    "gaussian_bound_std": bound_std,
                    "gaussian_bound_variance": bound_std**2,
                    "gaussian_bound_sd_lower": bound_mean - bound_std,
                    "gaussian_bound_sd_upper": bound_mean + bound_std,
                    "independent_runs": len(subset),
                    "outer_repeats": int(settings["outer_repeats"]),
                    "noise_draws": int(settings["noise_draws"]),
                    "selected_pairs": int(settings["selected_pairs"]),
                    "seed": int(settings["seed"]),
                }
            )
    return pd.DataFrame(rows)


def axis_limits(estimates: pd.DataFrame, metrics: tuple[str, ...]) -> tuple[float, float]:
    columns = [
        name
        for metric in metrics
        for name in (metric, metric + "_sd_lower", metric + "_sd_upper")
    ]
    values = estimates[columns].to_numpy()
    positive = values[values > 0]
    if not positive.size:
        raise ValueError("A logarithmic panel needs positive estimates.")
    # Use the same data-driven limits for both sigma panels, including the STD bars.
    log_minimum = np.log10(positive.min())
    log_maximum = np.log10(positive.max())
    margin = max(0.12, 0.04 * (log_maximum - log_minimum))
    return 10.0**(log_minimum - margin), 10.0**(log_maximum + margin)


def draw_sensitivity(estimates: pd.DataFrame, sweep: str, output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 5.25,
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": "#78888D", "axes.linewidth": 0.65,
    })
    if sweep == "c":
        selected = estimates[np.isclose(estimates["epsilon"], 0.1)]
        parameters = (1.25, 1.5, 2.0, 4.0)
        symbol = "c"
        filename = "CR_DRO_c_sensitivity"
    elif sweep == "epsilon":
        selected = estimates[np.isclose(estimates["c"], 2.0)]
        parameters = (0.0, 0.01, 0.1, 1.0)
        symbol = r"\epsilon"
        filename = "CR_DRO_epsilon_sensitivity"
    elif sweep == "alpha":
        selected = estimates
        parameters = CVAR_ALPHAS
        symbol = r"\alpha"
        filename = "CVaR_alpha_sensitivity"
    else:
        raise ValueError("The parameter sweep must be c, epsilon, or alpha.")

    gap_metric = "cvar_gap" if sweep == "alpha" else "dro_gap"
    bound_metric = "gaussian_cvar_bound" if sweep == "alpha" else "gaussian_l2_bound"
    floor, ceiling = axis_limits(selected, (gap_metric, bound_metric))
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.55, 2.75))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.15, top=0.78, wspace=0.27)
    for axis, sigma in zip(axes, SIGMAS):
        for parameter, color, marker in zip(parameters, COLORS, MARKERS):
            subset = selected[
                np.isclose(selected["sigma"], sigma)
                & np.isclose(selected[sweep], parameter)
            ].sort_values("n_train")
            if tuple(subset["n_train"]) != SAMPLE_SIZES:
                raise ValueError("Every plotted setting must include all seven sample sizes.")
            for metric, kind, linestyle in [
                (bound_metric, "bound", "-"),
                (gap_metric, "gap", (0, (5, 3))),
            ]:
                means = subset[metric].to_numpy()
                deviations = subset[metric + "_std"].to_numpy()
                visible_lower = np.maximum(means - deviations, floor)
                lower_errors = means - np.minimum(visible_lower, means)
                legend_label = rf"${symbol}={parameter:g}$ " + (
                    "upper bound" if kind == "bound" else "gap"
                )
                axis.errorbar(
                    subset["n_train"], np.where(means > 0, means, np.nan),
                    yerr=np.vstack([lower_errors, deviations]),
                    label=legend_label, color=color,
                    marker=marker, linestyle=linestyle,
                    markerfacecolor="white" if kind == "bound" else color,
                    linewidth=1.3, markersize=4.1, markeredgewidth=1.0,
                    elinewidth=0.9, capsize=3.3, capthick=0.9, barsabove=True,
                )
                for n_train in subset.loc[subset[metric + "_sd_lower"] <= 0, "n_train"]:
                    axis.annotate(
                        "", xy=(n_train, floor * 1.04), xytext=(n_train, floor * 2.1),
                        arrowprops={"arrowstyle": "->", "color": color, "lw": 0.9}, zorder=4,
                    )
        axis.axhline(1.0, color="#909A9E", linestyle=":", linewidth=0.8, zorder=1)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlim(400, 40800)
        axis.set_ylim(floor, ceiling)
        axis.xaxis.set_major_locator(FixedLocator(SAMPLE_SIZES[::2]))
        axis.xaxis.set_major_formatter(FixedFormatter([f"{n:,}" for n in SAMPLE_SIZES[::2]]))
        axis.xaxis.set_minor_locator(FixedLocator(SAMPLE_SIZES[1::2]))
        axis.xaxis.set_minor_formatter(NullFormatter())
        lower_decade = int(np.floor(np.log10(floor)))
        upper_decade = int(np.floor(np.log10(ceiling)))
        ticks = [10.0**power for power in range(lower_decade, upper_decade + 1)]
        ticks += [
            multiple * 10.0**power
            for power in {lower_decade, upper_decade}
            for multiple in (2, 5)
        ]
        axis.yaxis.set_major_locator(FixedLocator(sorted(
            tick for tick in ticks if floor <= tick <= ceiling
        )))
        axis.yaxis.set_major_formatter(LogFormatterSciNotation(
            labelOnlyBase=False, minor_thresholds=(np.inf, np.inf),
        ))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.tick_params(length=2.8, width=0.65, pad=3)
        axis.tick_params(axis="y", labelleft=True)
        axis.set_axisbelow(True)
        axis.grid(which="both", axis="both", color="#D8DDDF", linewidth=0.45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(rf"$\sigma={sigma:.2f}$", fontsize=6.3, pad=3)
        axis.set_xlabel(r"Training sample size $n$", labelpad=2)
        axis.set_ylabel("Gap / upper bound", labelpad=3)
    style_handles = [
        Line2D([], [], color="black", linestyle=(0, (5, 3)), linewidth=1.3,
               label="Generalization gap (dashed)"),
        Line2D([], [], color="black", linestyle="-", linewidth=1.3,
               label="Upper bound (solid)"),
    ]
    fig.legend(
        handles=style_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=2, fontsize=7.2, frameon=False, handlelength=5.0,
        borderpad=0.15, borderaxespad=0.1, columnspacing=2.0,
    )
    parameter_handles = [
        Line2D([], [], color=color, marker=marker, linestyle="None",
               markersize=4.5, markeredgewidth=1.0, label=rf"${symbol}={parameter:g}$")
        for parameter, color, marker in zip(parameters, COLORS, MARKERS)
    ]
    fig.legend(
        handles=parameter_handles, loc="upper center", bbox_to_anchor=(0.5, 0.91),
        ncol=4, fontsize=7.2, frameon=False, handlelength=1.0,
        borderpad=0.15, borderaxespad=0.1, columnspacing=2.0,
    )
    fig.savefig(output_dir / (filename + ".pdf"))
    fig.savefig(output_dir / (filename + ".png"), dpi=300)
    plt.close(fig)


def save_mc_figure(estimates: pd.DataFrame, output_dir: Path) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogFormatterSciNotation

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 5.25,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": "#78888D",
            "axes.linewidth": 0.65,
        }
    )
    endpoint_columns = [
        "mc_gap_sd_lower", "mc_gap_sd_upper",
        "gaussian_bound_sd_lower", "gaussian_bound_sd_upper",
    ]
    interval_values = estimates[["mc_gap", "gaussian_bound"] + endpoint_columns].to_numpy()
    positive_values = interval_values[interval_values > 0]
    maximum_upper = estimates[["mc_gap_sd_upper", "gaussian_bound_sd_upper"]].to_numpy().max()
    log_lower = np.log10(positive_values.min())
    log_upper = np.log10(maximum_upper)
    log_margin = max(0.12, 0.04 * (log_upper - log_lower))
    plot_floor = 10.0 ** (log_lower - log_margin)
    plot_ceiling = 10.0 ** (log_upper + log_margin)
    lower_decade = int(np.floor(np.log10(plot_floor)))
    upper_decade = int(np.floor(np.log10(plot_ceiling)))
    y_ticks = [10.0**power for power in range(lower_decade, upper_decade + 2)]
    y_ticks.extend(
        multiple * 10.0**power
        for power in {lower_decade, upper_decade}
        for multiple in (2, 5)
    )
    y_ticks = sorted({tick for tick in y_ticks if plot_floor <= tick <= plot_ceiling})
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.55, 2.55))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.84, wspace=0.27)
    for axis, sigma in zip(axes, SIGMAS):
        subset = estimates[estimates["sigma"] == sigma].sort_values("n_train")
        for value, lower, upper, color, marker, style, label in (
            (
                "gaussian_bound", "gaussian_bound_sd_lower", "gaussian_bound_sd_upper",
                "#A74714", "s", "-", "Upper bound (solid)",
            ),
            (
                "mc_gap", "mc_gap_sd_lower", "mc_gap_sd_upper",
                "#155778", "o", (0, (5, 3)), "Generalization gap (dashed)",
            ),
        ):
            lower_visible = np.maximum(subset[lower], plot_floor)
            # Caps make small Gaussian-bound standard deviations visible.
            axis.errorbar(
                subset["n_train"], subset[value].where(subset[value] > 0),
                yerr=np.vstack(
                    [subset[value] - np.minimum(lower_visible, subset[value]),
                     subset[upper] - subset[value]]
                ),
                label=label, color=color, marker=marker, linestyle=style,
                markerfacecolor="white" if marker == "s" else color,
                linewidth=1.3, markersize=4.1, markeredgewidth=1.0,
                elinewidth=0.9, capsize=3.3, capthick=0.9, barsabove=True,
            )
            for n_train in subset.loc[subset[lower] <= 0, "n_train"]:
                axis.annotate(
                    "", xy=(n_train, plot_floor * 1.04), xytext=(n_train, plot_floor * 2.1),
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 0.9}, zorder=4,
                )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlim(400, 40800)
        axis.set_ylim(plot_floor, plot_ceiling)
        axis.xaxis.set_major_locator(FixedLocator(SAMPLE_SIZES[::2]))
        axis.xaxis.set_major_formatter(FixedFormatter([f"{n:,}" for n in SAMPLE_SIZES[::2]]))
        axis.xaxis.set_minor_locator(FixedLocator(SAMPLE_SIZES[1::2]))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.yaxis.set_major_locator(FixedLocator(y_ticks))
        axis.yaxis.set_major_formatter(
            LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
        )
        axis.yaxis.set_minor_locator(NullLocator())
        axis.tick_params(length=2.8, width=0.65, pad=3)
        axis.tick_params(axis="y", labelleft=True)
        axis.set_axisbelow(True)
        axis.grid(which="both", axis="both", color="#D8DDDF", linewidth=0.45)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(rf"$\sigma={sigma:.2f}$", fontsize=6.3, pad=3)
        axis.set_xlabel(r"Training sample size $n$", labelpad=2)
        axis.set_ylabel("Gap / upper bound", labelpad=3)
    legend_handles = [
        Line2D([], [], color="#155778", linestyle=(0, (5, 3)), linewidth=1.3,
               label="Generalization gap (dashed)"),
        Line2D([], [], color="#A74714", linestyle="-", linewidth=1.3,
               label="Upper bound (solid)"),
    ]
    legend = fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.99),
        ncol=2, fontsize=7.2, frameon=True,
        facecolor="white", edgecolor="0.8", framealpha=0.9,
        handlelength=5.0, labelspacing=0.4, borderpad=0.15, borderaxespad=0.1,
    )
    legend.get_frame().set_linewidth(0.4)
    fig.savefig(output_dir / "Gap_and_bound.pdf")
    fig.savefig(output_dir / "Gap_and_bound.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/flights-200k.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--independent-runs", type=int, default=20)
    parser.add_argument("--outer-repeats", type=int, default=32)
    parser.add_argument("--noise-draws", type=int, default=16)
    parser.add_argument("--selected-pairs", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Reuse compatible completed context checkpoints.")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if min(args.independent_runs, args.outer_repeats, args.noise_draws, args.selected_pairs) < 2:
        parser.error("All repeat counts and --selected-pairs must be at least two")
    if args.selected_pairs > min(SAMPLE_SIZES):
        parser.error("--selected-pairs must not exceed the smallest sample size")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be a nonnegative 32-bit integer")
    args.data_path = args.data_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    args.data_path = obtain_data(args.data_path)
    features, labels = load_data(args.data_path)
    checkpoint_directory, run_token, protocol = prepare_checkpoints(args)
    shape = (args.independent_runs, len(SAMPLE_SIZES), len(SIGMAS), args.outer_repeats)
    train_values = np.full(shape + (len(PARAMETER_SETTINGS),), np.nan)
    validation_values = np.full_like(train_values, np.nan)
    cvar_train_values = np.full(shape + (len(CVAR_ALPHAS),), np.nan)
    cvar_validation_values = np.full_like(cvar_train_values, np.nan)
    mc_signed_gaps = np.full(shape, np.nan)
    information = np.full(shape + (args.selected_pairs,), np.nan)
    selected_pair_indices = np.empty(shape[:3] + (args.selected_pairs,), dtype=np.int64)
    contexts_per_run = len(SAMPLE_SIZES) * len(SIGMAS) * args.outer_repeats
    completed_by_run = np.zeros(args.independent_runs, dtype=int)
    tasks = []

    for run_index in range(args.independent_runs):
        for size_index, n_train in enumerate(SAMPLE_SIZES):
            for sigma_index in range(len(SIGMAS)):
                setting_seed = np.random.SeedSequence([args.seed, run_index, n_train, sigma_index])
                children = setting_seed.spawn(args.outer_repeats + 1)
                pair_rng = np.random.default_rng(children[0])
                pair_indices = np.sort(pair_rng.choice(n_train, size=args.selected_pairs, replace=False))
                selected_pair_indices[run_index, size_index, sigma_index] = pair_indices
                for context_index, context_seed in enumerate(children[1:]):
                    key = (run_index, size_index, sigma_index, context_index)
                    checkpoint_path = checkpoint_directory / (
                        f"run_{run_index + 1:02d}_n_{n_train}_"
                        f"sigma_{sigma_index}_context_{context_index:03d}.json"
                    )
                    cached = None
                    if args.resume:
                        cached = read_checkpoint(checkpoint_path, key, run_token, args.selected_pairs)
                    if cached is not None:
                        train_values[key] = cached["train_values"]
                        validation_values[key] = cached["validation_values"]
                        cvar_train_values[key] = cached["cvar_train_values"]
                        cvar_validation_values[key] = cached["cvar_validation_values"]
                        mc_signed_gaps[key] = cached["mc_signed_gap"]
                        information[key] = cached["parameter_information"]
                        completed_by_run[run_index] += 1
                    else:
                        tasks.append(delayed(run_context)(
                            features, labels, key, context_seed, pair_indices,
                            args.noise_draws, checkpoint_path, run_token,
                        ))

    total_contexts = args.independent_runs * contexts_per_run
    print(
        f"Starting {args.independent_runs} independent complete experiments.\n"
        f"Each run has {len(SAMPLE_SIZES)} sample sizes, {len(SIGMAS)} noise levels, "
        f"{args.outer_repeats} contexts per setting, {args.noise_draws} noise draws "
        f"per context, and {args.selected_pairs} selected pairs.\n"
        f"MC, the seven DRO settings, and four CVaR levels share each context's models "
        f"and Gaussian draws.\n"
        f"Total contexts: {total_contexts:,}. Contexts to compute: {len(tasks):,}. "
        f"Model fits to compute: {len(tasks) * (args.selected_pairs + 1):,}.\n"
        f"Using {args.jobs} workers, one numerical-library thread per worker.",
        flush=True,
    )
    if tasks:
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=args.jobs, return_as="generator_unordered")(tasks)
            for key, payload in results:
                train_values[key] = payload["train_values"]
                validation_values[key] = payload["validation_values"]
                cvar_train_values[key] = payload["cvar_train_values"]
                cvar_validation_values[key] = payload["cvar_validation_values"]
                mc_signed_gaps[key] = payload["mc_signed_gap"]
                information[key] = payload["parameter_information"]
                run_index = key[0]
                completed_by_run[run_index] += 1
                completed = completed_by_run[run_index]
                if completed == contexts_per_run or completed % 32 == 0:
                    print(
                        f"Independent run {run_index + 1}: {completed}/{contexts_per_run} "
                        f"contexts completed ({time.perf_counter() - started:.1f} seconds).",
                        flush=True,
                    )
    if not all(np.isfinite(values).all() for values in [
        train_values, validation_values, cvar_train_values, cvar_validation_values,
        mc_signed_gaps, information,
    ]):
        raise RuntimeError("Every context must finish before the estimates are saved.")

    independent_runs = summarize_runs(train_values, validation_values, information, args)
    estimates = summarize(independent_runs)
    cvar_independent_runs = summarize_cvar_runs(
        cvar_train_values, cvar_validation_values, information, args
    )
    cvar_estimates = summarize(
        cvar_independent_runs, grouping=("n_train", "sigma", "alpha"), metrics=CVAR_METRICS
    )
    mc_independent_runs = summarize_mc_runs(
        mc_signed_gaps, information, args.noise_draws, args.seed
    )
    mc_estimates = summarize_mc(mc_independent_runs)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 5.25,
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": "#78888D", "axes.linewidth": 0.65,
    })
    with TemporaryDirectory(prefix="flights-output-", dir=args.output_dir) as folder:
        temporary_dir = Path(folder)
        independent_runs.to_csv(temporary_dir / "independent_runs.csv", index=False)
        estimates.to_csv(temporary_dir / "estimates.csv", index=False)
        cvar_independent_runs.to_csv(temporary_dir / "cvar_independent_runs.csv", index=False)
        cvar_estimates.to_csv(temporary_dir / "cvar_estimates.csv", index=False)
        mc_independent_runs.to_csv(temporary_dir / "mc_independent_runs.csv", index=False)
        mc_estimates.to_csv(temporary_dir / "mc_estimates.csv", index=False)
        np.savez_compressed(
            temporary_dir / "context_estimates.npz",
            parameter_information=information,
            train_values=train_values,
            validation_values=validation_values,
            cvar_train_values=cvar_train_values,
            cvar_validation_values=cvar_validation_values,
            mc_signed_gaps=mc_signed_gaps,
            selected_pair_indices=selected_pair_indices,
            sample_sizes=np.asarray(SAMPLE_SIZES),
            sigmas=np.asarray(SIGMAS),
            parameter_settings=np.asarray(PARAMETER_SETTINGS),
            cvar_alphas=np.asarray(CVAR_ALPHAS),
            mc_bins=np.asarray(N_BINS),
            run_ids=np.arange(1, args.independent_runs + 1),
            metadata=np.asarray(json.dumps(protocol, sort_keys=True)),
        )
        draw_sensitivity(estimates, "c", temporary_dir)
        draw_sensitivity(estimates, "epsilon", temporary_dir)
        draw_sensitivity(cvar_estimates, "alpha", temporary_dir)
        save_mc_figure(mc_estimates, temporary_dir)
        for output in sorted(temporary_dir.iterdir()):
            output.replace(args.output_dir / output.name)
    print(f"Results and all four MC/CVaR/CR-DRO figures saved to {args.output_dir}", flush=True)
    print(f"Finished in {time.perf_counter() - started:.1f} seconds.", flush=True)


if __name__ == "__main__":
    main()
