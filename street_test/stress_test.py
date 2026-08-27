from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to prevent GUI popups
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    fowlkes_mallows_score,
    completeness_score,
    v_measure_score,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 0. Global configuration
# ============================================================

@dataclass
class ExperimentConfig:
    n_repeats: int = 30
    random_seed: int = 20260824

    # Fixed IZGC parameters
    R: float = 30.0
    theta: int = 5
    delta_g_star: float = 2.0   # replace after independent calibration
    is_noise_enabled: bool = True

    # canonical space used by the manuscript
    canonical_range: float = 100.0

    # experiment levels
    noise_levels: Tuple[float, ...] = (0.00, 0.05, 0.10, 0.20, 0.30, 0.40)
    overlap_levels: Tuple[float, ...] = (0.00, 0.10, 0.20, 0.30, 0.40, 0.50)
    density_ratios: Tuple[int, ...] = (1, 2, 4, 8, 12, 16)
    ring_gaps: Tuple[float, ...] = (14.0, 10.0, 7.0, 5.0, 3.0, 1.5)

    # sample sizes
    spiral_points_per_cluster: int = 300
    blob_base_points: int = 500
    ring_points_per_cluster: int = 350


METRIC_COLUMNS = [
    "ARI", "AMI", "FMI",
    "Macro_F1", "Macro_Jaccard",
    "Completeness", "V_measure"
]


# ============================================================
# 1. Utility functions
# ============================================================

def normalize_to_100(X: np.ndarray) -> np.ndarray:
    """Normalize coordinates to [0,100]x[0,100], matching the manuscript."""
    scaler = MinMaxScaler(feature_range=(0.0, 100.0))
    return scaler.fit_transform(X)


def relabel_contiguous(y: np.ndarray) -> np.ndarray:
    """
    Relabel arbitrary labels to 0..K-1, preserving -1 as noise.
    """
    y = np.asarray(y).copy()
    out = np.full_like(y, fill_value=-1)
    valid = sorted([int(v) for v in np.unique(y) if v != -1])
    mapping = {old: new for new, old in enumerate(valid)}
    for old, new in mapping.items():
        out[y == old] = new
    return out


def hungarian_align_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Align predicted cluster IDs to ground-truth labels using Hungarian matching.

    Noise label -1 is retained as -1 and excluded from matching.
    Unmatched predicted clusters receive new labels beyond max ground-truth label.

    This enables permutation-invariant Macro-F1 and Macro-Jaccard.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    aligned = np.full_like(y_pred, -1)

    mask = y_pred != -1
    if mask.sum() == 0:
        return aligned

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred[mask])

    # contingency matrix: rows=true, cols=pred
    M = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for i, t in enumerate(true_labels):
        for j, p in enumerate(pred_labels):
            M[i, j] = np.sum((y_true == t) & (y_pred == p))

    row_ind, col_ind = linear_sum_assignment(-M)

    pred_to_true = {}
    for r, c in zip(row_ind, col_ind):
        pred_to_true[pred_labels[c]] = true_labels[r]

    next_label = int(true_labels.max()) + 1 if len(true_labels) else 0
    for p in pred_labels:
        if p in pred_to_true:
            aligned[y_pred == p] = pred_to_true[p]
        else:
            aligned[y_pred == p] = next_label
            next_label += 1

    aligned[y_pred == -1] = -1
    return aligned


def macro_f1_jaccard(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Macro-F1 and Macro-Jaccard after Hungarian alignment.

    Ground-truth classes are treated as the evaluation classes.
    Predicted noise (-1) counts as a false negative for the corresponding true class.
    Extra predicted clusters count indirectly through false negatives / false positives.
    """
    y_true = np.asarray(y_true)
    y_aligned = hungarian_align_labels(y_true, y_pred)

    classes = np.unique(y_true)
    f1s, jacs = [], []

    for c in classes:
        tp = np.sum((y_true == c) & (y_aligned == c))
        fp = np.sum((y_true != c) & (y_aligned == c))
        fn = np.sum((y_true == c) & (y_aligned != c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        jac = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        f1s.append(f1)
        jacs.append(jac)

    return float(np.mean(f1s)), float(np.mean(jacs))


def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all reviewer-requested metrics.

    ARI/AMI/FMI/Completeness/V-measure are directly permutation-invariant.
    Macro-F1 and Macro-Jaccard use Hungarian label alignment.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    f1, jac = macro_f1_jaccard(y_true, y_pred)

    return {
        "ARI": adjusted_rand_score(y_true, y_pred),
        "AMI": adjusted_mutual_info_score(y_true, y_pred),
        "FMI": fowlkes_mallows_score(y_true, y_pred),
        "Macro_F1": f1,
        "Macro_Jaccard": jac,
        "Completeness": completeness_score(y_true, y_pred),
        "V_measure": v_measure_score(y_true, y_pred),
    }


# ============================================================
# 2. Synthetic datasets
# ============================================================

def generate_spiral(
    seed: int,
    n_per_cluster: int = 300,
    n_clusters: int = 3,
    noise_ratio: float = 0.0,
    arm_noise: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Three intertwined spirals + uniform background noise.

    Noise points are assigned label -1 in y_true.
    NOTE:
    For external clustering metrics, whether noise should be treated as an
    independent class depends on the manuscript definition.
    In this framework noise is kept as ground-truth class -1.
    """
    rng = np.random.default_rng(seed)

    X_parts, y_parts = [], []

    for k in range(n_clusters):
        t = np.linspace(0.5, 4.5 * np.pi, n_per_cluster)
        r = np.linspace(2.0, 28.0, n_per_cluster)
        phase = 2 * np.pi * k / n_clusters

        x = r * np.cos(t + phase)
        y = r * np.sin(t + phase)

        pts = np.c_[x, y]
        pts += rng.normal(scale=arm_noise, size=pts.shape)

        X_parts.append(pts)
        y_parts.append(np.full(n_per_cluster, k, dtype=int))

    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    if noise_ratio > 0:
        # noise_ratio = n_noise / total_after_noise
        n_clean = len(X)
        n_noise = int(round(noise_ratio * n_clean / max(1e-12, 1.0 - noise_ratio)))

        mins = X.min(axis=0) - 4
        maxs = X.max(axis=0) + 4
        noise = rng.uniform(mins, maxs, size=(n_noise, 2))

        X = np.vstack([X, noise])
        y = np.concatenate([y, np.full(n_noise, -1, dtype=int)])

    return normalize_to_100(X), y


def generate_overlapping_blobs(
    seed: int,
    overlap: float,
    n_per_cluster: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two Gaussian blobs with continuously controlled overlap.

    overlap in [0, 0.5]:
        0.0 -> well separated
        0.5 -> strongly overlapping

    The center separation decreases as overlap increases.
    """
    rng = np.random.default_rng(seed)

    overlap = float(np.clip(overlap, 0.0, 0.5))
    sigma = 1.0

    # separation from 6 sigma to 1.2 sigma
    d = 6.0 - (overlap / 0.5) * 4.8

    mu1 = np.array([-d / 2, 0.0])
    mu2 = np.array([ d / 2, 0.0])

    cov = np.array([[sigma**2, 0.0], [0.0, sigma**2]])

    X1 = rng.multivariate_normal(mu1, cov, n_per_cluster)
    X2 = rng.multivariate_normal(mu2, cov, n_per_cluster)

    X = np.vstack([X1, X2])
    y = np.r_[np.zeros(n_per_cluster, dtype=int),
              np.ones(n_per_cluster, dtype=int)]

    return normalize_to_100(X), y


def generate_density_heterogeneous_blobs(
    seed: int,
    ratio: int,
    base_points: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Three separated blobs with heterogeneous sampling densities.

    Density ratio is implemented through sample-count imbalance:
        cluster sizes = [base_points, base_points/ratio, base_points/sqrt(ratio)]

    This keeps geometry broadly comparable while progressively reducing support
    for sparse clusters.
    """
    rng = np.random.default_rng(seed)
    ratio = max(1, int(ratio))

    n1 = base_points
    n2 = max(30, int(round(base_points / ratio)))
    n3 = max(30, int(round(base_points / math.sqrt(ratio))))

    centers = np.array([
        [-4.0, -1.0],
        [ 4.0, -1.0],
        [ 0.0,  4.5],
    ])

    covs = [
        np.array([[1.00, 0.10], [0.10, 0.80]]),
        np.array([[1.00, 0.00], [0.00, 1.00]]),
        np.array([[0.80, 0.00], [0.00, 1.20]]),
    ]

    sizes = [n1, n2, n3]
    Xs, ys = [], []

    for k, (mu, cov, n) in enumerate(zip(centers, covs, sizes)):
        Xk = rng.multivariate_normal(mu, cov, n)
        Xs.append(Xk)
        ys.append(np.full(n, k, dtype=int))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return normalize_to_100(X), y


def generate_nested_rings(
    seed: int,
    gap: float,
    n_per_ring: int = 350,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Three nested rings. The radial gap between adjacent rings is controlled by `gap`.
    Smaller gap => harder topological separation.

    Ring thickness is fixed; therefore progressively smaller gaps create touching /
    near-touching boundaries.
    """
    rng = np.random.default_rng(seed)

    # radii before normalization
    r1 = 10.0
    r2 = r1 + gap + 3.0
    r3 = r2 + gap + 3.0
    radii = [r1, r2, r3]

    Xs, ys = [], []
    for k, r in enumerate(radii):
        ang = rng.uniform(0, 2 * np.pi, n_per_ring)
        radial_jitter = rng.normal(0, 0.9, n_per_ring)
        rr = r + radial_jitter

        x = rr * np.cos(ang)
        y = rr * np.sin(ang)
        pts = np.c_[x, y]

        Xs.append(pts)
        ys.append(np.full(n_per_ring, k, dtype=int))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    return normalize_to_100(X), y


# ============================================================
# 3. Algorithms
# ============================================================

def run_izgc(
    X: np.ndarray,
    R: float,
    theta: int,
    delta_g: float,
    is_noise_enabled: bool = True,
) -> np.ndarray:
    """
    Adapter for ContourBasedClusteringWithoutZ22 implementation.
    """
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from scipy.spatial import cKDTree
    import ContourBasedClusteringWithoutZ22 as ContourBasedClusteringWithoutZ

    # Compute density for the points
    points = np.array(X)
    tree = cKDTree(points)
    density = np.array([len(tree.query_ball_point(p, R)) for p in points])  # zi = density
    combined_data = [(*point, density[i], i) for i, point in enumerate(points)]

    # Initialize the contour clustering model
    # merge_small_clusters has opposite boolean logic to is_noise_enabled:
    # merge_small_clusters = True means small clusters are merged into nearest large ones
    # merge_small_clusters = False means small clusters are not merged, marking them as noise (-1)
    model = ContourBasedClusteringWithoutZ.ContourBasedClusteringWithoutZ(
        combined_data,
        grid_size=delta_g,
        max_buffer_distance=0,
        min_cluster_size=theta,
        interactive=False,
        merge_small_clusters=not is_noise_enabled
    )

    # Perform clustering
    model.perform_clustering()

    # Get the resulting predicted labels
    labels = model.get_predicted_labels()

    return np.asarray(labels, dtype=int)


def run_dbscan(X: np.ndarray) -> np.ndarray:
    """
    Baseline example only.
    These fixed values are NOT claimed to be publication-optimal.
    For the final paper, define a fair baseline protocol separately.
    """
    return DBSCAN(eps=3.0, min_samples=5).fit_predict(X)


def run_kmeans(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)


def run_gmm(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    return GaussianMixture(
        n_components=k,
        n_init=10,
        random_state=seed
    ).fit_predict(X)


# ============================================================
# 4. Independent calibration of delta_g*
# ============================================================

def make_calibration_dataset(seed: int = 12345) -> Tuple[np.ndarray, np.ndarray]:
    """
    Independent calibration dataset.

    IMPORTANT:
    Do not use any of the four stress-test datasets to select delta_g*.
    This example uses a moderately difficult 3-blob dataset only for calibration.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[-4, 0], [0, 4], [4, 0]], dtype=float)
    Xs, ys = [], []

    for k, mu in enumerate(centers):
        n = 300
        cov = np.array([[1.0, 0.15], [0.15, 1.0]])
        Xk = rng.multivariate_normal(mu, cov, n)
        Xs.append(Xk)
        ys.append(np.full(n, k, dtype=int))

    X = normalize_to_100(np.vstack(Xs))
    y = np.concatenate(ys)
    return X, y


def calibrate_delta_g(
    config: ExperimentConfig,
    search_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Select delta_g* ONCE using the independent calibration dataset.

    Default criterion: ARI.
    You can alternatively use a multi-metric mean rank or mean normalized score.
    """
    X, y = make_calibration_dataset(config.random_seed + 999)

    rows = []
    for dg in search_grid:
        pred = run_izgc(
            X,
            R=config.R,
            theta=config.theta,
            delta_g=float(dg),
            is_noise_enabled=config.is_noise_enabled,
        )
        met = evaluate_clustering(y, pred)
        rows.append({"delta_g": float(dg), **met})

    df = pd.DataFrame(rows)
    return df.sort_values("ARI", ascending=False).reset_index(drop=True)


# ============================================================
# 5. Experiment runner
# ============================================================

def run_one_setting(
    experiment_name: str,
    level_name: str,
    level_value,
    generator: Callable[[int], Tuple[np.ndarray, np.ndarray]],
    config: ExperimentConfig,
) -> pd.DataFrame:
    rows = []

    for rep in range(config.n_repeats):
        seed = config.random_seed + rep + abs(hash((experiment_name, str(level_value)))) % 1_000_000

        X, y_true = generator(seed)

        y_pred = run_izgc(
            X,
            R=config.R,
            theta=config.theta,
            delta_g=config.delta_g_star,
            is_noise_enabled=config.is_noise_enabled,
        )

        met = evaluate_clustering(y_true, y_pred)

        rows.append({
            "Experiment": experiment_name,
            "Difficulty": level_name,
            "Level": level_value,
            "Repeat": rep + 1,
            "Seed": seed,
            "N": len(X),
            "True_K": len([v for v in np.unique(y_true) if v != -1]),
            "Pred_K": len([v for v in np.unique(y_pred) if v != -1]),
            "Pred_Noise_Ratio": float(np.mean(y_pred == -1)),
            **met
        })

    return pd.DataFrame(rows)


def run_all_stress_tests(config: ExperimentConfig) -> pd.DataFrame:
    frames = []

    # 1) Noise robustness
    for noise in config.noise_levels:
        frames.append(
            run_one_setting(
                experiment_name="Noise robustness",
                level_name="noise_ratio",
                level_value=noise,
                generator=lambda seed, noise=noise: generate_spiral(
                    seed,
                    n_per_cluster=config.spiral_points_per_cluster,
                    noise_ratio=noise,
                ),
                config=config,
            )
        )

    # 2) Boundary overlap
    for overlap in config.overlap_levels:
        frames.append(
            run_one_setting(
                experiment_name="Boundary overlap",
                level_name="overlap",
                level_value=overlap,
                generator=lambda seed, overlap=overlap: generate_overlapping_blobs(
                    seed,
                    overlap=overlap,
                    n_per_cluster=config.blob_base_points,
                ),
                config=config,
            )
        )

    # 3) Density heterogeneity
    for ratio in config.density_ratios:
        frames.append(
            run_one_setting(
                experiment_name="Density heterogeneity",
                level_name="density_ratio",
                level_value=ratio,
                generator=lambda seed, ratio=ratio: generate_density_heterogeneous_blobs(
                    seed,
                    ratio=ratio,
                    base_points=config.blob_base_points,
                ),
                config=config,
            )
        )

    # 4) Topological separation
    for gap in config.ring_gaps:
        frames.append(
            run_one_setting(
                experiment_name="Topological separation",
                level_name="ring_gap",
                level_value=gap,
                generator=lambda seed, gap=gap: generate_nested_rings(
                    seed,
                    gap=gap,
                    n_per_ring=config.ring_points_per_cluster,
                ),
                config=config,
            )
        )

    return pd.concat(frames, ignore_index=True)


# ============================================================
# 6. Summary and figures
# ============================================================

def summarize_results(raw: pd.DataFrame) -> pd.DataFrame:
    agg_dict = {}
    for m in METRIC_COLUMNS:
        agg_dict[f"{m}_mean"] = (m, "mean")
        agg_dict[f"{m}_std"] = (m, "std")

    agg_dict.update({
        "Pred_K_mean": ("Pred_K", "mean"),
        "Pred_K_std": ("Pred_K", "std"),
        "Pred_Noise_Ratio_mean": ("Pred_Noise_Ratio", "mean"),
        "N_mean": ("N", "mean"),
    })

    summary = (
        raw.groupby(["Experiment", "Difficulty", "Level"], as_index=False)
           .agg(**agg_dict)
    )
    return summary


def plot_degradation_curves(summary: pd.DataFrame, outdir: Path) -> None:
    """
    One figure per experiment, one curve per metric.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    for exp in summary["Experiment"].unique():
        sub = summary[summary["Experiment"] == exp].copy()
        sub = sub.sort_values("Level")

        fig, ax = plt.subplots(figsize=(7.2, 5.2))

        for metric in ["ARI", "AMI", "FMI", "Macro_F1", "Macro_Jaccard", "V_measure"]:
            ax.plot(
                sub["Level"].astype(float),
                sub[f"{metric}_mean"],
                marker="o",
                label=metric
            )

        ax.set_title(exp)
        ax.set_xlabel(sub["Difficulty"].iloc[0])
        ax.set_ylabel("Score")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()

        fname = exp.lower().replace(" ", "_") + "_curves.png"
        fig.savefig(outdir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_dataset_examples(config: ExperimentConfig, outdir: Path) -> None:
    """
    Save representative examples for manuscript visualization.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    examples = [
        ("spiral_noise_30", *generate_spiral(
            config.random_seed,
            n_per_cluster=config.spiral_points_per_cluster,
            noise_ratio=0.30
        )),
        ("overlap_40", *generate_overlapping_blobs(
            config.random_seed,
            overlap=0.40,
            n_per_cluster=config.blob_base_points
        )),
        ("density_ratio_16", *generate_density_heterogeneous_blobs(
            config.random_seed,
            ratio=16,
            base_points=config.blob_base_points
        )),
        ("nested_gap_3", *generate_nested_rings(
            config.random_seed,
            gap=3.0,
            n_per_ring=config.ring_points_per_cluster
        )),
    ]

    for name, X, y in examples:
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        for lab in np.unique(y):
            mask = y == lab
            if lab == -1:
                ax.scatter(X[mask, 0], X[mask, 1], s=7, alpha=0.45, label="noise")
            else:
                ax.scatter(X[mask, 0], X[mask, 1], s=7, alpha=0.7, label=f"C{lab+1}")
        ax.set_title(name)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"{name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# 7. Friedman ranking
# ============================================================

def friedman_mean_ranks(
    all_algorithm_summary: pd.DataFrame,
    metric: str = "ARI",
) -> pd.DataFrame:
    """
    Expected input columns:
        Algorithm, Experiment, Level, <metric>

    Each Experiment x Level is treated as one benchmark condition.
    Higher metric is better -> rank 1 is best.

    For publication, repeat this function for all seven metrics.
    """
    df = all_algorithm_summary.copy()

    required = {"Algorithm", "Experiment", "Level", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for Friedman ranking: {missing}")

    df["Condition"] = (
        df["Experiment"].astype(str) + "__" + df["Level"].astype(str)
    )

    pivot = df.pivot_table(
        index="Condition",
        columns="Algorithm",
        values=metric,
        aggfunc="mean"
    )

    ranks = pivot.rank(axis=1, ascending=False, method="average")
    mean_rank = ranks.mean(axis=0).sort_values()

    return (
        mean_rank.rename("Mean_Rank")
                 .reset_index()
                 .rename(columns={"index": "Algorithm"})
    )


# ============================================================
# 8. Optional baseline comparison scaffold
# ============================================================

def evaluate_baseline_on_dataset(
    X: np.ndarray,
    y_true: np.ndarray,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """
    Minimal examples only.
    Final manuscript should apply a pre-declared FAIR baseline protocol.

    K-means and GMM are given true K because the manuscript currently does so.
    DBSCAN below uses a fixed example parameter and should be replaced with your
    chosen protocol.
    """
    k = len([v for v in np.unique(y_true) if v != -1])

    preds = {
        "K-means++": run_kmeans(X, k=k, seed=seed),
        "GMM": run_gmm(X, k=k, seed=seed),
        "DBSCAN": run_dbscan(X),
    }

    return {alg: evaluate_clustering(y_true, pred) for alg, pred in preds.items()}


# ============================================================
# 9. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="results_stress_test")
    parser.add_argument("--delta-g", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate delta_g* once on independent calibration dataset and exit."
    )
    args = parser.parse_args()

    config = ExperimentConfig(
        n_repeats=args.repeats,
        delta_g_star=args.delta_g,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    if args.calibrate:
        grid = np.arange(0.5, 5.0001, 0.05)
        cal = calibrate_delta_g(config, grid)
        cal.to_csv(outdir / "delta_g_calibration.csv", index=False)

        best = cal.iloc[0]
        print("\nBest independent calibration result:")
        print(best[["delta_g", "ARI", "AMI", "FMI"]].to_string())
        print(
            "\nIMPORTANT: freeze this delta_g* and use exactly the same value "
            "for all subsequent stress tests."
        )
        return

    # visual examples
    plot_dataset_examples(config, outdir / "dataset_examples")

    # stress tests
    raw = run_all_stress_tests(config)
    raw.to_csv(outdir / "stress_test_raw.csv", index=False)

    summary = summarize_results(raw)
    summary.to_csv(outdir / "stress_test_summary.csv", index=False)

    plot_degradation_curves(summary, outdir / "figures")

    print("\nFinished.")
    print(f"Fixed IZGC parameters: R={config.R}, theta={config.theta}, "
          f"delta_g*={config.delta_g_star}")
    print(f"Raw results    : {outdir / 'stress_test_raw.csv'}")
    print(f"Summary results: {outdir / 'stress_test_summary.csv'}")
    print(f"Figures        : {outdir / 'figures'}")


if __name__ == "__main__":
    main()
