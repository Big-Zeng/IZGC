from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

SPIRAL_CSV = Path("F:\科研\论文\分区论文\distrubuteArea\distributeAreaCode\data/spiral.csv")

# Same seed as the original noise experiment
RANDOM_SEED = 20260824

# Paper figure: 0%, 10%, 20%, 30%, 40%, 50%
SELECTED_NOISE = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]

OUTPUT_DIR = Path("paper_figures")
OUTPUT_BASENAME = "spiral_noise_0_10_20_30_40_50"


# ============================================================
# Unified paper style
# ============================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 8.5,

    "axes.linewidth": 0.8,

    # Keep fonts editable in vector PDF
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Same preprocessing as your original experiment
# ============================================================

def normalize_coordinates(X: np.ndarray) -> np.ndarray:
    """Min-max normalization to [0, 100], identical to the experiment."""
    x_min, y_min = X.min(axis=0)
    x_max, y_max = X.max(axis=0)

    x_range = x_max - x_min
    y_range = y_max - y_min

    X_norm = np.zeros_like(X, dtype=float)

    X_norm[:, 0] = (
        (X[:, 0] - x_min) / x_range * 100
        if x_range > 0 else 0
    )
    X_norm[:, 1] = (
        (X[:, 1] - y_min) / y_range * 100
        if y_range > 0 else 0
    )

    return X_norm


def load_spiral_csv(csv_path: Path):
    """
    spiral.csv must contain:
        x, y, target
    """
    df = pd.read_csv(csv_path)

    X = df[["x", "y"]].astype(float).to_numpy()
    y = df["target"].to_numpy()

    X = normalize_coordinates(X)

    return X, y


# ============================================================
# Exact original noise-generation method
# ============================================================

def add_noise_to_dataset(
    X: np.ndarray,
    y_true: np.ndarray,
    noise_ratio: float,
    seed: int,
):
    """
    Same definition as the original noise experiment:

        noise_ratio = N_noise / (N_clean + N_noise)

    Noise points are uniformly sampled within:
        [X.min - 4, X.max + 4]

    Noise label = -1.
    """

    if noise_ratio <= 0:
        return X.copy(), y_true.copy()

    rng = np.random.default_rng(seed)

    n_clean = len(X)

    n_noise = int(
        round(
            noise_ratio * n_clean
            / max(1e-12, 1.0 - noise_ratio)
        )
    )

    mins = X.min(axis=0) - 4
    maxs = X.max(axis=0) + 4

    noise = rng.uniform(
        mins,
        maxs,
        size=(n_noise, 2),
    )

    X_noisy = np.vstack([X, noise])

    y_noisy = np.concatenate([
        y_true,
        np.full(n_noise, -1, dtype=int),
    ])

    return X_noisy, y_noisy


# ============================================================
# Figure
# Reference style:
# plot_izgc_transition_clusters(...)
# ============================================================

def plot_noise_examples():
    X_base, y_base = load_spiral_csv(SPIRAL_CSV)

    # Same overall spatial range for every panel.
    # This makes the density change directly comparable.
    mins = X_base.min(axis=0) - 4
    maxs = X_base.max(axis=0) + 4

    # 2 rows x 3 columns
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 9.0),  # 4.5 x 4.5 per panel
    )

    # Flatten axes for easier iteration
    axes_flat = axes.flatten()

    for ax, noise_ratio in zip(axes_flat, SELECTED_NOISE):

        X, y_true = add_noise_to_dataset(
            X_base,
            y_base,
            noise_ratio,
            RANDOM_SEED,
        )

        # ----------------------------------------------------
        # Clean original clusters
        # Similar to plot_izgc_transition_clusters:
        # default Matplotlib cluster colors
        # ----------------------------------------------------
        clean_labels = [
            lab for lab in np.unique(y_true)
            if lab != -1
        ]

        for lab in clean_labels:
            mask = y_true == lab

            ax.scatter(
                X[mask, 0],
                X[mask, 1],

                # Original IZGC transition code uses s=7.
                # Slightly enlarged for paper readability.
                s=14,

                alpha=0.75,
                edgecolors="none",
            )

        # ----------------------------------------------------
        # Noise
        # Match IZGC transition style:
        # gray + marker="x"
        # ----------------------------------------------------
        noise_mask = y_true == -1

        if np.any(noise_mask):
            ax.scatter(
                X[noise_mask, 0],
                X[noise_mask, 1],

                # Slightly larger than the reference s=7
                s=15,

                alpha=0.35,
                c="gray",
                marker="x",

                linewidths=1.5,
            )

        # ----------------------------------------------------
        # Panel formatting
        # ----------------------------------------------------

        ax.set_title(
            f"Noise={noise_ratio:.0%}",
            fontsize=15,
            pad=6,
        )

        # Restore X/Y labels and coordinate ticks.
        ax.set_xlabel("X", fontsize=15)
        ax.set_ylabel("Y", fontsize=15)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="both", labelsize=15)

        # Ensure all panels have exactly the same scale
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])

        ax.set_aspect("equal", adjustable="box")

    # No (a), (b), (c)
    # No repeated X/Y labels
    # No legend, following the reference transition figure

    fig.tight_layout(
        pad=0.8,
        w_pad=0.8,
        h_pad=1.2,  # Add vertical spacing between rows
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Vector format preferred for papers
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    # 600 dpi raster version
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    # Optional vector SVG
    svg_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.svg"
    fig.savefig(
        svg_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Print point counts for verification
    # --------------------------------------------------------
    print(f"Clean Spiral points: {len(X_base)}")

    for noise_ratio in SELECTED_NOISE:
        X, y = add_noise_to_dataset(
            X_base,
            y_base,
            noise_ratio,
            RANDOM_SEED,
        )

        n_noise = int(np.sum(y == -1))
        n_clean = int(np.sum(y != -1))

        print(
            f"Noise={noise_ratio:.0%}: "
            f"clean={n_clean}, "
            f"noise={n_noise}, "
            f"total={len(X)}"
        )

    print("\nSaved:")
    print(f"  {pdf_path}")
    print(f"  {png_path}  (600 dpi)")
    print(f"  {svg_path}")


if __name__ == "__main__":
    plot_noise_examples()