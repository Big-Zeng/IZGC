from pathlib import Path

import argparse
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# Paper style
# ============================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Algorithm styles
# ============================================================

ALG_COLORS = {
    "IZGC": "#e74c3c",
    "K-means++": "#3498db",
    "CLIQUE": "#f39c12",
    "BANG": "#9b59b6",
    "DBSCAN": "#2ecc71",
    "GMM": "#e67e22",
    "DPC-FWSN": "#1abc9c",
}

ALG_MARKERS = {
    "IZGC": "o",
    "K-means++": "s",
    "CLIQUE": "D",
    "BANG": "v",
    "DBSCAN": "^",
    "GMM": "P",
    "DPC-FWSN": "X",
}


# ============================================================
# Spiral ground-truth number of clusters
# ============================================================

TRUE_K = 3


# ============================================================
# Four independent figures
# ============================================================

FIGURE_SPECS = [
    {
        "column": "Inlier_ARI_mean",
        "title": "Inlier ARI",
        "ylabel": "Inlier ARI",
        "filename": "noise_inlier_ari_mean",
        "performance_metric": True,
    },
    {
        "column": "Inlier_Macro_F1_mean",
        "title": "Inlier Macro-F1",
        "ylabel": "Inlier Macro-F1",
        "filename": "noise_inlier_macro_f1_mean",
        "performance_metric": True,
    },
    {
        "column": "Inlier_V_measure_mean",
        "title": "Inlier V-measure",
        "ylabel": "Inlier V-measure",
        "filename": "noise_inlier_v_measure_mean",
        "performance_metric": True,
    },
    {
        "column": "Pred_K_mean",
        "title": r"Predicted Cluster Number ($K_{\mathrm{pred}}$)",
        "ylabel": r"$K_{\mathrm{pred}}$",
        "filename": "noise_kpred_mean",
        "performance_metric": False,
    },
]


def check_required_columns(df: pd.DataFrame) -> None:
    required = {
        "Algorithm",
        "noise_ratio",
        "Inlier_ARI_mean",
        "Inlier_Macro_F1_mean",
        "Inlier_V_measure_mean",
        "Pred_K_mean",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        print("\nERROR: missing required columns:")
        for col in missing:
            print(f"  - {col}")

        print("\nAvailable columns:")
        for col in df.columns:
            print(f"  {col}")

        raise ValueError(
            "The summary CSV does not contain all required mean columns."
        )


def get_algorithm_order(summary: pd.DataFrame):
    preferred = [
        "IZGC",
        "K-means++",
        "CLIQUE",
        "BANG",
        "DBSCAN",
        "GMM",
        "DPC-FWSN",
    ]

    available = list(summary["Algorithm"].unique())

    order = [alg for alg in preferred if alg in available]

    for alg in available:
        if alg not in order:
            order.append(alg)

    return order


def choose_k_tick_step(k_min: float, k_max: float) -> int:
    """
    Choose a sparse integer tick interval for K_pred.

    Aim for roughly 5-8 y ticks instead of one tick for every integer.
    """
    span = max(1.0, k_max - k_min)

    if span <= 6:
        return 1
    elif span <= 15:
        return 2
    elif span <= 30:
        return 5
    elif span <= 60:
        return 10
    else:
        return int(math.ceil(span / 6.0 / 5.0) * 5)


def style_common_axis(
    ax,
    summary: pd.DataFrame,
    title: str,
    ylabel: str,
):
    noise_levels = np.sort(
        summary["noise_ratio"].astype(float).unique()
    )


    ax.set_xlabel(
        "Noise Ratio",
        fontsize=15,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=15,
    )

    ax.set_xticks(noise_levels)

    ax.set_xticklabels(
        [f"{x:.0%}" for x in noise_levels]
    )

    ax.tick_params(
        axis="both",
        labelsize=12,
        length=3,
    )

    ax.grid(
        True,
        alpha=0.22,
        linewidth=0.6,
    )


def plot_one_metric(
    summary: pd.DataFrame,
    algorithms,
    spec,
    output_dir: Path,
):
    column = spec["column"]

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for alg in algorithms:
        sub = (
            summary[
                summary["Algorithm"] == alg
            ]
            .sort_values("noise_ratio")
        )

        if sub.empty:
            continue

        ax.plot(
            sub["noise_ratio"].astype(float),
            sub[column].astype(float),

            marker=ALG_MARKERS.get(
                alg,
                "o",
            ),

            color=ALG_COLORS.get(
                alg,
                None,
            ),

            linewidth=1.8,
            markersize=5.5,

            label=alg,
        )

    style_common_axis(
        ax,
        summary,
        spec["title"],
        spec["ylabel"],
    )

    # --------------------------------------------------------
    # Performance metrics
    # --------------------------------------------------------

    if spec["performance_metric"]:

        ax.set_ylim(
            -0.02,
            1.02,
        )

        ax.set_yticks(
            np.arange(
                0.0,
                1.01,
                0.2,
            )
        )

    # --------------------------------------------------------
    # K_pred
    # --------------------------------------------------------

    else:

        k_values = (
            summary[column]
            .astype(float)
            .dropna()
        )

        if not k_values.empty:

            data_min = float(k_values.min())
            data_max = float(k_values.max())

            # Include true K in the displayed range
            k_min = min(
                data_min,
                TRUE_K,
            )

            k_max = max(
                data_max,
                TRUE_K,
            )

            margin = max(
                1.0,
                0.08 * (k_max - k_min + 1),
            )

            lower = max(
                0,
                math.floor(k_min - margin),
            )

            upper = math.ceil(
                k_max + margin
            )

            step = choose_k_tick_step(
                lower,
                upper,
            )

            first_tick = (
                math.ceil(lower / step)
                * step
            )

            yticks = np.arange(
                first_tick,
                upper + step,
                step,
            )

            ax.set_ylim(
                lower,
                upper,
            )

            ax.set_yticks(
                yticks
            )

        # Ground-truth K = 3
        ax.axhline(
            y=TRUE_K,
            linestyle="--",
            linewidth=1.4,
            color="black",
            alpha=0.75,
            label=rf"True $K={TRUE_K}$",
        )

    # Shared legend within each independent figure
    ax.legend(
        frameon=False,
        fontsize=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    base = output_dir / spec["filename"]

    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        base.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        base.with_suffix(".svg"),
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {base.with_suffix('.pdf')}")
    print(f"Saved: {base.with_suffix('.png')}")
    print(f"Saved: {base.with_suffix('.svg')}")


def plot_all_four(
    summary_csv: Path,
    output_dir: Path,
):
    summary = pd.read_csv(
        summary_csv
    )

    check_required_columns(
        summary
    )

    summary["noise_ratio"] = (
        summary["noise_ratio"]
        .astype(float)
    )

    algorithms = get_algorithm_order(
        summary
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loaded: {summary_csv}")
    print(f"True K: {TRUE_K}")
    print()

    for spec in FIGURE_SPECS:
        plot_one_metric(
            summary,
            algorithms,
            spec,
            output_dir,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot four separate representative "
            "mean-metric figures for the Spiral noise experiment."
        )
    )

    parser.add_argument(
        "--summary",
        type=str,
        default=(
            "../results_noise_spiral/"
            "noise_experiment_summary.csv"
        ),
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="paper_figures",
    )

    args = parser.parse_args()

    plot_all_four(
        summary_csv=Path(args.summary),
        output_dir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()
