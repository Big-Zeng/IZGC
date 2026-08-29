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
# Overlap experiment: True K = 2
# ============================================================
TRUE_K = 2

D_VALUES_DESC = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]


# ============================================================
# Four figure specs (与噪声图对应)
# ============================================================
FIGURE_SPECS = [
    {
        "column": "ARI_mean",
        "ylabel": "ARI",
        "filename": "overlap_ari_mean",
        "performance_metric": True,
    },
    {
        "column": "Macro_F1_mean",
        "ylabel": "Macro-F1",
        "filename": "overlap_macro_f1_mean",
        "performance_metric": True,
    },
    {
        "column": "V_measure_mean",
        "ylabel": "V-measure",
        "filename": "overlap_v_measure_mean",
        "performance_metric": True,
    },
    {
        "column": "Pred_K_mean",
        "ylabel": r"$K_{\mathrm{pred}}$",
        "filename": "overlap_kpred_mean",
        "performance_metric": False,
    },
]


def get_algorithm_order(summary: pd.DataFrame):
    preferred = [
        "IZGC", "K-means++", "CLIQUE", "BANG", "DBSCAN", "GMM", "DPC-FWSN",
    ]
    available = list(summary["Algorithm"].unique())
    order = [a for a in preferred if a in available]
    for a in available:
        if a not in order:
            order.append(a)
    return order


def choose_k_tick_step(k_min: float, k_max: float) -> int:
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


def style_common_axis(ax, d_values, ylabel):
    ax.set_xlabel(r"Center Distance $d$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(d_values)
    ax.set_xticklabels([f"{int(d)}" for d in d_values])
    ax.invert_xaxis()
    ax.tick_params(axis="both", labelsize=12, length=3)
    ax.grid(True, alpha=0.22, linewidth=0.6)


def plot_one_metric(summary, algorithms, spec, output_dir):
    column = spec["column"]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for alg in algorithms:
        sub = (
            summary[summary["Algorithm"] == alg]
            .sort_values("d", ascending=False)   # d 从大到小
        )
        if sub.empty:
            continue

        ax.plot(
            sub["d"].astype(float),
            sub[column].astype(float),
            marker=ALG_MARKERS.get(alg, "o"),
            color=ALG_COLORS.get(alg, None),
            linewidth=1.8,
            markersize=5.5,
            label=alg,
        )

    style_common_axis(ax, D_VALUES_DESC, spec["ylabel"])

    # ---- Performance metrics (0~1) ----
    if spec["performance_metric"]:
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))

    # ---- Pred_K ----
    else:
        k_values = summary[column].astype(float).dropna()
        if not k_values.empty:
            data_min, data_max = float(k_values.min()), float(k_values.max())
            k_min = min(data_min, TRUE_K)
            k_max = max(data_max, TRUE_K)
            margin = max(1.0, 0.08 * (k_max - k_min + 1))
            lower = max(0, math.floor(k_min - margin))
            upper = math.ceil(k_max + margin)
            step = choose_k_tick_step(lower, upper)
            first_tick = math.ceil(lower / step) * step
            yticks = np.arange(first_tick, upper + step, step)
            ax.set_ylim(lower, upper)
            ax.set_yticks(yticks)

        ax.axhline(
            y=TRUE_K, linestyle="--", linewidth=1.4,
            color="black", alpha=0.75,
            label=rf"True $K={TRUE_K}$",
        )

    # ---- Legend: 图外下方 ----
    ax.legend(
        frameon=False,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    base = output_dir / spec["filename"]
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {base.with_suffix('.pdf')}")
    print(f"Saved: {base.with_suffix('.png')}")
    print(f"Saved: {base.with_suffix('.svg')}")


def plot_all_four(summary_csv: Path, output_dir: Path):
    summary = pd.read_csv(summary_csv)
    summary["d"] = summary["d"].astype(float)
    algorithms = get_algorithm_order(summary)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded: {summary_csv}")
    print(f"True K: {TRUE_K}")
    print(f"Algorithms: {algorithms}")
    print(f"d values: {D_VALUES_DESC}")
    print()

    for spec in FIGURE_SPECS:
        plot_one_metric(summary, algorithms, spec, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Plot four metric figures for the Overlap experiment (varying d)."
    )
    parser.add_argument(
        "--summary", type=str,
        default="../results_overlap/overlap_exp_d_summary.csv",
    )
    parser.add_argument(
        "--outdir", type=str,
        default="paper_figures",
    )
    args = parser.parse_args()

    plot_all_four(
        summary_csv=Path(args.summary),
        output_dir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()