from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# ============================================================
# IZGC results from the 12 synthetic benchmark datasets
# ============================================================

DATA = [
    # Dataset, K_true, K_pred, AMI, ARI, FMI, Macro-F1, V-measure

    # Non-uniform multi-shape
    ["Jain",        2,  2, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    ["Ring",        2,  2, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    ["Cth",         4,  4, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    ["Ls",          6,  6, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    ["LineBlobs",   3,  3, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    ["3c",          3,  4, 0.9772, 0.9122, 0.9765, 0.9372, 0.9412],

    # Complex and multi-morphic
    ["Flame",       2,  2, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    ["Spiral",      3,  3, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    ["Complex8",    8, 13, 0.9604, 0.9938, 0.9752, 0.9632, 0.9685],
    ["Complex9",    9,  9, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
    ["R15",        15, 15, 0.9938, 0.9928, 0.9967, 0.9967, 0.9942],
    ["Aggregation", 7,  7, 0.9864, 0.9913, 0.9949, 0.9950, 0.9866],
]


df = pd.DataFrame(
    DATA,
    columns=[
        "Dataset",
        "K_true",
        "K_pred",
        "AMI",
        "ARI",
        "FMI",
        "Macro_F1",
        "V_measure",
    ],
)


# ============================================================
# Average datasets sharing the same K_true
# ============================================================

metric_cols = [
    "AMI",
    "ARI",
    "FMI",
    "Macro_F1",
    "V_measure",
    "K_pred",
]

mean_by_k = (
    df.groupby("K_true", as_index=False)[metric_cols]
      .mean()
      .sort_values("K_true")
)


print("Mean results grouped by K_true:")
print(mean_by_k.to_string(index=False))


# ============================================================
# Paper style
# ============================================================

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 1 x 2 figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9.0, 3.8),
)


# ============================================================
# (a) Mean performance variation with cluster count
# ============================================================

ax = axes[0]

metric_specs = [
    ("AMI",       "o", "AMI"),
    ("ARI",       "s", "ARI"),
    ("FMI",       "^", "FMI"),
    ("Macro_F1",  "D", "Macro-F1"),
    ("V_measure", "v", "V-measure"),
]

for metric, marker, label in metric_specs:
    ax.plot(
        mean_by_k["K_true"],
        mean_by_k[metric],
        marker=marker,
        linewidth=1.7,
        markersize=5.2,
        label=label,
    )

ax.set_xlabel(
    r"Ground-truth cluster number $K_{\mathrm{true}}$",
    fontsize=12,
)

ax.set_ylabel(
    "Mean clustering score",
    fontsize=12,
)

# Use the full metric range for a conservative presentation.
ax.set_ylim(
    0.85,
    1.02,
)

ax.set_xticks(
    mean_by_k["K_true"]
)

ax.tick_params(
    axis="both",
    labelsize=12,
    length=3,
)

ax.grid(
    True,
    alpha=0.20,
    linewidth=0.6,
)

ax.legend(
    frameon=False,
    fontsize=12,
    ncol=2,
)


# ============================================================
# (b) Mean structural recovery
# ============================================================

ax = axes[1]

ax.plot(
    mean_by_k["K_true"],
    mean_by_k["K_pred"],
    marker="o",
    linewidth=1.8,
    markersize=5.5,
    label=r"Mean $K_{\mathrm{pred}}$",
)

k_min = int(mean_by_k["K_true"].min())
k_max = int(mean_by_k["K_true"].max())

ax.plot(
    [k_min, k_max],
    [k_min, k_max],
    linestyle="--",
    linewidth=1.4,
    label=r"Ideal: $K_{\mathrm{pred}}=K_{\mathrm{true}}$",
)
# Label structural exceptions
for _, row in df.iterrows():

    if row["K_pred"] != row["K_true"]:

        ax.annotate(
            row["Dataset"],
            (
                row["K_true"],
                row["K_pred"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=12,
        )
ax.set_xlabel(
    r"Ground-truth cluster number $K_{\mathrm{true}}$",
    fontsize=12,
)

ax.set_ylabel(
    r"Mean predicted cluster number $\overline{K}_{\mathrm{pred}}$",
    fontsize=12,
)

ax.set_xticks(
    mean_by_k["K_true"]
)

ax.tick_params(
    axis="both",
    labelsize=12,
    length=3,
)

ax.grid(
    True,
    alpha=0.20,
    linewidth=0.6,
)

ax.legend(
    frameon=False,
    fontsize=12,
)


# ============================================================
# Export
# ============================================================

fig.tight_layout(
    w_pad=1.3,
)

OUTPUT_DIR = Path("paper_figures")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

base = OUTPUT_DIR / "cluster_scale_robustness_mean"

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

# Also save the aggregated values for the paper/reviewer response.
mean_by_k.to_csv(
    OUTPUT_DIR / "cluster_scale_robustness_mean.csv",
    index=False,
)

print("\nSaved:")
print(base.with_suffix(".pdf"))
print(base.with_suffix(".png"))
print(base.with_suffix(".svg"))
print(OUTPUT_DIR / "cluster_scale_robustness_mean.csv")
