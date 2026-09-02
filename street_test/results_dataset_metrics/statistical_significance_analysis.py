"""
Statistical Significance Analysis for IZGC

Methods:
    Friedman test
    Nemenyi post-hoc test

Input:
    dataset_metrics.csv

Output:
    average_rank_results.csv
    nemenyi_izgc_only_results.csv

    average_rank_table.tex
    nemenyi_izgc_only_table.tex

"""

import pandas as pd
import numpy as np

from scipy.stats import friedmanchisquare
from scipy.stats import studentized_range



# ==========================================================
# Configuration
# ==========================================================

INPUT_FILE = "dataset_metrics.csv"


ALGORITHMS = [
    "IZGC",
    "K-means++",
    "CLIQUE",
    "BANG",
    "DBSCAN",
    "GMM",
    "DPC-FWSN"
]


METRICS = [
    "AMI",
    "ARI",
    "FMI",
    "Macro_F1",
    "Macro_Jaccard",
    "Completeness",
    "V_measure"
]


ALPHA = 0.05



# ==========================================================
# Utility
# ==========================================================


def format_value(x):

    if isinstance(x, float):
        return f"{x:.3f}"

    return str(x)



# ==========================================================
# Average Rank
# ==========================================================


def calculate_average_rank(df, metric):


    table = df.pivot(
        index="Dataset",
        columns="Algorithm",
        values=metric
    )


    # higher score -> better rank

    ranks = table.rank(
        axis=1,
        ascending=False,
        method="average"
    )


    mean_rank = (
        ranks.mean()
        .sort_values()
    )


    return mean_rank



# ==========================================================
# Friedman Test
# ==========================================================


def friedman_test(df, metric):


    table = df.pivot(
        index="Dataset",
        columns="Algorithm",
        values=metric
    )


    values = []


    for alg in ALGORITHMS:

        values.append(
            table[alg].values
        )


    chi2, p = friedmanchisquare(
        *values
    )


    return chi2, p



# ==========================================================
# Nemenyi Test
# Only IZGC vs other algorithms
# ==========================================================


def nemenyi_test_izgc_only(
        mean_rank,
        n_dataset,
        metric):


    k = len(mean_rank)


    # Critical difference

    q_alpha = (
        studentized_range.ppf(
            1 - ALPHA,
            k,
            np.inf
        )
        /
        np.sqrt(2)
    )


    CD = (
        q_alpha *
        np.sqrt(
            k*(k+1)
            /
            (6*n_dataset)
        )
    )


    results=[]


    izgc_rank = mean_rank["IZGC"]



    for alg in mean_rank.index:


        if alg == "IZGC":
            continue


        diff = abs(
            izgc_rank -
            mean_rank[alg]
        )


        results.append({

            "Metric":
            metric,


            "Comparison":
            f"IZGC vs {alg}",


            "Rank_difference":
            round(diff,3),


            "Critical_difference":
            round(CD,3),


            "Significant":
            "Yes"
            if diff > CD
            else
            "No"

        })


    return pd.DataFrame(results)



# ==========================================================
# Save Latex
# ==========================================================


def save_latex(
        df,
        filename,
        caption,
        label):


    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:


        f.write(
r"""
\begin{table*}[ht]
\centering
\caption{
"""
+
caption
+
r"""
}
\label{
"""
+
label
+
r"""
}

\scriptsize

\begin{tabular}{lllll}

\toprule
"""
        )


        f.write(
            " & ".join(df.columns)
            +
            "\\\\\n"
        )


        f.write(
r"""
\midrule
"""
        )


        for _,row in df.iterrows():


            values = [
                format_value(x)
                for x in row.values
            ]


            f.write(
                " & ".join(values)
                +
                "\\\\\n"
            )


        f.write(
r"""
\bottomrule

\end{tabular}

\end{table*}
"""
        )



# ==========================================================
# Main
# ==========================================================


if __name__ == "__main__":


    df = pd.read_csv(
        INPUT_FILE
    )


    print("="*60)
    print("Statistical Significance Analysis")
    print("="*60)


    print(
        "Datasets:",
        df["Dataset"].nunique()
    )

    print(
        "Algorithms:",
        df["Algorithm"].nunique()
    )



    all_rank_results=[]

    all_nemenyi_results=[]



    # ======================================================
    # Each metric
    # ======================================================


    for metric in METRICS:


        print("\nMetric:", metric)


        # -----------------------------
        # Friedman
        # -----------------------------

        chi2,p = friedman_test(
            df,
            metric
        )


        print(
            "Friedman Chi-square:",
            round(chi2,3)
        )


        print(
            "p-value:",
            round(p,5)
        )



        # -----------------------------
        # Rank
        # -----------------------------

        rank = calculate_average_rank(
            df,
            metric
        )


        rank_df = pd.DataFrame({

            "Metric":
            metric,

            "Algorithm":
            rank.index,


            "Average_Rank":
            [
                round(x,3)
                for x in rank.values
            ]

        })


        all_rank_results.append(
            rank_df
        )



        # -----------------------------
        # Nemenyi IZGC only
        # -----------------------------

        nemenyi_df = (
            nemenyi_test_izgc_only(
                rank,
                df["Dataset"].nunique(),
                metric
            )
        )


        all_nemenyi_results.append(
            nemenyi_df
        )



    # ======================================================
    # Merge
    # ======================================================


    rank_results = pd.concat(
        all_rank_results,
        ignore_index=True
    )


    nemenyi_results = pd.concat(
        all_nemenyi_results,
        ignore_index=True
    )



    # ======================================================
    # Save CSV
    # ======================================================


    rank_results.to_csv(
        "average_rank_results.csv",
        index=False
    )


    nemenyi_results.to_csv(
        "nemenyi_izgc_only_results.csv",
        index=False
    )


    print("\nSaved CSV files.")



    # ======================================================
    # Save Latex
    # ======================================================


    save_latex(

        rank_results,

        "average_rank_table.tex",

        "Average ranks of seven clustering algorithms across benchmark datasets.",

        "tab:S3_rank"

    )



    save_latex(

        nemenyi_results,

        "nemenyi_izgc_only_table.tex",

        "Nemenyi post-hoc comparisons between IZGC and competing algorithms.",

        "tab:S3_nemenyi"

    )


    print("\nSaved Latex tables.")

    print("\nFinished.")