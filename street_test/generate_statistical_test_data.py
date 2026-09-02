"""
Statistical Significance Analysis for IZGC

Workflow:

1. Run seven clustering algorithms
2. Collect seven evaluation metrics
3. Perform Friedman test
4. Perform Wilcoxon signed-rank test
5. Generate tables for Supplementary Material S3

Algorithms:
IZGC, K-means++, CLIQUE, BANG, DBSCAN, GMM, DPC-FWSN

Metrics:
AMI, ARI, FMI, Macro_F1,
Macro_Jaccard, Completeness, V_measure
"""


import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd


from scipy.stats import (
    friedmanchisquare,
    wilcoxon
)


# ============================================================
# Import your existing experiment module
# ============================================================

from dataset_metrics import (
    DATASETS,
    ALGORITHMS,
    METRIC_COLUMNS,

    load_dataset,

    evaluate_clustering,

    run_izgc,
    run_kmeans,
    run_dbscan,
    run_gmm,
    run_clique,
    run_bang,
    run_dpc_fwsn,

    IZGC_PARAMS,
    CLIQUE_PARAMS,
    BANG_PARAMS,
    DBSCAN_PARAMS,
    GMM_N_COMPONENTS,
    GMM_N_INIT,
    DPC_K,
    KMEANS_N_INIT,
    KMEANS_MAX_ITER,
    RANDOM_SEED,

    PYCLUSTERING_AVAILABLE,
    MATLAB_AVAILABLE,

    MatlabRunner
)



# ============================================================
# Run experiments
# ============================================================


def run_all_algorithms():


    results=[]


    matlab_runner=None


    if MATLAB_AVAILABLE:

        matlab_runner = MatlabRunner(
            method="auto"
        )



    data_dir = (
        Path(__file__)
        .resolve()
        .parent
        /
        "../data"
    )


    for csv_name, dataset_name in DATASETS:


        print("\n")
        print("="*80)

        print(
            "Dataset:",
            dataset_name
        )

        print("="*80)



        csv_path = (
            data_dir /
            csv_name
        )


        X,y_raw,y_true = load_dataset(
            str(csv_path)
        )


        true_k = len(
            np.unique(y_true)
        )


        # ====================================================
        # IZGC
        # ====================================================

        print("Running IZGC")

        p = IZGC_PARAMS[dataset_name]


        y_pred = run_izgc(
            X,
            p["grid_size"],
            p["R"],
            p["theta"]
        )


        metrics = evaluate_clustering(
            y_true,
            y_pred
        )


        results.append(
            make_row(
                dataset_name,
                "IZGC",
                metrics
            )
        )



        # ====================================================
        # K-means++
        # ====================================================

        print("Running K-means++")


        y_pred = run_kmeans(
            X,
            true_k,
            n_init=KMEANS_N_INIT,
            max_iter=KMEANS_MAX_ITER,
            random_state=RANDOM_SEED
        )


        metrics=evaluate_clustering(
            y_true,
            y_pred
        )


        results.append(
            make_row(
                dataset_name,
                "K-means++",
                metrics
            )
        )



        # ====================================================
        # CLIQUE
        # ====================================================

        if PYCLUSTERING_AVAILABLE:


            print("Running CLIQUE")


            p=CLIQUE_PARAMS[dataset_name]


            y_pred=run_clique(
                X,
                p["intervals"],
                p["density_threshold"]
            )


            metrics=evaluate_clustering(
                y_true,
                y_pred
            )


            results.append(
                make_row(
                    dataset_name,
                    "CLIQUE",
                    metrics
                )
            )



        # ====================================================
        # BANG
        # ====================================================

        if PYCLUSTERING_AVAILABLE:


            print("Running BANG")


            p=BANG_PARAMS[dataset_name]


            y_pred=run_bang(
                X,
                p["levels"],
                p["density_threshold"]
            )


            metrics=evaluate_clustering(
                y_true,
                y_pred
            )


            results.append(
                make_row(
                    dataset_name,
                    "BANG",
                    metrics
                )
            )



        # ====================================================
        # DBSCAN
        # ====================================================


        print("Running DBSCAN")


        p=DBSCAN_PARAMS[dataset_name]


        y_pred=run_dbscan(
            X,
            p["eps"],
            p["min_samples"]
        )


        metrics=evaluate_clustering(
            y_true,
            y_pred
        )


        results.append(
            make_row(
                dataset_name,
                "DBSCAN",
                metrics
            )
        )



        # ====================================================
        # GMM
        # ====================================================


        print("Running GMM")


        n=GMM_N_COMPONENTS[dataset_name]

        if n is None:

            n=true_k


        y_pred=run_gmm(
            X,
            n,
            n_init=GMM_N_INIT,
            random_state=RANDOM_SEED
        )


        metrics=evaluate_clustering(
            y_true,
            y_pred
        )


        results.append(
            make_row(
                dataset_name,
                "GMM",
                metrics
            )
        )



        # ====================================================
        # DPC-FWSN
        # ====================================================


        if MATLAB_AVAILABLE:


            print("Running DPC-FWSN")


            y_pred,matlab_runner = run_dpc_fwsn(
                X,
                y_raw,
                DPC_K[dataset_name],
                matlab_runner
            )


            metrics=evaluate_clustering(
                y_true,
                y_pred
            )


            results.append(
                make_row(
                    dataset_name,
                    "DPC-FWSN",
                    metrics
                )
            )



    if matlab_runner:

        matlab_runner.close()



    return pd.DataFrame(results)



# ============================================================
# create row
# ============================================================


def make_row(
        dataset,
        algorithm,
        metrics):


    row={

        "Dataset":
        dataset,

        "Algorithm":
        algorithm

    }


    for m in METRIC_COLUMNS:

        row[m]=metrics[m]


    return row




# ============================================================
# Friedman test
# ============================================================


def friedman_test(df):


    output=[]


    for metric in METRIC_COLUMNS:


        table=df.pivot(
            index="Dataset",
            columns="Algorithm",
            values=metric
        )


        values=[
            table[a].values
            for a in ALGORITHMS
            if a in table.columns
        ]


        stat,p=friedmanchisquare(
            *values
        )


        output.append({

            "Metric":metric,

            "Chi_square":stat,

            "p_value":p

        })


    return pd.DataFrame(output)




# ============================================================
# Wilcoxon IZGC vs second best
# ============================================================


def wilcoxon_test(df):


    output=[]


    for metric in METRIC_COLUMNS:


        table=df.pivot(
            index="Dataset",
            columns="Algorithm",
            values=metric
        )


        rank=(
            table.mean()
            .sort_values(
                ascending=False
            )
        )


        second=rank.index[1]


        stat,p=wilcoxon(
            table["IZGC"],
            table[second]
        )


        output.append({

            "Metric":metric,

            "Comparison":
            f"IZGC vs {second}",

            "p_value":p

        })


    return pd.DataFrame(output)




# ============================================================
# Main
# ============================================================


if __name__=="__main__":


    outdir=Path(
        "results_statistical"
    )

    outdir.mkdir(
        exist_ok=True
    )


    print(
        "Start experiments..."
    )


    df=run_all_algorithms()


    df.to_csv(
        outdir /
        "statistical_raw_results.csv",
        index=False
    )


    print(
        "Running Friedman test..."
    )


    friedman=friedman_test(df)


    friedman.to_csv(
        outdir /
        "friedman_results.csv",
        index=False
    )



    print(
        "Running Wilcoxon test..."
    )


    wilcoxon_result=wilcoxon_test(df)


    wilcoxon_result.to_csv(
        outdir /
        "wilcoxon_results.csv",
        index=False
    )



    print("\nFinished!")