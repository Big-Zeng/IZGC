"""
Generate IZGC clustering visualization
with the same style as experiments.
"""

import matplotlib
matplotlib.use("Agg")


import os
import sys
import numpy as np


sys.path.append(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


import ContourBasedClusteringWithoutZ22 as ContourClustering


from algorithm_comparison.utils import (
    load_dataset,
    normalize_coordinates,
    plot_clustering_result
)



# ==========================
# datasets
# ==========================

DATASETS = [

    "jain",
    "ring",
    "cth",
    "ls",
    "lineblobs",
    "2d_3c"

]


DATA_DIR = "../data"

OUTPUT_DIR = "IZGC_results"



# ==========================
# IZGC parameters
# ==========================

IZGC_PARAMS = {


    "flame":
        3.160,


    "spiral":
        2.460,


    "complex8":
        1.1440,


    "complex9":
        0.813,


    "R15":
        0.803,


    "Aggregation":
        1.491,
    "jain":
    2.826,
    "ring":
    3.043,
    "cth":
    2.004,
    "ls":
    2.170,
    "lineblobs":
    2.923,
    "2d_3c":
    1.742

}



# 固定参数
MAX_BUFFER_DISTANCE = 0

MIN_CLUSTER_SIZE = 5

DENSITY_RADIUS = 30



# ==========================
# Run IZGC
# ==========================

def run_izgc(
        X,
        grid_size
):


    from scipy.spatial import cKDTree


    tree = cKDTree(X)


    density=np.array(

        [
            len(
                tree.query_ball_point(
                    p,
                    DENSITY_RADIUS
                )
            )

            for p in X

        ]

    )


    combined_data=[

        (
            *p,
            density[i],
            i
        )

        for i,p in enumerate(X)

    ]


    cluster = (
        ContourClustering
        .ContourBasedClusteringWithoutZ
        (
            combined_data,

            grid_size=grid_size,

            max_buffer_distance=
            MAX_BUFFER_DISTANCE,

            min_cluster_size=
            MIN_CLUSTER_SIZE,

            merge_small_clusters=True
        )
    )


    cluster.perform_clustering()


    labels = (
        cluster
        .get_predicted_labels()
    )


    return np.array(labels)



# ==========================
# main
# ==========================


if __name__=="__main__":


    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    for name in DATASETS:


        print(
            "\nProcessing",
            name
        )


        # load
        path=os.path.join(
            DATA_DIR,
            name+".csv"
        )


        X,y=load_dataset(
            path
        )


        # same normalization as paper
        X=normalize_coordinates(
            X
        )


        # parameter

        dg=IZGC_PARAMS[name]


        # clustering

        labels=run_izgc(
            X,
            dg
        )


        print(
            name,
            "clusters:",
            len(np.unique(labels))
        )


        # =========================
        # use original plotting
        # =========================

        save_path=os.path.join(

            OUTPUT_DIR,

            f"{name}_IZGC.png"

        )


        plot_clustering_result(

            X,

            labels,

            f"IZGC ({name})",

            save_path,

            dpi=600

        )


    print(
        "\nFinished!"
    )