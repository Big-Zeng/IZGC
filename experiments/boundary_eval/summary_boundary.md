# Boundary-quality experiment - results

## Overall (mean over 6 scenes x 20 seeds)

| method | BF | BPrec | BRec | HD95 (%diag) | IoU | ARI | k_pred | extra | missed |
|---|---|---|---|---|---|---|---|---|---|
| IZGC-Full | 0.716 | 0.650 | 0.811 | 4.76 | 0.786 | 0.896 | 2.19 | 0.30 | 0.11 |
| IZGC-NoZP | 0.595 | 0.702 | 0.528 | 12.43 | 0.523 | 0.000 | 0.98 | 0.00 | 1.02 |
| K-means++ | 0.802 | 0.824 | 0.785 | 4.50 | 0.771 | 0.791 | 1.97 | 0.00 | 0.03 |
| GMM | 0.830 | 0.865 | 0.802 | 4.54 | 0.766 | 0.782 | 1.97 | 0.00 | 0.03 |
| DBSCAN | 0.841 | 0.908 | 0.794 | 5.94 | 0.735 | 0.741 | 2.06 | 0.33 | 0.27 |
| CLIQUE | 0.794 | 0.857 | 0.748 | 6.08 | 0.663 | 0.591 | 2.62 | 0.93 | 0.32 |
| BANG | 0.842 | 0.878 | 0.812 | 4.19 | 0.736 | 0.813 | 3.79 | 1.90 | 0.11 |

## Per scene (BF / IoU)

| scene | IZGC-Full | IZGC-NoZP | K-means++ | GMM | DBSCAN | CLIQUE | BANG |
|---|---|---|---|---|---|---|---|
| density_3 | 0.691 / 0.782 | 0.641 / 0.536 | 0.943 / 0.784 | 0.949 / 0.784 | 0.949 / 0.784 | 0.935 / 0.779 | 0.943 / 0.782 |
| density_8 | 0.535 / 0.651 | 0.647 / 0.533 | 0.770 / 0.651 | 0.904 / 0.659 | 0.904 / 0.659 | 0.893 / 0.656 | 0.863 / 0.651 |
| gap_2 | 0.849 / 0.807 | 0.536 / 0.504 | 0.884 / 0.914 | 0.874 / 0.914 | 0.821 / 0.780 | 0.719 / 0.568 | 0.834 / 0.771 |
| gap_4 | 0.843 / 0.917 | 0.532 / 0.504 | 0.888 / 0.916 | 0.882 / 0.916 | 0.850 / 0.830 | 0.803 / 0.736 | 0.769 / 0.651 |
| nonconvex_c | 0.744 / 0.817 | 0.696 / 0.537 | 0.694 / 0.705 | 0.621 / 0.670 | 0.689 / 0.565 | 0.633 / 0.527 | 0.832 / 0.802 |
| nonconvex_u | 0.638 / 0.739 | 0.516 / 0.525 | 0.632 / 0.654 | 0.752 / 0.653 | 0.830 / 0.791 | 0.778 / 0.710 | 0.811 / 0.755 |

## Statistical tests

### metric: BF

| metric | comparison | statistic | p_value | p_holm | note |
|---|---|---|---|---|---|
| BF | Friedman (all methods) | 243.1 | 1.208e-49 |  | 7 methods, 120 blocks |
| BF | IZGC-Full vs IZGC-NoZP | 1111 | 0 | 0 | paired Wilcoxon, Holm |
| BF | IZGC-Full vs K-means++ | 1304 | 0 | 0 | paired Wilcoxon, Holm |
| BF | IZGC-Full vs GMM | 1244 | 0 | 0 | paired Wilcoxon, Holm |
| BF | IZGC-Full vs DBSCAN | 1172 | 0 | 0 | paired Wilcoxon, Holm |
| BF | IZGC-Full vs CLIQUE | 2125 | 0.0002 | 0.0002 | paired Wilcoxon, Holm |
| BF | IZGC-Full vs BANG | 973 | 0 | 0 | paired Wilcoxon, Holm |

### metric: IoU_pen

| metric | comparison | statistic | p_value | p_holm | note |
|---|---|---|---|---|---|
| IoU_pen | Friedman (all methods) | 336.8 | 1.075e-69 |  | 7 methods, 120 blocks |
| IoU_pen | IZGC-Full vs IZGC-NoZP | 9 | 0 | 0 | paired Wilcoxon, Holm |
| IoU_pen | IZGC-Full vs K-means++ | 742 | 0.0003 | 0.0009 | paired Wilcoxon, Holm |
| IoU_pen | IZGC-Full vs GMM | 500 | 0.0059 | 0.0059 | paired Wilcoxon, Holm |
| IoU_pen | IZGC-Full vs DBSCAN | 408 | 0.0003 | 0.0006 | paired Wilcoxon, Holm |
| IoU_pen | IZGC-Full vs CLIQUE | 450 | 0 | 0 | paired Wilcoxon, Holm |
| IoU_pen | IZGC-Full vs BANG | 833 | 0 | 0 | paired Wilcoxon, Holm |

### metric: HD95pct

| metric | comparison | statistic | p_value | p_holm | note |
|---|---|---|---|---|---|
| HD95pct | Friedman (all methods) | 275.6 | 1.383e-56 |  | 7 methods, 120 blocks |
| HD95pct | IZGC-Full vs IZGC-NoZP | 141 | 0 | 0 | paired Wilcoxon, Holm |
| HD95pct | IZGC-Full vs K-means++ | 3507 | 0.9925 | 0.9925 | paired Wilcoxon, Holm |
| HD95pct | IZGC-Full vs GMM | 3486 | 0.9475 | 1 | paired Wilcoxon, Holm |
| HD95pct | IZGC-Full vs DBSCAN | 3193 | 0.3939 | 1 | paired Wilcoxon, Holm |
| HD95pct | IZGC-Full vs CLIQUE | 2670 | 0.024 | 0.12 | paired Wilcoxon, Holm |
| HD95pct | IZGC-Full vs BANG | 3237 | 0.4627 | 1 | paired Wilcoxon, Holm |
