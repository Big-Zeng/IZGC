# Boundary-quality experiment - results

## Overall (mean over 6 scenes x 20 seeds)

| method | BF | BPrec | BRec | HD95 (%diag) | IoU | ARI | k_pred | extra | missed |
|---|---|---|---|---|---|---|---|---|---|
| IZGC-Full | 0.521 | 0.468 | 0.595 | 4.19 | 0.777 | 0.874 | 2.34 | 0.47 | 0.12 |
| IZGC-NoZP | 0.595 | 0.702 | 0.528 | 12.43 | 0.523 | 0.000 | 0.98 | 0.00 | 1.02 |
| K-means++ | 0.802 | 0.824 | 0.785 | 4.50 | 0.771 | 0.791 | 1.97 | 0.00 | 0.03 |
| GMM | 0.819 | 0.850 | 0.793 | 4.76 | 0.757 | 0.758 | 1.97 | 0.00 | 0.03 |
| DBSCAN | 0.841 | 0.906 | 0.799 | 5.88 | 0.745 | 0.765 | 2.36 | 0.60 | 0.24 |
| CLIQUE | 0.794 | 0.858 | 0.749 | 6.08 | 0.663 | 0.591 | 2.62 | 0.93 | 0.32 |
| BANG | 0.842 | 0.878 | 0.812 | 4.20 | 0.737 | 0.814 | 3.72 | 1.86 | 0.14 |

## Per scene (BF / IoU)

| scene | IZGC-Full | IZGC-NoZP | K-means++ | GMM | DBSCAN | CLIQUE | BANG |
|---|---|---|---|---|---|---|---|
| density_3 | 0.523 / 0.784 | 0.641 / 0.536 | 0.943 / 0.784 | 0.949 / 0.784 | 0.943 / 0.783 | 0.935 / 0.779 | 0.945 / 0.783 |
| density_8 | 0.303 / 0.659 | 0.647 / 0.533 | 0.770 / 0.651 | 0.904 / 0.659 | 0.865 / 0.646 | 0.896 / 0.658 | 0.863 / 0.651 |
| gap_2 | 0.747 / 0.888 | 0.536 / 0.504 | 0.884 / 0.914 | 0.877 / 0.915 | 0.830 / 0.785 | 0.719 / 0.568 | 0.834 / 0.771 |
| gap_4 | 0.343 / 0.851 | 0.532 / 0.504 | 0.888 / 0.916 | 0.888 / 0.916 | 0.894 / 0.915 | 0.803 / 0.736 | 0.767 / 0.658 |
| nonconvex_c | 0.709 / 0.798 | 0.696 / 0.537 | 0.694 / 0.705 | 0.617 / 0.655 | 0.684 / 0.551 | 0.633 / 0.527 | 0.832 / 0.802 |
| nonconvex_u | 0.499 / 0.680 | 0.516 / 0.525 | 0.632 / 0.654 | 0.678 / 0.611 | 0.830 / 0.789 | 0.778 / 0.710 | 0.811 / 0.755 |

## Statistical tests

### metric: BF

| metric | comparison | statistic | p_value | p_holm | note |
|---|---|---|---|---|---|
| BF | Friedman (all methods) | 289.7 | 1.294e-59 |  | 7 methods, 120 blocks |
| BF | IZGC-Full vs IZGC-NoZP | 1859 | 0 | 0 | paired Wilcoxon, Holm |
| BF | IZGC-Full vs K-means++ | 312 | 0 | 0 | paired Wilcoxon, Holm |
| BF | IZGC-Full vs GMM | 484 | 0 | 0 | paired Wilcoxon, Holm |
| BF | IZGC-Full vs DBSCAN | 315 | 0 | 0 | paired Wilcoxon, Holm |
| BF | IZGC-Full vs CLIQUE | 607 | 0 | 0 | paired Wilcoxon, Holm |
| BF | IZGC-Full vs BANG | 63 | 0 | 0 | paired Wilcoxon, Holm |

### metric: IoU_pen

| metric | comparison | statistic | p_value | p_holm | note |
|---|---|---|---|---|---|
| IoU_pen | Friedman (all methods) | 285.7 | 9.332e-59 |  | 7 methods, 120 blocks |
| IoU_pen | IZGC-Full vs IZGC-NoZP | 28 | 0 | 0 | paired Wilcoxon, Holm |
| IoU_pen | IZGC-Full vs K-means++ | 1050 | 0.0219 | 0.0438 | paired Wilcoxon, Holm |
| IoU_pen | IZGC-Full vs GMM | 590 | 0.0602 | 0.0602 | paired Wilcoxon, Holm |
| IoU_pen | IZGC-Full vs DBSCAN | 924 | 0.0013 | 0.004 | paired Wilcoxon, Holm |
| IoU_pen | IZGC-Full vs CLIQUE | 666 | 0 | 0 | paired Wilcoxon, Holm |
| IoU_pen | IZGC-Full vs BANG | 1144 | 0 | 0.0002 | paired Wilcoxon, Holm |

### metric: HD95pct

| metric | comparison | statistic | p_value | p_holm | note |
|---|---|---|---|---|---|
| HD95pct | Friedman (all methods) | 294.3 | 1.332e-60 |  | 7 methods, 120 blocks |
| HD95pct | IZGC-Full vs IZGC-NoZP | 0 | 0 | 0 | paired Wilcoxon, Holm |
| HD95pct | IZGC-Full vs K-means++ | 3434 | 0.8372 | 1 | paired Wilcoxon, Holm |
| HD95pct | IZGC-Full vs GMM | 3417 | 0.8017 | 1 | paired Wilcoxon, Holm |
| HD95pct | IZGC-Full vs DBSCAN | 3506 | 0.9904 | 0.9904 | paired Wilcoxon, Holm |
| HD95pct | IZGC-Full vs CLIQUE | 2282 | 0.001 | 0.0049 | paired Wilcoxon, Holm |
| HD95pct | IZGC-Full vs BANG | 2899 | 0.1006 | 0.4022 | paired Wilcoxon, Holm |
