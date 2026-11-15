# Boundary-quality experiment

Evaluates whether the regions returned by a method match the *true* boundaries
of the generating geometry, which label-based indices such as ARI cannot
measure.

## Why an extra benchmark

The 12 classic benchmarks (Flame, Spiral, Jain, ...) provide points and labels
only.  Reconstructing a "true" boundary from those labels with an alpha shape
or a convex hull would make the evaluation circular: the method whose output is
closest to the reconstruction wins by construction.  Every scene here is
therefore defined by the polygon used to *generate* the points, so the
reference boundary is analytic.

## Six scenes in three families

| Scene | Family | What it probes |
|---|---|---|
| `nonconvex_c` | A non-convex | a C-shaped ring whose open side is filled by cluster 2: a convex or linearly separable region model must fail |
| `nonconvex_u` | A non-convex | a U-shaped block with cluster 2 in the trough between its arms |
| `gap_4` | B narrow gap | two irregular blobs 4.0 units apart |
| `gap_2` | B narrow gap | the same pair 2.0 units apart, i.e. about one inter-point spacing (spurious bridging) |
| `density_3` | C heterogeneous | mean density ratio 2.94 between clusters, plus a within-cluster core/rim gradient |
| `density_8` | C heterogeneous | mean density ratio 8.27, same geometry family |

Each cluster holds 300-500 points (the exact per-cluster counts are written to
`scene_manifest.json`).  Family A and B are sampled at ONE common
points-per-area rate, so a non-convex or narrow-gap test measures geometry
rather than a density contrast; family C is the only one with a deliberate
density contrast.  The narrow gap is calibrated on the real geometry: the two
polygons are translated until their closest-point distance equals the nominal
value, which is not the same as the centre-to-centre arithmetic for irregular
outlines.

## Protocol

1. **Parameter search exactly as in the paper.**  Search ranges are those of
   supplementary Table S4, step sizes those of
   `algorithm_comparison/run_all.py`; the objective is ARI, the paper's own
   `OPTIMIZE_METRIC`.  IZGC searches only the grid step,
   $\Delta_g \in [0.5, 5]$ with step 0.5, at the published $R=30$, $\theta=5$.
   DBSCAN sweeps 2,450 configurations, CLIQUE 4,850, BANG 987, GMM 44.  The
   search runs on validation seed 9001, which is never reported, and ARI never
   looks at a boundary, so the boundary scores cannot have been tuned.

   | method | parameters swept |
   |---|---|
   | IZGC | $\Delta_g$ 0.5:0.5:5.0 (R = 30, theta = 5 fixed) |
   | K-means++ | k fixed to the true count |
   | GMM | covariance {full, tied, diag, spherical} x n_init 1:5:50 |
   | DBSCAN | eps 1:1:50, min_samples 2:1:50 |
   | CLIQUE | intervals 4:1:100, density threshold 1:1:50 |
   | BANG | levels 4:1:50, density threshold 0.01:0.05:1 |

2. **One representation for all methods.**  IZGC contributes its own accepted
   contours; point-label methods are polygonised with the same alpha shape,
   whose alpha is chosen on the validation seed by area agreement (IoU) so the
   baselines obtain an outline of their own region rather than a crippled one.
   Boundary F1 and HD95 are computed on the boundary curves, IoU on the
   enclosed areas.

3. **Common geometry.**  One grid (1 unit), one projection, one boundary
   tolerance (1.5 units, i.e. 1.06 % of the scene diagonal); tolerance
   sensitivity at 1.0 / 1.5 / 2.5 units is reported alongside.

4. **Extra and missed regions cost.**  Matching is Hungarian on IoU.  In the
   reported `IoU_pen` the denominator contains all ground-truth area plus the
   area of unmatched predicted regions, and unmatched ground-truth boundary
   points stay in the recall term, so neither over- nor under-segmentation is
   free.

## Metrics

| Metric | Meaning | Better |
|---|---|---|
| `BF` | Boundary F1 within tolerance, boundary curves only | higher |
| `BPrec`, `BRec` | its precision and recall components | higher |
| `HD95pct` | 95th-percentile symmetric Hausdorff distance, % of the scene diagonal | lower |
| `IoU_pen` | penalised IoU over matched and unmatched regions | higher |
| `k_pred`, `extra`, `missed` | predicted count and the two error types | - |

## Ablation

| Configuration | Automatic zero-value points | Contours |
|---|---|---|
| `IZGC-Full` | yes | yes |
| `IZGC-NoZP` | no | yes |

`IZGC-NoZP` overrides `generate_virtual_points` in a subclass, so the released
algorithm is untouched.  The manual-barrier arm is not used here: none of the
six scenes contains a hard barrier, so it would duplicate `IZGC-Full`.

## Files

```
scene_gen.py           six scenes, analytic polygons, manifest
regions.py             grid, rasterisation, alpha shapes, Hungarian matching
metrics.py             Boundary F1, HD95, IoU (penalised and matched)
adapters.py            uniform method interface, ablation subclass, paper grids
run_boundary.py        parameter search -> frozen parameters -> test seeds
summarise_boundary.py  tables, Friedman, paired Wilcoxon (Holm)
plot_boundary.py       qualitative figure
results_boundary.csv   one row per scene x seed x method
params_selected.json   frozen parameters per scene and method
alpha_selected.json    alpha of the shared polygonisation per scene
scene_manifest.json    per-cluster counts, areas, densities, gap widths
```

## Run

```bash
python scene_gen.py                            # print the six scenes
python run_boundary.py --seeds 20 --workers 12 # search + evaluation
python summarise_boundary.py
python plot_boundary.py
```

Runtime: the parameter search dominates (5.0 x 10^4 configurations); on 12
worker processes it takes about 15-20 minutes, after which the results are
cached in `params_selected.json`.

## Notes on scale

Coordinates are already in a fixed 0-100 square, matching the normalisation the
manuscript uses for every baseline, so the paper's parameter ranges apply
without rescaling.  `uniform_scale` implements the single-factor rescaling
`s = 100 / max(range_x, range_y)` for external data, which preserves distances
and aspect ratio.
