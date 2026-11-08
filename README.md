# IZGC — Isoline Zero-point Grid Clustering

Reference implementation of **IZGC**, the constraint-aware spatial clustering framework described in
*"An Intelligent Constraint-Aware Spatial Clustering Framework via Density Field Reconstruction"*
(Zeng, Lin, Pan & Chu, submitted to *Computers & Geosciences*).

IZGC reconstructs a smooth density field from an unlabelled point set on a locally extended grid,
inserts zero-valued points into empty grid cells so that low-density regions become explicit, extracts
isolines from the reconstructed field, and converts the contour hierarchy into cluster regions. Points
are then assigned to density centres, which yields a partition that follows the topological boundary of
each cluster instead of an assumed convex shape.

This repository contains the method, the baseline algorithms used for comparison, the benchmark datasets,
and the scripts behind the experiments, tables and figures in the manuscript.

---

## Repository layout

| Path | Contents |
|------|----------|
| `ContourBasedClusteringWithoutZ22.py` | **Reference implementation of IZGC** — the exact version used for all reported results |
| `compareTest2.py` | Minimal single-dataset demo: IZGC vs. K-means++ vs. DBSCAN, with metrics and a four-panel figure |
| `clustering_metrics.py` | AMI, ARI and FMI implementations used by the demo |
| `algorithm_comparison/` | Head-to-head benchmarking harness (`alg_*.py` wrappers, `run_all.py`) |
| `algorithm_comparison/utils.py` | Data loading, normalisation, evaluation and plotting helpers |
| `data/` | Benchmark point sets in CSV format (`x, y, target`) |
| `data/real_data/` | Case-study inputs for the municipal delivery-zone experiment (coordinates only) |
| `street_test/` | Scripts behind the noise, overlap and per-dataset metric tables, plus figure generation |
| `addedExperiment/` | Stress-test script with the full seven-metric protocol |
| `docs/README.zh-CN.md` | Notes in Chinese (development history, parameter tuning hints) |

Files whose names carry other version suffixes (`…WithoutZ.py`, `…WithoutZ2.py`, `…WithoutZ3.py`,
`…WithoutZ4.py`, `…WithoutZ21.py`, `…WithoutZ22_v1.py`, `…WithoutZ22_map.py`) are earlier development
snapshots. They are kept for provenance only and are **not** required to reproduce the paper; the
reference implementation is `ContourBasedClusteringWithoutZ22.py`.

---

## Requirements

* Python 3.9 or newer (tested with CPython 3.11.0 on Windows 11 and Linux)
* `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `shapely`

Optional, only for specific baselines and analyses:

| Package | Needed for |
|---------|-----------|
| `pyclustering` | BANG baseline (`algorithm_comparison/alg_bang.py`) |
| `SALib` | Sobol global sensitivity analysis (`sobol_sensitivity_test.py`) |
| MATLAB R2020a+ with the DPC-FWSN sources | DPC-FWSN baseline (`algorithm_comparison/alg_dpc_fwsn.py`); skipped automatically when MATLAB is unavailable |

## Installation

```bash
git clone https://github.com/Big-Zeng/IZGC.git
cd IZGC
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Quick start

Run the minimal demo from the repository root:

```bash
python compareTest2.py
```

It loads `data/flame.csv`, normalises the coordinates to `[0, 100]`, runs IZGC, K-means++ and DBSCAN,
prints the metrics and writes three figures to the repository root
(`contour_areas.png`, `cluster_result.png`, `clustering_comparison.png`).

Output on the Flame dataset (Python 3.11.0, scikit-learn 1.6.1):

```
KMeans Clustering Results:      Inertia 170054.88   Silhouette  0.3806   ARI 0.4534
DBSCAN Clustering Results:      Inertia 168684.88   Silhouette -0.1091   ARI 0.0412
Contour-Based Clustering:       Inertia 191916.44   Silhouette  0.3323   ARI 1.0000
```

The demo runs in well under a minute on a laptop. To run without a display (servers, CI, batch jobs):

```bash
MPLBACKEND=Agg python compareTest2.py
```

`MPLBACKEND=Agg` makes the interactive figure windows return immediately, so no window has to be closed
by hand. To switch a dataset, edit the two settings at the top of `compareTest2.py`:

```python
DATASET = "data/flame.csv"     # any CSV with columns x, y, target
NORMALIZE_COORDS = True        # scale both axes to [0, 100]
```

## Input data format

Every dataset is a plain CSV with a header row and at least three columns:

```csv
x,y,target
1.85,27.8,1
1.35,26.65,1
1.40,23.25,2
```

* `x`, `y` — two-dimensional coordinates (any unit; normalisation is applied by the calling script)
* `target` — ground-truth label, used for the external validation indices only

Files in `data/real_data/` hold the case-study delivery-point coordinates and carry `x, y` without
`target`; supervised indices are therefore not computed for them.

---

## Benchmark datasets

The synthetic benchmarks used in the manuscript are all in `data/`:

| Dataset | File | Dataset | File |
|---------|------|---------|------|
| Flame | `flame.csv` | Aggregation | `Aggregation.csv` |
| Jain | `jain.csv` | Compound | `Compound.csv` |
| Spiral | `spiral.csv` | R15 | `R15.csv` |
| Cth | `cth.csv` | D31 | `D31.csv` |
| Complex8 | `complex8.csv` | Complex9 | `complex9.csv` |
| S1–S4 | `s1.csv`, `s2.csv`, … | A2, A3 | `a2.csv`, `a3.csv` |

Additional point sets (`ring.csv`, `ls.csv`, `lineblobs.csv`, `2d_3c.csv`, `zelnik4.csv`,
`cluto-t5-8k.csv`, …) are retained so that the individual benchmark configurations reported in the paper
can be re-run.

---

## Reproducing the results in the paper

| Experiment in the manuscript | Command | Output |
|------------------------------|---------|--------|
| Algorithm comparison across benchmark datasets | `cd algorithm_comparison && python run_all.py` | one figure per algorithm plus a metrics table in `algorithm_comparison/results/` |
| Per-dataset seven-metric comparison | `python street_test/dataset_metrics.py` | `street_test/results_dataset_metrics/` |
| Noise robustness study | `python street_test/noise_experiment_spiral.py` | `street_test/results_noise_spiral/` |
| Overlap robustness study | `python street_test/overlap_experiment.py` | `street_test/results_overlap/` |
| Calibrated stress test (Δg calibration plus repeated runs) | `python addedExperiment/run_izgc_stress_test.py --repeats 30` | `results_stress_test/` |
| Case-study zoning maps | `python algorithm_comparison/run_contour_unsupervised.py` | `algorithm_comparison/results_unsupervised/` |
| Global sensitivity analysis (Sobol) | `python sobol_sensitivity_test.py` | `sobol_results/` |
| Grid-spacing sweep | `python test_grid_size_sweep.py` | `grid_size_sweep_results.png` and `.txt` in the current directory |

`run_all.py` executes IZGC, K-means++, GMM, DBSCAN, CLIQUE, BANG and DPC-FWSN on a single dataset and
writes one figure per algorithm plus a metrics table. DPC-FWSN requires MATLAB and is skipped with a
warning when the engine is not found, so the remaining baselines still run.

The seven-metric protocol used in the tables is AMI, ARI, FMI, Macro-F1, Macro-Jaccard, Completeness and
V-measure. Macro-F1 and Macro-Jaccard are computed after Hungarian label alignment, which makes them
permutation-invariant.

---

## Main parameters

IZGC has three primary hyperparameters — the grid step Δg, the search radius R and the minimum cluster
scale θ — plus one assignment buffer used when attaching boundary points to density centres.

| Code parameter | Paper symbol | Demo default | Effect |
|----------------|--------------|--------------|--------|
| `grid_size` | Δg | `3.160` | Spacing of the interpolation grid. Smaller values give finer cluster boundaries at higher cost |
| `radius` | R | `30` | Neighbourhood radius of the `cKDTree` density estimate, in normalised units |
| `min_cluster_size` | θ | `5` | Clusters below this size are merged into the nearest large cluster |
| `max_buffer_distance` | — | `0` | Buffer used when assigning boundary points to density centres |

Δg is selected per dataset: the reported experiments use values between 1.74 and 3.04, and the exact
value for each dataset is recorded in the header of `street_test/dataset_metrics.py`. Setting
`merge_small_clusters=False` labels the points of small clusters as noise (`-1`) instead of merging them.

```python
import ContourBasedClusteringWithoutZ22 as izgc

model = izgc.ContourBasedClusteringWithoutZ22(
    combined_data,               # [(x, y, density, index), ...]
    grid_size=3.160,             # Δg
    max_buffer_distance=0,
    min_cluster_size=5,          # θ
    merge_small_clusters=True,
    interactive=False,
)
labels = model.perform_clustering()
```

`interactive=True` opens the intermediate figures and waits for them to be closed; use `False` for batch
runs. The density estimate uses the `radius` argument of the calling wrapper
(`VRPAnalyzer.run_contour_clustering(radius=30, ...)` in `compareTest2.py`), not a constructor argument.

Δg is scale dependent. Because coordinates are normalised to a 100 × 100 box before clustering, a given
Δg is comparable across datasets; if normalisation is disabled, Δg and R must be rescaled to the
coordinate units of the data.

---

## Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| ARI | [−1, 1] | Agreement with ground truth, corrected for chance |
| AMI | [0, 1] | Shared information between partitions, normalised |
| FMI | [0, 1] | Pairwise precision/recall trade-off |
| Macro-F1 | [0, 1] | Class-balanced F1 after optimal label alignment |
| V-measure | [0, 1] | Harmonic mean of homogeneity and completeness |
| Silhouette | [−1, 1] | Internal cohesion/separation, computed on non-noise points |
| Inertia | ≥ 0 | Within-cluster sum of squared distances |

Internal indices (Silhouette, Inertia) are reported for reference only: they favour compact, convex
clusters and are therefore biased against non-convex ground truth.

---

## Case-study data

The municipal delivery-zone experiment uses delivery point sets supplied by a commercial partner. The
release in `data/real_data/` contains coordinate pairs only (`x, y`, in degrees of longitude and
latitude); customer, address, order and time attributes are not part of this repository. Because no
ground-truth partition exists for these areas, only internal indices are reported for the case study,
and the visual assessment of the resulting zones is qualitative.

---

## License

Released under the **GNU General Public License v3.0** — see [`LICENSE`](LICENSE).

Copyright (C) 2025–2026 Xiangfei Zeng, Zhinan Lin, Jeng-Shyang Pan, Shu-Chuan Chu.

You may redistribute and/or modify this code under the terms of the GPL-3.0 as published by the Free
Software Foundation. The program is distributed in the hope that it will be useful, but **without any
warranty**; without even the implied warranty of merchantability or fitness for a particular purpose.
Derivative works distributed to others must remain under the same licence.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{zeng2026izgc,
  title   = {An Intelligent Constraint-Aware Spatial Clustering Framework via Density Field Reconstruction},
  author  = {Zeng, Xiangfei and Lin, Zhinan and Pan, Jeng-Shyang and Chu, Shu-Chuan},
  journal = {Computers \& Geosciences},
  year    = {2026},
  note    = {Manuscript submitted for publication}
}
```

## Contact

* Xiangfei Zeng — xiangfeizeng@ygu.edu.cn
* Jeng-Shyang Pan (corresponding author) — 100112@nuist.edu.cn

Bug reports and questions are welcome through the issue tracker.

---

*中文说明见 [`docs/README.zh-CN.md`](docs/README.zh-CN.md).*
