"""Boundary and region metrics.

Boundary F1 and HD95 are computed on the boundary *curves* (densified polygon
rings); IoU is computed on the *areas* those curves enclose.  All three use the
same grid, the same projection and the same tolerance, fixed before any result
was inspected.

The definitions are globally penalising by construction: extra predicted
regions contribute boundary points that cannot be matched, and missed regions
leave ground-truth boundary points unmatched, so neither is free.
"""

from __future__ import annotations

import numpy as np

import regions as rg

TOL = 1.5  # boundary tolerance, scene units (1.5 % of the 141.4 diagonal)


def boundary_scores(gt_pts, pred_pts, tol=TOL):
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return 0.0, 0.0, 0.0
    d_pred = rg.nn_distances(pred_pts, gt_pts)   # predicted -> ground truth
    d_gt = rg.nn_distances(gt_pts, pred_pts)     # ground truth -> predicted
    prec = float((d_pred <= tol).mean())
    rec = float((d_gt <= tol).mean())
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec


def hd95(gt_pts, pred_pts, diag):
    if len(gt_pts) == 0 or len(pred_pts) == 0:
        return 100.0
    d = np.r_[rg.nn_distances(pred_pts, gt_pts), rg.nn_distances(gt_pts, pred_pts)]
    return float(np.percentile(d, 95) / diag * 100.0)


def region_scores(gt_masks, pred_masks):
    """Hungarian-matched region scores.

    IoU_pen  intersection sum over matched pairs divided by the ground-truth
             area plus the area of unmatched predicted regions; missed regions
             stay in the denominator as ground truth, so both error types cost.
    IoU_match area-weighted mean IoU over matched pairs only (optimistic).
    """
    pairs, gk, pk, M = rg.match_regions(gt_masks, pred_masks)
    gt_area = sum(m.sum() for m in gt_masks.values())
    inter = sum(np.logical_and(gt_masks[g], pred_masks[p]).sum() for g, p, _ in pairs)
    pred_area = sum(m.sum() for m in pred_masks.values())
    matched_pred = {p for _, p, _ in pairs}
    extra_area = sum(v.sum() for k, v in pred_masks.items() if k not in matched_pred)
    denom = gt_area + extra_area
    iou_pen = float(inter / denom) if denom else 0.0
    if pairs:
        w = np.array([gt_masks[g].sum() for g, _, _ in pairs], float)
        iou_match = float(np.average([s for _, _, s in pairs], weights=w))
    else:
        iou_match = 0.0
    return iou_pen, iou_match, len(pairs), len(gk), len(pk)


def leakage(pred_union: np.ndarray, band_mask: np.ndarray) -> float:
    """Share of the predicted area that falls inside the barrier buffer."""
    area = pred_union.sum()
    if area == 0 or band_mask is None:
        return 0.0
    return float(np.logical_and(pred_union, band_mask).sum() / area)


def evaluate(scene, pred_polys, pred_labels, grid, tol=TOL,
             tol_list=(1.0, 1.5, 2.5), representation="native"):
    """Full metric set for one method on one scene.

    representation="native"    IZGC contours / alpha shapes, i.e. each method
                               in its own best form
    representation="unified"   every method's labels pass through the same
                               raster -> 0.5-isoline step, which removes the
                               representation advantage of alpha shapes
    """
    gt_masks = {i: rg.polygon_mask(p, grid) for i, p in enumerate(scene["polys"], 1)}
    labelmap = rg.rasterize_labels(scene["X"], np.asarray(pred_labels), grid)
    pred_masks = rg.label_masks(labelmap)

    gt_pts = np.vstack([rg.boundary_points(p, grid.step) for p in scene["polys"]])
    if representation == "unified":
        pred_pts = (np.vstack([rg.mask_boundary_points(m, grid) for m in pred_masks.values()])
                    if pred_masks else np.empty((0, 2)))
    else:
        pred_pts = (np.vstack([rg.boundary_points(p, grid.step) for p in pred_polys])
                    if pred_polys else np.empty((0, 2)))

    bf, bp, br = boundary_scores(gt_pts, pred_pts, tol)
    iou_pen, iou_match, matched, k_gt, k_pred = region_scores(gt_masks, pred_masks)
    pred_union = labelmap > 0
    band = scene.get("barrier_band")
    band_mask = rg.polygon_mask(band, grid) if band is not None else None

    sens = {}
    for t in tol_list:
        sens[f"BF@{t:g}"] = boundary_scores(gt_pts, pred_pts, t)[0]

    return dict(
        BF=bf, BPrec=bp, BRec=br,
        HD95pct=hd95(gt_pts, pred_pts, grid.diag),
        IoU_pen=iou_pen, IoU_match=iou_match,
        k_gt=k_gt, k_pred=k_pred, k_matched=matched,
        extra=max(0, k_pred - matched), missed=max(0, k_gt - matched),
        leak=leakage(pred_union, band_mask),
        noise_frac=float((np.asarray(pred_labels) < 0).mean()) if len(pred_labels) else 0.0,
        n_pred_boundary=len(pred_pts),
        **sens,
    )
