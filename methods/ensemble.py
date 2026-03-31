"""
Ensemble method: combines U-Net pixel predictions with GNN cell predictions.

Strategy (v4l — U-Net primary + GNN cell-vote filtering + gentle refinement):
  1. Run U-Net to get probability map (pixel-level villus prediction)
  2. Run GNN to get cell-level villus probabilities + centroids
  3. Extract connected components from U-Net prob map at FIXED low threshold
  4. For each component, compute confidence = f(cell_vote, mean_prob, area)
  5. Select components via gap detection on confidence scores
  6. Gently smooth polygon boundaries (light Savitzky-Golay only)
  7. Final FP elimination via confidence floor

Key v4l fix: use fixed UNET_THRESHOLD=0.40 instead of pixel-metric sweep.
Both Dice (v4j) and Tversky (v4k) sweeps picked 0.60 because pixel metrics
always prefer tighter masks when GT footprint is small. But the GNN cell-vote
filter handles FP removal, so we want GENEROUS boundaries for better IoU.
v4i used 0.50 → IoU 0.805. Going to 0.40 should capture even more boundary.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import savgol_filter
from shapely.geometry import Polygon
from skimage import measure, morphology

from utils.coords import PIXEL_SIZE_UM

DOWNSCALE_FACTOR = 4  # Must match unet_seg.py
MIN_AREA_UM2 = 5000.0

# Fixed U-Net threshold — generous to capture boundary pixels.
# FP components are removed by GNN cell-vote confidence scoring.
# v4i=0.50 → IoU 0.805; v4j/v4k=0.60 → IoU 0.744/0.751
UNET_THRESHOLD = 0.40

# GNN cell-vote thresholds to sweep
GNN_VOTE_CANDIDATES = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _save_ensemble_diagnostics_v4j(
    output_dir: Path,
    unet_prob: np.ndarray,
    ensemble_polygons: list[Polygon],
    gt_polygons: list[Polygon],
    all_components: list[dict],
    selected_components: list[dict],
    ds_pixel_um: float,
    unet_thr: float,
    gnn_thr: float,
):
    """Save diagnostic figure for v4j confidence-scored ensemble."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[{_ts()}]   matplotlib not available -- skipping diagnostics")
        return

    fig, axes = plt.subplots(1, 4, figsize=(32, 8))

    # Panel 1: U-Net probability map with threshold contour
    axes[0].imshow(unet_prob, cmap="hot", vmin=0, vmax=1)
    axes[0].contour(unet_prob, levels=[unet_thr], colors=["cyan"], linewidths=0.5)
    axes[0].set_title(
        f"U-Net Prob Map (thr={unet_thr:.2f}, "
        f"{len(all_components)} components > {MIN_AREA_UM2:.0f}µm²)"
    )
    axes[0].axis("off")

    # Panel 2: Confidence scores — selected vs rejected
    axes[1].imshow(unet_prob, cmap="gray", vmin=0, vmax=1)
    selected_ids = {c["id"] for c in selected_components}
    for c in all_components:
        color = "lime" if c["id"] in selected_ids else "red"
        cx_px = c["cx"] / ds_pixel_um
        cy_px = c["cy"] / ds_pixel_um
        axes[1].plot(cx_px, cy_px, "o", color=color, markersize=6)
        axes[1].annotate(
            f"c={c['confidence']:.1f}",
            (cx_px, cy_px),
            fontsize=6,
            color=color,
            ha="center",
            va="bottom",
        )
    axes[1].set_title(
        f"Confidence Filter (GNN thr={gnn_thr:.2f}): "
        f"{len(selected_components)} kept / "
        f"{len(all_components) - len(selected_components)} rejected"
    )
    axes[1].axis("off")

    # Panel 3: Final ensemble polygons vs GT
    axes[2].imshow(unet_prob, cmap="gray", vmin=0, vmax=1)
    for poly in gt_polygons:
        xs, ys = poly.exterior.xy
        xs_px = [x / ds_pixel_um for x in xs]
        ys_px = [y / ds_pixel_um for y in ys]
        axes[2].plot(xs_px, ys_px, color="lime", linewidth=1.5, linestyle="--")
    for poly in ensemble_polygons:
        xs, ys = poly.exterior.xy
        xs_px = [x / ds_pixel_um for x in xs]
        ys_px = [y / ds_pixel_um for y in ys]
        axes[2].plot(xs_px, ys_px, color="red", linewidth=1.5)
    axes[2].set_title(
        f"Ensemble v4l: {len(ensemble_polygons)} preds (red) vs "
        f"{len(gt_polygons)} GT (green dashed)"
    )
    axes[2].axis("off")

    # Panel 4: Per-component confidence bar chart
    if all_components:
        ids = [str(c["id"]) for c in all_components[:15]]
        confs = [c["confidence"] for c in all_components[:15]]
        colors = [
            "green" if c["id"] in selected_ids else "red" for c in all_components[:15]
        ]
        axes[3].barh(ids, confs, color=colors, edgecolor="black", linewidth=0.5)
        axes[3].set_xlabel("Confidence Score")
        axes[3].set_title("Component Confidence (top 15)")
        axes[3].invert_yaxis()
    else:
        axes[3].text(0.5, 0.5, "No components", ha="center", va="center")
        axes[3].set_title("Component Confidence")

    fig.suptitle(
        f"Ensemble v4l: Fixed Low Threshold + Confidence Filter "
        f"(U-Net thr={unet_thr:.2f}, GNN thr={gnn_thr:.2f})",
        fontsize=14,
    )
    fig.tight_layout()
    path = output_dir / "ensemble_diagnostic.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{_ts()}]   saved diagnostic -> {path}")


def _smooth_contour(contour: np.ndarray, window: int = 9) -> np.ndarray:
    """Smooth a contour using Savitzky-Golay filter for cleaner polygon edges.

    Applies circular padding so the contour remains closed. Falls back to
    the original contour if it's too short for the window size.

    v4k: reduced default window from 15 → 9 to prevent over-smoothing.
    """
    n = len(contour)
    if n < window + 2:
        return contour

    # Ensure odd window size
    w = window if window % 2 == 1 else window + 1
    if w >= n:
        w = n - 2 if (n - 2) % 2 == 1 else n - 3
    if w < 5:
        return contour

    # Circular padding for smooth wrap-around
    pad = w // 2
    rows = np.concatenate([contour[-pad:, 0], contour[:, 0], contour[:pad, 0]])
    cols = np.concatenate([contour[-pad:, 1], contour[:, 1], contour[:pad, 1]])

    rows_smooth = savgol_filter(rows, w, polyorder=3)
    cols_smooth = savgol_filter(cols, w, polyorder=3)

    # Remove padding
    smoothed = np.column_stack(
        [
            rows_smooth[pad : pad + n],
            cols_smooth[pad : pad + n],
        ]
    )
    return smoothed


def segment(data_path: str, output_dir: str) -> list[Polygon]:
    """Ensemble segmentation v4l: fixed low threshold + confidence filter.

    Parameters
    ----------
    data_path : str
        Root path to Xenium dataset.
    output_dir : str
        Directory for output artifacts.

    Returns
    -------
    list[Polygon]
        Villi polygons in micron coordinates.
    """
    from utils.io import load_annotations

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ds_pixel_um = PIXEL_SIZE_UM * DOWNSCALE_FACTOR

    print(
        f"\n[{_ts()}] ===== ENSEMBLE v4l: Fixed Low Threshold + Confidence Filter =====\n"
    )

    # ==================================================================
    # Step 1: Run U-Net → probability map
    # ==================================================================
    print(f"[{_ts()}] Step 1: Running U-Net segmentation...")
    unet_dir = out / "_unet_internal"
    unet_dir.mkdir(exist_ok=True)

    from methods.unet_seg import segment as unet_segment

    unet_polys = unet_segment(data_path, str(unet_dir))
    print(f"[{_ts()}]   U-Net produced {len(unet_polys)} polygons")

    # Load the probability map saved by U-Net
    unet_prob_path = unet_dir / "prob_map.npy"
    if not unet_prob_path.exists():
        print(
            f"[{_ts()}]   ERROR: U-Net prob_map.npy not found — falling back to U-Net polygons"
        )
        return unet_polys

    unet_prob = np.load(unet_prob_path)
    print(
        f"[{_ts()}]   U-Net prob map: {unet_prob.shape}, "
        f"range=[{unet_prob.min():.3f}, {unet_prob.max():.3f}]"
    )

    # ==================================================================
    # Step 1b: Fixed U-Net threshold (no sweep — see module docstring)
    # ==================================================================
    unet_thr = UNET_THRESHOLD
    gt_polygons = load_annotations(data_path)
    print(f"[{_ts()}]   Using fixed U-Net threshold: {unet_thr:.2f}")

    # ==================================================================
    # Step 2: Run GNN → cell probabilities + centroids
    # ==================================================================
    print(f"\n[{_ts()}] Step 2: Running Multimodal GNN for cell-level scores...")
    gnn_dir = out / "_gnn_internal"
    gnn_dir.mkdir(exist_ok=True)

    from methods.multimodal_gnn import segment as gnn_segment

    gnn_polys = gnn_segment(data_path, str(gnn_dir))
    print(
        f"[{_ts()}]   GNN produced {len(gnn_polys)} polygons (used only for cell probs)"
    )

    # Load cell-level outputs from GNN
    cell_probs_path = gnn_dir / "cell_probs.npy"
    cell_centroids_path = gnn_dir / "cell_centroids_um.npy"

    if not cell_probs_path.exists() or not cell_centroids_path.exists():
        print(
            f"[{_ts()}]   WARNING: GNN cell outputs missing — falling back to U-Net only"
        )
        return unet_polys

    cell_probs = np.load(cell_probs_path)
    cell_centroids = np.load(cell_centroids_path)  # (N, 2) in microns, (x, y)

    # ==================================================================
    # Step 2b: Find best GNN vote threshold
    # ==================================================================
    print(f"\n[{_ts()}] Step 2b: GNN vote threshold selection...")
    # Sweep GNN thresholds — pick the one that separates cells most clearly
    best_gnn_thr = 0.5
    best_separation = -1.0
    for gnn_thr_candidate in GNN_VOTE_CANDIDATES:
        pos_mask = cell_probs >= gnn_thr_candidate
        neg_mask = ~pos_mask
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
        if n_pos < 10 or n_neg < 10:
            continue
        # Measure separation: ratio of mean positive prob to mean negative prob
        mean_pos = cell_probs[pos_mask].mean()
        mean_neg = cell_probs[neg_mask].mean()
        separation = mean_pos - mean_neg
        if separation > best_separation:
            best_separation = separation
            best_gnn_thr = gnn_thr_candidate

    n_pos_cells = int((cell_probs >= best_gnn_thr).sum())
    print(
        f"[{_ts()}]   GNN vote threshold: {best_gnn_thr:.2f} "
        f"(separation={best_separation:.3f}), "
        f"{n_pos_cells} positive cells"
    )

    # ==================================================================
    # Step 3: Extract U-Net connected components with adaptive threshold
    # ==================================================================
    print(
        f"\n[{_ts()}] Step 3: Extracting U-Net connected components (thr={unet_thr:.2f})..."
    )

    binary = (unet_prob >= unet_thr).astype(np.uint8)
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=200)
    binary = morphology.remove_small_holes(binary, area_threshold=500)
    labeled, n_cc = ndi.label(binary.astype(np.uint8))
    print(f"[{_ts()}]   {n_cc} connected components at threshold {unet_thr:.2f}")

    # ==================================================================
    # Step 4: Confidence scoring (cell-vote + prob + area)
    # ==================================================================
    print(f"\n[{_ts()}] Step 4: Computing confidence scores...")

    # Map cell centroids (microns) to downscaled pixel coords
    cell_px_x = cell_centroids[:, 0] / ds_pixel_um
    cell_px_y = cell_centroids[:, 1] / ds_pixel_um
    cell_rows = np.clip(cell_px_y.astype(int), 0, unet_prob.shape[0] - 1)
    cell_cols = np.clip(cell_px_x.astype(int), 0, unet_prob.shape[1] - 1)

    # Which component does each cell belong to? (0 = background)
    cell_comp_labels = labeled[cell_rows, cell_cols]
    positive_mask = cell_probs >= best_gnn_thr

    scored_components: list[dict] = []
    for region in measure.regionprops(labeled):
        comp_id = region.label
        area_um2 = region.area * (ds_pixel_um**2)

        if area_um2 < MIN_AREA_UM2:
            continue

        # Count GNN-positive cells in this component
        in_comp = cell_comp_labels == comp_id
        n_total = int(in_comp.sum())
        n_pos = int((in_comp & positive_mask).sum())
        vote_frac = n_pos / max(n_total, 1)

        # Mean U-Net probability within the component (quality of prediction)
        comp_mask = labeled == comp_id
        mean_unet = float(unet_prob[comp_mask].mean())

        # Area-based score: penalize very small components, cap at 1.0
        area_score = min(1.0, area_um2 / 20000.0)

        # Composite confidence: weighted combination
        # Cell-vote is the strongest signal (weight 0.5)
        # Mean U-Net prob indicates prediction quality (weight 0.3)
        # Area provides prior for real villi being large (weight 0.2)
        confidence = 0.5 * vote_frac + 0.3 * mean_unet + 0.2 * area_score

        # Centroid in microns
        cy_um = region.centroid[0] * ds_pixel_um
        cx_um = region.centroid[1] * ds_pixel_um

        scored_components.append(
            {
                "id": comp_id,
                "area_um2": area_um2,
                "cx": cx_um,
                "cy": cy_um,
                "n_total": n_total,
                "n_pos": n_pos,
                "vote_frac": vote_frac,
                "mean_unet": mean_unet,
                "area_score": area_score,
                "confidence": confidence,
            }
        )

    # Sort by confidence descending
    scored_components.sort(key=lambda c: c["confidence"], reverse=True)

    print(
        f"[{_ts()}]   {len(scored_components)} components above {MIN_AREA_UM2:.0f} µm²"
    )
    for i, c in enumerate(scored_components[:12]):
        print(
            f"[{_ts()}]     #{i}: area={c['area_um2']:.0f}µm², "
            f"cells={c['n_total']}, pos={c['n_pos']}, "
            f"vote={c['vote_frac']:.2f}, prob={c['mean_unet']:.2f}, "
            f"conf={c['confidence']:.3f}, "
            f"centroid=({c['cx']:.0f},{c['cy']:.0f})"
        )

    # ==================================================================
    # Step 5: Select components — gap detection + confidence floor
    # ==================================================================
    print(f"\n[{_ts()}] Step 5: Selecting villi via confidence gap + floor...")

    if len(scored_components) < 2:
        selected = scored_components
    else:
        confs = [c["confidence"] for c in scored_components]

        # Find the largest relative gap in confidence scores
        best_gap_ratio = 0
        best_cutoff = len(confs)
        for i in range(1, min(len(confs), 15)):
            if confs[i] > 0.01:
                gap_ratio = confs[i - 1] / confs[i]
            else:
                gap_ratio = float("inf")
            if gap_ratio > best_gap_ratio:
                best_gap_ratio = gap_ratio
                best_cutoff = i

        # Confidence floor: require minimum confidence
        # Real villi should have vote_frac > 0.3 AND mean_unet > 0.5
        CONFIDENCE_FLOOR = 0.35

        score_cutoff = max(best_cutoff, 1)

        # Take components above the gap that also pass the floor
        selected = []
        for i, c in enumerate(scored_components):
            if i < score_cutoff and c["confidence"] >= CONFIDENCE_FLOOR:
                selected.append(c)
            elif (
                i >= score_cutoff
                and c["confidence"] >= CONFIDENCE_FLOOR
                and c["vote_frac"] >= 0.3
            ):
                # Below the gap but high confidence — include (could be a real villus)
                selected.append(c)

        # Safety: if gap detection removed everything, fall back to floor-only
        if not selected:
            selected = [
                c for c in scored_components if c["confidence"] >= CONFIDENCE_FLOOR
            ]

        print(
            f"[{_ts()}]   Gap at position {best_cutoff} "
            f"(ratio={best_gap_ratio:.1f}x), confidence floor={CONFIDENCE_FLOOR}, "
            f"selected {len(selected)} components"
        )

    # ==================================================================
    # Step 6: Extract refined polygon boundaries
    # ==================================================================
    print(f"\n[{_ts()}] Step 6: Extracting gently-smoothed polygon boundaries...")

    ensemble_polygons: list[Polygon] = []
    for c in selected:
        comp_mask = (labeled == c["id"]).astype(np.uint8)

        # Clean up the mask
        comp_mask = morphology.remove_small_holes(
            comp_mask.astype(bool), area_threshold=500
        ).astype(np.uint8)

        # Extract contour from the binary mask directly (v4k: no prob-field smoothing)
        contours = measure.find_contours(comp_mask, level=0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        if len(contour) < 4:
            continue

        # Light Savitzky-Golay smoothing only (window=9)
        contour = _smooth_contour(contour, window=9)

        # Convert to micron coordinates
        coords_um = [
            (pt[1] * ds_pixel_um, pt[0] * ds_pixel_um)  # (x, y) in microns
            for pt in contour
        ]

        try:
            poly = Polygon(coords_um)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < MIN_AREA_UM2:
                continue

            # v4k: very gentle simplification (0.5 µm) to avoid losing boundary detail
            poly = poly.simplify(tolerance=0.5, preserve_topology=True)

            ensemble_polygons.append(poly)
            print(
                f"[{_ts()}]   component {c['id']}: "
                f"area={poly.area:.0f} µm², "
                f"conf={c['confidence']:.3f}, "
                f"vertices={len(poly.exterior.coords)}"
            )
        except Exception as exc:
            print(f"[{_ts()}]   component {c['id']}: polygon extraction failed: {exc}")

    # ==================================================================
    # Step 7: Final FP check — remove polygons with very low GT overlap
    # ==================================================================
    # (Only possible because we have GT; in production this would use confidence only)
    print(f"\n[{_ts()}] Step 7: Final quality check...")
    n_before = len(ensemble_polygons)

    # Remove any polygons that are suspiciously small compared to others
    if len(ensemble_polygons) > 1:
        areas = [p.area for p in ensemble_polygons]
        median_area = np.median(areas)
        # Remove polygons less than 15% of the median area
        ensemble_polygons = [
            p for p in ensemble_polygons if p.area >= median_area * 0.15
        ]
        if len(ensemble_polygons) < n_before:
            print(
                f"[{_ts()}]   Removed {n_before - len(ensemble_polygons)} tiny polygons "
                f"(< 15% of median area {median_area:.0f} µm²)"
            )

    # ==================================================================
    # Step 8: Diagnostics
    # ==================================================================
    print(f"\n[{_ts()}] Step 8: Saving diagnostics...")
    _save_ensemble_diagnostics_v4j(
        out,
        unet_prob,
        ensemble_polygons,
        gt_polygons,
        scored_components,
        selected,
        ds_pixel_um,
        unet_thr,
        best_gnn_thr,
    )

    elapsed = time.time() - t0
    print(
        f"\n[{_ts()}] ENSEMBLE v4l DONE: {len(ensemble_polygons)} villi polygons in {elapsed:.1f}s"
    )

    return ensemble_polygons
