"""
Ensemble method: combines U-Net pixel predictions with GNN cell predictions.

Strategy (v4i — U-Net primary + GNN cell-vote filtering):
  1. Run U-Net to get probability map (pixel-level villus prediction)
  2. Run GNN to get cell-level villus probabilities + centroids
  3. Extract connected components from U-Net probability map
  4. For each component, count GNN-positive cells inside → "cell vote" score
  5. Keep components with high cell-vote scores (real villi), discard FPs
  6. Extract polygon boundaries from kept components

The insight: U-Net finds ALL villi but over-detects (~20 components), while
GNN cell-level probabilities accurately distinguish real villi from tissue
noise. Using GNN as a spatial filter (not for instance segmentation) is the
key to high precision without losing recall.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage import measure, morphology

from utils.coords import PIXEL_SIZE_UM

DOWNSCALE_FACTOR = 4  # Must match unet_seg.py
MIN_AREA_UM2 = 5000.0


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _save_ensemble_diagnostics_v4i(
    output_dir: Path,
    unet_prob: np.ndarray,
    ensemble_polygons: list[Polygon],
    gt_polygons: list[Polygon],
    all_components: list[dict],
    selected_components: list[dict],
    ds_pixel_um: float,
):
    """Save diagnostic figure for v4i cell-vote ensemble."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[{_ts()}]   matplotlib not available -- skipping diagnostics")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Panel 1: U-Net probability map with all components
    axes[0].imshow(unet_prob, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title(
        f"U-Net Prob Map ({len(all_components)} components > {MIN_AREA_UM2:.0f}µm²)"
    )
    axes[0].axis("off")

    # Panel 2: Cell-vote scores — selected vs rejected
    axes[1].imshow(unet_prob, cmap="gray", vmin=0, vmax=1)
    selected_ids = {c["id"] for c in selected_components}
    for c in all_components:
        color = "lime" if c["id"] in selected_ids else "red"
        # Draw centroid marker
        cx_px = c["cx"] / ds_pixel_um
        cy_px = c["cy"] / ds_pixel_um
        axes[1].plot(cx_px, cy_px, "o", color=color, markersize=6)
        axes[1].annotate(
            f"{c['n_pos']}",
            (cx_px, cy_px),
            fontsize=7,
            color=color,
            ha="center",
            va="bottom",
        )
    axes[1].set_title(
        f"Cell-Vote Filter: {len(selected_components)} kept (green) / "
        f"{len(all_components) - len(selected_components)} rejected (red)"
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
        f"Ensemble v4i: {len(ensemble_polygons)} preds (red) vs "
        f"{len(gt_polygons)} GT (green dashed)"
    )
    axes[2].axis("off")

    fig.suptitle("Ensemble v4i: U-Net Primary + GNN Cell-Vote Filter", fontsize=14)
    fig.tight_layout()
    path = output_dir / "ensemble_diagnostic.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{_ts()}]   saved diagnostic -> {path}")


def segment(data_path: str, output_dir: str) -> list[Polygon]:
    """Ensemble segmentation: U-Net primary + GNN cell-vote filtering.

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

    print(f"\n[{_ts()}] ===== ENSEMBLE v4i: U-Net Primary + GNN Cell-Vote =====\n")

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
    n_pos_cells = int((cell_probs >= 0.5).sum())
    print(
        f"[{_ts()}]   GNN cell probs: {len(cell_probs)} cells, "
        f"{n_pos_cells} positive (>= 0.5)"
    )

    # ==================================================================
    # Step 3: Extract U-Net connected components (NO watershed)
    # ==================================================================
    print(f"\n[{_ts()}] Step 3: Extracting U-Net connected components...")

    thresh = 0.5
    binary = (unet_prob >= thresh).astype(np.uint8)
    binary = morphology.remove_small_objects(binary.astype(bool), min_size=200)
    binary = morphology.remove_small_holes(binary, area_threshold=500)
    labeled, n_cc = ndi.label(binary.astype(np.uint8))
    print(f"[{_ts()}]   {n_cc} connected components at threshold {thresh}")

    # ==================================================================
    # Step 4: Score each component by GNN cell-vote
    # ==================================================================
    print(f"\n[{_ts()}] Step 4: Scoring components by GNN cell-vote...")

    # Map cell centroids (microns) to downscaled pixel coords
    cell_px_x = cell_centroids[:, 0] / ds_pixel_um
    cell_px_y = cell_centroids[:, 1] / ds_pixel_um
    cell_rows = np.clip(cell_px_y.astype(int), 0, unet_prob.shape[0] - 1)
    cell_cols = np.clip(cell_px_x.astype(int), 0, unet_prob.shape[1] - 1)

    # Which component does each cell belong to? (0 = background)
    cell_comp_labels = labeled[cell_rows, cell_cols]
    positive_mask = cell_probs >= 0.5

    # Score each component
    scored_components: list[dict] = []
    for region in measure.regionprops(labeled):
        comp_id = region.label
        area_um2 = region.area * (ds_pixel_um**2)

        if area_um2 < MIN_AREA_UM2:
            continue

        # Count GNN-positive cells in this component
        in_comp = cell_comp_labels == comp_id
        n_pos = int((in_comp & positive_mask).sum())

        # Mean U-Net probability within the component
        comp_mask = labeled == comp_id
        mean_unet = float(unet_prob[comp_mask].mean())

        # Centroid in microns
        cy_um = region.centroid[0] * ds_pixel_um
        cx_um = region.centroid[1] * ds_pixel_um

        scored_components.append(
            {
                "id": comp_id,
                "area_um2": area_um2,
                "cx": cx_um,
                "cy": cy_um,
                "n_pos": n_pos,
                "mean_unet": mean_unet,
                "score": n_pos * mean_unet,
            }
        )

    # Sort by score descending
    scored_components.sort(key=lambda c: c["score"], reverse=True)

    print(
        f"[{_ts()}]   {len(scored_components)} components above {MIN_AREA_UM2:.0f} µm²"
    )
    for i, c in enumerate(scored_components[:10]):
        print(
            f"[{_ts()}]     #{i}: area={c['area_um2']:.0f}µm², "
            f"pos_cells={c['n_pos']}, score={c['score']:.1f}, "
            f"centroid=({c['cx']:.0f},{c['cy']:.0f})"
        )

    # ==================================================================
    # Step 5: Select components — natural gap detection
    # ==================================================================
    print(f"\n[{_ts()}] Step 5: Selecting villi via score gap detection...")

    if len(scored_components) < 2:
        selected = scored_components
    else:
        scores = [c["score"] for c in scored_components]

        # Find the largest relative gap in the sorted scores
        # (drop in score from one component to the next)
        best_gap_ratio = 0
        best_cutoff = len(scores)
        for i in range(1, min(len(scores), 15)):
            if scores[i] > 0:
                gap_ratio = scores[i - 1] / scores[i]
            else:
                gap_ratio = float("inf")
            if gap_ratio > best_gap_ratio:
                best_gap_ratio = gap_ratio
                best_cutoff = i

        # Also apply a minimum score threshold (at least 50 positive cells)
        min_score = 50 * 0.5  # n_pos * mean_unet (conservative)
        score_cutoff = max(best_cutoff, 1)

        # Take components above the gap
        selected = scored_components[:score_cutoff]

        # Additionally include any remaining components above min_score
        for c in scored_components[score_cutoff:]:
            if c["score"] >= min_score and c["n_pos"] >= 50:
                selected.append(c)

        print(
            f"[{_ts()}]   Gap detected at position {best_cutoff} "
            f"(ratio={best_gap_ratio:.1f}x), selected {len(selected)} components"
        )

    # ==================================================================
    # Step 6: Extract polygon boundaries from selected components
    # ==================================================================
    print(f"\n[{_ts()}] Step 6: Extracting polygon boundaries...")

    ensemble_polygons: list[Polygon] = []
    for c in selected:
        comp_mask = (labeled == c["id"]).astype(np.uint8)

        # Clean up the mask
        comp_mask = morphology.remove_small_holes(
            comp_mask.astype(bool), area_threshold=500
        ).astype(np.uint8)

        # Extract contour
        contours = measure.find_contours(comp_mask, level=0.5)
        if not contours:
            continue

        # Use the largest contour
        contour = max(contours, key=len)
        if len(contour) < 4:
            continue

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
            ensemble_polygons.append(poly)
            print(
                f"[{_ts()}]   component {c['id']}: "
                f"area={poly.area:.0f} µm², "
                f"pos_cells={c['n_pos']}"
            )
        except Exception as exc:
            print(f"[{_ts()}]   component {c['id']}: polygon extraction failed: {exc}")

    # ==================================================================
    # Step 7: Diagnostics
    # ==================================================================
    print(f"\n[{_ts()}] Step 7: Saving diagnostics...")
    gt_polygons = load_annotations(data_path)
    _save_ensemble_diagnostics_v4i(
        out,
        unet_prob,
        ensemble_polygons,
        gt_polygons,
        scored_components,
        selected,
        ds_pixel_um,
    )

    elapsed = time.time() - t0
    print(
        f"\n[{_ts()}] ENSEMBLE DONE: {len(ensemble_polygons)} villi polygons in {elapsed:.1f}s"
    )

    return ensemble_polygons
