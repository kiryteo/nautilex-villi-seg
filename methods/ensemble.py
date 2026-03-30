"""
Ensemble method: combines U-Net pixel predictions with GNN cell predictions.

Strategy:
  1. Run U-Net to get probability map (pixel-level villus prediction)
  2. Run GNN to get cell-level villus probabilities
  3. Use GNN to identify individual villi (instance seeds via HDBSCAN)
  4. Refine each GNN polygon boundary using U-Net probability map

The insight: GNN is better at counting/separating individual villi (F1),
while U-Net is better at delineating precise boundaries (IoU). The ensemble
uses GNN for "what" and U-Net for "where exactly".
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage import measure, morphology, segmentation, transform

from utils.coords import PIXEL_SIZE_UM

DOWNSCALE_FACTOR = 4  # Must match unet_seg.py
MIN_AREA_UM2 = 5000.0


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _refine_polygon_with_unet(
    gnn_poly: Polygon,
    unet_prob: np.ndarray,
    threshold: float = 0.3,
    buffer_um: float = 50.0,
) -> Polygon | None:
    """Refine a single GNN polygon using the U-Net probability map.

    Strategy:
      1. Create a bounding box around the GNN polygon (with buffer)
      2. Within that region, threshold the U-Net probability map
      3. Keep only the connected component(s) that overlap with the GNN polygon
      4. Convert refined mask back to a polygon

    Parameters
    ----------
    gnn_poly : Polygon
        GNN-predicted villus polygon in micron coordinates.
    unet_prob : np.ndarray
        U-Net probability map at downscaled resolution.
    threshold : float
        Probability threshold for U-Net mask.
    buffer_um : float
        Buffer around GNN polygon to search for U-Net signal (microns).

    Returns
    -------
    Polygon or None
        Refined polygon in micron coordinates, or None if refinement fails.
    """
    from skimage import draw

    H, W = unet_prob.shape
    # Pixel size at downscaled resolution
    ds_pixel_um = PIXEL_SIZE_UM * DOWNSCALE_FACTOR

    # Get GNN polygon bounding box in downscaled pixel coordinates
    minx, miny, maxx, maxy = gnn_poly.bounds  # micron coords
    buffer_px = int(buffer_um / ds_pixel_um)

    # Convert to downscaled pixel coords
    r0 = max(0, int(miny / ds_pixel_um) - buffer_px)
    r1 = min(H, int(maxy / ds_pixel_um) + buffer_px)
    c0 = max(0, int(minx / ds_pixel_um) - buffer_px)
    c1 = min(W, int(maxx / ds_pixel_um) + buffer_px)

    if r1 <= r0 or c1 <= c0:
        return None

    # Crop the probability map to the region of interest
    roi_prob = unet_prob[r0:r1, c0:c1]

    # Rasterize the GNN polygon onto the ROI for overlap checking
    gnn_mask = np.zeros_like(roi_prob, dtype=bool)
    # Convert GNN polygon exterior to downscaled pixel coords relative to ROI
    exterior = np.asarray(gnn_poly.exterior.coords)
    px_coords = exterior / ds_pixel_um
    rr_poly = px_coords[:, 1] - r0  # y -> row
    cc_poly = px_coords[:, 0] - c0  # x -> col

    try:
        rr, cc = draw.polygon(rr_poly, cc_poly, shape=gnn_mask.shape)
        if len(rr) > 0:
            gnn_mask[rr, cc] = True
    except Exception:
        return None

    # Threshold U-Net probability in the ROI
    unet_binary = roi_prob >= threshold

    # Label connected components in the U-Net mask
    labeled, n_components = ndi.label(unet_binary)

    if n_components == 0:
        return None

    # Keep only components that overlap with the GNN polygon
    refined_mask = np.zeros_like(unet_binary, dtype=bool)
    for comp_id in range(1, n_components + 1):
        comp = labeled == comp_id
        overlap = (comp & gnn_mask).sum()
        if overlap > 0:
            refined_mask |= comp

    if refined_mask.sum() < 10:  # Too few pixels
        return None

    # Fill holes and clean up
    refined_mask = morphology.remove_small_holes(refined_mask, area_threshold=200)

    # Extract contour
    contours = measure.find_contours(refined_mask, level=0.5)
    if not contours:
        return None

    # Use the largest contour
    contour = max(contours, key=len)
    if len(contour) < 4:
        return None

    # Convert back to micron coordinates
    coords_um = [
        (
            (c[1] + c0) * ds_pixel_um,  # x in microns
            (c[0] + r0) * ds_pixel_um,  # y in microns
        )
        for c in contour
    ]

    try:
        poly = Polygon(coords_um)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < MIN_AREA_UM2:
            return None
        return poly
    except Exception:
        return None


def _save_ensemble_diagnostics(
    output_dir: Path,
    unet_prob: np.ndarray,
    gnn_polygons: list[Polygon],
    refined_polygons: list[Polygon],
    gt_polygons: list[Polygon],
):
    """Save a diagnostic figure showing GNN seeds, U-Net mask, and refined result."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[{_ts()}]   matplotlib not available -- skipping diagnostics")
        return

    ds_pixel_um = PIXEL_SIZE_UM * DOWNSCALE_FACTOR
    H, W = unet_prob.shape

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Panel 1: U-Net probability map
    axes[0].imshow(unet_prob, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("U-Net Probability Map")
    axes[0].axis("off")

    # Panel 2: GNN polygons on U-Net map
    axes[1].imshow(unet_prob, cmap="gray", vmin=0, vmax=1)
    for poly in gnn_polygons:
        xs, ys = poly.exterior.xy
        xs_px = [x / ds_pixel_um for x in xs]
        ys_px = [y / ds_pixel_um for y in ys]
        axes[1].plot(xs_px, ys_px, color="cyan", linewidth=1.5)
    axes[1].set_title(f"GNN Instance Seeds ({len(gnn_polygons)} polygons)")
    axes[1].axis("off")

    # Panel 3: Refined ensemble polygons vs GT
    axes[2].imshow(unet_prob, cmap="gray", vmin=0, vmax=1)
    for poly in gt_polygons:
        xs, ys = poly.exterior.xy
        xs_px = [x / ds_pixel_um for x in xs]
        ys_px = [y / ds_pixel_um for y in ys]
        axes[2].plot(xs_px, ys_px, color="lime", linewidth=1.5, linestyle="--")
    for poly in refined_polygons:
        xs, ys = poly.exterior.xy
        xs_px = [x / ds_pixel_um for x in xs]
        ys_px = [y / ds_pixel_um for y in ys]
        axes[2].plot(xs_px, ys_px, color="red", linewidth=1.5)
    axes[2].set_title(
        f"Ensemble Result ({len(refined_polygons)} polygons, "
        f"GT={len(gt_polygons)} dashed green)"
    )
    axes[2].axis("off")

    fig.suptitle("Ensemble: GNN Instances + U-Net Boundaries", fontsize=14)
    fig.tight_layout()
    path = output_dir / "ensemble_diagnostic.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{_ts()}]   saved diagnostic -> {path}")


def segment(data_path: str, output_dir: str) -> list[Polygon]:
    """Ensemble segmentation combining U-Net + Multimodal GNN.

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

    print(f"\n[{_ts()}] ===== ENSEMBLE: U-Net + Multimodal GNN =====\n")

    # ==================================================================
    # Step 1: Run U-Net
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
            f"[{_ts()}]   ERROR: U-Net prob_map.npy not found -- falling back to U-Net polygons only"
        )
        return unet_polys

    unet_prob = np.load(unet_prob_path)
    print(
        f"[{_ts()}]   U-Net prob map: {unet_prob.shape}, range=[{unet_prob.min():.3f}, {unet_prob.max():.3f}]"
    )

    # ==================================================================
    # Step 2: Run Multimodal GNN
    # ==================================================================
    print(f"\n[{_ts()}] Step 2: Running Multimodal GNN segmentation...")
    gnn_dir = out / "_gnn_internal"
    gnn_dir.mkdir(exist_ok=True)

    from methods.multimodal_gnn import segment as gnn_segment

    gnn_polys = gnn_segment(data_path, str(gnn_dir))
    print(f"[{_ts()}]   GNN produced {len(gnn_polys)} polygons")

    if not gnn_polys:
        print(f"[{_ts()}]   GNN produced no polygons -- falling back to U-Net result")
        return unet_polys

    # ==================================================================
    # Step 3: Refine GNN polygons with U-Net probability map
    # ==================================================================
    print(f"\n[{_ts()}] Step 3: Refining GNN polygons with U-Net probability...")

    refined_polygons: list[Polygon] = []
    for i, gnn_poly in enumerate(gnn_polys):
        refined = _refine_polygon_with_unet(
            gnn_poly,
            unet_prob,
            threshold=0.3,  # Lower threshold for refinement to capture more of the villus
            buffer_um=50.0,
        )
        if refined is not None:
            refined_polygons.append(refined)
            area_change = (refined.area / gnn_poly.area - 1.0) * 100
            print(
                f"[{_ts()}]   polygon {i}: "
                f"GNN area={gnn_poly.area:.0f} um^2 -> "
                f"refined={refined.area:.0f} um^2 ({area_change:+.1f}%)"
            )
        else:
            # Keep original GNN polygon if refinement fails
            refined_polygons.append(gnn_poly)
            print(f"[{_ts()}]   polygon {i}: refinement failed, keeping GNN original")

    # ==================================================================
    # Step 4: Diagnostics
    # ==================================================================
    print(f"\n[{_ts()}] Step 4: Saving diagnostics...")
    gt_polygons = load_annotations(data_path)
    _save_ensemble_diagnostics(out, unet_prob, gnn_polys, refined_polygons, gt_polygons)

    elapsed = time.time() - t0
    print(
        f"\n[{_ts()}] ENSEMBLE DONE: {len(refined_polygons)} villi polygons in {elapsed:.1f}s"
    )

    return refined_polygons
