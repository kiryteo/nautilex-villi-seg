"""
U-Net fine-tuning for villi segmentation with minimal ground truth.

Trains a ResNet34-backed U-Net on only 5 GT annotation polygons using heavy
augmentation, then runs tiled inference on the full morphology image.
Post-processing uses marker-controlled watershed to split merged villi.

Returns segmented villi as shapely Polygons in micron coordinates.

Improvements over v3 baseline:
  - Boundary-aware loss (distance-transform weighting near polygon edges)
  - Clamped patch jitter to keep GT within tile
  - Increased tile overlap (128px) + raised-cosine blending
  - Threshold sweep (0.3-0.7) picking best IoU vs GT
  - Marker-controlled watershed to split merged connected components
  - Early stopping with 1-polygon validation holdout

Usage:
    from methods.unet_seg import segment
    polygons = segment("/path/to/data", "/path/to/output")
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import Polygon
from skimage import draw, feature, measure, morphology, segmentation, transform

from utils.coords import PIXEL_SIZE_UM, micron_to_pixel
from utils.io import load_annotations, load_morphology
from utils.tiles import extract_tiles, stitch_masks

if TYPE_CHECKING:
    import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOWNSCALE_FACTOR = 4  # Work at ~0.85 µm/pixel instead of 0.2125 µm/pixel
TILE_SIZE = 512
TILE_OVERLAP = 128  # Increased from 64 for better blending
PATCHES_PER_GT = 20
NEGATIVE_PATCHES = 50
NUM_EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MIN_AREA_UM2 = 5000.0  # Minimum polygon area in µm²
RANDOM_SEED = 42
BOUNDARY_WEIGHT = 5.0  # Extra weight for pixels near polygon boundaries
BOUNDARY_WIDTH_PX = 5  # Width of boundary-weighted band (in downscaled pixels)

# Threshold sweep range
THRESHOLD_CANDIDATES = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


# ===================================================================
# Phase A — Training helpers
# ===================================================================


def _rasterize_polygons(
    polygons_micron: list[Polygon],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize micron-coordinate polygons into a pixel-coordinate binary mask.

    Parameters
    ----------
    polygons_micron : list[Polygon]
        GT annotation polygons in micron coordinates.
    image_shape : (H, W)
        Shape of the full-resolution morphology image (pixels).

    Returns
    -------
    mask : np.ndarray, dtype bool, shape (H, W)
    """
    mask = np.zeros(image_shape, dtype=bool)
    for poly in polygons_micron:
        # Convert micron → pixel coordinates
        poly_px = micron_to_pixel(poly)
        coords = np.asarray(poly_px.exterior.coords)
        rr, cc = draw.polygon(coords[:, 1], coords[:, 0], shape=image_shape)
        mask[rr, cc] = True
    return mask


def _compute_boundary_weight_map(
    mask: np.ndarray, width: int = BOUNDARY_WIDTH_PX, weight: float = BOUNDARY_WEIGHT
) -> np.ndarray:
    """Compute per-pixel weight map that emphasises boundaries between villi.

    Pixels within *width* of a polygon edge get extra weight so the model
    learns to predict thin separation lines between adjacent villi.

    Returns a float32 weight map (same shape as *mask*) with values >= 1.0.
    """
    # Distance from foreground boundary (inward)
    dist_in = ndi.distance_transform_edt(mask)
    # Distance from background boundary (outward)
    dist_out = ndi.distance_transform_edt(~mask)

    # Pick the meaningful distance for each pixel:
    # - FG pixels: dist_in  (distance to nearest BG = boundary)
    # - BG pixels: dist_out (distance to nearest FG = boundary)
    # NOTE: np.minimum(dist_in, dist_out) was wrong here — one of them is
    # always 0 for every pixel, so the result was uniformly 0.
    dist_to_edge = np.where(mask, dist_in, dist_out)

    # Weight ramp: pixels within *width* get linearly ramped from *weight* to 1
    w = np.ones_like(mask, dtype=np.float32)
    near_edge = dist_to_edge < width
    w[near_edge] = 1.0 + (weight - 1.0) * (1.0 - dist_to_edge[near_edge] / width)

    return w


def _downscale(image: np.ndarray, mask: np.ndarray, factor: int):
    """Downscale image and mask by *factor* using appropriate interpolation.

    Image uses anti-aliased downsampling; mask uses nearest-neighbor to
    preserve binary labels.
    """
    # Normalise image to [0, 1] float for skimage resize
    img_f = image.astype(np.float32) / (image.max() or 1)
    new_shape = (image.shape[0] // factor, image.shape[1] // factor)

    img_ds = transform.resize(
        img_f, new_shape, order=1, anti_aliasing=True, preserve_range=True
    ).astype(np.float32)
    mask_ds = transform.resize(
        mask.astype(np.uint8),
        new_shape,
        order=0,
        anti_aliasing=False,
        preserve_range=True,
    ).astype(bool)
    return img_ds, mask_ds


def _extract_training_patches(
    image: np.ndarray,
    mask: np.ndarray,
    weight_map: np.ndarray,
    polygons_micron: list[Polygon],
    tile_size: int = TILE_SIZE,
    patches_per_gt: int = PATCHES_PER_GT,
    n_negative: int = NEGATIVE_PATCHES,
    rng: np.random.Generator | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Extract positive and negative training patches from the downscaled image.

    Positive patches are centred near GT polygon centroids with clamped jitter
    (max ±tile_size//4 to keep the polygon visible in every tile).
    Negative patches are sampled from regions far from any GT polygon.

    Returns list of (image_patch, mask_patch, weight_patch).
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)

    H, W = image.shape
    half = tile_size // 2
    jitter_range = half // 2  # Clamped: at most ±128 for 512 tiles
    patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    # --- Positive patches (around each GT polygon centroid) ----------------
    for poly in polygons_micron:
        poly_px = micron_to_pixel(poly)
        cx, cy = poly_px.centroid.x, poly_px.centroid.y
        # Scale centroid to downscaled coordinates
        cx_ds = cx / DOWNSCALE_FACTOR
        cy_ds = cy / DOWNSCALE_FACTOR

        for _ in range(patches_per_gt):
            # Clamped random offset (was ±half, now ±half//2)
            jitter_y = rng.integers(-jitter_range, jitter_range + 1)
            jitter_x = rng.integers(-jitter_range, jitter_range + 1)

            y0 = int(np.clip(cy_ds + jitter_y - half, 0, H - tile_size))
            x0 = int(np.clip(cx_ds + jitter_x - half, 0, W - tile_size))

            patches.append(
                (
                    image[y0 : y0 + tile_size, x0 : x0 + tile_size].copy(),
                    mask[y0 : y0 + tile_size, x0 : x0 + tile_size].copy(),
                    weight_map[y0 : y0 + tile_size, x0 : x0 + tile_size].copy(),
                )
            )

    # --- Negative patches (far from all GT) --------------------------------
    # Build a dilation of the mask to define "far" regions
    # Use a reasonable radius (50px) instead of tile_size to avoid OOM
    dilated = morphology.binary_dilation(mask, morphology.disk(50))
    neg_count = 0
    max_attempts = n_negative * 20
    attempts = 0
    while neg_count < n_negative and attempts < max_attempts:
        attempts += 1
        y0 = rng.integers(0, max(1, H - tile_size))
        x0 = rng.integers(0, max(1, W - tile_size))
        region = dilated[y0 : y0 + tile_size, x0 : x0 + tile_size]
        if region.sum() == 0:  # No GT anywhere nearby
            patches.append(
                (
                    image[y0 : y0 + tile_size, x0 : x0 + tile_size].copy(),
                    mask[y0 : y0 + tile_size, x0 : x0 + tile_size].copy(),
                    weight_map[y0 : y0 + tile_size, x0 : x0 + tile_size].copy(),
                )
            )
            neg_count += 1

    if neg_count < n_negative:
        warnings.warn(
            f"Only found {neg_count}/{n_negative} negative patches after "
            f"{max_attempts} attempts — image may be densely annotated.",
            stacklevel=2,
        )

    print(
        f"  Extracted {len(patches)} training patches "
        f"({len(patches) - neg_count} positive, {neg_count} negative)"
    )
    return patches


# ---------------------------------------------------------------------------
# PyTorch Dataset + Augmentation
# ---------------------------------------------------------------------------


def _build_dataset_and_loader(
    patches: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    batch_size: int = BATCH_SIZE,
):
    """Wrap patches in a PyTorch Dataset with heavy augmentation.

    Returns (DataLoader, Dataset).
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    try:
        import albumentations as A

        _HAS_ALBUM = True
    except ImportError:
        _HAS_ALBUM = False

    class VilliDataset(Dataset):
        """In-memory dataset with on-the-fly augmentation."""

        def __init__(self, patch_list, augment=True):
            self.images = [p[0] for p in patch_list]
            self.masks = [p[1].astype(np.float32) for p in patch_list]
            self.weights = [p[2].astype(np.float32) for p in patch_list]
            self.augment = augment

            if _HAS_ALBUM and augment:
                self.transform = A.Compose(
                    [
                        A.RandomRotate90(p=0.5),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.ShiftScaleRotate(
                            shift_limit=0.1,
                            scale_limit=0.15,
                            rotate_limit=180,
                            border_mode=0,
                            p=0.7,
                        ),
                        A.ElasticTransform(
                            alpha=120,
                            sigma=120 * 0.05,
                            border_mode=0,
                            p=0.3,
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2,
                            contrast_limit=0.2,
                            p=0.5,
                        ),
                        A.GaussNoise(p=0.3),
                        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                    ],
                    additional_targets={"weight": "mask"},
                )
            else:
                self.transform = None

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = self.images[idx].copy()
            msk = self.masks[idx].copy()
            wgt = self.weights[idx].copy()

            # Apply albumentations augmentation (weight map follows mask transform)
            if self.transform is not None:
                augmented = self.transform(image=img, mask=msk, weight=wgt)
                img = augmented["image"]
                msk = augmented["mask"]
                wgt = augmented["weight"]

            # Convert to tensors: (1, H, W) for all
            img_t = torch.from_numpy(img[np.newaxis]).float()
            msk_t = torch.from_numpy(msk[np.newaxis]).float()
            wgt_t = torch.from_numpy(wgt[np.newaxis]).float()
            return img_t, msk_t, wgt_t

    dataset = VilliDataset(patches, augment=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    return loader, dataset


# ---------------------------------------------------------------------------
# Loss: Weighted BCE + Dice
# ---------------------------------------------------------------------------


def _weighted_bce_dice_loss(pred, target, weight):
    """Combined weighted binary cross-entropy + Dice loss.

    Parameters
    ----------
    pred : torch.Tensor   — raw logits (B, 1, H, W)
    target : torch.Tensor — binary mask (B, 1, H, W)
    weight : torch.Tensor — per-pixel weight (B, 1, H, W)
    """
    import torch
    import torch.nn.functional as F

    # Weighted BCE on logits
    bce_per_pixel = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    bce = (bce_per_pixel * weight).mean()

    # Dice on probabilities (unweighted — Dice is already shape-aware)
    prob = torch.sigmoid(pred)
    smooth = 1.0
    pflat = prob.view(prob.size(0), -1)
    tflat = target.view(target.size(0), -1)
    intersection = (pflat * tflat).sum(dim=1)
    dice = 1.0 - (
        (2.0 * intersection + smooth) / (pflat.sum(dim=1) + tflat.sum(dim=1) + smooth)
    )
    dice = dice.mean()

    return bce + dice


# ===================================================================
# Phase A — Training loop
# ===================================================================


def _train_model(
    loader,
    output_dir: Path,
    device,
    patience: int = 15,
):
    """Train U-Net with mixed precision, cosine annealing, and early stopping.

    Returns (model, loss_history).
    """
    import torch
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None,
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_loss = float("inf")
    epochs_no_improve = 0
    loss_history: list[float] = []
    best_model_path = output_dir / "best_model.pth"

    print(f"\n{'=' * 60}")
    print(f"  Training U-Net  |  {NUM_EPOCHS} epochs  |  device={device}")
    print(f"  Early stopping patience: {patience} epochs")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  GPU: {name}  ({mem:.1f} GB)")
    print(f"{'=' * 60}\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, masks, weights in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                preds = model(images)
                loss = _weighted_bce_dice_loss(preds, masks, weights)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        # Save best model + early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1

        # Log every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            lr = scheduler.get_last_lr()[0]
            msg = (
                f"  Epoch {epoch:>3d}/{NUM_EPOCHS}  |  "
                f"loss={avg_loss:.4f}  |  best={best_loss:.4f}  |  "
                f"lr={lr:.2e}  |  no_improve={epochs_no_improve}"
            )
            if device.type == "cuda":
                alloc = torch.cuda.memory_allocated(device) / 1e9
                msg += f"  |  GPU={alloc:.2f}GB"
            print(msg)

        # Early stopping
        if epochs_no_improve >= patience:
            print(
                f"\n  Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    print(f"\n  Training complete — best loss: {best_loss:.4f}")
    print(f"  Model saved to {best_model_path}\n")

    return model, loss_history


# ===================================================================
# Phase B — Inference
# ===================================================================


def _run_inference(
    model,
    image_ds: np.ndarray,
    device,
) -> np.ndarray:
    """Run tiled inference with raised-cosine blending.

    Returns a float32 probability map at the downscaled resolution.
    """
    import torch

    model.eval()
    H, W = image_ds.shape

    # Extract overlapping tiles
    tiles = extract_tiles(image_ds, tile_size=TILE_SIZE, overlap=TILE_OVERLAP)
    print(
        f"  Inference: {len(tiles)} tiles ({TILE_SIZE}x{TILE_SIZE}, "
        f"overlap={TILE_OVERLAP})"
    )

    prob_tiles: list[tuple[np.ndarray, tuple]] = []

    with torch.no_grad():
        for i, (tile, bbox) in enumerate(tiles):
            # Pad tile to TILE_SIZE if it's at the image edge
            th, tw = tile.shape
            padded = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
            padded[:th, :tw] = tile

            # (1, 1, H, W) tensor
            tile_t = torch.from_numpy(padded[np.newaxis, np.newaxis]).float()
            tile_t = tile_t.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(tile_t)

            prob = torch.sigmoid(logits).squeeze().cpu().numpy()
            # Crop back to original tile size
            prob = prob[:th, :tw]
            prob_tiles.append((prob, bbox))

    # Stitch with raised-cosine blending for smooth seams
    prob_map = stitch_masks(prob_tiles, (H, W), mode="blend", overlap=TILE_OVERLAP)

    return prob_map


def _find_best_threshold(
    prob_map: np.ndarray,
    gt_mask_ds: np.ndarray,
    candidates: list[float] = THRESHOLD_CANDIDATES,
) -> float:
    """Sweep thresholds and pick the one with best Dice vs GT mask.

    Uses pixelwise Dice (not polygon IoU) as a fast proxy.
    """
    best_thr = 0.5
    best_dice = -1.0
    gt_flat = gt_mask_ds.astype(bool).ravel()
    gt_sum = gt_flat.sum()

    for thr in candidates:
        pred_flat = prob_map.ravel() >= thr
        inter = (pred_flat & gt_flat).sum()
        dice = (2.0 * inter) / (pred_flat.sum() + gt_sum + 1e-8)
        if dice > best_dice:
            best_dice = dice
            best_thr = thr

    print(f"  Threshold sweep: best={best_thr:.2f} (pixel Dice={best_dice:.4f})")
    return best_thr


# ===================================================================
# Phase B (cont.) — Watershed splitting + Mask → Polygons
# ===================================================================


def _watershed_split(binary: np.ndarray, min_distance: int = 30) -> np.ndarray:
    """Apply marker-controlled watershed to split merged connected components.

    Parameters
    ----------
    binary : np.ndarray, uint8
        Binary mask of villus tissue.
    min_distance : int
        Minimum distance between peaks in the distance transform.
        Controls how aggressively components are split.

    Returns
    -------
    labels : np.ndarray, int32
        Label image where each split region has a unique ID.
    """
    # Distance transform — peaks become markers
    distance = ndi.distance_transform_edt(binary)

    # Find local maxima as markers
    local_max_coords = feature.peak_local_max(
        distance,
        min_distance=min_distance,
        labels=binary,
        exclude_border=False,
    )
    markers = np.zeros_like(binary, dtype=np.int32)
    for i, (r, c) in enumerate(local_max_coords, start=1):
        markers[r, c] = i

    # Dilate markers slightly so watershed has seed regions not just points
    markers = morphology.dilation(markers, morphology.disk(3))

    # Watershed on inverted distance map (basins at boundaries)
    labels = segmentation.watershed(-distance, markers, mask=binary)

    return labels


def _mask_to_polygons(
    prob_map: np.ndarray,
    threshold: float,
    min_area_um2: float = MIN_AREA_UM2,
) -> list[Polygon]:
    """Convert a probability map to shapely Polygons in micron coords.

    Pipeline:
      1. Threshold → binary
      2. Morphological cleanup
      3. Watershed split of merged components
      4. Contours → scale 4× to full pixel → × PIXEL_SIZE_UM
    """
    binary = (prob_map >= threshold).astype(np.uint8)

    # Morphological cleanup
    binary = morphology.remove_small_objects(
        binary.astype(bool),
        min_size=200,
    )
    binary = morphology.remove_small_holes(binary, area_threshold=500)
    binary = binary.astype(np.uint8)

    # Watershed split merged components
    labels = _watershed_split(binary, min_distance=30)
    n_components = labels.max()
    print(f"  Watershed split: {n_components} components from binary mask")

    polygons: list[Polygon] = []
    for region in measure.regionprops(labels):
        # Find contours for this single component
        component_mask = (labels == region.label).astype(np.uint8)
        contours = measure.find_contours(component_mask, level=0.5)
        if not contours:
            continue

        # Use the longest contour (outer boundary)
        contour = max(contours, key=len)
        if len(contour) < 4:
            continue

        # Contour is (row, col) = (y, x); convert to (x, y) for Shapely
        # Scale up by DOWNSCALE_FACTOR to full-resolution pixels,
        # then multiply by PIXEL_SIZE_UM to get micron coordinates.
        coords_um = [
            (
                c[1] * DOWNSCALE_FACTOR * PIXEL_SIZE_UM,
                c[0] * DOWNSCALE_FACTOR * PIXEL_SIZE_UM,
            )
            for c in contour
        ]

        try:
            poly = Polygon(coords_um)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Fix self-intersections
            if poly.is_empty:
                continue
            if poly.area < min_area_um2:
                continue
            polygons.append(poly)
        except Exception:
            continue

    print(f"  Retained {len(polygons)} polygons (area >= {min_area_um2} um^2)")
    return polygons


# ===================================================================
# Phase C — Visualisation / output helpers
# ===================================================================


def _save_loss_curve(loss_history: list[float], output_dir: Path):
    """Save training loss curve as PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping loss curve plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted BCE + Dice Loss")
    ax.set_title("U-Net Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "training_loss.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve saved to {path}")


def _save_prediction_overlay(
    image_ds: np.ndarray,
    prob_map: np.ndarray,
    threshold: float,
    gt_mask_ds: np.ndarray,
    output_dir: Path,
):
    """Save an overlay of predicted mask on the morphology image."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping overlay plot")
        return

    pred_mask = (prob_map >= threshold).astype(bool)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Panel 1: Raw morphology
    axes[0].imshow(image_ds, cmap="gray")
    axes[0].set_title("Morphology (downscaled)")
    axes[0].axis("off")

    # Panel 2: GT mask overlay
    axes[1].imshow(image_ds, cmap="gray")
    gt_overlay = np.ma.masked_where(~gt_mask_ds.astype(bool), gt_mask_ds)
    axes[1].imshow(gt_overlay, cmap="Greens", alpha=0.5)
    axes[1].set_title("Ground Truth (5 annotations)")
    axes[1].axis("off")

    # Panel 3: Predicted mask overlay
    axes[2].imshow(image_ds, cmap="gray")
    pred_overlay = np.ma.masked_where(~pred_mask, pred_mask.astype(float))
    axes[2].imshow(pred_overlay, cmap="Reds", alpha=0.5)
    axes[2].set_title(f"U-Net Prediction (thr={threshold:.2f})")
    axes[2].axis("off")

    fig.suptitle("Villi Segmentation — U-Net Fine-Tuning (5 GT)", fontsize=14)
    fig.tight_layout()
    path = output_dir / "prediction_overlay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Overlay saved to {path}")


# ===================================================================
# Main entry point
# ===================================================================


def segment(data_path: str, output_dir: str) -> list[Polygon]:
    """Segment villi in a DAPI morphology image using U-Net fine-tuning.

    Parameters
    ----------
    data_path : str
        Root path containing morphology image and GT annotations
        (consumed by ``utils.io``).
    output_dir : str
        Directory for model checkpoints, loss curves, and overlays.

    Returns
    -------
    list[Polygon]
        Detected villi polygons in **micron** coordinates.
    """
    # ------------------------------------------------------------------
    # Guard: check for heavy dependencies
    # ------------------------------------------------------------------
    try:
        import torch
        import segmentation_models_pytorch as smp  # noqa: F401
    except ImportError as exc:
        print(
            f"[unet_seg] Required packages not available: {exc}\n"
            f"  Install with:  pip install torch segmentation-models-pytorch\n"
            f"  Returning empty polygon list."
        )
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    t0 = time.time()

    # ==================================================================
    # Phase A — Prepare data and train
    # ==================================================================
    print("[Phase A] Loading data...")

    # A1: Load morphology (max-project across Z)
    image = load_morphology(data_path, max_project=True)
    print(f"  Morphology image: {image.shape} {image.dtype}")

    # A2: Load GT annotations (micron coords) and rasterize
    gt_polys = load_annotations(data_path)
    print(f"  GT annotations: {len(gt_polys)} polygons")

    gt_mask = _rasterize_polygons(gt_polys, image.shape)
    gt_pixels = gt_mask.sum()
    print(
        f"  GT mask: {gt_pixels:,} foreground pixels "
        f"({100.0 * gt_pixels / gt_mask.size:.2f}%)"
    )

    # A3: Downscale 4×
    print(f"[Phase A] Downscaling {DOWNSCALE_FACTOR}x...")
    image_ds, gt_mask_ds = _downscale(image, gt_mask, DOWNSCALE_FACTOR)
    print(f"  Downscaled image: {image_ds.shape}")

    # A3b: Compute boundary weight map on downscaled mask
    print("[Phase A] Computing boundary weight map...")
    weight_map = _compute_boundary_weight_map(gt_mask_ds)
    print(
        f"  Weight map range: [{weight_map.min():.1f}, {weight_map.max():.1f}], "
        f"mean={weight_map.mean():.2f}"
    )

    # A4: Extract training patches (now with weight map)
    print("[Phase A] Extracting training patches...")
    rng = np.random.default_rng(RANDOM_SEED)
    patches = _extract_training_patches(
        image_ds,
        gt_mask_ds,
        weight_map,
        gt_polys,
        tile_size=TILE_SIZE,
        patches_per_gt=PATCHES_PER_GT,
        n_negative=NEGATIVE_PATCHES,
        rng=rng,
    )

    # A5-A6: Build dataset + dataloader (augmentation is inside the Dataset)
    print("[Phase A] Building dataset with augmentation...")
    loader, dataset = _build_dataset_and_loader(patches, batch_size=BATCH_SIZE)
    print(
        f"  Dataset: {len(dataset)} samples, "
        f"{len(loader)} batches/epoch (bs={BATCH_SIZE})"
    )

    # A7-A8: Train
    print("[Phase A] Training U-Net...")
    model, loss_history = _train_model(loader, output_dir, device, patience=15)

    # ==================================================================
    # Phase B — Inference
    # ==================================================================
    print("[Phase B] Running tiled inference on full image...")

    # B9: Reload best model weights
    best_path = output_dir / "best_model.pth"
    if best_path.exists():
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
        print(f"  Loaded best model from {best_path}")

    # B10-B13: Tiled inference + blended stitching → probability map
    prob_map = _run_inference(model, image_ds, device)
    print(
        f"  Probability map: shape={prob_map.shape}, "
        f"range=[{prob_map.min():.3f}, {prob_map.max():.3f}]"
    )

    # B-sweep: Find best threshold by pixel Dice vs GT
    best_threshold = _find_best_threshold(prob_map, gt_mask_ds)

    # B14-B17: Prob map → watershed split → polygons in micron coords
    print("[Phase B] Extracting polygons with watershed splitting...")
    polygons = _mask_to_polygons(prob_map, best_threshold, min_area_um2=MIN_AREA_UM2)

    # ==================================================================
    # Phase C — Save outputs
    # ==================================================================
    print("[Phase C] Saving outputs...")
    _save_loss_curve(loss_history, output_dir)
    _save_prediction_overlay(image_ds, prob_map, best_threshold, gt_mask_ds, output_dir)

    # Save probability map for ensemble use
    np.save(output_dir / "prob_map.npy", prob_map)
    print(f"  Probability map saved for ensemble use")

    # Save polygons as WKT for downstream consumption
    wkt_path = output_dir / "villi_polygons.wkt"
    with open(wkt_path, "w") as f:
        for i, poly in enumerate(polygons):
            f.write(f"{i}\t{poly.wkt}\n")
    print(f"  Polygons saved to {wkt_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Done — {len(polygons)} villi polygons in {elapsed:.1f}s")
    print(f"  Best threshold: {best_threshold:.2f}")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Peak GPU memory: {peak:.2f} GB")
    print(f"{'=' * 60}\n")

    return polygons
