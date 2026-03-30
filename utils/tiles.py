"""Tile utilities for processing large images in patches.

Supports extracting overlapping tiles and stitching them back with either
max-pooling or weighted (raised-cosine) blending for smooth seams.
"""

from __future__ import annotations
import numpy as np


def extract_tiles(
    image: np.ndarray,
    tile_size: int = 2048,
    overlap: int = 256,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Extract overlapping tiles from a 2-D image.
    Returns list of (tile, (y0, x0, y1, x1)) in pixel coords.
    """
    H, W = image.shape[:2]
    stride = tile_size - overlap
    tiles = []
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            tile = image[y0:y1, x0:x1]
            # Pad if smaller than tile_size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded
            tiles.append((tile, (y0, x0, y1, x1)))
    return tiles


def _raised_cosine_weight(h: int, w: int, margin: int) -> np.ndarray:
    """Create a 2-D raised-cosine weight map for smooth tile blending.

    Pixels in the interior have weight 1.0; pixels within *margin* of any
    edge ramp smoothly from 0 to 1 using a raised-cosine profile.
    """
    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)

    m = min(margin, h // 2, w // 2)
    if m > 0:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, m, dtype=np.float32)))
        wy[:m] = np.minimum(wy[:m], ramp)
        wy[-m:] = np.minimum(wy[-m:], ramp[::-1])
        wx[:m] = np.minimum(wx[:m], ramp)
        wx[-m:] = np.minimum(wx[-m:], ramp[::-1])

    return wy[:, None] * wx[None, :]


def stitch_masks(
    tiles: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    image_shape: tuple[int, int],
    mode: str = "max",
    overlap: int = 0,
) -> np.ndarray:
    """Stitch tile masks back into a full image.

    Parameters
    ----------
    mode : str
        'max' — per-pixel maximum (good for binary masks).
        'blend' — weighted average using raised-cosine window in the
                   overlap region. Requires *overlap* > 0.
        'label' — center-crop priority to avoid overlap artifacts.
    overlap : int
        Tile overlap in pixels. Only used when mode='blend'.
    """
    H, W = image_shape
    if mode == "blend" and overlap > 0:
        accum = np.zeros((H, W), dtype=np.float64)
        weight_sum = np.zeros((H, W), dtype=np.float64)
        for mask, (y0, x0, y1, x1) in tiles:
            h, w = y1 - y0, x1 - x0
            tile_data = mask[:h, :w].astype(np.float64)
            weight = _raised_cosine_weight(h, w, margin=overlap).astype(np.float64)
            accum[y0:y1, x0:x1] += tile_data * weight
            weight_sum[y0:y1, x0:x1] += weight
        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-8)
        return (accum / weight_sum).astype(np.float32)
    elif mode == "max":
        out = np.zeros((H, W), dtype=np.float32)
        for mask, (y0, x0, y1, x1) in tiles:
            h, w = y1 - y0, x1 - x0
            out[y0:y1, x0:x1] = np.maximum(
                out[y0:y1, x0:x1], mask[:h, :w].astype(np.float32)
            )
        return out
    else:  # label — center-crop priority
        out = np.zeros((H, W), dtype=np.int32)
        for mask, (y0, x0, y1, x1) in tiles:
            h, w = y1 - y0, x1 - x0
            region = mask[:h, :w]
            empty = out[y0:y1, x0:x1] == 0
            out[y0:y1, x0:x1] = np.where(empty, region, out[y0:y1, x0:x1])
        return out
