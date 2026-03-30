"""
Nautilex villi segmentation — multi-method entrypoint.

Env vars:
  DATA_PATH  — mount path for Xenium dataset
  OUTPUT_DIR — result path for Beaker
  METHOD     — which method(s) to run: "all", "classical", "gpu",
               or a single name: density, morphology, graph,
               sam2, stagate, unet, multimodal
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

# ── environment ──────────────────────────────────────────────────────────────
DATA_PATH = os.environ.get("DATA_PATH", "/data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
METHOD = os.environ.get("METHOD", "all").lower().strip()

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def log(msg: str):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


# ── environment banner ───────────────────────────────────────────────────────
log("Starting segment.py")
log(f"Python   : {sys.version.split()[0]}")
log(f"DATA_PATH: {DATA_PATH}")
log(f"OUTPUT_DIR: {OUTPUT_DIR}")
log(f"METHOD   : {METHOD}")

gpu_available = False
try:
    import torch

    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    log(
        f"PyTorch  : {torch.__version__}  |  CUDA: {gpu_available}  |  GPUs: {gpu_count}"
    )
    for i in range(gpu_count):
        log(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    log("PyTorch not installed — CPU-only run")

# ── method registry ──────────────────────────────────────────────────────────
CLASSICAL = ["density", "morphology", "graph"]
GPU_METHODS = ["sam2", "stagate", "unet", "multimodal", "ensemble"]
ALL_METHODS = CLASSICAL + GPU_METHODS


def resolve_methods(method_str: str) -> list[str]:
    if method_str == "all":
        return ALL_METHODS
    if method_str == "classical":
        return CLASSICAL
    if method_str == "gpu":
        return GPU_METHODS
    names = [m.strip() for m in method_str.split(",")]
    for n in names:
        if n not in ALL_METHODS:
            log(f"WARNING: unknown method '{n}', skipping")
    return [n for n in names if n in ALL_METHODS]


def import_method(name: str):
    """Dynamically import methods/<name>.py or methods/<name>_seg.py."""
    if name == "multimodal":
        mod_name = "methods.multimodal_gnn"
    elif name == "ensemble":
        mod_name = "methods.ensemble"
    elif name in ("sam2", "stagate", "unet"):
        mod_name = f"methods.{name}_seg"
    else:
        mod_name = f"methods.{name}"
    import importlib

    return importlib.import_module(mod_name)


# ── validation setup ─────────────────────────────────────────────────────────
from utils.io import load_annotations, load_cells
from utils.validate import evaluate
from utils.output import (
    save_geojson,
    assign_cells_to_villi,
    save_cell_villus_map,
    save_metrics,
)

gt_polys = load_annotations(DATA_PATH)
log(f"Loaded {len(gt_polys)} GT annotation polygons")

cells_df = load_cells(DATA_PATH)
log(f"Loaded {len(cells_df)} cells")

# ── run methods ──────────────────────────────────────────────────────────────
methods_to_run = resolve_methods(METHOD)
log(f"Methods to run: {methods_to_run}")

summary: dict = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "data_path": DATA_PATH,
    "n_gt_polygons": len(gt_polys),
    "n_cells": len(cells_df),
    "gpu_available": gpu_available,
    "methods": {},
}

for method_name in methods_to_run:
    log(f"\n{'=' * 60}")
    log(f"RUNNING: {method_name}")
    log(f"{'=' * 60}")

    method_dir = Path(OUTPUT_DIR) / method_name
    method_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        mod = import_method(method_name)
        pred_polys = mod.segment(DATA_PATH, str(method_dir))
        elapsed = time.time() - t0

        log(f"{method_name}: produced {len(pred_polys)} polygons in {elapsed:.1f}s")

        # Validate against GT
        if gt_polys and pred_polys:
            metrics = evaluate(pred_polys, gt_polys)
        else:
            metrics = {
                "n_gt": len(gt_polys),
                "n_pred": len(pred_polys),
                "mean_iou": 0.0,
            }
        metrics["elapsed_seconds"] = round(elapsed, 2)

        # Save outputs
        save_geojson(pred_polys, method_dir / "villi.geojson")
        save_metrics(metrics, method_dir / "metrics.json")

        # Cell assignment
        if pred_polys:
            labels = assign_cells_to_villi(cells_df, pred_polys)
            save_cell_villus_map(cells_df, labels, method_dir / "cell_villus_map.csv")
            metrics["n_cells_assigned"] = int((labels >= 0).sum())

        log(
            f"{method_name} metrics: mean_IoU={metrics.get('mean_iou', 0):.3f}, "
            f"F1@0.5={metrics.get('F1@0.5', 0):.3f}, "
            f"n_pred={len(pred_polys)}"
        )

        summary["methods"][method_name] = {
            "status": "success",
            "n_polygons": len(pred_polys),
            **metrics,
        }

    except Exception as e:
        elapsed = time.time() - t0
        log(f"ERROR in {method_name} after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        summary["methods"][method_name] = {
            "status": "error",
            "error": str(e),
            "elapsed_seconds": round(elapsed, 2),
        }

# ── comparison summary ───────────────────────────────────────────────────────
log(f"\n{'=' * 60}")
log("SUMMARY")
log(f"{'=' * 60}")
log(
    f"{'Method':<15} {'Status':<8} {'Polygons':>8} {'Mean IoU':>10} {'F1@0.5':>8} {'Time':>8}"
)
log("-" * 65)

for name, info in summary["methods"].items():
    if info["status"] == "success":
        log(
            f"{name:<15} {'OK':<8} {info['n_polygons']:>8} "
            f"{info.get('mean_iou', 0):>10.3f} {info.get('F1@0.5', 0):>8.3f} "
            f"{info.get('elapsed_seconds', 0):>7.1f}s"
        )
    else:
        log(
            f"{name:<15} {'FAIL':<8} {'—':>8} {'—':>10} {'—':>8} "
            f"{info.get('elapsed_seconds', 0):>7.1f}s"
        )

# Save master summary
summary_path = Path(OUTPUT_DIR) / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))
log(f"\nWrote {summary_path}")
log("Done.")
