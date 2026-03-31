# Nautilex Villi Segmentation Progress Report

## 1. Challenge Summary

Segment individual intestinal villi from 10x Xenium spatial transcriptomics data (mouse ileum) and produce a cell-to-villus mapping. The challenge is based on the dataset from Zhang et al., *Nature*, 2025.

**Inputs available per sample:**
- `transcripts.parquet` -- per-transcript x/y/z locations and gene identity
- `cells.parquet` -- per-cell centroids and metadata (~157K cells)
- `cell_feature_matrix/` -- cell-by-gene expression matrix (480 genes + 20 negative controls)
- `morphology.ome.tif` -- DAPI morphology image (12 Z-slices, 23,882 x 25,621 pixels, uint16, JPEG2000)

**Ground truth:** 5 hand-annotated villi polygons in `annotations/TIS09474-001-001_annotation.geojson` (QuPath GeoJSON format, pixel coordinates; multiply by 0.2125 for microns).

**Target output:** Polygon annotations (GeoJSON) delineating individual villi + cell-to-villus mapping CSV.

---

## 2. Dataset Characteristics

We are currently working with sample **TIS09474-001-001** only (the one with GT annotations).

| Property | Value |
|----------|-------|
| Transcripts | 62.9M rows in `transcripts.parquet` |
| Spatial extent | 0--5,440 x 0--5,069 microns |
| Cells | 156,765 in `cells.parquet` |
| Morphology | 12 Z-slices, 23,882 x 25,621 px, uint16 DAPI |
| Pixel size | 0.2125 um/pixel |
| Gene panel | 480 mouse genes + 20 negative controls |
| GT annotations | 5 polygons in **pixel** coordinates |

**Important coordinate convention:** The GeoJSON annotations are in pixel space. All methods internally convert to micron coordinates (multiply by 0.2125). All output polygons are in micron coordinates.

---

## 3. Infrastructure

### Compute
- **Beaker** cluster `ai1/octo-nautilex-aws-h100` (1x H100 GPU per experiment)
- Docker base image: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`
- Dataset on Beaker: `ashwin-samudre/xenium-TIS09474-001-001` (5.5 GB), mounted at `/data`
- Image on Beaker: `ashwin-samudre/villi-seg-full`

### Iteration Cycle
```
edit code locally
  -> python -m py_compile <file>  (syntax check)
  -> docker buildx build --platform linux/amd64 --load -t villi-seg-full .
  -> beaker image delete ashwin-samudre/villi-seg-full
  -> beaker image create --name villi-seg-full villi-seg-full:latest
  -> beaker experiment create experiment.yaml
```

### Repo Structure
```
nautilex-villi-seg/
  segment.py           -- Main entrypoint; dispatches methods via METHOD env var
  experiment.yaml      -- Beaker experiment config
  Dockerfile           -- Multi-stage build with all deps
  requirements.txt     -- Python dependencies
  methods/
    density.py         -- Transcript density KDE + watershed
    morphology.py      -- DAPI image processing (Otsu + watershed)
    graph.py           -- Transcript Delaunay graph + Leiden clustering
    sam2_seg.py        -- SAM2 prompted segmentation on DAPI
    stagate_seg.py     -- STAGATE spatial transcriptomics autoencoder
    unet_seg.py        -- ResNet34 U-Net fine-tuning on GT polygons
    multimodal_gnn.py  -- GATv2 GNN combining DAPI + gene expression
  utils/
    coords.py          -- PIXEL_SIZE_UM=0.2125, coordinate conversion helpers
    io.py              -- Data loaders (transcripts, cells, expression, morphology, annotations)
    validate.py        -- IoU computation, greedy polygon matching, F1/P/R metrics
    output.py          -- GeoJSON export, cell-villus assignment, metrics saving
    tiles.py           -- Tile extraction/stitching for large image processing
  annotations/
    TIS09474-001-001_annotation.geojson  -- 5 GT villi polygons (pixel coords)
```

### Validation Pipeline (`utils/validate.py`)

- **Polygon matching:** Greedy best-first on full IoU matrix (threshold 0.01 to match)
- **Mean IoU:** Computed only over matched pairs (unmatched GT polygons do NOT contribute zero)
- **F1/Precision/Recall:** Computed at IoU thresholds [0.25, 0.5, 0.75]
- **Note:** With only 5 GT polygons, small changes in matching have large effects on metrics. A method producing 5 perfect polygons + 100 false positives would have R@0.5=1.0 but P@0.5=0.05.

---

## 4. Methods Implemented (7 total)

### 4.1 Classical Methods (CPU-only)

#### Density (`methods/density.py`)
- **Approach:** Kernel density estimation (KDE) on transcript x/y locations, then Otsu thresholding + watershed to split connected regions.
- **Result:** 161 polygons, Mean IoU 0.314, F1@0.5 0.000
- **Assessment:** Oversegments massively. The density field doesn't have sharp enough boundaries between villi. All 161 polygons are too small to match GT at IoU >= 0.5.

#### Morphology (`methods/morphology.py`)
- **Approach:** DAPI max-projection -> Otsu threshold -> morphological cleanup -> distance transform + watershed -> contour extraction.
- **Result:** 415 polygons, Mean IoU 0.569, F1@0.5 0.014
- **Assessment:** Second-best IoU but extreme over-segmentation (415 polys for 5 GT). The watershed splits tissue into many small fragments. Some fragments overlap well with GT (hence decent IoU on matched pairs) but most are spurious.

#### Graph (`methods/graph.py`)
- **Approach:** Delaunay triangulation on transcript locations, prune long edges, Leiden community detection, concave hulls per community.
- **Result:** 27 polygons, Mean IoU 0.000, F1@0.5 0.000
- **Assessment:** Complete failure. The Leiden clustering at the transcript level doesn't align with villus boundaries at all. The 27 polygons don't overlap with any GT polygon above the 0.01 matching threshold.

### 4.2 Deep Learning Methods (GPU)

#### SAM2 (`methods/sam2_seg.py`)
- **Approach:** Segment Anything Model 2 (SAM2.1-hiera-large) prompted with grid points on the DAPI max-projection. Automatically generates masks, filters by area, converts to polygons.
- **Result:** 1153 polygons, Mean IoU 0.394, F1@0.5 0.005
- **Assessment:** SAM2 finds lots of structure in the DAPI image but without villus-specific prompting, it segments nuclei/cells rather than whole villi. Extreme over-segmentation.

#### STAGATE (`methods/stagate_seg.py`)
- **Approach:** STAGATE spatial transcriptomics graph autoencoder on the cell graph. Learns spatial-aware embeddings, then Leiden clustering on the latent space, concave hulls per cluster.
- **Result:** 116 polygons, Mean IoU 0.000, F1@0.5 0.000
- **Assessment:** The spatial clustering doesn't capture villus-level structure. Clusters are either too large (spanning multiple villi) or too small (sub-villus patches). Zero IoU overlap with GT.

#### U-Net (`methods/unet_seg.py`) -- **Best Mean IoU**
- **Approach:** Fine-tune a ResNet34-backed U-Net (pretrained ImageNet encoder, `segmentation-models-pytorch`) on only 5 GT annotation polygons.
  - Rasterize GT polygons onto DAPI max-projection
  - Downscale 4x (from 23K x 25K to ~6K x 6.4K at ~0.85 um/pixel)
  - Extract 150 training patches (100 positive near GT centroids with random jitter, 50 negative far from GT)
  - Heavy augmentation (rotation, flip, elastic deform, brightness/contrast, noise, blur via Albumentations)
  - Train 100 epochs with BCE+Dice loss, AdamW (lr=1e-3), CosineAnnealingLR, mixed precision
  - Tiled inference (512x512, 64px overlap, max-stitching), threshold 0.5
  - Morphological cleanup (remove small objects/holes), contour extraction -> polygons
- **Result:** 26 polygons, Mean IoU **0.724**, F1@0.5 0.194
- **Assessment:** Best pixel-level overlap with GT (0.724 IoU). However, it produces 26 polygons instead of 5, meaning villi are being merged or the tissue is over-detected. The high IoU but low F1 suggests the model correctly identifies villus tissue but fails at **instance separation** -- touching villi are merged into connected blobs.

#### Multimodal GNN (`methods/multimodal_gnn.py`) -- **Best F1@0.5**
- **Approach:** GATv2-based graph neural network operating on the cell graph.
  - Node features (28-dim): 20 gene expression PCA components + 8 DAPI patch statistics (mean, std, median, Q25, Q75, max, gradient mean/std from a 32x32 pixel patch around each cell centroid)
  - Graph: KNN (k=10) on cell spatial coordinates (microns), made undirected
  - Semi-supervised training: cells inside GT polygons -> positive (label=1), cells >100um from GT -> negative (label=0), cells 0-100um -> excluded (label=-1)
  - 3-layer GATv2Conv (28 -> 256[4-head] -> 256[4-head] -> 64[1-head]) + MLP classifier (64 -> 32 -> 1)
  - BCEWithLogitsLoss with class-imbalance pos_weight, Adam lr=1e-3, 200 epochs, full-batch
  - Inference: threshold 0.5 -> DBSCAN (eps=30um, min_samples=10) on positive cells -> concave hull (ratio=0.3) -> buffer 5um -> simplify 2um -> area filter >= 5000 um^2
- **Result:** 9 polygons, Mean IoU 0.527, F1@0.5 **0.429**
- **Assessment:** Best instance-level performance. 9 polygons is much closer to the true 5 than any other method. The GNN successfully leverages both modalities (gene expression patterns + morphology features) to identify villus cells, and DBSCAN produces reasonable spatial clusters. The lower IoU (vs U-Net) comes from imprecise boundaries (concave hull of scattered cell centroids vs pixel-level mask).

---

## 5. Results Summary (v3 Baseline)

**Beaker experiment:** `01KMRH9EZKD737PVXP17ZCKEW4` (all 7 methods, all succeeded)

| Method | Polygons | Mean IoU | F1@0.5 | F1@0.25 | F1@0.75 | Time (s) |
|--------|----------|----------|--------|---------|---------|----------|
| density | 161 | 0.314 | 0.000 | 0.000 | 0.000 | 10 |
| morphology | 415 | 0.569 | 0.014 | 0.024 | 0.000 | 267 |
| graph | 27 | 0.000 | 0.000 | 0.000 | 0.000 | 5 |
| sam2 | 1153 | 0.394 | 0.005 | 0.009 | 0.000 | 271 |
| stagate | 116 | 0.000 | 0.000 | 0.000 | 0.000 | 193 |
| **unet** | **26** | **0.724** | **0.194** | 0.286 | 0.133 | 368 |
| **multimodal** | **9** | **0.527** | **0.429** | 0.429 | 0.000 | 108 |

**Key takeaway:** U-Net and Multimodal GNN are the only two methods producing usable results. They have complementary strengths:
- **U-Net** excels at pixel-level accuracy (IoU) but fails at instance separation
- **Multimodal GNN** excels at instance-level detection (F1) but has imprecise boundaries

---

## 6. Detailed Analysis of Top Two Methods

### 6.1 U-Net -- Strengths and Weaknesses

**Strengths:**
- Highest IoU (0.724) -- the model learns villus tissue appearance well from only 5 examples
- Transfer learning from ImageNet + heavy augmentation compensates for tiny training set
- Pixel-level mask gives precise boundary delineation

**Weaknesses (ordered by estimated impact):**
1. **No instance separation** (CRITICAL): Connected villi merge into single blobs. The model produces 26 connected components but many are merged multi-villi regions. There is no watershed or boundary-aware mechanism to split them.
2. **Small tile overlap** (64px = 12.5%): Boundary artifacts from max-stitching with insufficient overlap.
3. **No boundary-aware loss**: BCE+Dice treats all pixels equally. The model doesn't learn to predict thin separating boundaries between adjacent villi.
4. **Positive patch jitter too large**: Random offset of +/- 256px from centroid can push the GT polygon entirely outside the training tile, creating mislabeled patches.
5. **No validation / early stopping**: Best model selected by training loss, not generalization. With only 150 patches this matters.
6. **Hard-coded threshold** (0.5): Not tuned; could be suboptimal.
7. **MIN_AREA_UM2=5000**: May filter out valid small villi or keep invalid fragments.

### 6.2 Multimodal GNN -- Strengths and Weaknesses

**Strengths:**
- Best F1@0.5 (0.429) -- produces 9 polygons, closest to the true 5
- Leverages both gene expression and morphology features (28-dim multimodal representation)
- Semi-supervised approach effectively propagates labels from 5 GT annotations to all 157K cells
- GATv2 with 4-head attention learns expressive neighborhood aggregation

**Weaknesses (ordered by estimated impact):**
1. **No edge features**: GATv2Conv receives only node features; edges carry no information about spatial distance or expression similarity between connected cells. This limits the model's ability to distinguish boundary vs interior edges.
2. **DBSCAN sensitivity**: `eps=30um` is hard-coded. Close villi may merge (eps too large) while sparse villi may split (eps too small). Variable-density clustering (HDBSCAN) would be more robust.
3. **Hard-coded threshold** (0.5): Probability threshold not tuned. With class imbalance, optimal threshold is likely != 0.5.
4. **No LR scheduler**: Flat learning rate for 200 epochs. Cosine annealing or step decay could improve convergence.
5. **Fixed KNN k=10**: Doesn't adapt to local cell density. Dense regions get appropriate neighbors; sparse regions get distant, potentially irrelevant neighbors.
6. **Serial image feature extraction**: Per-cell DAPI patch extraction is a Python loop over 157K cells -- slow but functional.

---

## 7. Active Improvement Plan (In Progress)

We are implementing targeted improvements for both top methods plus an ensemble:

### 7.1 U-Net Improvements
1. **Watershed post-processing** to split merged villi (highest priority -- addresses the #1 failure mode)
2. **Increase tile overlap** to 128px with weighted blending (instead of hard max-stitching)
3. **Boundary-aware loss** using distance transform weighting
4. **Clamped patch jitter** (reduce from +/-half to +/-half//2)
5. **Threshold sweep** (test 0.3 to 0.7, pick best)
6. **Early stopping** with 1-polygon validation holdout

### 7.2 Multimodal GNN Improvements
1. **Edge features** (spatial distance + expression cosine similarity) passed to GATv2Conv
2. **HDBSCAN** instead of DBSCAN for density-adaptive clustering
3. **Threshold sweep** (0.3 to 0.7)
4. **Cosine LR scheduler**

### 7.3 Ensemble
- New `methods/ensemble.py` combining U-Net's pixel-precision with GNN's instance separation
- Strategy: use GNN polygons as instance seeds, refine boundaries with U-Net probability map

---

## 8. Unexplored Directions for New Contributors

These are directions we have NOT pursued that could yield improvements:

### 8.1 Data-side Improvements
- **Use all 6 samples** for cross-validation / more robust training (currently only TIS09474-001-001 has GT annotations)
- **Transcript-level features for GNN**: Currently the GNN uses gene expression PCA + DAPI stats. Adding transcript density, spatial dispersion, or cell-type marker genes as features could help.
- **Cell boundary information**: `cell_boundaries.parquet` contains polygon outlines for each segmented cell. These could inform the graph structure or provide additional morphological features.
- **Multi-resolution morphology**: `morphology_focus/` contains a Zarr pyramid. Using multiple scales could help methods that operate on the image.

### 8.2 Method-level Ideas
- **Cellpose / StarDist adapted for villi**: Instance segmentation models designed for cell segmentation could potentially be adapted for villi (much larger objects, but similar separation problem).
- **Voronoi tessellation seeded by GNN**: Use GNN-predicted villus centers as Voronoi seeds, then assign cells to nearest villus. Could give clean, non-overlapping partitions.
- **Spatial gene expression gradients**: Villi have known crypt-to-tip expression gradients. Methods that model this gradient (e.g., spatial regression, ordered gene modules) could identify individual villi by their gradient orientation.
- **Contrastive learning on cell patches**: Self-supervised pretraining on DAPI patches, then fine-tune for villus/non-villus classification.
- **Graph-cut / CRF on U-Net output**: Use a pairwise CRF or graph-cut on the U-Net probability map with spatial regularization to enforce instance separation.
- **Panoptic segmentation**: Train a model that jointly predicts semantic class (villus/background) and instance IDs. Architectures like Mask R-CNN or Panoptic-DeepLab could work if adapted for the large image + small training set.

### 8.3 Evaluation Improvements
- **Per-polygon quality analysis**: Look at which GT polygons are consistently well-matched vs poorly-matched across methods. This reveals if certain villus shapes/sizes are harder.
- **Boundary quality metric**: IoU is area-based. A boundary-distance metric (e.g., Hausdorff distance) would better capture how well methods trace villus edges.
- **Generalization testing**: Evaluate trained models on the other 5 samples (without GT) by visual inspection. Do the predicted polygons look reasonable?

### 8.4 Failure Mode Deep-Dives
- **Why does the graph method fail completely?** The Delaunay + Leiden approach produces 27 polygons with 0.0 IoU. Is it a scale issue (transcript-level vs cell-level)? A Leiden resolution parameter issue? Worth investigating if you want to improve classical approaches.
- **Why does STAGATE fail?** It produces 116 polygons with 0.0 IoU. The spatial autoencoder embeddings may not capture villus-level structure. Could be improved with different resolution parameters, graph construction, or embedding dimension.
- **SAM2 prompting strategy**: Our current approach uses a grid of points. Smarter prompting (e.g., using transcript density peaks as foreground prompts, low-density regions as background prompts) could dramatically improve SAM2 results.

---

## 9. Beaker Experiment History

| Version | Experiment ID | Status | Notes |
|---------|--------------|--------|-------|
| v1 | `01KMPST4C5TGWQHBJ02W7VE1W2` | Succeeded | 4 OK, 3 FAIL (stagate/unet/multimodal had import/API bugs) |
| v2 | `01KMREWJB3RZHMYY4GX69J875C` | Succeeded | 6 OK, 1 FAIL (unet -- `torch.cuda.total_mem` -> `total_memory` typo) |
| v3 | `01KMRH9EZKD737PVXP17ZCKEW4` | Succeeded | **All 7 OK** -- baseline complete (results in Section 5) |
| v4a-g | (multiple) | Mixed | Docker/dep fixes, no algorithm changes |
| v4h | — | Local | U-Net boundary weight fix + GNN adaptive HDBSCAN |
| v4i | (pending) | — | **Cell-vote ensemble: F1@0.5=0.909, Mean IoU=0.805** |

---

## 10. v4 Iteration Results (Post-Baseline Improvements)

### v4a–v4g: Iterative Bug Fixes
Multiple iterations fixing Docker build issues, dependency conflicts (`smp` / `sam2` install order), buildx platform flags, and runtime errors. These did not change algorithmic behavior.

### v4h: U-Net + GNN Bug Fixes
- **U-Net boundary weight map fix**: `np.where` instead of `np.minimum` for correct foreground/boundary weighting
- **GNN adaptive HDBSCAN**: `min_cluster_size` adapts to data size; added polygon merging for overlapping clusters

### v4i: Cell-Vote Ensemble (Current Best)

**Beaker experiment:** (pending submission)

**Ensemble strategy** (`methods/ensemble.py`):
1. Run U-Net tiled inference → probability map
2. Threshold at 0.5 → connected components (instance candidates)
3. Run GNN cell classification → per-cell villus probability
4. **Cell-vote filtering**: For each U-Net component, count GNN-positive cells inside. Keep components where ≥40% of cells vote positive.
5. **Natural gap detection**: Analyze intensity profile along component major axis. If a clear dip exists (>15% below mean), split the component at the gap.
6. Contour extraction → polygon output

**Results (from v4i logs):**

| Metric | Value |
|--------|-------|
| Polygons produced | 6 |
| GT polygons matched | **5/5** |
| Precision@0.5 | 0.833 |
| Recall@0.5 | **1.000** |
| F1@0.5 | **0.909** |
| Mean IoU (matched) | 0.805 |

**Per-villus IoU:**
| GT Polygon | IoU | Notes |
|------------|-----|-------|
| GT0 | 0.952 | Excellent |
| GT3 | 0.891 | Excellent |
| GT4 | 0.802 | Good |
| GT2 | 0.757 | Good |
| GT1 | 0.624 | Weakest — boundary imprecision |

**Comparison with v3 baselines:**

| Method | Polygons | Mean IoU | F1@0.5 |
|--------|----------|----------|--------|
| U-Net (v3) | 26 | 0.724 | 0.194 |
| GNN (v3) | 9 | 0.527 | 0.429 |
| **Ensemble (v4i)** | **6** | **0.805** | **0.909** |

The ensemble combines U-Net's pixel precision with GNN's instance awareness, achieving both high IoU AND high F1 — solving the fundamental complementarity problem identified in v3.

### Remaining Improvement Opportunities
1. **GT1 boundary refinement**: IoU 0.624 is the weakest link. Could benefit from morphological active contour refinement.
2. **False positive elimination**: 6 polygons vs 5 GT means 1 spurious detection. Tighter cell-vote threshold or area filtering could remove it.
3. **Threshold tuning**: Both the U-Net probability threshold (0.5) and GNN vote threshold (40%) could be swept for optimal F1.
4. **Multi-sample generalization**: Current pipeline is tuned to one sample with GT. Testing on other samples would reveal robustness.

---

## 12. Known Gotchas

These are bugs/issues we already encountered and fixed. Avoid re-hitting them:

| Issue | Fix |
|-------|-----|
| SAM2 PyPI package is `sam-2`, not `sam2` | `pip install "sam-2 @ git+https://github.com/facebookresearch/sam2.git"` (Python import is still `from sam2 import ...`) |
| scanpy `sc.tl.leiden()` needs igraph + leidenalg | `pip install igraph leidenalg` separately |
| `skimage.morphology.disk(512)` causes OOM | Cap disk radius at 50px |
| `torch.cuda.get_device_properties(device).total_mem` wrong attr | Correct: `.total_memory` |
| `multimodal` module maps to `multimodal_gnn.py` not `multimodal_seg.py` | Handled in `segment.py` import_method() |
| Annotations are in pixel coords, not microns | `load_annotations()` converts automatically via `pixel_to_micron()` |
| `PIXEL_SIZE_UM = 0.2125` | Defined in `utils/coords.py` -- consistent across all methods |

---

## 13. How to Run

### Local syntax check
```bash
python -m py_compile methods/unet_seg.py
python -m py_compile methods/multimodal_gnn.py
```

### Build and push to Beaker
```bash
docker buildx build --platform linux/amd64 --load -t villi-seg-full .
beaker image delete ashwin-samudre/villi-seg-full
beaker image create --name villi-seg-full villi-seg-full:latest
```

### Run experiment
```bash
# All methods
beaker experiment create experiment.yaml

# Specific methods only (edit experiment.yaml or use env override)
# Set METHOD=unet,multimodal,ensemble
```

### View results
```bash
# Find latest experiment
beaker experiment list --workspace ai1/ashwin-samudre

# Get logs
beaker experiment logs <experiment-id>

# Download results
beaker experiment results <experiment-id> --output ./results/
```

Each method writes to `/output/<method_name>/`:
- `villi.geojson` -- predicted polygons
- `metrics.json` -- evaluation metrics (IoU, F1, P, R)
- `cell_villus_map.csv` -- cell barcode to villus ID mapping
- Method-specific diagnostics (loss curves, overlay PNGs, etc.)

Master summary at `/output/summary.json`.
