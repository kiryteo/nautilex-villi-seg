# ── base: PyTorch + CUDA for GPU methods ─────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# System deps for shapely, scipy, tifffile, building PyG extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ \
        libgeos-dev \
        libffi-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Pin PyTorch to prevent dependency upgrades past CUDA 12.4 ────────────────
# The base image provides torch 2.5.1+cu124. We pin it to prevent
# segmentation-models-pytorch or sam2 from upgrading to a newer torch.
ENV PIP_NO_DEPS_FOR_TORCH=1

# Core scientific stack first (rarely changes)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        numpy pandas pyarrow scipy scikit-image scikit-learn \
        tifffile imagecodecs shapely matplotlib h5py anndata scanpy

# PyTorch Geometric (needs torch already installed)
RUN pip install --no-cache-dir \
        torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu124.html && \
    pip install --no-cache-dir torch-geometric

# Segmentation models + SAM2
# Leiden clustering deps (needed by scanpy's sc.tl.leiden)
# HDBSCAN for multimodal GNN clustering
RUN pip install --no-cache-dir igraph leidenalg hdbscan

RUN pip install --no-cache-dir --no-deps segmentation-models-pytorch>=0.3.3 && \
    pip install --no-cache-dir albumentations efficientnet-pytorch pretrainedmodels timm
RUN pip install --no-cache-dir --no-deps "sam-2 @ git+https://github.com/facebookresearch/sam2.git" && \
    pip install --no-cache-dir hydra-core iopath

# ── Download SAM2 checkpoint ─────────────────────────────────────────────────
RUN mkdir -p /models && \
    python -c "from sam2.build_sam import build_sam2; print('SAM2 package OK')" && \
    pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import hf_hub_download; \
               hf_hub_download('facebook/sam2.1-hiera-large', 'sam2.1_hiera_large.pt', local_dir='/models')" \
    || echo "SAM2 checkpoint download failed — will skip SAM2 method at runtime"

# ── Copy application code ────────────────────────────────────────────────────
COPY utils/ utils/
COPY methods/ methods/
COPY segment.py .

# Result path
RUN mkdir -p /output

ENV SAM2_CHECKPOINT=/models/sam2.1_hiera_large.pt

ENTRYPOINT ["python", "segment.py"]
