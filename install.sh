#!/usr/bin/env bash
set -euo pipefail


# ---- GPU PyTorch (CUDA 11.8) ----
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ---- core scientific stack ----
pip install numpy pandas scipy
pip install -U scikit-learn

# ---- transformers & helpers ----
pip install transformers einops

pip install torch_geometric
