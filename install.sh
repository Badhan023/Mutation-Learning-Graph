#!/usr/bin/env bash
set -euo pipefail


# ---- GPU PyTorch ----
pip install torch torchvision torchaudio

# ---- core scientific stack ----
pip install numpy pandas scipy
pip install -U scikit-learn

# ---- transformers & helpers ----
pip install transformers einops

pip install torch_geometric
