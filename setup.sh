#!/bin/bash
# CS 7150 Environment Setup (Linux)
# Run this script once to create the conda environment with all dependencies.
#
# Usage:
#   bash setup.sh          # CUDA 11.8 (default, recommended for Explorer NVIDIA GPUs)
#   bash setup.sh cpu      # CPU-only (no NVIDIA GPU)

set -e

module load cuda/12.1.1 anaconda3/2024.06

CUDA_VERSION="${1:-cu118}"

echo "=== CS 7150 Environment Setup ==="
echo "PyTorch CUDA variant: ${CUDA_VERSION}"
echo ""

# Step 1: Create conda environment from yml
echo "[1/3] Creating conda environment..."
conda env create -f pa_env.yml || conda env update -f pa_env.yml
echo ""

# Step 2: Activate and install PyTorch via pip
echo "[2/3] Installing PyTorch (${CUDA_VERSION})..."
eval "$(conda shell.bash hook)"
source activate cs7150
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
echo ""

# Step 3: Verify
echo "[3/3] Verifying installation..."
python -c "
import numpy as np
import matplotlib
import torch
import torchvision

print(f'  NumPy:       {np.__version__}')
print(f'  Matplotlib:  {matplotlib.__version__}')
print(f'  PyTorch:     {torch.__version__}')
print(f'  Torchvision: {torchvision.__version__}')
print(f'  CUDA:        {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:         {torch.cuda.get_device_name(0)}')
"

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with: source activate cs7150"