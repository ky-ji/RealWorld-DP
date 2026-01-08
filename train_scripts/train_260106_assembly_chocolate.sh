#!/bin/bash

# Training script for CogAct real-world robot data
# Uses Diffusion Transformer policy with your converted CogAct dataset

# GPU Configuration
# Set which GPU to use (0, 1, 2, 3, or multiple GPUs like "0,1")
GPU_ID=${1:-0}  # Default to GPU 0 if not specified

echo "============================================================"
echo "Training Diffusion Policy"
echo "============================================================"
echo ""
echo "GPU: $GPU_ID"
echo ""


# Activate conda environment
echo "Activating robodiff environment..."
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate robodiff

# Check if dataset exists
DATASET_PATH="/data0/assembly_chocolate/20260104_no_depth_320_180_8d.zarr/20260104_no_depth_320_180_8d.zarr"


echo "✓ Dataset found"
echo ""

# Run training with 8D dataset
echo "Starting training on GPU $GPU_ID..."
python train.py --config-name=train_assembly_chocolate

echo ""
echo "============================================================"
echo "Training completed or interrupted"
echo "============================================================"
