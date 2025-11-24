#!/bin/bash

# Training script for CogAct real-world robot data
# Uses Diffusion Transformer policy with your converted CogAct dataset

# GPU Configuration
# Set which GPU to use (0, 1, 2, 3, or multiple GPUs like "0,1")
GPU_ID=${1:-0}  # Default to GPU 0 if not specified

echo "============================================================"
echo "Training CogAct Robot Policy"
echo "============================================================"
echo ""
echo "GPU: $GPU_ID"
echo ""
echo "Dataset Configuration:"
echo "  Path: /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full_320x180.zarr"
echo "  Episodes: ~153 (or less if using clean subset)"
echo "  Image resolution: 320x180 (optimized for training speed)"
echo "  Action space: 7D (x, y, z, qx, qy, qz, qw)"
echo ""
echo "Model: Diffusion Transformer"
echo "  - 8 layers, 4 heads, 256 embedding dim"
echo "  - ~20M parameters"
echo "  - 90% crop (162x288 from 180x320)"
echo ""
echo "Training Settings:"
echo "  - 600 epochs"
echo "  - Batch size: 128"
echo "  - Learning rate: 1e-4 with cosine schedule"
echo "  - Checkpoint every 50 epochs"
echo "  - Delta action mode: enabled"
echo ""
echo "Logs: wandb (online mode)"
echo "Output: data/outputs/[timestamp]_train_diffusion_transformer_hybrid_cogact_robot_7d/"
echo ""
echo "============================================================"
echo ""

# Activate conda environment
echo "Activating robodiff environment..."
source ~/storage/anaconda3/etc/profile.d/conda.sh
conda activate robodiff

# Check if dataset exists
DATASET_PATH="/home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full_320x180.zarr"
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset not found at $DATASET_PATH"
    echo "Please run the conversion script first:"
    echo "  ./convert_full_320x180.sh"
    echo ""
    echo "Or manually:"
    echo "  python scripts/convert_cogact_to_zarr.py --input /home/kyji/public/dataset/cogact/1124/trajectories --output $DATASET_PATH --resolution 320 180"
    exit 1
fi

echo "✓ Dataset found"
echo ""

# Run training
echo "Starting training on GPU $GPU_ID..."
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --config-name=train_cogact_robot

echo ""
echo "============================================================"
echo "Training completed or interrupted"
echo "============================================================"
