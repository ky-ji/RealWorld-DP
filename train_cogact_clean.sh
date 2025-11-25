#!/bin/bash

# Training script for CogAct CLEAN data (episode 65 onwards)
# Uses only high-quality episodes for better training results

# GPU Configuration
# Set which GPU to use (0, 1, 2, 3, or multiple GPUs like "0,1")
GPU_ID=${1:-0}  # Default to GPU 0 if not specified

echo "============================================================"
echo "Training CogAct Robot Policy (CLEAN DATA)"
echo "============================================================"
echo ""
echo "GPU: $GPU_ID"
echo ""
echo "Dataset Configuration:"
echo "  Path: /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_clean_320x180_8d.zarr"
echo "  Episodes: ~96 (episode_0066 onwards)"
echo "  Image resolution: 320x180 (optimized for training speed)"
echo "  Action space: 8D (x, y, z, qx, qy, qz, qw, gripper)"
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
echo "Logs: wandb (project: diffusion_policy_cogact_clean)"
echo "Output: data/outputs/[timestamp]_train_diffusion_transformer_hybrid_cogact_robot_7d_clean/"
echo "Note: Model will learn gripper control (8D action)"
echo ""
echo "============================================================"
echo ""

# Activate conda environment
echo "Activating robodiff environment..."
source ~/storage/anaconda3/etc/profile.d/conda.sh
conda activate robodiff

# Check if dataset exists
DATASET_PATH="/home/kyji/public/dataset/cogact/1124/diffusion_policy_data_clean_320x180_8d.zarr"
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: 8D clean dataset not found at $DATASET_PATH"
    echo "Please run the conversion script first:"
    echo "  python scripts/convert_cogact_to_zarr.py \\"
    echo "    --input /home/kyji/public/dataset/cogact/1124/trajectories \\"
    echo "    --output $DATASET_PATH \\"
    echo "    --resolution 320 180 \\"
    echo "    --start-episode 66"
    exit 1
fi

echo "✓ Clean dataset found"
echo ""

# Run training with 8D dataset
echo "Starting training on GPU $GPU_ID..."
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --config-name=train_cogact_robot_clean \
    task.dataset.zarr_path="$DATASET_PATH"

echo ""
echo "============================================================"
echo "Training completed or interrupted"
echo "============================================================"
