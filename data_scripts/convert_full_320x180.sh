#!/bin/bash

# Convert full dataset to 320x180 for fast training
# This resolution is optimal for robot manipulation tasks

echo "============================================================"
echo "Converting CogAct Full Dataset to 320x180"
echo "============================================================"
echo ""
echo "Input:  /home/kyji/public/dataset/cogact/1124/trajectories"
echo "Output: /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full_320x180_8d.zarr"
echo "Resolution: 320x180 (optimal for training speed and performance)"
echo ""
echo "This will take approximately 15-30 minutes..."
echo ""
echo "============================================================"
echo ""

# Activate conda environment
source ~/storage/anaconda3/etc/profile.d/conda.sh
conda activate robodiff

# Run conversion with 8D action (pose + gripper)
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full_320x180_8d.zarr \
    --resolution 320 180




echo ""
echo "============================================================"
echo "Conversion complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Dataset includes 8D action (7D pose + 1D gripper)"
echo "2. Use train_cogact.sh to start training"
echo "3. Model will learn gripper control!"
echo ""
