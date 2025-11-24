#!/bin/bash

# Convert full dataset to 320x180 for fast training
# This resolution is optimal for robot manipulation tasks

echo "============================================================"
echo "Converting CogAct Full Dataset to 320x180"
echo "============================================================"
echo ""
echo "Input:  /home/kyji/public/dataset/cogact/1124/trajectories"
echo "Output: /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full_320x180.zarr"
echo "Resolution: 320x180 (optimal for training speed and performance)"
echo ""
echo "This will take approximately 15-30 minutes..."
echo ""
echo "============================================================"
echo ""

# Activate conda environment
source ~/storage/anaconda3/etc/profile.d/conda.sh
conda activate robodiff

# Run conversion
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full_320x180.zarr \
    --resolution 320 180

echo ""
echo "============================================================"
echo "Conversion complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. Update cogact_robot_7d.yaml to use the new dataset path"
echo "2. Update image_shape to [3, 180, 320]"
echo "3. Restart training with much faster data loading!"
echo ""
