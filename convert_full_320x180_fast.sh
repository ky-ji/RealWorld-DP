#!/bin/bash

# Fast conversion: Directly resize from full.zarr (1080x1920) to 320x180
# This avoids re-reading original images and should be faster

echo "============================================================"
echo "Fast Convert: Full Dataset to 320x180"
echo "============================================================"
echo ""
echo "Input:  /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full.zarr"
echo "Output: /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full_320x180.zarr"
echo "Resolution: 320x180"
echo ""
echo "Estimated time: 5-10 minutes (much faster than from scratch)"
echo ""
echo "============================================================"
echo ""

# Activate conda environment
source ~/storage/anaconda3/etc/profile.d/conda.sh
conda activate robodiff

# Run fast conversion script
python << 'PYTHON_SCRIPT'
import zarr
import numpy as np
import cv2
from tqdm import tqdm
import shutil
import os

# Paths
input_path = '/home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full.zarr'
output_path = '/home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full_320x180.zarr'

# Remove output if exists
if os.path.exists(output_path):
    print(f"Removing existing output: {output_path}")
    shutil.rmtree(output_path)

print(f"Loading source zarr: {input_path}")
src_store = zarr.open(input_path, 'r')

# Get dimensions
n_frames = src_store['data']['image'].shape[0]
print(f"Total frames: {n_frames}")

# Create output zarr
print(f"Creating output zarr: {output_path}")
dst_store = zarr.open(output_path, 'w')

# Copy metadata
print("Copying metadata...")
if 'meta' in src_store:
    zarr.copy(src_store['meta'], dst_store, name='meta')

# Create data group
dst_data = dst_store.create_group('data')

# Process each key in data
src_data = src_store['data']
print(f"Processing data keys: {list(src_data.keys())}")

for key in src_data.keys():
    src_array = src_data[key]
    
    if key == 'image':
        # Resize images from 1080x1920 to 180x320
        print(f"\nResizing images: {src_array.shape} -> (N, 180, 320, 3)")
        
        # Create output array
        dst_array = dst_data.create_dataset(
            key,
            shape=(n_frames, 180, 320, 3),
            chunks=(3, 180, 320, 3),
            dtype=np.uint8,
            compressor=zarr.Blosc(cname='lz4', clevel=5, shuffle=zarr.Blosc.SHUFFLE)
        )
        
        # Resize in batches
        batch_size = 50
        for i in tqdm(range(0, n_frames, batch_size), desc="Resizing images"):
            end_idx = min(i + batch_size, n_frames)
            batch = src_array[i:end_idx]
            
            # Resize each image in batch
            resized_batch = []
            for img in batch:
                # img is HWC: 1080x1920x3
                resized = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA)
                resized_batch.append(resized)
            
            dst_array[i:end_idx] = np.array(resized_batch)
    else:
        # Copy other arrays as-is
        print(f"Copying {key}: {src_array.shape}")
        zarr.copy(src_array, dst_data, name=key)

print("\n" + "="*60)
print("Conversion complete!")
print("="*60)
print(f"\nOutput: {output_path}")
print(f"Image shape: (N, 180, 320, 3)")
print("\nYou can now start training with the optimized dataset!")

PYTHON_SCRIPT

echo ""
echo "============================================================"
echo "Done! Dataset ready for training."
echo "============================================================"
