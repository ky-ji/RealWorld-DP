#!/usr/bin/env python3
"""
Convert CogAct dataset to Diffusion Policy format (Zarr ReplayBuffer)

CogAct data structure:
- trajectories/
  - episode_XXXX/
    - data.pkl: Contains robot_eef_pose, robot_gripper, action, action_gripper, timestamp, image_index
    - images/frame_XXXX.jpg: RGB images
    - meta.json: Episode metadata

Diffusion Policy format:
- zarr ReplayBuffer with:
  - data/
    - image: (N, H, W, C) uint8
    - robot_eef_pose: (N, 7) float32 [x, y, z, qx, qy, qz, qw]
    - action: (N, 8) float32 [x, y, z, qx, qy, qz, qw, gripper]
  - meta/
    - episode_ends: (n_episodes,) int64
"""

import os
import sys
import pickle
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import zarr
import numcodecs
import cv2
from tqdm import tqdm

# Add parent directory to path to import diffusion_policy modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k

register_codecs()


def load_episode_data(episode_path: Path, 
                      target_resolution: Optional[tuple] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Load a single episode from CogAct format
    
    Args:
        episode_path: Path to episode directory
        target_resolution: (width, height) for resizing images, None to keep original
    
    Returns:
        Dictionary with arrays for each data field
    """
    # Load pickle data
    with open(episode_path / "data.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Load metadata
    with open(episode_path / "meta.json", "r") as f:
        meta = json.load(f)
    
    n_steps = meta['n_steps']
    
    # Check if Images folder exists and rename to images
    images_dir_upper = episode_path / "Images"
    images_dir_lower = episode_path / "images"
    
    if images_dir_upper.exists() and not images_dir_lower.exists():
        images_dir_upper.rename(images_dir_lower)
        print(f"  -> Renamed 'Images' to 'images' in {episode_path.name}")
    
    # Load images
    images_dir = images_dir_lower
    image_files = sorted(list(images_dir.glob("frame_*.jpg")))
    
    # Handle mismatch between image count and steps
    if len(image_files) != n_steps:
        print(f"Warning: {episode_path.name} has {len(image_files)} images but {n_steps} steps")
        # Use the minimum of the two to avoid index errors
        n_images_to_load = min(len(image_files), n_steps)
        if n_images_to_load < n_steps:
            print(f"  -> Skipping episode due to insufficient images")
            return None
    else:
        n_images_to_load = n_steps
    
    # Read first image to get dimensions
    first_img = cv2.imread(str(image_files[0]))
    first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
    original_height, original_width = first_img.shape[:2]
    
    # Determine output resolution
    if target_resolution is not None:
        output_width, output_height = target_resolution
    else:
        output_height, output_width = original_height, original_width
    
    # Allocate image array
    images = np.zeros((n_steps, output_height, output_width, 3), dtype=np.uint8)
    
    # Load images (only up to n_images_to_load to avoid index errors)
    for i in range(n_images_to_load):
        img = cv2.imread(str(image_files[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if target_resolution is not None:
            img = cv2.resize(img, (output_width, output_height), interpolation=cv2.INTER_AREA)
        
        images[i] = img
    
    # If we have extra images, use the last loaded image for remaining frames
    if n_images_to_load < n_steps:
        for i in range(n_images_to_load, n_steps):
            images[i] = images[n_images_to_load - 1]
    
    # Ensure images array is C-contiguous (required by JPEG2K encoder)
    images = np.ascontiguousarray(images)
    
    # Prepare output dictionary
    # Combine action pose (7D) with action gripper (1D) to form 8D action
    action_pose = data['action'].astype(np.float32)  # (N, 7)
    action_gripper = data['action_gripper'].astype(np.float32)  # (N,)
    if action_gripper.ndim == 1:
        action_gripper = action_gripper[:, np.newaxis]  # (N, 1)
    action_8d = np.concatenate([action_pose, action_gripper], axis=-1)  # (N, 8)
    
    episode_data = {
        'image': images,  # (N, H, W, 3) uint8
        'robot_eef_pose': data['robot_eef_pose'].astype(np.float32),  # (N, 7)
        'action': action_8d,  # (N, 8) [pose + gripper]
    }
    
    # Add gripper state as part of observation (optional)
    if 'robot_gripper' in data:
        robot_gripper = data['robot_gripper'].astype(np.float32)
        if robot_gripper.ndim == 1:
            robot_gripper = robot_gripper[:, np.newaxis]  # (N, 1)
        episode_data['robot_gripper_state'] = robot_gripper
    
    # Add timestamp if available
    if 'timestamp' in data:
        episode_data['timestamp'] = data['timestamp'].astype(np.float64)
    
    return episode_data


def convert_dataset(
    input_dir: str,
    output_zarr_path: str,
    target_resolution: Optional[tuple] = None,
    start_episode: Optional[int] = None,
    max_episodes: Optional[int] = None,
    image_compressor: str = 'blosc',
    compression_level: int = 5
):
    """
    Convert entire CogAct dataset to Diffusion Policy zarr format
    
    Args:
        input_dir: Path to trajectories directory containing episode folders
        output_zarr_path: Output path for zarr file
        target_resolution: (width, height) for resizing images, None to keep original
        start_episode: Starting episode number (e.g., 65 to start from episode_0065)
        max_episodes: Maximum number of episodes to convert (for testing)
        image_compressor: 'jpeg2k' or 'blosc'
        compression_level: Compression level (0-100 for jpeg2k, 0-9 for blosc)
    """
    input_path = Path(input_dir)
    output_path = Path(output_zarr_path)
    
    # Get all episode directories
    episode_dirs = sorted(list(input_path.glob("episode_*")))
    
    # Filter by start episode
    if start_episode is not None:
        episode_dirs = [ep for ep in episode_dirs 
                       if int(ep.name.split('_')[1]) >= start_episode]
        print(f"Starting from episode_{start_episode:04d}")
    
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
    
    print(f"Found {len(episode_dirs)} episodes to convert")
    
    if len(episode_dirs) == 0:
        print("No episodes found!")
        return
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing output if it exists to avoid conflicts
    if output_path.exists():
        print(f"Removing existing output: {output_path}")
        shutil.rmtree(output_path)
    
    # Set up compressor
    if image_compressor == 'jpeg2k':
        img_compressor = Jpeg2k(level=compression_level)
    else:
        img_compressor = numcodecs.Blosc(cname='lz4', clevel=compression_level)
    
    # Create replay buffer
    store = zarr.DirectoryStore(str(output_path))
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=store)
    
    # Convert each episode
    print("Converting episodes...")
    skipped_episodes = []
    for episode_dir in tqdm(episode_dirs):
        try:
            # Load episode data
            episode_data = load_episode_data(episode_dir, target_resolution=target_resolution)
            
            # Skip if episode data is None (due to data issues)
            if episode_data is None:
                skipped_episodes.append(episode_dir.name)
                continue
            
            # Add to replay buffer with appropriate compressors
            compressors = {
                'image': img_compressor,
                'robot_eef_pose': 'default',
                'action': 'default',
            }
            
            if 'robot_gripper_state' in episode_data:
                compressors['robot_gripper_state'] = 'default'
            if 'timestamp' in episode_data:
                compressors['timestamp'] = 'default'
            
            replay_buffer.add_episode(episode_data, compressors=compressors)
            
        except Exception as e:
            print(f"\nError processing {episode_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nConversion complete!")
    print(f"Total episodes: {replay_buffer.n_episodes}")
    print(f"Total steps: {replay_buffer.n_steps}")
    if skipped_episodes:
        print(f"\nSkipped {len(skipped_episodes)} episodes due to data issues:")
        for ep in skipped_episodes:
            print(f"  - {ep}")
    print(f"\nOutput saved to: {output_path}")
    
    # Print data structure
    print("\nData structure:")
    for key in replay_buffer.data.keys():
        arr = replay_buffer[key]
        if hasattr(arr, 'shape'):
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CogAct dataset to Diffusion Policy zarr format"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/kyji/public/dataset/cogact/1124/trajectories",
        help="Path to input trajectories directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/kyji/public/dataset/cogact/1124/diffusion_policy_data.zarr",
        help="Path to output zarr directory"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="Target resolution for images (width height), e.g., --resolution 640 480"
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=None,
        help="Starting episode number (e.g., 65 to start from episode_0065)"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to convert (for testing)"
    )
    parser.add_argument(
        "--compressor",
        type=str,
        choices=['jpeg2k', 'blosc'],
        default='blosc',
        help="Image compressor to use (default: blosc, more stable)"
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=5,
        help="Compression level (0-100 for jpeg2k, 0-9 for blosc, default: 5)"
    )
    
    args = parser.parse_args()
    
    target_res = None
    if args.resolution is not None:
        target_res = tuple(args.resolution)
    
    convert_dataset(
        input_dir=args.input,
        output_zarr_path=args.output,
        target_resolution=target_res,
        start_episode=args.start_episode,
        max_episodes=args.max_episodes,
        image_compressor=args.compressor,
        compression_level=args.compression_level
    )


if __name__ == "__main__":
    main()
