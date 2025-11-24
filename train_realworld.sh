#!/bin/bash

# Training script for real-world 7D robot data
# This will train a Diffusion Transformer policy using your collected real-world dataset

echo "Starting training with Diffusion Transformer..."
echo "Dataset: /home/kyji/storage_net/realworld_eval/realworld_data/1119/diffusion_policy_dataset"
echo "Episodes: 50"
echo "Total steps: 6844"
echo "Action space: 7D (x, y, z, rx, ry, rz, gripper)"
echo ""
echo "Model: Diffusion Transformer"
echo "  - 8 layers, 4 heads, 256 embedding dim"
echo "  - ~20M parameters"
echo ""
echo "Training will run for 600 epochs"
echo "Logs will be saved to wandb (offline mode)"
echo "Checkpoints will be saved every 50 epochs"
echo ""

# Run training
python train.py --config-name=train_real_robot_7d
