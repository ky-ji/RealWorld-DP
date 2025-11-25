#!/bin/bash

# 从 7D Action 模型微调到 8D Action 模型（包含 gripper 控制）
# 用法: bash finetune_7d_to_8d.sh [GPU_ID] [NUM_EPOCHS]
# 示例: bash finetune_7d_to_8d.sh 0 200

set -e  # 遇到错误立即退出

echo "============================================================"
echo "从 7D 模型微调到 8D 模型（包含 gripper 控制）"
echo "============================================================"

# GPU 配置
GPU_ID=${1:-0}  # 默认使用 GPU 0
NUM_EPOCHS=${2:-200}  # 默认 200 epochs（微调）
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo ""
echo "配置："
echo "  GPU: $GPU_ID"
echo "  训练轮数: $NUM_EPOCHS epochs"
echo ""

# 配置路径
OLD_CKPT="data/outputs/2025.11.24/04.23.47_train_diffusion_transformer_hybrid_cogact_robot_7d/checkpoints/epoch=0550-train_loss=0.017.ckpt"
NEW_CKPT="data/pretrained/cogact_7d_to_8d_init.ckpt"
DATASET_PATH="/home/kyji/public/dataset/cogact/1124/diffusion_policy_data_clean_320x180_8d.zarr"

# 激活 conda 环境
echo "激活 robodiff 环境..."
source ~/storage/anaconda3/etc/profile.d/conda.sh
conda activate robodiff

# ============================================================
# 步骤 1: 检查数据集
# ============================================================
echo ""
echo "步骤 1/3: 检查 8D 数据集"
echo "------------------------------------------------------------"

if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ 错误: 找不到 8D 数据集: $DATASET_PATH"
    echo ""
    echo "请先运行数据转换："
    echo "  python scripts/convert_cogact_to_zarr.py \\"
    echo "      --input /home/kyji/public/dataset/cogact/1124/trajectories \\"
    echo "      --output $DATASET_PATH \\"
    echo "      --resolution 320 180 --start-episode 66"
    exit 1
fi

echo "✓ 数据集已存在: $DATASET_PATH"

# ============================================================
# 步骤 2: 适配旧模型参数
# ============================================================
echo ""
echo "步骤 2/3: 适配模型参数（7D → 8D）"
echo "------------------------------------------------------------"

if [ -f "$NEW_CKPT" ]; then
    echo "⚠ 适配后的模型已存在: $NEW_CKPT"
    read -p "是否重新生成？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳过模型适配，使用现有模型"
    else
        python scripts/finetune_from_7d_to_8d.py \
            --old_ckpt "$OLD_CKPT" \
            --new_ckpt "$NEW_CKPT"
    fi
else
    if [ ! -f "$OLD_CKPT" ]; then
        echo "❌ 错误: 找不到旧模型 checkpoint: $OLD_CKPT"
        exit 1
    fi
    
    echo "旧模型: $OLD_CKPT"
    echo "新模型: $NEW_CKPT"
    echo ""
    
    python scripts/finetune_from_7d_to_8d.py \
        --old_ckpt "$OLD_CKPT" \
        --new_ckpt "$NEW_CKPT"
fi

echo "✓ 模型适配完成"

# ============================================================
# 步骤 3: 开始训练
# ============================================================
echo ""
echo "步骤 3/3: 开始微调训练"
echo "------------------------------------------------------------"
echo ""
echo "训练配置："
echo "  - GPU: $GPU_ID"
echo "  - 数据集: $DATASET_PATH"
echo "  - 预训练权重: $NEW_CKPT"
echo "  - 训练轮数: $NUM_EPOCHS"
echo "  - Action 维度: 8D (pose + gripper)"
echo ""
echo "日志输出: data/outputs/[timestamp]_train_diffusion_transformer_hybrid_cogact_robot_7d/"
echo ""
echo "============================================================"
echo ""

# 运行训练，传递配置参数
# 注意：CUDA_VISIBLE_DEVICES 已经设置，所以 device 应该是 cuda:0
python train.py --config-name=train_cogact_robot \
    task.dataset.zarr_path="$DATASET_PATH" \
    training.device="cuda:0" \
    training.resume=False \
    training.num_epochs=$NUM_EPOCHS \
    training.checkpoint_every=10 \
    exp_name="finetune_8d_gripper"

echo ""
echo "============================================================"
echo "✓ 训练完成！"
echo "============================================================"
echo ""
echo "下一步："
echo "1. 查看训练日志和 checkpoint"
echo "2. 更新推理服务器配置 (mmalb_dp/server_config.py)"
echo "3. 测试新模型的夹爪控制"
echo ""
