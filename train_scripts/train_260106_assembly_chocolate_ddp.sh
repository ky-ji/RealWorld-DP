#!/bin/bash

# ============================================================
# 多卡DDP训练启动脚本
# 支持指定GPU数量和设备ID
# ============================================================

# 使用说明函数
usage() {
    echo "使用方法:"
    echo "  bash train_scripts/train_ddp.sh [GPU数量] [GPU设备ID(可选)]"
    echo ""
    echo "示例:"
    echo "  bash train_scripts/train_ddp.sh 8              # 使用8卡（默认GPU 0-7）"
    echo "  bash train_scripts/train_ddp.sh 4              # 使用4卡（默认GPU 0-3）"
    echo "  bash train_scripts/train_ddp.sh 4 0,1,2,3      # 指定使用GPU 0,1,2,3"
    echo "  bash train_scripts/train_ddp.sh 4 2,3,4,5      # 指定使用GPU 2,3,4,5"
    echo "  bash train_scripts/train_ddp.sh 1              # 单卡训练（测试用）"
    echo ""
    exit 1
}

# 检查参数
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
fi

# 获取GPU数量（默认8）
NUM_GPUS=${1:-8}

# 获取GPU设备ID（可选）
GPU_DEVICES=${2:-""}

# 验证GPU数量
if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || [ "$NUM_GPUS" -lt 1 ]; then
    echo "❌ 错误: GPU数量必须是大于0的整数"
    usage
fi

echo "============================================================"
echo "多卡DDP训练 - Diffusion Policy"
echo "============================================================"
echo ""
echo "GPU数量: $NUM_GPUS"

# 如果指定了GPU设备ID
if [ -n "$GPU_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_DEVICES
    echo "指定GPU: $GPU_DEVICES"
    
    # 验证指定的GPU数量是否匹配
    IFS=',' read -ra GPUS <<< "$GPU_DEVICES"
    ACTUAL_GPU_COUNT=${#GPUS[@]}
    if [ "$ACTUAL_GPU_COUNT" -ne "$NUM_GPUS" ]; then
        echo "⚠️  警告: 指定的GPU数量($ACTUAL_GPU_COUNT)与参数($NUM_GPUS)不匹配"
        echo "    将使用实际GPU数量: $ACTUAL_GPU_COUNT"
        NUM_GPUS=$ACTUAL_GPU_COUNT
    fi
else
    echo "使用GPU: 默认前${NUM_GPUS}个可用GPU"
fi

echo "配置文件: train_assembly_chocolate_ddp.yaml"
echo ""

# 激活conda环境
echo "激活robodiff环境..."
source ~/tools/miniconda3/etc/profile.d/conda.sh
conda activate robodiff


# 设置CUDA库路径（解决libcuda.so找不到的问题）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
# 设置NCCL的CUDA路径,避免在运行时搜索
export NCCL_CUDA_PATH=/usr

# 设置NCCL环境变量（优化通信性能）
export NCCL_DEBUG=WARN  # 可以设置为INFO查看详细信息
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=eth0

# 启动训练
echo "启动 ${NUM_GPUS}卡 DDP训练..."
echo "命令: torchrun --nproc_per_node=$NUM_GPUS --master_port=50000 train_ddp.py --config-name=train_assembly_chocolate_ddp"
echo ""

# 使用torchrun启动分布式训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=50000 \
    train_ddp.py \
    --config-name=train_assembly_chocolate_ddp

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练完成"
else
    echo "❌ 训练中断或出错 (退出码: $EXIT_CODE)"
fi
echo "============================================================"

exit $EXIT_CODE

