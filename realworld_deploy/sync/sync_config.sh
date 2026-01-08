#!/bin/bash
# ============================================================
# 同步配置文件
# 将代码同步到真机主机（robot_inference）或服务器（server）
# ============================================================

# ==================== 基础配置 ====================
# 获取脚本所在目录
SYNC_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# realworld_deploy 目录
REALWORLD_DEPLOY_DIR="$(cd "${SYNC_SCRIPT_DIR}/.." && pwd)"

# ==================== 真机主机配置 ====================
# 连接真机主机的配置
ROBOT_HOST="192.168.1.100"       # 真机主机地址（请修改为你的地址）
ROBOT_USER="user"                 # SSH 用户名
ROBOT_PORT=22                     # SSH 端口
ROBOT_SSH_KEY=""                  # SSH 私钥路径（留空使用默认密钥）

# 真机主机上的目标路径
ROBOT_TARGET_DIR="/home/user/robot_inference"  # 请修改为你想要的路径

# ==================== 推理服务器配置 ====================
# 连接推理服务器的配置
SERVER_HOST="115.190.134.186"    # 服务器地址
SERVER_USER="jikangye"           # SSH 用户名
SERVER_PORT=22                   # SSH 端口
SERVER_SSH_KEY="${REALWORLD_DEPLOY_DIR}/robot_inference/keys/id_server"

# 服务器上的目标路径
SERVER_TARGET_DIR="/home/jikangye/workspace/baselines/vla-baselines/RealWorld-DP"

# ==================== 同步配置 ====================
# 排除的文件和目录
EXCLUDE_PATTERNS=(
    "__pycache__"
    "*.pyc"
    "*.pyo"
    "*.pyd"
    ".git"
    "*.log"
    "log/"
    "logs/"
    "*.ckpt"
    "*.pth"
    "data/"
    ".ipynb_checkpoints"
    "*.egg-info"
)

# 构建 rsync 排除参数
build_exclude_args() {
    local args=""
    for pattern in "${EXCLUDE_PATTERNS[@]}"; do
        args="${args} --exclude=${pattern}"
    done
    echo "${args}"
}

# ==================== 辅助函数 ====================
# 显示配置信息
show_config() {
    echo "========================================"
    echo "同步配置"
    echo "========================================"
    echo "本地 realworld_deploy 目录: ${REALWORLD_DEPLOY_DIR}"
    echo ""
    echo "真机主机配置:"
    echo "  主机: ${ROBOT_USER}@${ROBOT_HOST}:${ROBOT_PORT}"
    echo "  目标目录: ${ROBOT_TARGET_DIR}"
    echo ""
    echo "推理服务器配置:"
    echo "  主机: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}"
    echo "  目标目录: ${SERVER_TARGET_DIR}"
    echo "========================================"
}

# 导出变量
export SYNC_SCRIPT_DIR
export REALWORLD_DEPLOY_DIR
export ROBOT_HOST ROBOT_USER ROBOT_PORT ROBOT_SSH_KEY ROBOT_TARGET_DIR
export SERVER_HOST SERVER_USER SERVER_PORT SERVER_SSH_KEY SERVER_TARGET_DIR

