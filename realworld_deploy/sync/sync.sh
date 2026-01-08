#!/bin/bash
# ============================================================
# 通用同步脚本
# 支持同步到真机主机(robot)或推理服务器(server)
# ============================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 加载配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sync_config.sh"

# 显示帮助
show_help() {
    echo "用法: $0 <目标> [选项]"
    echo ""
    echo "目标:"
    echo "  robot       同步 robot_inference 代码到真机主机"
    echo "  server      同步 server 代码到推理服务器"
    echo "  all         同步所有代码到两个目标"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示此帮助信息"
    echo "  -d, --dry-run       仅显示将要同步的文件（不执行）"
    echo "  -c, --config        显示当前配置"
    echo "  --robot-dir DIR     覆盖真机目标目录"
    echo "  --server-dir DIR    覆盖服务器目标目录"
    echo ""
    echo "示例:"
    echo "  $0 robot                      # 同步到真机主机"
    echo "  $0 server                     # 同步到推理服务器"
    echo "  $0 robot -d                   # 预览同步到真机的文件"
    echo "  $0 robot --robot-dir /tmp/test  # 同步到指定目录"
    echo ""
    echo "配置文件: ${SCRIPT_DIR}/sync_config.sh"
}

# 构建 SSH 命令
build_ssh_cmd() {
    local host=$1
    local user=$2
    local port=$3
    local password=$4
    
    local cmd="ssh -p $port ${user}@${host}"
    echo "$cmd"
}

# 构建 rsync 命令
build_rsync_cmd() {
    local host=$1
    local user=$2
    local port=$3
    local password=$4
    local dry_run=$5
    
    local ssh_cmd="ssh -p $port"
    local ssh_opts="-e \"$ssh_cmd\""
    
    local cmd="rsync -avz --progress"
    if [ "$dry_run" = "true" ]; then
        cmd="$cmd --dry-run"
    fi
    
    echo "$cmd $ssh_opts"
}

# 检查 SSH 连接
check_ssh() {
    local host=$1
    local user=$2
    local port=$3
    local password=$4
    local name=$5
    
    echo -e "${BLUE}[检查] ${name} SSH 连接...${NC}"
    echo -e "${YELLOW}  请输入密码:${NC}"
    
    local ssh_cmd=$(build_ssh_cmd "$host" "$user" "$port" "$password")
    
    # 不重定向输出，让用户看到密码提示
    if eval "$ssh_cmd \"echo 'ok'\"" 2>&1; then
        echo -e "${GREEN}✓ ${name} 连接成功${NC}"
        return 0
    else
        echo -e "${RED}✗ ${name} 连接失败${NC}"
        echo "  请检查配置: ${host}:${port}"
        return 1
    fi
}

# 同步到真机主机
sync_to_robot() {
    local dry_run=$1
    local target_dir="${ROBOT_TARGET_DIR}"
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}同步 robot_inference 到真机主机${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "目标: ${ROBOT_USER}@${ROBOT_HOST}:${target_dir}"
    echo ""
    
    # 检查连接
    if ! check_ssh "$ROBOT_HOST" "$ROBOT_USER" "$ROBOT_PORT" "$ROBOT_PASSWORD" "真机主机"; then
        return 1
    fi
    
    # 创建远程目录
    echo -e "${BLUE}[创建] 远程目录...${NC}"
    local ssh_cmd=$(build_ssh_cmd "$ROBOT_HOST" "$ROBOT_USER" "$ROBOT_PORT" "$ROBOT_PASSWORD")
    eval "$ssh_cmd \"mkdir -p ${target_dir}\""
    
    # 构建排除参数
    local exclude_args=$(build_exclude_args)
    
    # 同步 robot_inference 目录
    echo -e "${BLUE}[同步] robot_inference/...${NC}"
    local rsync_base="rsync -avz --progress"
    if [ "$dry_run" = "true" ]; then
        rsync_base="$rsync_base --dry-run"
        echo -e "${YELLOW}(预览模式 - 不会实际传输文件)${NC}"
    fi
    
    local ssh_cmd="ssh -p ${ROBOT_PORT}"
    local ssh_opt="-e \"$ssh_cmd\""
    
    eval $rsync_base $ssh_opt $exclude_args \
        "${REALWORLD_DEPLOY_DIR}/robot_inference/" \
        "${ROBOT_USER}@${ROBOT_HOST}:${target_dir}/"
    
    echo ""
    echo -e "${GREEN}✓ robot_inference 同步完成！${NC}"
    echo ""
    echo "使用方法:"
    echo "  ssh ${ROBOT_USER}@${ROBOT_HOST}"
    echo "  cd ${target_dir}"
    echo "  python inference_client.py"
}

# 同步到推理服务器
sync_to_server() {
    local dry_run=$1
    local target_dir="${SERVER_TARGET_DIR}"
    
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}同步代码到推理服务器${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "目标: ${SERVER_USER}@${SERVER_HOST}:${target_dir}"
    echo ""
    
    # 检查连接
    if ! check_ssh "$SERVER_HOST" "$SERVER_USER" "$SERVER_PORT" "$SERVER_PASSWORD" "推理服务器"; then
        return 1
    fi
    
    # 创建远程目录
    echo -e "${BLUE}[创建] 远程目录...${NC}"
    local ssh_cmd=$(build_ssh_cmd "$SERVER_HOST" "$SERVER_USER" "$SERVER_PORT" "$SERVER_PASSWORD")
    eval "$ssh_cmd \"mkdir -p ${target_dir}/realworld_deploy\""
    
    # 构建排除参数
    local exclude_args=$(build_exclude_args)
    
    # 同步整个 realworld_deploy 目录
    echo -e "${BLUE}[同步] realworld_deploy/...${NC}"
    local rsync_base="rsync -avz --progress"
    if [ "$dry_run" = "true" ]; then
        rsync_base="$rsync_base --dry-run"
        echo -e "${YELLOW}(预览模式 - 不会实际传输文件)${NC}"
    fi
    
    local ssh_cmd="ssh -p ${SERVER_PORT}"
    local ssh_opt="-e \"$ssh_cmd\""
    
    eval $rsync_base $ssh_opt $exclude_args \
        "${REALWORLD_DEPLOY_DIR}/" \
        "${SERVER_USER}@${SERVER_HOST}:${target_dir}/realworld_deploy/"
    
    echo ""
    echo -e "${GREEN}✓ 服务器代码同步完成！${NC}"
    echo ""
    echo "启动推理服务器:"
    echo "  ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
    echo "  cd ${target_dir}"
    echo "  python realworld_deploy/server/dp_inference_server_ssh.py"
}

# 主函数
main() {
    local target=""
    local dry_run="false"
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -d|--dry-run)
                dry_run="true"
                shift
                ;;
            -c|--config)
                show_config
                exit 0
                ;;
            --robot-dir)
                ROBOT_TARGET_DIR="$2"
                shift 2
                ;;
            --server-dir)
                SERVER_TARGET_DIR="$2"
                shift 2
                ;;
            robot|server|all)
                target="$1"
                shift
                ;;
            *)
                echo -e "${RED}未知参数: $1${NC}"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查目标
    if [ -z "$target" ]; then
        echo -e "${RED}错误: 请指定同步目标 (robot/server/all)${NC}"
        echo ""
        show_help
        exit 1
    fi
    
    # 执行同步
    case $target in
        robot)
            sync_to_robot "$dry_run"
            ;;
        server)
            sync_to_server "$dry_run"
            ;;
        all)
            sync_to_robot "$dry_run"
            sync_to_server "$dry_run"
            ;;
    esac
}

main "$@"

