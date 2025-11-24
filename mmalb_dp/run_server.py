#!/usr/bin/env python3
"""
推理服务器启动脚本（GPU 服务器端）
运行在 192.168.31.212
等待真机客户端连接进行推理
"""

import sys
import signal
from pathlib import Path

from dp_inference_server import DPInferenceServer
from server_config import SERVER_IP, SERVER_PORT, CHECKPOINT_PATH


def signal_handler(sig, frame):
    """处理 Ctrl+C"""
    print("\n\n检测到 Ctrl+C，正在停止...")
    sys.exit(0)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("Diffusion Policy 推理服务器启动脚本")
    print("="*70)
    print("\n配置信息:")
    print(f"  服务器: {SERVER_IP}:{SERVER_PORT}")
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print("\n功能:")
    print("  1. 加载 DP 模型")
    print("  2. 等待真机客户端连接")
    print("  3. 接收观测 (图像 + 7D 状态)")
    print("  4. 推理得到动作")
    print("  5. 发送 7D 动作给客户端")
    print("\n按 Ctrl+C 停止程序")
    print("="*70 + "\n")
    
    # 检查 checkpoint 文件
    if not Path(CHECKPOINT_PATH).exists():
        print(f"✗ 错误: Checkpoint 文件不存在")
        print(f"  路径: {CHECKPOINT_PATH}")
        print(f"  请在 server_config.py 中修改 CHECKPOINT_PATH")
        sys.exit(1)
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # 创建服务器
        server = DPInferenceServer()
        
        # 运行
        server.run()
        
    except Exception as e:
        print(f"\n✗ 程序出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
