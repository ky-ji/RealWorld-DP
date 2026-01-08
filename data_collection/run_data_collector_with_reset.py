#!/usr/bin/env python3
"""
数据采集器管理脚本（真机端）
自动处理 Home 按钮复原功能：
1. 初始化机械臂到标准位置
2. 运行数据采集器
3. 检测到 Home 按钮时，kill 数据采集器
4. 运行 init_joints.py 复原机械臂
5. 重新启动数据采集器

注意：
- 此脚本运行在真机端（192.168.31.178）
- Joycon 发送端（joycon_socket_sender.py）运行在仿真端（192.168.31.212）
- Joycon 发送端保持运行，不需要重启
"""
import subprocess
import sys
import time
import signal


def run_init_joints():
    """运行初始化关节脚本"""
    print("\n" + "="*60)
    print("🏠 运行机械臂复原...")
    print("="*60)
    
    work_dir = "/home/kyji/Desktop/Workspace/robot/joycon-robotics/lee_2"
    
    try:
        # 运行 init_joints.py
        result = subprocess.run(
            [sys.executable, "init_joints.py"],
            cwd=work_dir,
            check=True,
            capture_output=False,
            text=True
        )
        print("✓ 机械臂复原完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 机械臂复原失败 (退出码: {e.returncode})")
        return False
    except FileNotFoundError as e:
        print(f"✗ 文件未找到: {e}")
        return False
    except Exception as e:
        print(f"✗ 运行复原脚本出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_data_collector():
    """运行数据采集器"""
    print("\n" + "="*60)
    print("🚀 启动数据采集器...")
    print("="*60)
    
    try:
        # 启动数据采集器进程
        process = subprocess.Popen(
            [sys.executable, "robot_data_collector.py"],
            cwd="/home/kyji/Desktop/Workspace/robot/joycon-robotics/lee_2",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        print(f"✓ 数据采集器已启动 (PID: {process.pid})")
        print("监听输出中...")
        print("-"*60)
        
        # 实时输出数据采集器的日志
        for line in process.stdout:
            print(line, end='')
            
            # 检测到 Home 按钮退出信号
            if "检测到 Home 按钮" in line and "退出程序" in line:
                print("\n" + "="*60)
                print("🏠 检测到 Home 按钮，准备复原...")
                print("="*60)
                
                # 等待进程自然退出
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # 如果5秒内没退出，强制kill
                    print("强制终止数据采集器...")
                    process.kill()
                    process.wait()
                
                print(f"✓ 数据采集器已停止")
                return True  # 返回 True 表示需要复原
        
        # 进程正常退出（不是因为 Home 按钮）
        process.wait()
        print("\n" + "="*60)
        print("数据采集器已退出")
        print("="*60)
        return False  # 返回 False 表示不需要复原
        
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在停止...")
        if process:
            process.terminate()
            process.wait()
        return False
    except Exception as e:
        print(f"✗ 运行数据采集器出错: {e}")
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("数据采集器管理脚本（真机端）")
    print("="*60)
    print("功能：")
    print("  - 首次启动时初始化机械臂")
    print("  - 自动运行数据采集器")
    print("  - 检测 Home 按钮并自动复原")
    print("  - 复原后自动重启数据采集器")
    print("\n注意：")
    print("  - 请确保仿真端的 joycon_socket_sender.py 已经运行")
    print("  - Joycon 保持连接，不会重启")
    print("="*60)
    
    try:
        # 首次启动：初始化机械臂
        print("\n" + "="*60)
        print("🔧 首次启动：初始化机械臂到标准位置...")
        print("="*60)
        
        success = run_init_joints()
        if not success:
            print("\n✗ 初始化失败，退出程序")
            return
        
        print("\n等待 2 秒...")
        time.sleep(2)
        
        # 进入主循环
        while True:
            # 运行数据采集器
            need_reset = run_data_collector()
            
            if not need_reset:
                # 正常退出，不需要复原
                print("\n程序正常退出")
                break
            
            # 需要复原
            print("\n等待 1 秒...")
            time.sleep(1)
            
            # 运行复原脚本
            success = run_init_joints()
            
            if not success:
                print("\n复原失败，退出程序")
                break
            
            # 等待一段时间再重启
            print("\n等待 2 秒后重启数据采集器...")
            time.sleep(2)
            
            print("\n" + "="*60)
            print("🔄 重新启动数据采集器...")
            print("="*60)
    
    except KeyboardInterrupt:
        print("\n\n检测到 Ctrl+C，退出程序")
    except Exception as e:
        print(f"\n程序出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
