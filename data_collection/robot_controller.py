"""
机械臂控制器模块
封装frankx的Robot和ImpedanceMotion，提供高层控制接口
"""
import sys
import os
from time import sleep
from threading import Lock, Thread
from typing import Optional, Tuple

# 添加frankx路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../frankx'))
from frankx import Affine, ImpedanceMotion, Robot, JointMotion


class RobotController:
    """机械臂控制器"""
    
    def __init__(self, 
                 host: str = '172.16.0.2',
                 translational_stiffness: float = 200.0,
                 rotational_stiffness: float = 20.0):
        """
        初始化机械臂控制器
        
        Args:
            host: 机械臂IP地址
            translational_stiffness: 平移刚度（越小越柔和）
            rotational_stiffness: 旋转刚度（越小越柔和）
        """
        self.host = host
        self.robot = None
        self.gripper = None
        self.impedance_motion = None
        self.robot_thread = None
        self.control_lock = Lock()
        
        # 刚度参数
        self.translational_stiffness = translational_stiffness
        self.rotational_stiffness = rotational_stiffness
        
        # 状态标志
        self.is_initialized = False
        self.is_running = False
        self.is_resetting = False  # 复原中标志
        
        print(f"[机械臂控制器] 初始化")
        print(f"  IP: {host}")
        print(f"  平移刚度: {translational_stiffness}")
        print(f"  旋转刚度: {rotational_stiffness}")
    
    def connect(self) -> bool:
        """连接到机械臂"""
        try:
            print(f"[机械臂控制器] 正在连接到机械臂 {self.host}...")
            self.robot = Robot(self.host)
            self.robot.set_default_behavior()
            self.robot.recover_from_errors()
            self.robot.set_dynamic_rel(0.06)  # 设置动态速度为0.06（降低5倍）
            
            # 获取夹爪并设置参数
            self.gripper = self.robot.get_gripper()
            self.gripper.gripper_force = 10.0  # 设置夹爪力为10N（默认20N）
            self.gripper.gripper_speed = 0.05  # 设置夹爪速度
            print(f"[机械臂控制器] ✓ 夹爪已连接 (力: 10N, 速度: 0.05m/s)")
            
            # 获取当前位姿
            current_pose = self.robot.current_pose()
            print(f"[机械臂控制器] ✓ 连接成功")
            print(f"  当前位置: X={current_pose.x:.3f}, Y={current_pose.y:.3f}, Z={current_pose.z:.3f}")
            print(f"  当前姿态: A={current_pose.a:.3f}, B={current_pose.b:.3f}, C={current_pose.c:.3f}")
            
            return True
        except Exception as e:
            print(f"[机械臂控制器] ✗ 连接失败: {e}")
            return False
    
    def start_impedance_control(self) -> bool:
        """启动阻抗控制"""
        if not self.robot:
            print(f"[机械臂控制器] 错误: 机械臂未连接")
            return False
        
        try:
            # 创建阻抗运动
            self.impedance_motion = ImpedanceMotion(
                translational_stiffness=self.translational_stiffness,
                rotational_stiffness=self.rotational_stiffness
            )
            
            # 获取当前姿态作为初始目标
            current_pose = self.robot.current_pose()
            self.impedance_motion.target = current_pose
            
            # 启动异步运动
            print(f"[机械臂控制器] 启动阻抗控制...")
            self.robot_thread = self.robot.move_async(self.impedance_motion)
            
            # 等待激活（重试机制）
            max_retries = 10
            for i in range(max_retries):
                sleep(0.2)
                if self.impedance_motion.is_active:
                    print(f"[机械臂控制器] ✓ 阻抗控制已激活 (尝试 {i+1}/{max_retries})")
                    self.is_initialized = True
                    self.is_running = True
                    return True
            
            print(f"[机械臂控制器] ✗ 阻抗控制未激活 (已尝试 {max_retries} 次)")
            return False
                
        except Exception as e:
            print(f"[机械臂控制器] 启动阻抗控制失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def update_target(self, x: float, y: float, z: float, 
                     a: float, b: float, c: float) -> bool:
        """
        更新目标位姿（6D）
        
        Args:
            x, y, z: 目标位置（米）
            a, b, c: 目标姿态（弧度，欧拉角）
            
        Returns:
            True if successful, False otherwise
        """
        # 如果正在复原，直接返回True但不更新目标
        if self.is_resetting:
            return True
        
        if not self.is_running or not self.impedance_motion:
            print(f"[机械臂控制器] 警告: 阻抗控制未运行")
            return False
        
        if not self.impedance_motion.is_active:
            print(f"[机械臂控制器] 警告: 阻抗控制未激活")
            return False
        
        try:
            with self.control_lock:
                # 创建新的目标位姿
                new_target = Affine(x, y, z, a, b, c)
                
                # 更新目标
                self.impedance_motion.target = new_target
            
            return True
            
        except Exception as e:
            print(f"[机械臂控制器] 更新目标失败: {e}")
            return False
    
    def update_target_position_only(self, x: float, y: float, z: float) -> bool:
        """
        只更新目标位置，保持当前姿态
        
        Args:
            x, y, z: 目标位置（米）
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_running or not self.impedance_motion:
            return False
        
        try:
            with self.control_lock:
                current_target = self.impedance_motion.target
                new_target = Affine(
                    x, y, z,
                    current_target.a,
                    current_target.b,
                    current_target.c
                )
                self.impedance_motion.target = new_target
            return True
        except Exception as e:
            print(f"[机械臂控制器] 更新位置失败: {e}")
            return False
    
    def get_current_pose(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        获取当前位姿
        
        Returns:
            (x, y, z, a, b, c) or None if failed
        """
        if not self.robot:
            return None
        
        try:
            pose = self.robot.current_pose()
            return (pose.x, pose.y, pose.z, pose.a, pose.b, pose.c)
        except Exception as e:
            print(f"[机械臂控制器] 获取当前位姿失败: {e}")
            return None
    
    def get_target_pose(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        获取目标位姿
        
        Returns:
            (x, y, z, a, b, c) or None if failed
        """
        if not self.impedance_motion:
            return None
        
        try:
            with self.control_lock:
                target = self.impedance_motion.target
                return (target.x, target.y, target.z, target.a, target.b, target.c)
        except Exception as e:
            print(f"[机械臂控制器] 获取目标位姿失败: {e}")
            return None
    
    def set_gripper(self, width: float) -> bool:
        """
        设置夹爪宽度
        
        Args:
            width: 夹爪宽度 [0.0, 1.0]，0为完全闭合，1为完全打开
        
        Returns:
            是否成功
        """
        if not self.gripper:
            print(f"[机械臂控制器] 错误: 夹爪未初始化")
            return False
        
        try:
            # 将0-1映射到实际夹爪宽度（假设最大宽度为0.08m）
            actual_width = width * 0.08
            self.gripper.move_async(actual_width)
            return True
        except Exception as e:
            print(f"[机械臂控制器] 夹爪控制失败: {e}")
            return False
    
    def reset_to_initial_joints(self) -> bool:
        """
        复原到初始关节位置（完全重启方式）
        
        此方法会：
        1. 完全停止当前阻抗控制
        2. 断开所有连接
        3. 创建新连接并执行复原
        4. 重新启动阻抗控制
        
        相当于重启整个控制器
        
        Returns:
            是否成功
        """
        try:
            print(f"[机械臂控制器] 🏠 开始完全复原...")
            
            # 标记为复原中
            self.is_resetting = True
            
            # 1. 完全停止当前控制
            print(f"[机械臂控制器] 停止阻抗控制...")
            self.is_running = False
            
            if self.impedance_motion:
                try:
                    self.impedance_motion.finish()
                except:
                    pass
                self.impedance_motion = None
            
            if self.robot_thread:
                try:
                    self.robot_thread.join(timeout=1.0)
                except:
                    pass
                self.robot_thread = None
            
            # 2. 断开机器人连接
            print(f"[机械臂控制器] 断开机器人连接...")
            if self.robot:
                try:
                    del self.robot
                except:
                    pass
                self.robot = None
            
            if self.gripper:
                try:
                    del self.gripper
                except:
                    pass
                self.gripper = None
            
            sleep(0.5)
            
            # 3. 重新连接并复原
            print(f"[机械臂控制器] 重新连接机器人...")
            self.robot = Robot(self.host)
            self.robot.set_default_behavior()
            self.robot.recover_from_errors()
            
            # Franka Panda 标准初始位置（弧度）
            initial_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
            
            # 设置快速运动（50%速度）
            self.robot.set_dynamic_rel(0.5)
            
            # 移动到初始关节位置
            print(f"[机械臂控制器] 移动到初始位置...")
            success = self.robot.move(JointMotion(initial_joints))
            
            if not success:
                print(f"[机械臂控制器] ✗ 复原失败")
                self.is_resetting = False
                return False
            
            print(f"[机械臂控制器] ✓ 复原完成！")
            
            # 恢复正常速度
            self.robot.set_dynamic_rel(0.3)
            
            # 4. 重新获取夹爪
            print(f"[机械臂控制器] 重新连接夹爪...")
            self.gripper = self.robot.get_gripper()
            self.gripper.gripper_force = 10.0
            self.gripper.gripper_speed = 0.05
            
            # 5. 重新启动阻抗控制（目标设为当前初始位置）
            print(f"[机械臂控制器] 重新启动阻抗控制...")
            sleep(0.5)
            
            # 获取当前位姿（此时已经在初始位置）
            initial_pose = self.robot.current_pose()
            print(f"[机械臂控制器] 初始位姿: x={initial_pose.x:.3f}, y={initial_pose.y:.3f}, z={initial_pose.z:.3f}")
            
            self.impedance_motion = ImpedanceMotion(
                translational_stiffness=self.translational_stiffness,
                rotational_stiffness=self.rotational_stiffness
            )
            
            # 重要：目标设为当前初始位置，防止飞回原位
            self.impedance_motion.target = initial_pose
            
            self.robot_thread = self.robot.move_async(self.impedance_motion)
            
            # 等待激活（重试机制）
            max_retries = 10
            for i in range(max_retries):
                sleep(0.2)
                if self.impedance_motion.is_active:
                    self.is_running = True
                    self.is_initialized = True
                    print(f"[机械臂控制器] ✓ 阻抗控制已重启 (尝试 {i+1}/{max_retries})")
                    print(f"[机械臂控制器] ✓ 完全复原成功！阻抗控制目标已设为初始位置")
                    self.is_resetting = False
                    return True
            
            print(f"[机械臂控制器] ✗ 阻抗控制未激活 (已尝试 {max_retries} 次)")
            self.is_resetting = False
            return False
                
        except Exception as e:
            print(f"[机械臂控制器] 复原出错: {e}")
            import traceback
            traceback.print_exc()
            self.is_resetting = False
            return False
    
    def stop(self):
        """停止控制"""
        print(f"[机械臂控制器] 正在停止...")
        self.is_running = False
        
        # 停止阻抗控制
        if self.impedance_motion:
            try:
                self.impedance_motion.finish()
                print(f"[机械臂控制器] 阻抗控制已停止")
            except Exception as e:
                print(f"[机械臂控制器] 停止阻抗控制时出错: {e}")
        
        # 等待线程结束
        if self.robot_thread:
            try:
                self.robot_thread.join(timeout=2.0)
            except:
                pass
        
        print(f"[机械臂控制器] 已停止")
    
    def __del__(self):
        """析构函数"""
        if self.is_running:
            self.stop()
