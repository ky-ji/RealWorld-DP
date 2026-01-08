"""
机械臂数据采集器
运行在 192.168.31.178
负责接收Joycon数据并控制机械臂，同时记录完整的轨迹数据用于训练
"""
import socket
import time
import numpy as np
import pickle
import json
import cv2
from pathlib import Path
from threading import Thread, Lock, Event
from datetime import datetime
from typing import Optional, List, Dict

from socket_config import *
from joycon_data_protocol import JoyconDataPacket, is_heartbeat_packet
from coordinate_mapper import create_default_mapper, CoordinateMapper
from robot_controller import RobotController
from camera_manager import CameraManager


class TrajectoryEpisode:
    """单个轨迹episode数据"""
    
    def __init__(self, episode_id: int, save_dir: Path):
        self.episode_id = episode_id
        self.start_time = time.time()
        self.data_points = []
        self.save_dir = save_dir
        
        # 创建episode文件夹和images子文件夹
        self.episode_folder = save_dir / f'episode_{self.episode_id:04d}'
        self.images_folder = self.episode_folder / 'images'
        self.images_folder.mkdir(parents=True, exist_ok=True)
        
        self.image_count = 0
        
    def add_data_point(self, 
                       timestamp: float,
                       joycon_pose: np.ndarray,
                       joycon_gripper: float,
                       robot_obs_pose: np.ndarray,
                       robot_obs_gripper: float,
                       robot_action_pose: np.ndarray,
                       robot_action_gripper: float,
                       image_index: int = -1):
        """
        添加数据点
        
        Args:
            timestamp: 相对时间戳（秒）
            joycon_pose: Joycon 6D位姿 [x, y, z, roll, pitch, yaw]
            joycon_gripper: Joycon 夹爪值 [0.0, 1.0]
            robot_obs_pose: 机械臂观测位姿 [x, y, z, a, b, c]
            robot_obs_gripper: 机械臂观测夹爪宽度（米）
            robot_action_pose: 机械臂动作位姿 [x, y, z, a, b, c]
            robot_action_gripper: 机械臂动作夹爪宽度（米）
            image_index: 对应的图像索引，-1表示无图像
        """
        data_point = {
            'timestamp': timestamp,
            'joycon_pose': joycon_pose,
            'joycon_gripper': joycon_gripper,
            'robot_obs_pose': robot_obs_pose,
            'robot_obs_gripper': robot_obs_gripper,
            'robot_action_pose': robot_action_pose,
            'robot_action_gripper': robot_action_gripper,
            'image_index': image_index,
        }
        self.data_points.append(data_point)
    
    def save_image(self, image: np.ndarray) -> int:
        """
        保存图像到images文件夹
        
        Args:
            image: 图像数据 (BGR格式)
            
        Returns:
            int: 图像索引
        """
        image_path = self.images_folder / f'frame_{self.image_count:04d}.jpg'
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        current_index = self.image_count
        self.image_count += 1
        return current_index
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        n_steps = len(self.data_points)
        
        # 转换为numpy数组
        episode_data = {
            'episode_id': self.episode_id,
            'start_time': self.start_time,
            'duration': self.data_points[-1]['timestamp'] if n_steps > 0 else 0.0,
            'n_steps': n_steps,
            'n_images': self.image_count,
            
            # 时间戳
            'timestamp': np.array([d['timestamp'] for d in self.data_points]),
            
            # Joycon 数据（原始输入）
            'joycon_pose': np.array([d['joycon_pose'] for d in self.data_points]),
            'joycon_gripper': np.array([d['joycon_gripper'] for d in self.data_points]),
            
            # 机械臂观测（实际状态）
            'robot_obs_pose': np.array([d['robot_obs_pose'] for d in self.data_points]),
            'robot_obs_gripper': np.array([d['robot_obs_gripper'] for d in self.data_points]),
            
            # 机械臂动作（目标状态）
            'robot_action_pose': np.array([d['robot_action_pose'] for d in self.data_points]),
            'robot_action_gripper': np.array([d['robot_action_gripper'] for d in self.data_points]),
            
            # 图像索引
            'image_index': np.array([d['image_index'] for d in self.data_points]),
        }
        
        return episode_data
    
    def save(self):
        """
        保存episode数据
        
        文件结构：
        episode_0001/
          data.pkl
          meta.json
          images/
            frame_0000.jpg
            frame_0001.jpg
            ...
        """
        episode_data = self.to_dict()
        
        # 保存为pickle格式（完整数据）
        pkl_file = self.episode_folder / 'data.pkl'
        with open(pkl_file, 'wb') as f:
            pickle.dump(episode_data, f)
        
        # 保存元数据为JSON（便于查看）
        meta_data = {
            'episode_id': self.episode_id,
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'duration': episode_data['duration'],
            'n_steps': episode_data['n_steps'],
            'n_images': episode_data['n_images'],
            'data_shapes': {
                'timestamp': episode_data['timestamp'].shape,
                'joycon_pose': episode_data['joycon_pose'].shape,
                'robot_obs_pose': episode_data['robot_obs_pose'].shape,
                'robot_action_pose': episode_data['robot_action_pose'].shape,
                'image_index': episode_data['image_index'].shape,
            }
        }
        
        json_file = self.episode_folder / 'meta.json'
        with open(json_file, 'w') as f:
            json.dump(meta_data, f, indent=2)
        
        return self.episode_folder, pkl_file, json_file


class RobotDataCollector:
    """机械臂数据采集器"""
    
    def __init__(self, 
                 host: str = '', 
                 port: int = SOCKET_PORT,
                 robot_host: str = '172.16.0.2',
                 save_dir: str = 'data/trajectories',
                 enable_robot_control: bool = True,
                 enable_camera: bool = True,
                 camera_index: int = 0,
                 camera_resolution: tuple = (1920, 1080),
                 data_collection_freq: float = 10.0):
        """
        初始化数据采集器
        
        Args:
            host: 监听地址（''表示监听所有网卡）
            port: 端口号
            robot_host: 机械臂IP地址
            save_dir: 数据保存目录
            enable_robot_control: 是否启用机械臂控制
            enable_camera: 是否启用摄像头
            camera_index: 摄像头索引
            camera_resolution: 摄像头分辨率 (width, height)
            data_collection_freq: 数据采集频率 (Hz)
        """
        self.host = host
        self.port = port
        self.robot_host = robot_host
        self.enable_robot_control = enable_robot_control
        self.enable_camera = enable_camera
        self.data_collection_freq = data_collection_freq
        
        # 数据保存目录
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 摄像头
        self.camera_manager = None
        if enable_camera:
            self.camera_manager = CameraManager(
                camera_index=camera_index,
                resolution=camera_resolution,
                fps=30
            )
        
        # Socket相关
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.running = False
        
        # 线程控制
        self.receive_thread = None
        self.control_thread = None
        self.stop_event = Event()
        
        # 数据缓冲
        self.data_lock = Lock()
        self.latest_packet = None
        self.data_buffer = ""
        
        # 机械臂控制
        self.robot_controller = None
        self.coordinate_mapper = None
        self.initial_calibration_done = False
        self.joycon_initial_pose = None
        self.robot_initial_pose = None
        
        # 夹爪控制
        self.last_gripper_value = None
        self.cached_gripper_width = 0.08  # 缓存的夹爪宽度
        self.gripper_read_counter = 0  # 夹爪读取计数器
        
        # 按钮控制
        self.last_x_button_value = 0  # X按钮状态
        self.last_r_button_value = 0  # R按钮状态
        self.last_home_button_value = 0  # Home按钮状态
        
        # 防抖动：记录上次触发时间
        self.last_trigger_time = {}
        self.debounce_delay = 0.2  # 200ms 防抖动延迟
        
        # 数据采集
        self.current_episode: Optional[TrajectoryEpisode] = None
        self.episode_count = self._get_next_episode_id() - 1  # 减1，因为start_recording会+1
        self.is_recording = False
        self.recording_lock = Lock()
        
        # 统计信息
        self.packets_received = 0
        self.control_updates = 0
        self.total_data_points = 0
        self.last_receive_time = 0
        self.start_time = time.time()
        
        print(f"[数据采集器] 初始化完成")
        print(f"[数据采集器] Socket监听: {self.host if self.host else '0.0.0.0'}:{self.port}")
        print(f"[数据采集器] 数据保存目录: {self.save_dir.absolute()}")
        print(f"[数据采集器] 下一个Episode编号: {self.episode_count + 1}")
        print(f"[数据采集器] 机械臂控制: {'启用' if enable_robot_control else '禁用'}")
        print(f"[数据采集器] 摄像头: {'启用' if enable_camera else '禁用'}")
        print(f"[数据采集器] 数据采集频率: {data_collection_freq} Hz")
    
    def _get_next_episode_id(self) -> int:
        """
        检测已存在的episode文件夹，返回下一个可用的episode ID
        
        Returns:
            int: 下一个episode ID（从已有的最大ID+1开始）
        """
        existing_episodes = []
        
        # 扫描保存目录中的所有episode文件夹
        if self.save_dir.exists():
            for item in self.save_dir.iterdir():
                if item.is_dir() and item.name.startswith('episode_'):
                    try:
                        # 提取episode编号
                        episode_num = int(item.name.split('_')[1])
                        existing_episodes.append(episode_num)
                    except (IndexError, ValueError):
                        # 忽略格式不正确的文件夹
                        continue
        
        # 如果没有已存在的episode，从1开始
        if not existing_episodes:
            return 1
        
        # 从最大编号+1开始
        max_episode = max(existing_episodes)
        next_id = max_episode + 1
        
        print(f"[数据采集器] 检测到 {len(existing_episodes)} 个已存在的episode")
        print(f"[数据采集器] 最大编号: {max_episode}，下一个编号: {next_id}")
        
        return next_id
    
    def init_robot_control(self) -> bool:
        """初始化机械臂控制"""
        if not self.enable_robot_control:
            print(f"[数据采集器] 机械臂控制已禁用")
            return True
        
        try:
            # 创建坐标映射器
            self.coordinate_mapper = create_default_mapper()
            
            # 创建机械臂控制器
            self.robot_controller = RobotController(
                host=self.robot_host,
                translational_stiffness=200.0,
                rotational_stiffness=20.0
            )
            
            # 连接机械臂
            if not self.robot_controller.connect():
                return False
            
            # 保存机械臂初始位姿
            self.robot_initial_pose = self.robot_controller.get_current_pose()
            if self.robot_initial_pose:
                print(f"[数据采集器] 机械臂初始位姿: X={self.robot_initial_pose[0]:.3f}, "
                      f"Y={self.robot_initial_pose[1]:.3f}, Z={self.robot_initial_pose[2]:.3f}")
            else:
                print(f"[数据采集器] ✗ 无法获取机械臂初始位姿")
                return False
            
            # 启动阻抗控制
            if not self.robot_controller.start_impedance_control():
                return False
            
            print(f"[数据采集器] ✓ 机械臂控制初始化成功")
            print(f"[数据采集器] 等待第一个Joycon数据包进行校准...")
            return True
            
        except Exception as e:
            print(f"[数据采集器] ✗ 机械臂控制初始化失败: {e}")
            return False
    
    def start_server(self) -> bool:
        """启动服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"[数据采集器] ✓ 服务器启动成功，等待连接...")
            return True
        except Exception as e:
            print(f"[数据采集器] ✗ 服务器启动失败: {e}")
            return False
    
    def wait_for_client(self) -> bool:
        """等待客户端连接"""
        try:
            print(f"[数据采集器] 等待仿真端连接...")
            self.client_socket, self.client_address = self.server_socket.accept()
            print(f"[数据采集器] ✓ 客户端已连接: {self.client_address}")
            return True
        except Exception as e:
            print(f"[数据采集器] ✗ 等待连接失败: {e}")
            return False
    
    def receive_loop(self):
        """接收数据循环"""
        print(f"[数据采集器] 接收线程启动")
        
        while self.running and not self.stop_event.is_set():
            try:
                data = self.client_socket.recv(BUFFER_SIZE)
                if not data:
                    print(f"[数据采集器] 客户端断开连接")
                    break
                
                self.data_buffer += data.decode(ENCODING)
                
                while '\n' in self.data_buffer:
                    line, self.data_buffer = self.data_buffer.split('\n', 1)
                    
                    if line.strip():
                        self.process_message(line.strip())
                
            except Exception as e:
                if self.running:
                    print(f"[数据采集器] 接收数据错误: {e}")
                break
        
        print(f"[数据采集器] 接收线程结束")
    
    def process_message(self, message: str):
        """处理接收到的消息"""
        try:
            if is_heartbeat_packet(message):
                return
            
            packet = JoyconDataPacket.from_json(message)
            
            # 更新最新数据
            with self.data_lock:
                self.latest_packet = packet
                self.packets_received += 1
                self.last_receive_time = time.time()
            
        except Exception as e:
            print(f"[数据采集器] 处理消息错误: {e}")
    
    def start_recording(self):
        """开始记录新的episode"""
        with self.recording_lock:
            if self.is_recording:
                print(f"[数据采集器] ⚠️  已经在记录中")
                return
            
            self.episode_count += 1
            self.current_episode = TrajectoryEpisode(self.episode_count, self.save_dir)
            self.is_recording = True
            print(f"\n[数据采集器] 🔴 开始记录 Episode {self.episode_count}")
    
    def stop_recording(self):
        """停止记录并保存当前episode"""
        with self.recording_lock:
            if not self.is_recording or self.current_episode is None:
                print(f"[数据采集器] ⚠️  没有正在记录的episode")
                return
            
            self.is_recording = False
            
            # 保存episode
            if len(self.current_episode.data_points) > 0:
                episode_folder, pkl_file, json_file = self.current_episode.save()
                n_steps = len(self.current_episode.data_points)
                n_images = self.current_episode.image_count
                duration = self.current_episode.data_points[-1]['timestamp']
                
                print(f"[数据采集器] ✓ Episode {self.episode_count} 已保存")
                print(f"  数据点数: {n_steps}")
                print(f"  图像数: {n_images}")
                print(f"  持续时间: {duration:.2f}秒")
                print(f"  数据频率: {n_steps/duration:.1f} Hz")
                print(f"  图像频率: {n_images/duration:.1f} Hz")
                print(f"  保存位置: {episode_folder}")
                
                self.total_data_points += n_steps
            else:
                print(f"[数据采集器] ⚠️  Episode {self.episode_count} 没有数据")
            
            self.current_episode = None
    
    def control_loop(self):
        """机械臂控制循环（同时记录数据）"""
        if not self.enable_robot_control:
            return
        
        print(f"[数据采集器] 控制线程启动")
        print(f"\n{'='*60}")
        print(f"数据采集控制说明:")
        print(f"  Joycon R 按钮: 🎬 开始/停止录制（可连续录制多个轨迹）")
        print(f"  Joycon ZR 按钮: 🤏 控制夹爪开关")
        print(f"  Joycon Home 按钮: 🏠 复原机械臂到初始关节位置")
        print(f"  Joycon X 按钮: ❌ 退出程序")
        print(f"  控制频率: 100 Hz")
        print(f"  数据采集频率: {self.data_collection_freq} Hz")
        print(f"{'='*60}\n")
        
        # 计算数据采集间隔（控制周期数）
        control_freq = 100  # 100Hz控制频率
        data_collection_interval = int(control_freq / self.data_collection_freq)
        control_counter = 0
        
        print(f"[数据采集器] 每 {data_collection_interval} 个控制周期采集一次数据")
        
        while self.running and not self.stop_event.is_set():
            try:
                with self.data_lock:
                    packet = self.latest_packet
                
                if packet is None:
                    time.sleep(0.01)
                    continue
                
                # 初始校准
                if not self.initial_calibration_done:
                    self.joycon_initial_pose = packet.pose
                    print(f"[数据采集器] ✓ 初始校准完成")
                    print(f"  Joycon初始位置: X={self.joycon_initial_pose[0]:.3f}, "
                          f"Y={self.joycon_initial_pose[1]:.3f}, Z={self.joycon_initial_pose[2]:.3f}")
                    print(f"  机械臂初始位置: X={self.robot_initial_pose[0]:.3f}, "
                          f"Y={self.robot_initial_pose[1]:.3f}, Z={self.robot_initial_pose[2]:.3f}")
                    self.initial_calibration_done = True
                    time.sleep(0.02)
                    continue
                
                # 获取Joycon的6D位姿
                pose = packet.pose
                
                # 获取当前时间用于防抖动
                current_time = time.time()
                
                # 安全检查
                if self.joycon_initial_pose is None or self.robot_initial_pose is None:
                    time.sleep(0.01)
                    continue
                
                # 计算Joycon相对于初始位置的偏移
                joycon_dx = pose[0] - self.joycon_initial_pose[0]
                joycon_dy = pose[1] - self.joycon_initial_pose[1]
                joycon_dz = pose[2] - self.joycon_initial_pose[2]
                
                # 应用偏移到机械臂初始位置
                robot_x = self.robot_initial_pose[0] + joycon_dx
                robot_y = self.robot_initial_pose[1] + joycon_dy
                robot_z = self.robot_initial_pose[2] + joycon_dz
                
                # 姿态固定为初始姿态
                robot_a = self.robot_initial_pose[3]
                robot_b = self.robot_initial_pose[4]
                robot_c = self.robot_initial_pose[5]
                
                # 工作空间限制
                robot_x, robot_y, robot_z = self.coordinate_mapper.clamp_to_workspace(
                    robot_x, robot_y, robot_z
                )
                
                # 数据采集时序优化：先读取状态和图像，再发送新指令
                # 这样可以保证观测和动作的时间对齐
                
                # 1. 读取当前状态（在发送新指令之前）
                # 注意：在阻抗控制模式下，使用目标位姿作为观测
                # 因为 get_current_pose() 会与阻抗控制线程冲突
                current_pose = self.robot_controller.get_target_pose()
                if current_pose is None:
                    time.sleep(0.01)
                    continue
                
                # 2. 读取夹爪状态
                self.gripper_read_counter += 1
                current_gripper_width = self.cached_gripper_width
                
                if self.gripper_read_counter >= 10:
                    try:
                        gripper_state = self.robot_controller.gripper.read_once()
                        current_gripper_width = gripper_state.width
                        self.gripper_read_counter = 0
                    except Exception as e:
                        pass
                
                # 3. 如果需要采集数据，先采集图像（在发送新指令之前）
                should_collect_data = (control_counter % data_collection_interval == 0)
                captured_frame = None
                captured_obs_pose = None
                captured_obs_gripper = None
                
                if self.is_recording and self.current_episode is not None and should_collect_data:
                    # 采集图像（与当前状态同步）
                    if self.camera_manager is not None:
                        captured_frame = self.camera_manager.read_frame()
                    
                    # 记录观测状态（与图像同步）
                    captured_obs_pose = np.array(current_pose)
                    captured_obs_gripper = self.cached_gripper_width
                
                # 4. 更新机械臂目标位姿（发送新的动作指令）
                if self.robot_controller.update_target(
                    robot_x, robot_y, robot_z,
                    robot_a, robot_b, robot_c
                ):
                    self.control_updates += 1
                
                # 夹爪控制（ZR按钮）
                gripper_value = packet.gripper
                target_gripper_width = current_gripper_width  # 默认保持当前宽度
                
                if gripper_value == 0.0 or gripper_value == 1.0:
                    if gripper_value != self.last_gripper_value:
                        try:
                            if gripper_value == 0.0:
                                print(f"[数据采集器] 🎮 检测到 ZR 按钮", flush=True)
                                print(f"[数据采集器] 🔴 夹爪关闭", flush=True)
                                # 夹爪最小宽度设为 0.03m
                                self.robot_controller.gripper.move_async(0.03)
                                target_gripper_width = 0.03
                                self.cached_gripper_width = 0.03
                            elif gripper_value == 1.0:
                                print(f"[数据采集器] 🎮 检测到 ZR 按钮", flush=True)
                                print(f"[数据采集器] 🟢 夹爪打开", flush=True)
                                self.robot_controller.gripper.move_async(0.08)
                                target_gripper_width = 0.08
                                self.cached_gripper_width = 0.08
                            self.last_gripper_value = gripper_value
                        except Exception as e:
                            print(f"[数据采集器] ⚠️  夹爪控制失败: {e}", flush=True)
                
                # Home按钮检测（由外部管理脚本处理）
                home_button = packet.buttons.get('home', 0)
                if home_button == 1 and self.last_home_button_value == 0:
                    # 防抖动检查
                    last_time = self.last_trigger_time.get('home', 0)
                    if current_time - last_time >= self.debounce_delay:
                        print(f"\n[数据采集器] 🏠 检测到 Home 按钮 - 退出程序以便复原", flush=True)
                        # 如果正在录制，先停止并保存
                        if self.is_recording:
                            print(f"[数据采集器] 停止当前录制并保存...", flush=True)
                            self.stop_recording()
                        # 设置退出标志
                        self.running = False
                        break
                self.last_home_button_value = home_button
                
                # R按钮控制录制（边沿触发 + 防抖动）
                r_button = packet.buttons.get('r', 0)
                
                if r_button == 1 and self.last_r_button_value == 0:
                    # 防抖动检查
                    last_time = self.last_trigger_time.get('r', 0)
                    time_since_last = current_time - last_time
                    
                    if time_since_last >= self.debounce_delay:
                        # R按钮按下（边沿触发）
                        print(f"\n{'='*60}", flush=True)
                        print(f"[数据采集器] 🎮 检测到 R 按钮", flush=True)
                        print(f"{'='*60}", flush=True)
                        if not self.is_recording:
                            print(f"[数据采集器] 🎬 开始录制轨迹...", flush=True)
                            self.start_recording()
                        else:
                            print(f"[数据采集器] ⏹️  停止录制并保存...", flush=True)
                            self.stop_recording()
                            print(f"\n[数据采集器] ✓ 轨迹已保存！可以继续录制下一个轨迹", flush=True)
                            print(f"{'='*60}\n", flush=True)
                        self.last_trigger_time['r'] = current_time
                
                self.last_r_button_value = r_button
                
                # X按钮 - 退出程序（发送端和接收端都退出）
                x_button = packet.buttons.get('x', 0)
                if x_button == 1 and self.last_x_button_value == 0:
                    # 防抖动检查
                    last_time = self.last_trigger_time.get('x', 0)
                    if current_time - last_time >= self.debounce_delay:
                        print(f"\n[数据采集器] 🎮 检测到 X 按钮", flush=True)
                        print(f"[数据采集器] ❌ 退出程序...", flush=True)
                        # 如果正在录制，先停止并保存
                        if self.is_recording:
                            print(f"[数据采集器] 停止当前录制并保存...", flush=True)
                            self.stop_recording()
                        
                        # 发送退出信号给发送端
                        try:
                            exit_signal = json.dumps({"type": "exit", "message": "X button pressed"}) + "\n"
                            self.client_socket.sendall(exit_signal.encode(ENCODING))
                            print(f"[数据采集器] 已发送退出信号给发送端", flush=True)
                        except:
                            pass
                        
                        # 设置退出标志
                        self.running = False
                        break
                self.last_x_button_value = x_button
                
                # 其他按钮检测（仅打印，不执行操作）
                other_buttons = {
                    'a': 'A 按钮',
                    'b': 'B 按钮',
                    'y': 'Y 按钮',
                    'plus': '+ 按钮',
                    'stick_r_btn': '右摇杆按下',
                    'right_sr': 'SR 按钮',
                    'right_sl': 'SL 按钮',
                }
                
                for btn_key, btn_name in other_buttons.items():
                    btn_value = packet.buttons.get(btn_key, 0)
                    last_btn_value = getattr(self, f'last_{btn_key}_value', 0)
                    
                    if btn_value == 1 and last_btn_value == 0:
                        print(f"[数据采集器] 🎮 检测到 {btn_name}", flush=True)
                    
                    setattr(self, f'last_{btn_key}_value', btn_value)
                
                # 记录数据（使用之前捕获的观测数据，保证时间对齐）
                # should_collect_data 已经在第555行计算过了
                if self.is_recording and self.current_episode is not None and should_collect_data:
                    # 确保捕获的数据有效
                    if captured_obs_pose is None or captured_obs_gripper is None:
                        # 如果捕获失败，跳过这次数据采集
                        pass
                    else:
                        timestamp = time.time() - self.current_episode.start_time
                        
                        # 保存图像（使用之前捕获的图像）
                        image_index = -1
                        if captured_frame is not None:
                            image_index = self.current_episode.save_image(captured_frame)
                        
                        # Joycon数据（原始输入）
                        joycon_pose = np.array(pose)
                        joycon_gripper = gripper_value
                        
                        # 机械臂观测（使用捕获的观测，与图像同步）
                        # 这是在发送新动作指令之前的状态
                        robot_obs_pose = captured_obs_pose
                        robot_obs_gripper = captured_obs_gripper
                        
                        # 机械臂动作（新计算的目标状态）
                        # 这是发送给机器人的新指令
                        robot_action_pose = np.array([robot_x, robot_y, robot_z, robot_a, robot_b, robot_c])
                        robot_action_gripper = self.cached_gripper_width
                        
                        self.current_episode.add_data_point(
                            timestamp=timestamp,
                            joycon_pose=joycon_pose,
                            joycon_gripper=joycon_gripper,
                            robot_obs_pose=robot_obs_pose,
                            robot_obs_gripper=robot_obs_gripper,
                            robot_action_pose=robot_action_pose,
                            robot_action_gripper=robot_action_gripper,
                            image_index=image_index
                        )
                
                control_counter += 1
                
                # 控制频率（100Hz）
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[数据采集器] 控制循环错误: {e}")
                import traceback
                traceback.print_exc()
                
                # 如果正在记录，尝试保存当前数据
                if self.is_recording and self.current_episode is not None:
                    print(f"[数据采集器] ⚠️  检测到错误，尝试保存当前episode...")
                    try:
                        self.stop_recording()
                    except:
                        pass
                
                time.sleep(0.1)
        
        print(f"[数据采集器] 控制线程结束")
    
    def keyboard_listener(self):
        """键盘监听线程"""
        import sys
        import select
        import termios
        import tty
        
        print(f"[数据采集器] 键盘监听线程启动")
        
        # 保存终端设置
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            while self.running and not self.stop_event.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    if key == 'q':
                        print(f"\n[数据采集器] 检测到 Q 键 - 退出程序")
                        self.running = False
                        break
        
        finally:
            # 恢复终端设置
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        
        print(f"[数据采集器] 键盘监听线程结束")
    
    def start(self):
        """启动数据采集器"""
        # 启动摄像头
        if self.camera_manager is not None:
            if not self.camera_manager.start():
                print("[数据采集器] 摄像头启动失败，退出程序")
                return False
        
        # 初始化机械臂控制
        if self.enable_robot_control:
            if not self.init_robot_control():
                print("[数据采集器] 机械臂控制初始化失败，退出程序")
                return False
        
        # 启动服务器
        if not self.start_server():
            return False
        
        # 等待客户端连接
        if not self.wait_for_client():
            return False
        
        self.running = True
        self.start_time = time.time()
        
        # 启动接收线程
        self.receive_thread = Thread(target=self.receive_loop, daemon=True)
        self.receive_thread.start()
        
        # 启动控制线程
        if self.enable_robot_control:
            self.control_thread = Thread(target=self.control_loop, daemon=True)
            self.control_thread.start()
        
        # 启动键盘监听线程
        keyboard_thread = Thread(target=self.keyboard_listener, daemon=True)
        keyboard_thread.start()
        
        print("\n" + "="*60)
        print("[数据采集器] 系统启动成功！")
        print(f"[数据采集器] 客户端: {self.client_address}")
        print(f"[数据采集器] 机械臂控制: {'运行中' if self.enable_robot_control else '禁用'}")
        print(f"[数据采集器] 摄像头: {'运行中' if self.camera_manager is not None else '禁用'}")
        print(f"[数据采集器] 数据保存: {self.save_dir.absolute()}")
        print("="*60 + "\n")
        
        # 保持运行
        try:
            while self.running:
                time.sleep(0.5)
                
                # 每10秒打印一次统计信息
                if self.packets_received % 2000 == 0 and self.packets_received > 0:
                    elapsed = time.time() - self.start_time
                    rate = self.packets_received / elapsed if elapsed > 0 else 0
                    print(f"[数据采集器] 运行中 - 数据包: {self.packets_received}, "
                          f"速率: {rate:.1f} Hz, Episodes: {self.episode_count}, "
                          f"总数据点: {self.total_data_points}")
        
        except KeyboardInterrupt:
            print("\n[数据采集器] 检测到 Ctrl+C")
        
        return True
    
    def stop(self):
        """停止数据采集器"""
        print("\n[数据采集器] 正在停止...")
        
        # 如果正在记录，先保存
        if self.is_recording:
            print("[数据采集器] 保存当前episode...")
            self.stop_recording()
        
        self.running = False
        self.stop_event.set()
        
        # 等待线程结束
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=2.0)
        
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)
        
        # 关闭连接
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # 统计信息
        elapsed = time.time() - self.start_time
        rate = self.packets_received / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"[数据采集器] 采集统计:")
        print(f"  总运行时间: {elapsed:.1f}秒")
        print(f"  总数据包: {self.packets_received}")
        print(f"  平均速率: {rate:.1f} Hz")
        print(f"  Episodes数量: {self.episode_count}")
        print(f"  总数据点: {self.total_data_points}")
        if self.episode_count > 0:
            print(f"  平均每episode: {self.total_data_points/self.episode_count:.0f} 数据点")
        print(f"  数据保存位置: {self.save_dir.absolute()}")
        print(f"{'='*60}")
        
        # 停止机械臂控制
        if self.robot_controller:
            self.robot_controller.stop()
        
        # 停止摄像头
        if self.camera_manager is not None:
            self.camera_manager.stop()
        
        print("[数据采集器] 已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='机械臂数据采集器')
    parser.add_argument('--robot-host', default='172.16.0.2', help='机械臂IP地址')
    parser.add_argument('--save-dir', default='/home/kyji/Desktop/Workspace/robot/data/trajectories', help='数据保存目录')
    parser.add_argument('--no-control', action='store_true', help='禁用机械臂控制（仅记录）')
    parser.add_argument('--no-camera', action='store_true', help='禁用摄像头')
    parser.add_argument('--camera-index', type=int, default=0, help='摄像头索引')
    parser.add_argument('--data-freq', type=float, default=10.0, help='数据采集频率 (Hz)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("机械臂轨迹数据采集器")
    print(f"本机IP: {ROBOT_IP}")
    print(f"监听端口: {SOCKET_PORT}")
    print(f"机械臂IP: {args.robot_host}")
    print(f"数据保存: {args.save_dir}")
    print(f"控制模式: {'仅记录' if args.no_control else '控制+记录'}")
    print(f"摄像头: {'禁用' if args.no_camera else f'启用 (索引{args.camera_index})'}")
    print(f"数据采集频率: {args.data_freq} Hz")
    print("="*60)
    
    print("\n自动启动服务器...")
    
    collector = RobotDataCollector(
        robot_host=args.robot_host,
        save_dir=args.save_dir,
        enable_robot_control=not args.no_control,
        enable_camera=not args.no_camera,
        camera_index=args.camera_index,
        camera_resolution=(1920, 1080),
        data_collection_freq=args.data_freq
    )
    
    try:
        collector.start()
    except KeyboardInterrupt:
        print("\n[数据采集器] 检测到 Ctrl+C")
    except Exception as e:
        print(f"\n[数据采集器] 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.stop()


if __name__ == "__main__":
    main()
