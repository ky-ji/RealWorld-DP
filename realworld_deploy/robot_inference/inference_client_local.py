#!/usr/bin/env python3
"""
Polymetis 推理客户端（本地版本）
通过本地Socket直接连接推理服务器
"""

import socket
import json
import time
import numpy as np
import cv2
import base64
import threading
import math
import torch
from pathlib import Path
from typing import Optional, Tuple, Dict
from threading import Thread, Lock, Event
from collections import deque

# 设置路径（使代码可在任意目录运行）
import _path_setup

try:
    from polymetis import RobotInterface, GripperInterface
    print("✓ Polymetis 库导入成功")
except ImportError as e:
    print(f"✗ 无法导入 Polymetis 库: {e}")
    import sys
    sys.exit(1)

from cameras import create_camera
from inference_config_local import (
    SERVER_IP, SERVER_PORT,
    ROBOT_IP, ROBOT_PORT, GRIPPER_PORT,
    CAMERA_TYPE, CAMERA_INDEX, CAMERA_RESOLUTION, IMAGE_QUALITY, ENABLE_DEPTH,
    INFERENCE_FREQ, N_OBS_STEPS, CAMERA_FREQ,
    ACTION_SCALE, STEPS_PER_INFERENCE,
    GRIPPER_OPEN_WIDTH, GRIPPER_CLOSED_WIDTH, GRIPPER_SPEED, GRIPPER_FORCE,
    CARTESIAN_KX, CARTESIAN_KXD,
    SOCKET_TIMEOUT, BUFFER_SIZE, ENCODING
)


class ObservationBuffer:
    """观测缓冲区管理器（实现时间对齐）"""

    def __init__(self, n_obs_steps: int, control_freq: float, camera_freq: float = 30.0):
        self.n_obs_steps = n_obs_steps
        self.control_freq = control_freq
        self.camera_freq = camera_freq
        self.dt = 1.0 / control_freq
        self.k = math.ceil(n_obs_steps * (camera_freq / control_freq))

        self.image_buffer = deque(maxlen=self.k)
        self.image_timestamps = deque(maxlen=self.k)
        self.pose_buffer = deque(maxlen=self.k)  # 7D位姿
        self.gripper_buffer = deque(maxlen=self.k)  # 1D夹爪
        self.state_timestamps = deque(maxlen=self.k)
        self.lock = Lock()

    def add_image(self, image: np.ndarray, timestamp: float):
        with self.lock:
            self.image_buffer.append(image)
            self.image_timestamps.append(timestamp)

    def add_state(self, pose_7d: np.ndarray, gripper_1d: np.ndarray, timestamp: float):
        """添加状态（拆分为7D位姿 + 1D夹爪）"""
        with self.lock:
            self.pose_buffer.append(pose_7d)
            self.gripper_buffer.append(gripper_1d)
            self.state_timestamps.append(timestamp)

    def get_aligned_obs(self) -> Optional[Dict]:
        with self.lock:
            if len(self.image_buffer) < self.n_obs_steps or len(self.pose_buffer) < self.n_obs_steps:
                return None

            last_image_timestamp = self.image_timestamps[-1]
            last_state_timestamp = self.state_timestamps[-1]
            last_timestamp = max(last_image_timestamp, last_state_timestamp)
            obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * self.dt)

            image_timestamps_arr = np.array(list(self.image_timestamps))
            aligned_images = []
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(image_timestamps_arr < t)[0]
                idx = is_before_idxs[-1] if len(is_before_idxs) > 0 else 0
                aligned_images.append(self.image_buffer[idx])

            state_timestamps_arr = np.array(list(self.state_timestamps))
            aligned_poses = []
            aligned_grippers = []
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(state_timestamps_arr < t)[0]
                idx = is_before_idxs[-1] if len(is_before_idxs) > 0 else 0
                aligned_poses.append(self.pose_buffer[idx])
                aligned_grippers.append(self.gripper_buffer[idx])

            return {
                'images': np.stack(aligned_images, axis=0),
                'poses': np.stack(aligned_poses, axis=0),  # (n_obs_steps, 7)
                'grippers': np.stack(aligned_grippers, axis=0),  # (n_obs_steps, 1)
                'timestamps': obs_align_timestamps
            }


class DPFormatConverter:
    """DP 格式转换器

    配置文件格式：
    - robot_eef_pose: [7] = [x, y, z, qx, qy, qz, qw]
    - robot_gripper_state: [1] = [gripper]
    - action: [7] = [x, y, z, qx, qy, qz, qw] (gripper单独处理)
    """

    GRIPPER_OPEN = 1.0
    GRIPPER_CLOSED = 0.0
    GRIPPER_THRESHOLD = 0.5

    @staticmethod
    def polymetis_to_dp_state(ee_pos: np.ndarray, ee_quat: np.ndarray, gripper_open: bool) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (7D位姿, 1D夹爪) 以匹配配置文件"""
        pose_7d = np.concatenate([ee_pos, ee_quat]).astype(np.float32)  # [x,y,z,qx,qy,qz,qw]
        gripper_value = DPFormatConverter.GRIPPER_OPEN if gripper_open else DPFormatConverter.GRIPPER_CLOSED
        gripper_1d = np.array([gripper_value], dtype=np.float32)
        return pose_7d, gripper_1d

    @staticmethod
    def dp_to_polymetis_action(pose_7d: np.ndarray, gripper_1d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        """从 (7D位姿, 1D夹爪) 转换为 Polymetis 格式"""
        pose_7d = np.array(pose_7d).flatten()
        ee_pos = pose_7d[:3]  # [x, y, z]
        ee_quat = pose_7d[3:7]  # [qx, qy, qz, qw]
        gripper_value = float(gripper_1d[0]) if isinstance(gripper_1d, np.ndarray) else float(gripper_1d)
        gripper_open = gripper_value > DPFormatConverter.GRIPPER_THRESHOLD
        return ee_pos.astype(np.float32), ee_quat.astype(np.float32), gripper_open


class LocalSocketClient:
    """本地Socket客户端（直接连接，无需SSH隧道）"""

    def __init__(self, server_ip: str, server_port: int):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None

    def connect(self, timeout: float = 5.0) -> bool:
        """连接到本地服务器"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((self.server_ip, self.server_port))
            print(f"[本地连接] ✓ 已连接到 {self.server_ip}:{self.server_port}")
            return True
        except Exception as e:
            print(f"[本地连接] ✗ 连接失败: {e}")
            return False

    def send_data(self, data: Dict) -> bool:
        """发送数据"""
        try:
            msg = json.dumps(data) + '\n'
            self.socket.sendall(msg.encode(ENCODING))
            return True
        except Exception as e:
            print(f"[本地连接] 发送错误: {e}")
            return False

    def recv_data(self, timeout: float = 5.0) -> Optional[Dict]:
        """接收数据"""
        try:
            self.socket.settimeout(timeout)
            data = b''
            while True:
                chunk = self.socket.recv(BUFFER_SIZE)
                if not chunk:
                    return None
                data += chunk
                if b'\n' in data:
                    break
            msg = data.decode(ENCODING).strip()
            return json.loads(msg)
        except socket.timeout:
            return None
        except Exception as e:
            print(f"[本地连接] 接收错误: {e}")
            return None

    def close(self):
        """关闭连接"""
        if self.socket:
            self.socket.close()
            print(f"[本地连接] ✓ 连接已关闭")


class PolymetisInferenceClientLocal:
    """Polymetis 推理客户端（本地版本）"""

    def __init__(self, server_ip: str = SERVER_IP, server_port: int = SERVER_PORT,
                 robot_ip: str = ROBOT_IP, robot_port: int = ROBOT_PORT,
                 gripper_port: int = GRIPPER_PORT,
                 camera_type: str = CAMERA_TYPE, camera_index: int = CAMERA_INDEX,
                 camera_resolution: Tuple[int, int] = CAMERA_RESOLUTION,
                 inference_freq: float = INFERENCE_FREQ, n_obs_steps: int = N_OBS_STEPS,
                 camera_freq: float = CAMERA_FREQ, image_quality: int = IMAGE_QUALITY,
                 enable_depth: bool = ENABLE_DEPTH):

        self.local_client = LocalSocketClient(server_ip, server_port)
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.gripper_port = gripper_port
        self.robot = None
        self.gripper = None

        # 创建相机实例
        self.camera = create_camera(
            camera_type=camera_type,
            width=camera_resolution[0],
            height=camera_resolution[1],
            fps=int(camera_freq),
            enable_depth=enable_depth,
            camera_index=camera_index,
        )
        self.inference_freq = inference_freq
        self.inference_interval = 1.0 / inference_freq
        self.n_obs_steps = n_obs_steps
        self.camera_freq = camera_freq
        self.image_quality = image_quality

        self.obs_buffer = ObservationBuffer(n_obs_steps, inference_freq, camera_freq)
        self.running = False
        self.data_lock = Lock()
        self.latest_action = None
        self.action_received = Event()

        self.actions_received = 0
        self.observations_sent = 0
        self.gripper_open = True
        self.last_gripper_state = True

        self.control_thread = None
        self.current_target_pos = None
        self.current_target_quat = None
        self.target_lock = Lock()

        # 时间戳管理
        self.eval_t_start = None
        self.iter_idx = 0

        # 回退点过滤
        self.last_action_output = None
        self.backtrack_threshold = 0.00
        self.filtered_backtrack_count = 0

        self.trajectory_log = {
            'observations': [],
            'actions': [],
            'executed': []
        }

    def run(self):
        """运行客户端"""
        print("\n" + "="*70)
        print("Polymetis 推理客户端 (本地版本)")
        print("="*70)

        try:
            if not self.local_client.connect():
                print("连接服务器失败")
                return

            print("\n[客户端] 初始化机械臂...")
            self._initialize_robot()

            print("[客户端] 初始化摄像头...")
            if not self.camera.start():
                print("⚠️  摄像头启动失败，将不记录图像")

            self.running = True

            # 启动观测收集线程
            obs_thread = Thread(target=self._collect_observations, daemon=True)
            obs_thread.start()

            recv_thread = Thread(target=self._receive_loop, daemon=True)
            recv_thread.start()

            self.control_thread = Thread(target=self._control_loop, daemon=True)
            self.control_thread.start()

            self._inference_loop()

        except KeyboardInterrupt:
            print("\n[客户端] 检测到 Ctrl+C，正在停止...")
        except Exception as e:
            print(f"程序出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def _initialize_robot(self):
        """初始化机器人"""
        try:
            self.robot = RobotInterface(ip_address=self.robot_ip, port=self.robot_port)
            print(f"已连接到机器人")

            try:
                self.gripper = GripperInterface(ip_address=self.robot_ip, port=self.gripper_port)
                print(f"已连接到夹爪")
            except:
                self.gripper = None
                print(f"夹爪服务器未启动")

            print("启动笛卡尔阻抗控制...")
            print(f"  刚度 Kx: {CARTESIAN_KX}")
            print(f"  阻尼 Kxd: {CARTESIAN_KXD}")
            self.robot.start_cartesian_impedance(
                Kx=torch.Tensor(CARTESIAN_KX),
                Kxd=torch.Tensor(CARTESIAN_KXD)
            )

            ee_pos, ee_quat = self.robot.get_ee_pose()
            with self.target_lock:
                self.current_target_pos = ee_pos.cpu().numpy()
                self.current_target_quat = ee_quat.cpu().numpy()

            print("笛卡尔阻抗控制已启动")
        except Exception as e:
            print(f"初始化机器人失败: {e}")
            raise

    def _filter_backtracking_actions(self, action_sequence_filtered, verbose=False):
        """过滤回退的动作点，只保留沿轨迹前进的点"""
        if self.last_action_output is None or len(action_sequence_filtered) == 0:
            return action_sequence_filtered, 0

        last_pos, last_quat = self.last_action_output

        # 计算每个动作与上次执行动作的距离
        distances = []
        for single_action in action_sequence_filtered:
            if isinstance(single_action, dict):
                pose_7d = np.array(single_action['pose'])
            else:
                single_action_arr = np.array(single_action).flatten()
                if len(single_action_arr) >= 7:
                    pose_7d = single_action_arr[:7]
                else:
                    distances.append(float('inf'))
                    continue

            target_pos, target_quat, _ = DPFormatConverter.dp_to_polymetis_action(pose_7d, np.array([1.0]))
            pos_dist = np.linalg.norm(target_pos - last_pos)
            distances.append(pos_dist)

        if len(distances) == 0:
            return action_sequence_filtered, 0

        # 找到第一个距离大于阈值的点
        start_idx = 0
        min_distance_threshold = self.backtrack_threshold

        for i, dist in enumerate(distances):
            if dist > min_distance_threshold:
                start_idx = i
                break

        if start_idx == 0 and distances[0] <= min_distance_threshold:
            if verbose:
                print(f"[过滤] 所有动作都太接近上次位置，过滤整个chunk")
            return [], len(action_sequence_filtered)

        filtered_actions = action_sequence_filtered[start_idx:]
        num_filtered = start_idx

        if verbose and num_filtered > 0:
            print(f"[过滤] 过滤掉前 {num_filtered} 个回退点，保留 {len(filtered_actions)} 个前进点")

        return filtered_actions, num_filtered

    def _control_loop(self):
        """控制循环"""
        print("[控制线程] 已启动，频率: 30 Hz")
        rate = 1.0 / 30.0

        while self.running:
            try:
                loop_start = time.time()

                with self.target_lock:
                    if self.current_target_pos is not None and self.current_target_quat is not None:
                        target_pos = torch.from_numpy(self.current_target_pos).float()
                        target_quat = torch.from_numpy(self.current_target_quat).float()
                        self.robot.update_desired_ee_pose(position=target_pos, orientation=target_quat)

                elapsed = time.time() - loop_start
                sleep_time = rate - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except Exception as e:
                if not self.running:
                    break
                print(f"[控制线程] 错误: {e}")
                time.sleep(0.1)

    def _collect_observations(self):
        """收集观测数据"""
        print("[客户端] 观测收集线程已启动 (30Hz)")

        while self.running:
            try:
                current_time = time.time() - (self.eval_t_start if self.eval_t_start else time.time())

                # 读取相机帧
                frame_data = self.camera.read_frame()
                if frame_data['color'] is not None:
                    self.obs_buffer.add_image(frame_data['color'], current_time)

                ee_pos, ee_quat = self.robot.get_ee_pose()
                if ee_pos is not None and ee_quat is not None:
                    ee_pos_np = ee_pos.cpu().numpy()
                    ee_quat_np = ee_quat.cpu().numpy()
                    pose_7d, gripper_1d = DPFormatConverter.polymetis_to_dp_state(ee_pos_np, ee_quat_np, self.gripper_open)
                    self.obs_buffer.add_state(pose_7d, gripper_1d, current_time)

                time.sleep(1.0 / 30.0)
            except Exception as e:
                if self.running:
                    print(f"[客户端] 观测收集错误: {e}")

    def _receive_loop(self):
        """接收动作循环"""
        while self.running:
            try:
                data = self.local_client.recv_data(timeout=1.0)
                if data:
                    # 支持两种类型：'action' 和 'action_sequence'
                    if data.get('type') == 'action':
                        action = np.array(data.get('action'), dtype=np.float32)
                        self.trajectory_log['actions'].append({'step': self.actions_received, 'action': action.tolist()})

                        with self.data_lock:
                            self.latest_action = action
                            self.actions_received += 1
                        self.action_received.set()
                    elif data.get('type') == 'action_sequence':
                        # 接收动作序列
                        actions = np.array(data.get('actions'), dtype=np.float32)
                        self.trajectory_log['actions'].append({'step': self.actions_received, 'actions': actions.tolist()})

                        with self.data_lock:
                            self.latest_action = actions
                            self.actions_received += 1
                        self.action_received.set()
                        print(f"[客户端] 收到动作序列: shape={actions.shape}")
            except Exception as e:
                if self.running:
                    print(f"[客户端] 接收错误: {e}")

    def _inference_loop(self):
        """推理循环"""
        dt = self.inference_interval
        actual_inference_interval = dt * STEPS_PER_INFERENCE
        print(f"[客户端] 推理循环已启动")
        print(f"  基础频率: {1/dt:.1f}Hz (dt={dt:.3f}s)")
        print(f"  实际推理频率: {1/actual_inference_interval:.1f}Hz (每次执行{STEPS_PER_INFERENCE}步)")

        # 等待观测缓冲区填充
        print("[客户端] 等待观测缓冲区填充...")
        start_delay = 0.5
        self.eval_t_start = time.time() + start_delay
        t_start = time.monotonic() + start_delay
        time.sleep(start_delay)
        print("[客户端] ✓ 开始推理控制")

        frame_latency = 1/30

        while self.running:
            try:
                aligned_obs = self.obs_buffer.get_aligned_obs()
                if aligned_obs is None:
                    time.sleep(0.01)
                    continue

                # 记录观测
                last_pose = aligned_obs['poses'][-1]
                last_gripper = aligned_obs['grippers'][-1]
                self.trajectory_log['observations'].append({
                    'step': self.observations_sent,
                    'pose': last_pose.tolist(),
                    'gripper': last_gripper.tolist(),
                    'n_obs_steps': self.n_obs_steps
                })

                images_b64 = []
                for img in aligned_obs['images']:
                    _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
                    images_b64.append(base64.b64encode(img_encoded).decode('utf-8'))

                # 发送观测
                obs_msg = {
                    'type': 'observation',
                    'images': images_b64,
                    'poses': aligned_obs['poses'].astype(np.float32).tolist(),
                    'grippers': aligned_obs['grippers'].astype(np.float32).tolist(),
                    'timestamps': aligned_obs['timestamps'].tolist()
                }

                if self.local_client.send_data(obs_msg):
                    self.observations_sent += 1
                    print(f"[客户端] 发送观测 #{self.observations_sent} (iter={self.iter_idx})")

                # 计算本次推理周期结束时间
                t_cycle_end = t_start + (self.iter_idx + STEPS_PER_INFERENCE) * dt

                if self.action_received.wait(timeout=actual_inference_interval):
                    self.action_received.clear()

                    with self.data_lock:
                        action = self.latest_action.copy() if self.latest_action is not None else None

                    if action is not None:
                        if action.ndim == 1:
                            action_sequence = [action]
                        else:
                            action_sequence = action

                        # 获取观测时间戳
                        obs_timestamps = aligned_obs['timestamps']

                        # 为每个动作分配时间戳
                        dt = self.inference_interval
                        action_offset = 0
                        action_timestamps = (np.arange(len(action_sequence), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]

                        # 直接使用所有动作
                        action_sequence_filtered = action_sequence
                        action_timestamps_filtered = action_timestamps

                        # 过滤回退点
                        action_sequence_final, num_backtrack_filtered = self._filter_backtracking_actions(
                            action_sequence_filtered, verbose=True
                        )

                        if num_backtrack_filtered > 0:
                            self.filtered_backtrack_count += num_backtrack_filtered

                        # 如果所有动作都被过滤，跳过
                        if len(action_sequence_final) == 0:
                            if self.filtered_backtrack_count % 10 == 0:
                                print(f"[过滤] 已过滤 {self.filtered_backtrack_count} 个回退点")
                            wait_until = t_cycle_end - frame_latency
                            wait_time = wait_until - time.monotonic()
                            if wait_time > 0:
                                time.sleep(wait_time)
                            continue

                        # 更新时间戳
                        action_timestamps_final = action_timestamps_filtered[num_backtrack_filtered:]

                        print(f"[客户端] 收到 {len(action_sequence)} 个动作，移除 {num_backtrack_filtered} 个回退点，执行 {len(action_sequence_final)} 个")

                        # 执行过滤后的动作
                        for step_idx, single_action in enumerate(action_sequence_final):
                            if isinstance(single_action, dict):
                                pose_7d = np.array(single_action['pose'])
                                gripper_1d = np.array(single_action['gripper'])
                            else:
                                single_action = np.array(single_action).flatten()
                                if len(single_action) == 8:
                                    pose_7d = single_action[:7]
                                    gripper_1d = single_action[7:8]
                                elif len(single_action) == 7:
                                    pose_7d = single_action
                                    gripper_1d = np.array([1.0])
                                else:
                                    print(f"[客户端] 警告: 动作维度不匹配: {len(single_action)}")
                                    continue

                            target_pos, target_quat, target_gripper_open = DPFormatConverter.dp_to_polymetis_action(pose_7d, gripper_1d)

                            current_pos, current_quat = self.robot.get_ee_pose()
                            if current_pos is None or current_quat is None:
                                continue

                            current_pos_np = current_pos.cpu().numpy()
                            delta_pos = target_pos - current_pos_np
                            scaled_target_pos = current_pos_np + delta_pos * ACTION_SCALE
                            scaled_target_quat = target_quat

                            with self.target_lock:
                                self.current_target_pos = scaled_target_pos
                                self.current_target_quat = scaled_target_quat

                            # 夹爪控制
                            if self.gripper is not None and target_gripper_open != self.last_gripper_state:
                                action_name = "打开" if target_gripper_open else "关闭"
                                print(f"[客户端] 夹爪{action_name}...")
                                try:
                                    if target_gripper_open:
                                        self.gripper.goto(
                                            width=0.09,
                                            speed=0.3,
                                            force=1.0,
                                            blocking=True
                                        )
                                    else:
                                        self.gripper.grasp(
                                            speed=0.2,
                                            force=1.0,
                                            grasp_width=0.0,
                                            epsilon_inner=0.1,
                                            epsilon_outer=0.1,
                                            blocking=True
                                        )
                                    actual_width = self.gripper.get_state().width
                                    print(f"✓ 夹爪已{action_name} (实际宽度: {actual_width:.4f}m)")
                                    self.last_gripper_state = target_gripper_open
                                except Exception as e:
                                    print(f"✗ 夹爪控制失败: {e}")

                            self.gripper_open = target_gripper_open

                            self.trajectory_log['executed'].append({
                                'step': self.actions_received,
                                'action_step': step_idx,
                                'pos': scaled_target_pos.tolist(),
                                'quat': scaled_target_quat.tolist(),
                                'gripper_open': target_gripper_open,
                                'timestamp': action_timestamps_final[step_idx]
                            })

                            # 等待到动作的目标时间戳
                            target_time = action_timestamps_final[step_idx]
                            wait_time = target_time - time.time()
                            if wait_time > 0:
                                time.sleep(wait_time)

                            print(f"[客户端] 执行动作 #{self.actions_received} 步 {step_idx+1}/{len(action_sequence_final)}")

                            # 更新上一次执行的动作
                            if step_idx == len(action_sequence_final) - 1:
                                self.last_action_output = (target_pos.copy(), target_quat.copy())

                        # 更新迭代索引
                        self.iter_idx += len(action_sequence_final)

                        # 等待到下一个推理周期
                        wait_until = t_cycle_end - frame_latency
                        wait_time = wait_until - time.monotonic()
                        if wait_time > 0:
                            time.sleep(wait_time)
                        else:
                            print(f"[客户端] 警告: 推理+执行超时 {-wait_time:.3f}s")
                else:
                    time.sleep(0.01)

            except Exception as e:
                print(f"[客户端] 推理循环错误: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)

    def _save_trajectory_log(self):
        """保存轨迹日志"""
        try:
            from datetime import datetime

            def convert_to_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (np.bool_, np.integer, np.floating)):
                    return obj.item()
                else:
                    return obj

            serializable_log = convert_to_serializable(self.trajectory_log)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存到 log 目录
            log_dir = _path_setup.get_log_dir()
            log_file = log_dir / f"trajectory_log_{timestamp}.json"

            with open(log_file, 'w') as f:
                json.dump(serializable_log, f, indent=2)

            print(f"[客户端] 轨迹日志已保存: {log_file}")
        except Exception as e:
            print(f"[客户端] 保存轨迹日志失败: {e}")

    def stop(self):
        """停止客户端"""
        self.running = False

        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2.0)

        if self.camera:
            self.camera.stop()

        self.local_client.close()

        print(f"\n[客户端] ✓ 已停止")
        print(f"[客户端] 总共发送 {self.observations_sent} 个观测")
        print(f"[客户端] 总共接收 {self.actions_received} 个动作")
        print(f"[客户端] 回退点过滤: {self.filtered_backtrack_count} 个 (阈值: {self.backtrack_threshold*1000:.1f}mm)")
        print(f"[客户端] 轨迹连续性: 移除了chunk中的回退点，保留前进点")

        self._save_trajectory_log()


if __name__ == "__main__":
    client = PolymetisInferenceClientLocal()
    try:
        client.run()
    except KeyboardInterrupt:
        print("\n[客户端] 检测到 Ctrl+C，正在停止...")
        client.stop()
