#!/usr/bin/env python3
"""
Diffusion Policy 推理服务器（SSH 隧道版本） - 增强调试版
"""

import socket
import json
import torch
import numpy as np
import cv2
import base64
import threading
import time
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import hydra
import dill
import sys

# 添加 diffusion_policy 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict

from server_config import (
    SERVER_IP, SERVER_PORT, CHECKPOINT_PATH, USE_EMA,
    DEVICE, SCHEDULER_TYPE, NUM_INFERENCE_STEPS, INFERENCE_FREQ,
    SOCKET_TIMEOUT, BUFFER_SIZE, ENCODING, MAX_CLIENTS, VERBOSE,
    ACTION_SCALE, ACTION_SMOOTHING_ALPHA, MAX_DELTA_POSITION, 
    MAX_DELTA_ROTATION, ENABLE_ACTION_LIMIT
)

class DPInferenceServerSSH:
    """Diffusion Policy 推理服务器（SSH 隧道版本）"""
    
    def __init__(self,
                 checkpoint_path: str = CHECKPOINT_PATH,
                 use_ema: bool = USE_EMA,
                 device: str = DEVICE,
                 scheduler_type: str = SCHEDULER_TYPE,
                 num_inference_steps: int = NUM_INFERENCE_STEPS,
                 inference_freq: float = INFERENCE_FREQ,
                 server_ip: str = SERVER_IP,
                 server_port: int = SERVER_PORT,
                 max_clients: int = MAX_CLIENTS,
                 verbose: bool = VERBOSE):
        
        self.checkpoint_path = checkpoint_path
        self.use_ema = use_ema
        self.device = device
        self.scheduler_type = scheduler_type.upper()
        self.num_inference_steps = num_inference_steps
        self.inference_freq = inference_freq
        self.server_ip = server_ip
        self.server_port = server_port
        self.max_clients = max_clients
        self.verbose = verbose
        
        self.policy = None
        self.cfg = None
        self.running = False
        self.expected_image_shape = None
        self.obs_keys = None
        
        # Episode 时间基准（用于处理相对时间戳）
        self.episode_start_time = None  # 服务器端episode开始时间（绝对时间）
        self.first_client_timestamp = None  # 第一个客户端时间戳（相对时间）
        self.last_recv_timestamp = None  # 上一次接收消息的时间（用于计算消息间隔）
        
        # 动作限制和平滑
        self.action_scale = ACTION_SCALE
        self.smoothing_alpha = ACTION_SMOOTHING_ALPHA
        self.max_delta_position = MAX_DELTA_POSITION
        self.max_delta_rotation = MAX_DELTA_ROTATION
        self.enable_action_limit = ENABLE_ACTION_LIMIT
        self.prev_action = None  # 上一次输出的动作（用于平滑）
        self.current_state = None  # 当前机械臂状态（用于增量限制）
        
        # 轨迹记录
        self.inference_log = {
            'meta': {
                'checkpoint': str(checkpoint_path),
                'start_time': time.strftime("%Y-%m-%d %H:%M:%S")
            },
            'steps': []
        }
        
        print("[推理服务器] 初始化...")
        self._load_model()
    
    def _load_model(self):
        """加载模型 (代码与原版相同，省略部分重复注释)"""
        print(f"[推理服务器] 加载模型: {self.checkpoint_path}")
        try:
            payload = torch.load(open(self.checkpoint_path, 'rb'), pickle_module=dill)
            self.cfg = payload['cfg']
            cls = hydra.utils.get_class(self.cfg._target_)
            workspace = cls(self.cfg)
            workspace.load_payload(payload)
            self.policy = workspace.model
            if self.use_ema:
                self.policy = workspace.ema_model
            self.policy.eval().to(self.device)
            
            # Scheduler 设置
            if self.scheduler_type == "DDIM":
                from diffusers.schedulers.scheduling_ddim import DDIMScheduler
                self.policy.noise_scheduler = DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon')
            elif self.scheduler_type == "DDPM":
                from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
                self.policy.noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02, beta_schedule='squaredcos_cap_v2', clip_sample=True, prediction_type='epsilon', variance_type='fixed_small')
            
            self.policy.num_inference_steps = self.num_inference_steps
            
            # 提取观测信息
            shape_meta = self.cfg.task.shape_meta
            rgb_keys = [k for k, v in shape_meta['obs'].items() if v.get('type') == 'rgb']
            lowdim_keys = [k for k, v in shape_meta['obs'].items() if v.get('type') == 'low_dim']

            self.obs_keys = {
                'rgb': rgb_keys[0] if rgb_keys else None,
                'lowdim': lowdim_keys
            }

            if self.obs_keys['rgb']:
                image_shape = shape_meta['obs'][self.obs_keys['rgb']]['shape'] 
                self.expected_image_shape = (image_shape[1], image_shape[2])

            print(f"[推理服务器] ✓ 模型加载成功 | 图像尺寸: {self.expected_image_shape}")
            self._warmup_model()
            
        except Exception as e:
            print(f"[推理服务器] ✗ 模型加载失败: {e}")
            raise
    
    def _warmup_model(self):
        """预热模型"""
        print("[推理服务器] 预热模型...")
        
        try:
            batch_size = 1
            n_obs_steps = self.policy.n_obs_steps
            
            # 获取观测规格
            shape_meta = self.cfg.task.shape_meta
            obs_keys = shape_meta['obs']
            
            # 找到 RGB 观测键
            rgb_key = None
            for key, spec in obs_keys.items():
                if spec.get('type') == 'rgb':
                    rgb_key = key
                    break
            
            if rgb_key is None:
                raise ValueError("未找到 RGB 类型的观测键")
            
            image_shape = shape_meta['obs'][rgb_key]['shape']
            
            # 创建虚拟观测
            dummy_image = torch.randn(
                batch_size, n_obs_steps, *image_shape,
                device=self.device, dtype=torch.float32
            )
            
            obs_dict = {rgb_key: dummy_image}
            
            # 添加 low_dim 观测
            for key, spec in obs_keys.items():
                if spec.get('type') == 'low_dim':
                    low_dim_shape = spec['shape']
                    dummy_low_dim = torch.randn(
                        batch_size, n_obs_steps, *low_dim_shape,
                        device=self.device, dtype=torch.float32
                    )
                    obs_dict[key] = dummy_low_dim
            
            # 推理
            with torch.no_grad():
                self.policy.reset()
                result = self.policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
            
            print(f"[推理服务器] ✓ 模型预热成功 (输出形状: {action.shape})")
            
        except Exception as e:
            print(f"[推理服务器] ✗ 模型预热失败: {e}")
            raise

    def _limit_and_smooth_action(self, action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """
        对输出动作进行限制和平滑处理
        
        Args:
            action: 原始动作序列 (n_action_steps, action_dim)
            current_state: 当前机械臂状态 (pose + gripper)
        
        Returns:
            处理后的动作序列
        """
        if not self.enable_action_limit:
            return action
        
        action = action.copy()
        n_steps, action_dim = action.shape
        
        # 1. 应用动作缩放（相对于当前状态的增量）
        if self.action_scale < 1.0 and current_state is not None:
            # 对每个动作步骤，计算与当前状态的增量并缩放
            for i in range(n_steps):
                # 假设前6维是pose（xyz + rotation），第7维是gripper
                pose_dim = min(6, action_dim - 1) if action_dim >= 7 else action_dim
                
                # 计算pose部分的增量
                delta = action[i, :pose_dim] - current_state[:pose_dim]
                # 缩放增量
                scaled_delta = delta * self.action_scale
                # 应用缩放后的增量
                action[i, :pose_dim] = current_state[:pose_dim] + scaled_delta
        
        # 2. 限制位置和旋转的最大变化量
        if current_state is not None:
            for i in range(n_steps):
                # 使用上一步动作或当前状态作为参考
                if i == 0:
                    ref_state = current_state
                else:
                    ref_state = action[i - 1]
                
                # 限制位置变化 (假设前3维是xyz)
                if action_dim >= 3:
                    pos_delta = action[i, :3] - ref_state[:3]
                    pos_delta_norm = np.linalg.norm(pos_delta)
                    if pos_delta_norm > self.max_delta_position:
                        pos_delta = pos_delta * (self.max_delta_position / pos_delta_norm)
                        action[i, :3] = ref_state[:3] + pos_delta
                
                # 限制旋转变化 (假设3:6维是旋转)
                if action_dim >= 6:
                    rot_delta = action[i, 3:6] - ref_state[3:6]
                    rot_delta_norm = np.linalg.norm(rot_delta)
                    if rot_delta_norm > self.max_delta_rotation:
                        rot_delta = rot_delta * (self.max_delta_rotation / rot_delta_norm)
                        action[i, 3:6] = ref_state[3:6] + rot_delta
        
        # 3. 与上一次输出动作做指数移动平均平滑
        if self.smoothing_alpha > 0 and self.prev_action is not None:
            # 使用上一次输出动作的第一个动作作为平滑参考
            prev_first_action = self.prev_action[0] if len(self.prev_action.shape) > 1 else self.prev_action
            
            # 对第一个动作步骤应用平滑
            action[0] = self.smoothing_alpha * prev_first_action + (1 - self.smoothing_alpha) * action[0]
            
            # 可选：对后续步骤也应用逐渐减弱的平滑
            for i in range(1, min(n_steps, 3)):  # 只平滑前3步
                decay = self.smoothing_alpha * (0.5 ** i)  # 指数衰减
                if i < len(self.prev_action):
                    action[i] = decay * self.prev_action[i] + (1 - decay) * action[i]
        
        # 更新记录
        self.prev_action = action.copy()
        
        return action

    def start(self):
        """启动服务器"""
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.server_ip, self.server_port))
            server_socket.listen(self.max_clients)
            
            print(f"[推理服务器] ✓ 监听 {self.server_ip}:{self.server_port}")
            self.running = True
            
            while self.running:
                try:
                    client_socket, client_addr = server_socket.accept()
                    print(f"[推理服务器] 客户端连接: {client_addr}")
                    self._handle_client(client_socket, client_addr)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"[推理服务器] 错误: {e}")
            
            server_socket.close()
            self._save_inference_log()
            
        except Exception as e:
            print(f"[推理服务器] 启动失败: {e}")

    def _handle_client(self, client_socket: socket.socket, client_addr: tuple):
        try:
            client_socket.settimeout(SOCKET_TIMEOUT)
            buffer = b''
            
            while self.running:
                try:
                    data = client_socket.recv(BUFFER_SIZE)
                    if not data: break
                    buffer += data
                    
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        # --- 关键点：记录接收时间戳 ---
                        recv_timestamp = time.time()
                        msg = line.decode(ENCODING).strip()
                        if msg:
                            self._process_message(client_socket, msg, recv_timestamp)
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[推理服务器] 接收错误: {e}")
                    break
            client_socket.close()
        except Exception as e:
            print(f"[推理服务器] 客户端错误: {e}")

    def _process_message(self, client_socket: socket.socket, message: str, recv_timestamp: float):
        try:
            data = json.loads(message)
            
            if data.get('type') == 'reset':
                self.policy.reset()
                # 记录episode开始时间
                self.episode_start_time = time.time()
                self.first_client_timestamp = None  # 重置，等待第一个observation
                self.last_recv_timestamp = None  # 重置接收时间记录
                # 重置动作平滑相关变量
                self.prev_action = None
                self.current_state = None
                response = {'type': 'reset_ack'}
                client_socket.sendall((json.dumps(response) + '\n').encode(ENCODING))
                
            elif data.get('type') == 'observation':
                # --- 记录处理开始 ---
                process_start_time = time.time()
                
                # 解码数据
                images_b64 = data.get('images', [])
                poses_list = data.get('poses', [])
                grippers_list = data.get('grippers', [])
                client_timestamps = data.get('timestamps', [])  # 相对时间戳（用于observation对齐）
                client_send_timestamp = data.get('send_timestamp')  # 绝对时间戳（用于计算通信延迟）
                
                # 记录最新的客户端相对时间戳（用于显示/对齐）
                latest_client_relative_ts = client_timestamps[-1] if client_timestamps else 0
                
                # 计算消息间隔（服务器端测量，作为参考）
                message_interval = None
                if self.last_recv_timestamp is not None:
                    message_interval = recv_timestamp - self.last_recv_timestamp
                self.last_recv_timestamp = recv_timestamp
                
                # 计算通信延迟（使用客户端发送的绝对时间戳）
                if client_send_timestamp is not None:
                    transport_latency = recv_timestamp - client_send_timestamp
                    # 安全检查：如果延迟为负或异常大，可能是时钟不同步
                    if transport_latency < 0:
                        print(f"[警告] 检测到负延迟 {transport_latency*1000:.1f}ms，时钟可能不同步")
                    elif transport_latency > 5:
                        print(f"[警告] 检测到异常大延迟 {transport_latency*1000:.1f}ms")
                    elif self.verbose:
                        print(f"[通信延迟] {transport_latency*1000:.1f}ms")
                else:
                    # 兼容旧版客户端：使用消息间隔作为近似
                    transport_latency = message_interval if message_interval is not None else 0
                    print(f"[警告] 客户端未发送 send_timestamp，使用消息间隔作为近似: {transport_latency*1000:.1f}ms")

                # 解码图像
                images = []
                for img_b64 in images_b64:
                    img_data = base64.b64decode(img_b64)
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    if self.expected_image_shape is not None:
                        expected_h, expected_w = self.expected_image_shape
                        if image.shape[:2] != (expected_h, expected_w):
                            image = cv2.resize(image, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                    images.append(image)

                # 准备 env_obs
                poses = np.array(poses_list, dtype=np.float32)
                grippers = np.array(grippers_list, dtype=np.float32)
                
                env_obs = {}
                if self.obs_keys['rgb']:
                    env_obs[self.obs_keys['rgb']] = np.stack(images, axis=0).astype(np.uint8)
                
                for lowdim_key in self.obs_keys['lowdim']:
                    if 'pose' in lowdim_key.lower():
                        env_obs[lowdim_key] = poses
                    elif 'gripper' in lowdim_key.lower():
                        env_obs[lowdim_key] = grippers

                # --- 运行推理并传递时间信息 ---
                # 直接传递原始 base64，避免重新编码
                last_image_b64 = images_b64[-1] if images_b64 else None
                
                action = self._infer_action(
                    env_obs, 
                    np.array(client_timestamps), 
                    recv_timestamp,
                    client_send_timestamp,  # 客户端发送时的绝对时间戳
                    transport_latency,  # 已计算的通信延迟
                    process_start_time,
                    message_interval,  # 消息间隔（作为参考）
                    last_image_b64  # 原始 base64 图片（用于日志保存）
                )
                
                # 发送响应
                response = {'type': 'action', 'action': action.tolist()}
                msg = json.dumps(response) + '\n'
                client_socket.sendall(msg.encode(ENCODING))
                
                # --- 记录发送时间 ---
                send_timestamp = time.time()
                
                # 更新当前日志条目的发送时间和总延迟
                if self.inference_log['steps']:
                    self.inference_log['steps'][-1]['timing']['send_timestamp'] = send_timestamp
                    # 总延迟 = 发送时间 - 客户端发送时间（如果可用）
                    if client_send_timestamp is not None:
                        total_latency = send_timestamp - client_send_timestamp
                        self.inference_log['steps'][-1]['timing']['total_latency_ms'] = float(total_latency * 1000)
        
        except Exception as e:
            print(f"[推理服务器] 处理错误: {e}")
            import traceback
            traceback.print_exc()

    def _infer_action(self, env_obs: dict, timestamps: np.ndarray, 
                      recv_timestamp: float, client_send_timestamp: Optional[float],
                      transport_latency: float, process_start_time: float,
                      message_interval: Optional[float] = None,
                      raw_image_b64: Optional[str] = None) -> np.ndarray:
        try:
            # 记录开始推理时间
            infer_start_time = time.time()
            
            # 准备数据
            shape_meta = self.cfg.task.shape_meta
            env_obs['timestamp'] = timestamps
            obs_dict_np = get_real_obs_dict(env_obs=env_obs, shape_meta=shape_meta)
            obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            
            # 推理
            with torch.no_grad():
                result = self.policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()

            infer_end_time = time.time()

            # 处理动作维度 (7D -> 8D)
            if action.shape[-1] == 7:
                last_gripper = env_obs[self.obs_keys['lowdim'][-1]][-1]
                gripper_actions = np.tile(last_gripper, (action.shape[0], 1))
                action_with_gripper = np.concatenate([action, gripper_actions], axis=-1)
            else:
                action_with_gripper = action
            
            # --- 动作限制和平滑 ---
            # 获取当前状态（pose + gripper）
            current_state = np.concatenate([
                env_obs[self.obs_keys['lowdim'][0]][-1],  # pose
                env_obs[self.obs_keys['lowdim'][-1]][-1]  # gripper
            ]) if len(self.obs_keys['lowdim']) > 0 else None
            
            # 应用动作限制和平滑
            action_with_gripper = self._limit_and_smooth_action(action_with_gripper, current_state)
            
            if self.verbose and self.enable_action_limit:
                print(f"[动作限制] scale={self.action_scale:.2f}, smooth={self.smoothing_alpha:.2f}, "
                      f"max_pos={self.max_delta_position:.3f}, max_rot={self.max_delta_rotation:.3f}")

            # --- 保存日志 (含图片和时间戳) ---
            # 直接使用客户端发送的原始 base64，避免重新编码（零开销、无损）
            img_b64_str = raw_image_b64
            
            # 提取当前状态
            lowdim_obs_list = []
            for lowdim_key in self.obs_keys['lowdim']:
                if lowdim_key in env_obs:
                    lowdim_obs_list.append(env_obs[lowdim_key][-1])
            last_state = np.concatenate(lowdim_obs_list) if lowdim_obs_list else np.array([])

            # 提取最后一个相对时间戳用于显示
            last_relative_ts = float(timestamps[-1]) if len(timestamps) > 0 else 0.0
            
            current_step = {
                'step': len(self.inference_log['steps']),
                'timing': {
                    'client_send': float(client_send_timestamp) if client_send_timestamp is not None else None,  # 客户端发送时间（绝对时间戳）
                    'client_obs_relative': last_relative_ts,  # 客户端观测时间（相对时间戳，用于对齐）
                    'server_recv': float(recv_timestamp),             # 服务器收到时间
                    'process_start': float(process_start_time),       # 开始处理时间
                    'infer_start': float(infer_start_time),           # 开始模型推理
                    'infer_end': float(infer_end_time),               # 结束模型推理
                    'send_timestamp': 0.0,                            # 发送时间 (process_message中更新)
                    'transport_latency_ms': float(transport_latency * 1000),  # 通信延迟（毫秒）
                    'message_interval_ms': float(message_interval * 1000) if message_interval is not None else None,  # 消息间隔（毫秒）
                    'inference_latency_ms': float((infer_end_time - infer_start_time) * 1000)  # 推理延迟（毫秒）
                },
                'input': {
                    'state': last_state.astype(np.float32).tolist(),
                    'image_base64': img_b64_str  # 保存实际进入模型的图像
                },
                'action': {
                    'values': action_with_gripper.astype(np.float32).tolist()
                }
            }
            self.inference_log['steps'].append(current_step)
            
            return action_with_gripper.astype(np.float32)
        
        except Exception as e:
            print(f"[推理服务器] 推理错误: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((self.policy.n_action_steps, 8), dtype=np.float32)

    def _save_inference_log(self):
        try:
            log_dir = Path(__file__).parent / "log"
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 使用 zip压缩保存，因为包含图片Base64，文件会很大
            # 这里先保存普通json，建议用户定期清理
            log_file = log_dir / f"inference_log_{timestamp}.json"
            with open(log_file, 'w') as f:
                json.dump(self.inference_log, f, indent=2)
            print(f"[推理服务器] 日志已保存: {log_file}")
        except Exception as e:
            print(f"[错误] 保存日志失败: {e}")

if __name__ == "__main__":
    server = DPInferenceServerSSH()
    server.start()