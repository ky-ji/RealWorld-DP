"""
将采集的轨迹数据转换为 Diffusion Policy 训练格式

数据格式转换:
1. 图像: JPG序列 -> MP4视频
2. 低维数据: PKL -> Zarr ReplayBuffer
3. 组织结构: episode_xxxx/ -> videos/ + replay_buffer.zarr/

输出格式:
dataset_path/
├── replay_buffer.zarr/
│   ├── data/
│   │   ├── timestamp
│   │   ├── robot_eef_pose  # [x, y, z, rx, ry, rz, gripper]
│   │   └── action          # [x, y, z, rx, ry, rz, gripper]
│   └── meta/
│       ├── episode_ends
│       └── episode_lengths
└── videos/
    ├── 0/
    │   └── 0.mp4
    ├── 1/
    │   └── 0.mp4
    └── ...
"""

import numpy as np
import scipy.spatial.transform as st
import pickle
import json
import cv2
import zarr
from pathlib import Path
from typing import List, Dict
import argparse
from tqdm import tqdm


class DiffusionPolicyConverter:
    """转换为 Diffusion Policy 格式"""
    
    @staticmethod
    def frankx_euler_to_rotvec(euler_zyx: np.ndarray) -> np.ndarray:
        """
        Frankx欧拉角(ZYX) -> 旋转向量
        
        Args:
            euler_zyx: [a, b, c] where a=Yaw, b=Pitch, c=Roll (ZYX顺序)
        
        Returns:
            rotvec: [rx, ry, rz] 旋转向量
        """
        if euler_zyx.ndim == 1:
            rot = st.Rotation.from_euler('ZYX', euler_zyx)
            return rot.as_rotvec()
        else:
            # 批量转换
            rotvecs = []
            for euler in euler_zyx:
                rot = st.Rotation.from_euler('ZYX', euler)
                rotvecs.append(rot.as_rotvec())
            return np.array(rotvecs)
    
    @staticmethod
    def load_episode(episode_folder: Path) -> Dict:
        """加载单个episode数据"""
        pkl_file = episode_folder / 'data.pkl'
        with open(pkl_file, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def images_to_video(image_folder: Path, output_video: Path, fps: int = 30):
        """
        将图像序列转换为MP4视频
        
        Args:
            image_folder: 包含 frame_xxxx.jpg 的文件夹
            output_video: 输出视频路径
            fps: 帧率
        """
        # 获取所有图像文件
        image_files = sorted(image_folder.glob('frame_*.jpg'))
        
        if len(image_files) == 0:
            raise ValueError(f"No images found in {image_folder}")
        
        # 读取第一张图像获取尺寸
        first_image = cv2.imread(str(image_files[0]))
        height, width = first_image.shape[:2]
        
        # 创建视频写入器 (H.264编码)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
        # 写入所有图像
        for img_file in image_files:
            img = cv2.imread(str(img_file))
            out.write(img)
        
        out.release()
        return len(image_files)
    
    @staticmethod
    def convert_dataset(
        input_dir: str,
        output_dir: str,
        fps: int = 30,
        use_action_as_obs: bool = False,
        verbose: bool = True
    ):
        """
        批量转换数据集
        
        Args:
            input_dir: 输入目录（包含 episode_xxxx 文件夹）
            output_dir: 输出目录
            fps: 视频帧率
            use_action_as_obs: 是否使用 action 作为 observation（如果 obs 数据质量不好）
            verbose: 是否打印详细信息
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # 创建输出目录
        output_path.mkdir(parents=True, exist_ok=True)
        videos_dir = output_path / 'videos'
        videos_dir.mkdir(exist_ok=True)
        
        # 获取所有episode文件夹
        episode_folders = sorted(input_path.glob('episode_*'))
        
        if len(episode_folders) == 0:
            print(f"❌ 在 {input_dir} 中没有找到episode文件夹")
            return
        
        print(f"\n{'='*60}")
        print(f"数据转换: Diffusion Policy 格式")
        print(f"输入目录: {input_path}")
        print(f"输出目录: {output_path}")
        print(f"Episode数量: {len(episode_folders)}")
        print(f"视频帧率: {fps} FPS")
        print(f"{'='*60}\n")
        
        # 创建 Zarr store
        zarr_path = output_path / 'replay_buffer.zarr'
        root = zarr.open(str(zarr_path), mode='w')
        data_group = root.create_group('data')
        meta_group = root.create_group('meta')
        
        # 收集所有数据
        all_timestamps = []
        all_robot_eef_pose = []
        all_actions = []
        episode_ends = []
        
        total_steps = 0
        
        # 处理每个episode
        for ep_idx, episode_folder in enumerate(tqdm(episode_folders, desc="转换Episodes")):
            if verbose:
                print(f"\n处理 {episode_folder.name}...")
            
            # 1. 加载数据
            episode_data = DiffusionPolicyConverter.load_episode(episode_folder)
            n_steps = episode_data['n_steps']
            
            # 2. 转换图像为视频
            images_folder = episode_folder / 'images'
            episode_video_dir = videos_dir / str(ep_idx)
            episode_video_dir.mkdir(exist_ok=True)
            
            video_path = episode_video_dir / '0.mp4'  # Camera 0
            n_frames = DiffusionPolicyConverter.images_to_video(
                images_folder, video_path, fps=fps
            )
            
            if verbose:
                print(f"  ✓ 视频: {n_frames} 帧 -> {video_path}")
            
            # 3. 转换低维数据
            # 提取位置和姿态
            if use_action_as_obs:
                # 使用 action 作为 observation
                obs_position = episode_data['robot_action_pose'][:, :3]
                obs_euler_zyx = episode_data['robot_action_pose'][:, 3:6]
                obs_gripper = episode_data['robot_action_gripper']
            else:
                # 使用 obs
                obs_position = episode_data['robot_obs_pose'][:, :3]
                obs_euler_zyx = episode_data['robot_obs_pose'][:, 3:6]
                obs_gripper = episode_data['robot_obs_gripper']
            
            action_position = episode_data['robot_action_pose'][:, :3]
            action_euler_zyx = episode_data['robot_action_pose'][:, 3:6]
            action_gripper = episode_data['robot_action_gripper']
            
            # 欧拉角 -> 旋转向量
            obs_rotvec = DiffusionPolicyConverter.frankx_euler_to_rotvec(obs_euler_zyx)
            action_rotvec = DiffusionPolicyConverter.frankx_euler_to_rotvec(action_euler_zyx)
            
            # 组合为 7D: [x, y, z, rx, ry, rz, gripper]
            robot_eef_pose = np.concatenate([
                obs_position, 
                obs_rotvec, 
                obs_gripper.reshape(-1, 1)
            ], axis=1)
            
            action = np.concatenate([
                action_position, 
                action_rotvec, 
                action_gripper.reshape(-1, 1)
            ], axis=1)
            
            # 添加到总数据
            all_timestamps.append(episode_data['timestamp'])
            all_robot_eef_pose.append(robot_eef_pose)
            all_actions.append(action)
            
            total_steps += n_steps
            episode_ends.append(total_steps)
            
            if verbose:
                print(f"  ✓ 数据: {n_steps} 步")
                print(f"    - robot_eef_pose: {robot_eef_pose.shape}")
                print(f"    - action: {action.shape}")
        
        # 4. 合并所有数据并保存到 Zarr
        print(f"\n保存到 Zarr...")
        
        all_timestamps = np.concatenate(all_timestamps, axis=0)
        all_robot_eef_pose = np.concatenate(all_robot_eef_pose, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        episode_ends = np.array(episode_ends, dtype=np.int64)
        
        # 保存数据
        data_group.create_dataset('timestamp', data=all_timestamps, 
                                  chunks=(1,), dtype=np.float64)
        data_group.create_dataset('robot_eef_pose', data=all_robot_eef_pose, 
                                  chunks=(1, 7), dtype=np.float64)
        data_group.create_dataset('action', data=all_actions, 
                                  chunks=(1, 7), dtype=np.float64)
        
        # 保存元数据
        meta_group.create_dataset('episode_ends', data=episode_ends, 
                                  chunks=None, dtype=np.int64)
        
        print(f"  ✓ timestamp: {all_timestamps.shape}")
        print(f"  ✓ robot_eef_pose: {all_robot_eef_pose.shape}")
        print(f"  ✓ action: {all_actions.shape}")
        print(f"  ✓ episode_ends: {episode_ends.shape}")
        
        # 5. 保存配置信息
        config = {
            'n_episodes': len(episode_folders),
            'n_steps': total_steps,
            'fps': fps,
            'data_format': '7D',
            'data_description': {
                'robot_eef_pose': '[x, y, z, rx, ry, rz, gripper]',
                'action': '[x, y, z, rx, ry, rz, gripper]',
                'rotation_format': 'rotation_vector',
            },
            'episode_ends': episode_ends.tolist(),
            'use_action_as_obs': use_action_as_obs,
        }
        
        config_file = output_path / 'dataset_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ 转换完成！")
        print(f"  Episodes: {len(episode_folders)}")
        print(f"  总步数: {total_steps}")
        print(f"  平均步数/episode: {total_steps/len(episode_folders):.0f}")
        print(f"  输出目录: {output_path.absolute()}")
        print(f"  配置文件: {config_file}")
        print(f"{'='*60}\n")


def verify_conversion(output_dir: str):
    """验证转换结果"""
    output_path = Path(output_dir)
    
    print(f"\n{'='*60}")
    print(f"验证转换结果")
    print(f"{'='*60}\n")
    
    # 检查目录结构
    zarr_path = output_path / 'replay_buffer.zarr'
    videos_dir = output_path / 'videos'
    config_file = output_path / 'dataset_config.json'
    
    print("目录结构:")
    print(f"  ✓ replay_buffer.zarr: {zarr_path.exists()}")
    print(f"  ✓ videos/: {videos_dir.exists()}")
    print(f"  ✓ dataset_config.json: {config_file.exists()}")
    
    # 读取 Zarr 数据
    if zarr_path.exists():
        root = zarr.open(str(zarr_path), mode='r')
        print(f"\nZarr 数据:")
        print(f"  data/timestamp: {root['data']['timestamp'].shape}")
        print(f"  data/robot_eef_pose: {root['data']['robot_eef_pose'].shape}")
        print(f"  data/action: {root['data']['action'].shape}")
        print(f"  meta/episode_ends: {root['meta']['episode_ends'].shape}")
        print(f"  episode_ends: {root['meta']['episode_ends'][:]}")
    
    # 读取配置
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"\n配置信息:")
        print(f"  Episodes: {config['n_episodes']}")
        print(f"  总步数: {config['n_steps']}")
        print(f"  帧率: {config['fps']} FPS")
        print(f"  数据格式: {config['data_format']}")
    
    # 检查视频
    if videos_dir.exists():
        video_files = list(videos_dir.glob('*/*.mp4'))
        print(f"\n视频文件:")
        print(f"  总数: {len(video_files)}")
        for vf in video_files[:5]:  # 只显示前5个
            print(f"    - {vf.relative_to(output_path)}")
        if len(video_files) > 5:
            print(f"    ... 还有 {len(video_files)-5} 个")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='转换数据为 Diffusion Policy 格式')
    parser.add_argument('--input-dir', default='../../data/trajectories',
                       help='输入目录（包含 episode_xxxx 文件夹）')
    parser.add_argument('--output-dir', default='../../data/diffusion_policy_dataset',
                       help='输出目录')
    parser.add_argument('--fps', type=int, default=30,
                       help='视频帧率')
    parser.add_argument('--use-action-as-obs', action='store_true',
                       help='使用 action 作为 observation（如果 obs 数据质量不好）')
    parser.add_argument('--verify', action='store_true',
                       help='验证转换结果')
    parser.add_argument('--quiet', action='store_true',
                       help='静默模式')
    
    args = parser.parse_args()
    
    # 转换数据集
    DiffusionPolicyConverter.convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        use_action_as_obs=args.use_action_as_obs,
        verbose=not args.quiet
    )
    
    # 验证转换
    if args.verify:
        verify_conversion(args.output_dir)


if __name__ == "__main__":
    main()
