#!/usr/bin/env python3
"""
推理日志可视化工具

功能:
1. 加载和分析推理日志 (inference_log_*.json)
2. 可视化状态轨迹
3. 可视化预测动作
4. 分析推理性能和一致性

用法:
    python visualize_inference_log.py --log_path <path_to_log.json>
    python visualize_inference_log.py --log_dir <log_directory>  # 分析最新日志
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import warnings
import os

# 确保以 `python toolbox/xxx.py` 方式运行时也能 import 项目内模块
sys.path.insert(0, str(Path(__file__).parent.parent))



class InferenceLogVisualizer:
    """推理日志可视化器"""
    
    def __init__(self, log_path: str):
        """
        Args:
            log_path: 推理日志文件路径
        """
        self.log_path = log_path
        print(f"[可视化] 加载推理日志: {log_path}")
        
        with open(log_path, 'r') as f:
            self.log_data = json.load(f)
        
        self.steps = self.log_data.get('steps', [])
        print(f"[可视化] 总步数: {len(self.steps)}")
        
        self._extract_data()
    
    def _extract_data(self):
        """提取和整理数据"""
        self.states = []
        self.actions = []
        self.timestamps = []
        
        for step in self.steps:
            # 提取状态
            state = step.get('input', {}).get('state', [])
            self.states.append(state)
            
            # 提取时间戳
            timestamp = step.get('input', {}).get('timestamp', 0)
            self.timestamps.append(timestamp)
            
            # 提取动作（取预测的动作序列）
            action_data = step.get('action', {})
            action_values = action_data.get('values', [])
            self.actions.append(action_values)
        
        self.states = np.array(self.states) if self.states else np.array([])
        self.timestamps = np.array(self.timestamps)
        
        print(f"[可视化] 状态 shape: {self.states.shape}")
        print(f"[可视化] 动作 steps: {len(self.actions)}")
        
        if len(self.actions) > 0 and len(self.actions[0]) > 0:
            print(f"[可视化] 每步预测动作数: {len(self.actions[0])}")
            print(f"[可视化] 动作维度: {len(self.actions[0][0])}")
    
    def visualize_state_trajectory(self, save_path: str = None):
        """可视化状态轨迹
        
        Args:
            save_path: 保存路径（可选）
        """
        if len(self.states) == 0:
            print("[错误] 没有状态数据")
            return
        
        state_dim = self.states.shape[1]
        T = len(self.states)
        
        # 定义状态标签
        if state_dim == 8:
            labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
        elif state_dim == 7:
            labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        else:
            labels = [f'dim_{i}' for i in range(state_dim)]
        
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        fig.suptitle(f'Inference State Trajectory ({T} steps)\n{Path(self.log_path).name}', fontsize=14)
        
        # 1. 位置 (x, y, z) 时间序列
        ax1 = fig.add_subplot(gs[0, :2])
        for i, label in enumerate(['x', 'y', 'z']):
            if i < state_dim:
                ax1.plot(self.states[:, i], label=label, linewidth=2)
        ax1.set_xlabel('Inference Step')
        ax1.set_ylabel('Position (m)')
        ax1.set_title('End-Effector Position Changes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 3D 轨迹
        ax2 = fig.add_subplot(gs[0, 2], projection='3d')
        if state_dim >= 3:
            ax2.plot(self.states[:, 0], self.states[:, 1], self.states[:, 2], 
                    'b-', linewidth=2, alpha=0.7)
            ax2.scatter(self.states[0, 0], self.states[0, 1], self.states[0, 2], 
                       c='g', s=100, label='Start', marker='o')
            ax2.scatter(self.states[-1, 0], self.states[-1, 1], self.states[-1, 2], 
                       c='r', s=100, label='End', marker='s')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title('3D Trajectory')
            ax2.legend()
        
        # 3. 姿态变化 (四元数)
        ax3 = fig.add_subplot(gs[1, :2])
        if state_dim >= 7:
            for i, label in enumerate(['qx', 'qy', 'qz', 'qw']):
                ax3.plot(self.states[:, 3+i], label=label, linewidth=2, alpha=0.8)
            ax3.set_xlabel('Inference Step')
            ax3.set_ylabel('Quaternion Components')
            ax3.set_title('End-Effector Orientation Changes')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Gripper 状态
        ax4 = fig.add_subplot(gs[1, 2])
        if state_dim >= 8:
            ax4.plot(self.states[:, 7], 'b-', linewidth=2)
            ax4.fill_between(range(T), self.states[:, 7], alpha=0.3)
            ax4.set_xlabel('Inference Step')
            ax4.set_ylabel('Gripper State')
            ax4.set_title('Gripper Open/Close State')
            ax4.set_ylim(-0.1, 1.1)
            ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Gripper Data', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Gripper State (N/A)')
        
        # 5. 时间戳分析
        ax5 = fig.add_subplot(gs[2, :2])
        if len(self.timestamps) > 1:
            # 计算时间间隔
            time_diffs = np.diff(self.timestamps)
            ax5.bar(range(len(time_diffs)), time_diffs, alpha=0.7, color='steelblue')
            ax5.axhline(y=np.mean(time_diffs), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(time_diffs)*1000:.1f}ms')
            ax5.set_xlabel('Inference Step')
            ax5.set_ylabel('Time Interval (s)')
            ax5.set_title('Inference Time Intervals')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. 统计信息
        ax6 = fig.add_subplot(gs[2, 2])
        
        stats_text = f"Total Steps: {T}\n"
        if len(self.timestamps) > 1:
            total_time = self.timestamps[-1] - self.timestamps[0]
            stats_text += f"Total Time: {total_time:.2f}s\n"
            stats_text += f"Avg Frequency: {T/total_time:.1f} Hz\n\n"
        
        stats_text += "Position Range:\n"
        for i, label in enumerate(['x', 'y', 'z']):
            if i < state_dim:
                stats_text += f"  {label}: [{self.states[:, i].min():.3f}, {self.states[:, i].max():.3f}]\n"
        
        if state_dim >= 8:
            stats_text += f"\nGripper:\n"
            stats_text += f"  Range: [{self.states[:, 7].min():.3f}, {self.states[:, 7].max():.3f}]"
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.axis('off')
        ax6.set_title('Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[可视化] 图像已保存: {save_path}")
        
        plt.show()
    
    def visualize_action_predictions(self, step_indices: list = None, save_path: str = None):
        """可视化动作预测
        
        Args:
            step_indices: 要可视化的步骤索引列表，默认为均匀采样
            save_path: 保存路径（可选）
        """
        if len(self.actions) == 0:
            print("[错误] 没有动作数据")
            return
        
        # 选择要可视化的步骤
        if step_indices is None:
            n_samples = min(8, len(self.actions))
            step_indices = np.linspace(0, len(self.actions)-1, n_samples, dtype=int).tolist()
        
        n_plots = len(step_indices)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
        fig.suptitle(f'Action Prediction Sequence Visualization\n{Path(self.log_path).name}', fontsize=14)
        
        for plot_idx, step_idx in enumerate(step_indices):
            if step_idx >= len(self.actions):
                continue
                
            ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1)
            
            action_seq = np.array(self.actions[step_idx])  # (n_action_steps, action_dim)
            
            if len(action_seq) == 0:
                continue
            
            n_action_steps, action_dim = action_seq.shape
            
            # 绘制位置预测
            for i, (label, color) in enumerate(zip(['x', 'y', 'z'], ['r', 'g', 'b'])):
                if i < action_dim:
                    ax.plot(action_seq[:, i], label=label, color=color, linewidth=2, marker='o', markersize=4)
            
            # 标记当前状态
            if step_idx < len(self.states):
                current_state = self.states[step_idx]
                for i, (label, color) in enumerate(zip(['x', 'y', 'z'], ['r', 'g', 'b'])):
                    if i < len(current_state):
                        ax.axhline(y=current_state[i], color=color, linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Action Step')
            ax.set_ylabel('Position')
            ax.set_title(f'Step {step_idx}')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[可视化] 图像已保存: {save_path}")
        
        plt.show()
    
    def visualize_action_consistency(self, save_path: str = None):
        """分析动作预测的一致性
        
        检查连续步之间预测动作的一致性
        
        Args:
            save_path: 保存路径（可选）
        """
        if len(self.actions) < 2:
            print("[错误] 需要至少2步数据")
            return
        
        # 计算连续步之间的预测差异
        # 比较 step i 的 action[k] 和 step i+1 的 action[k-1]
        # 因为 step i+1 执行后，action[1] 应该接近 step i 的 action[0]
        
        first_action_diffs = []  # 每步预测的第一个动作的差异
        overlap_diffs = []  # 重叠部分的差异
        
        for i in range(len(self.actions) - 1):
            curr_actions = np.array(self.actions[i])
            next_actions = np.array(self.actions[i + 1])
            
            if len(curr_actions) == 0 or len(next_actions) == 0:
                continue
            
            # 第一个动作的差异
            first_diff = np.linalg.norm(curr_actions[0, :3] - next_actions[0, :3])
            first_action_diffs.append(first_diff)
            
            # 重叠部分的差异（curr[1:] vs next[:-1]）
            if len(curr_actions) > 1 and len(next_actions) > 1:
                min_len = min(len(curr_actions) - 1, len(next_actions) - 1)
                for j in range(min_len):
                    diff = np.linalg.norm(curr_actions[j+1, :3] - next_actions[j, :3])
                    overlap_diffs.append(diff)
        
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(f'Action Prediction Consistency Analysis\n{Path(self.log_path).name}', fontsize=14)
        
        # 1. 第一个动作的差异
        ax1 = fig.add_subplot(221)
        if first_action_diffs:
            ax1.plot(first_action_diffs, 'b-', linewidth=1, alpha=0.7)
            ax1.axhline(y=np.mean(first_action_diffs), color='r', linestyle='--',
                       label=f'平均: {np.mean(first_action_diffs):.4f}')
            ax1.set_xlabel('Inference Step')
            ax1.set_ylabel('Position Difference (m)')
            ax1.set_title('First Action Difference Between Consecutive Steps')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. 差异直方图
        ax2 = fig.add_subplot(222)
        if first_action_diffs:
            ax2.hist(first_action_diffs, bins=30, density=True, alpha=0.7, 
                    color='steelblue', edgecolor='black')
            ax2.axvline(x=np.mean(first_action_diffs), color='r', linestyle='--',
                       label=f'Mean: {np.mean(first_action_diffs):.4f}')
            ax2.set_xlabel('Position Difference (m)')
            ax2.set_ylabel('Density')
            ax2.set_title('Difference Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 实际执行 vs 预测
        ax3 = fig.add_subplot(223)
        if len(self.states) > 1:
            # 实际状态变化
            actual_diffs = np.linalg.norm(np.diff(self.states[:, :3], axis=0), axis=1)
            ax3.plot(actual_diffs, 'b-', label='Actual Movement', alpha=0.7)
            
            # 预测的第一步
            if first_action_diffs and len(first_action_diffs) == len(actual_diffs):
                ax3.plot(first_action_diffs, 'r--', label='Predicted Difference', alpha=0.7)
            
            ax3.set_xlabel('Inference Step')
            ax3.set_ylabel('Displacement (m)')
            ax3.set_title('Actual Movement vs Predicted Difference')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 统计信息
        ax4 = fig.add_subplot(224)
        
        stats_text = "Consistency Analysis:\n\n"
        if first_action_diffs:
            stats_text += f"First Action Difference:\n"
            stats_text += f"  Mean: {np.mean(first_action_diffs):.4f} m\n"
            stats_text += f"  Std: {np.std(first_action_diffs):.4f} m\n"
            stats_text += f"  Max: {np.max(first_action_diffs):.4f} m\n"
            stats_text += f"  Min: {np.min(first_action_diffs):.4f} m\n\n"
        
        if len(self.states) > 1:
            actual_diffs = np.linalg.norm(np.diff(self.states[:, :3], axis=0), axis=1)
            stats_text += f"Actual Movement:\n"
            stats_text += f"  Mean: {np.mean(actual_diffs):.4f} m\n"
            stats_text += f"  Std: {np.std(actual_diffs):.4f} m\n"
            stats_text += f"  Total: {np.sum(actual_diffs):.4f} m\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        ax4.set_title('Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[可视化] 图像已保存: {save_path}")
        
        plt.show()
    
    def visualize_action_sequence_detail(self, step_idx: int, save_path: str = None):
        """详细可视化单步的动作预测序列
        
        Args:
            step_idx: 步骤索引
            save_path: 保存路径（可选）
        """
        if step_idx >= len(self.actions):
            print(f"[错误] 步骤索引超出范围: {step_idx}")
            return
        
        action_seq = np.array(self.actions[step_idx])
        if len(action_seq) == 0:
            print(f"[错误] 步骤 {step_idx} 没有动作数据")
            return
        
        n_action_steps, action_dim = action_seq.shape
        
        # 定义动作标签
        if action_dim == 8:
            labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'gripper']
        elif action_dim == 7:
            labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        else:
            labels = [f'dim_{i}' for i in range(action_dim)]
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        fig.suptitle(f'Step {step_idx} Action Prediction Details\n{Path(self.log_path).name}', fontsize=14)
        
        # 1. 位置预测
        ax1 = fig.add_subplot(gs[0, :2])
        for i, label in enumerate(['x', 'y', 'z']):
            ax1.plot(action_seq[:, i], label=label, linewidth=2, marker='o')
        
        # 当前状态
        if step_idx < len(self.states):
            current_state = self.states[step_idx]
            for i, (label, color) in enumerate(zip(['x', 'y', 'z'], plt.cm.tab10.colors[:3])):
                ax1.axhline(y=current_state[i], color=color, linestyle='--', alpha=0.5)
                ax1.plot(0, current_state[i], 's', color=color, markersize=10)
        
        ax1.set_xlabel('Action Step')
        ax1.set_ylabel('Position (m)')
        ax1.set_title('Position Prediction Sequence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 3D 预测轨迹
        ax2 = fig.add_subplot(gs[0, 2], projection='3d')
        ax2.plot(action_seq[:, 0], action_seq[:, 1], action_seq[:, 2], 
                'b-', linewidth=2, marker='o')
        ax2.scatter(action_seq[0, 0], action_seq[0, 1], action_seq[0, 2], 
                   c='g', s=100, label='Predicted Start')
        ax2.scatter(action_seq[-1, 0], action_seq[-1, 1], action_seq[-1, 2], 
                   c='r', s=100, label='Predicted End')
        
        if step_idx < len(self.states):
            current_state = self.states[step_idx]
            ax2.scatter(current_state[0], current_state[1], current_state[2], 
                       c='orange', s=150, marker='*', label='Current State')
        
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('3D Prediction Trajectory')
        ax2.legend()
        
        # 3. 姿态预测 (四元数)
        ax3 = fig.add_subplot(gs[1, :2])
        if action_dim >= 7:
            for i, label in enumerate(['qx', 'qy', 'qz', 'qw']):
                ax3.plot(action_seq[:, 3+i], label=label, linewidth=2, marker='o')
            ax3.set_xlabel('Action Step')
            ax3.set_ylabel('Quaternion Components')
            ax3.set_title('Orientation Prediction Sequence')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Gripper 预测
        ax4 = fig.add_subplot(gs[1, 2])
        if action_dim >= 8:
            ax4.plot(action_seq[:, 7], 'b-', linewidth=2, marker='o')
            ax4.fill_between(range(n_action_steps), action_seq[:, 7], alpha=0.3)
            ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
            ax4.set_xlabel('Action Step')
            ax4.set_ylabel('Gripper State')
            ax4.set_title('Gripper Prediction')
            ax4.set_ylim(-0.1, 1.1)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No Gripper Data', ha='center', va='center', transform=ax4.transAxes)
        
        # 5. 动作变化 (速度)
        ax5 = fig.add_subplot(gs[2, :2])
        if n_action_steps > 1:
            velocities = np.diff(action_seq[:, :3], axis=0)
            for i, label in enumerate(['vx', 'vy', 'vz']):
                ax5.plot(velocities[:, i], label=label, linewidth=2, marker='o')
            ax5.set_xlabel('Action Step')
            ax5.set_ylabel('Velocity (Difference)')
            ax5.set_title('Position Change Velocity')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. 数值表格
        ax6 = fig.add_subplot(gs[2, 2])
        
        table_text = f"Action Prediction ({n_action_steps} steps, {action_dim}D):\n\n"
        table_text += "Step |    x    |    y    |    z    "
        if action_dim >= 8:
            table_text += "| gripper"
        table_text += "\n" + "-" * 45 + "\n"
        
        for i in range(min(n_action_steps, 8)):
            table_text += f"  {i}  | {action_seq[i, 0]:7.4f} | {action_seq[i, 1]:7.4f} | {action_seq[i, 2]:7.4f}"
            if action_dim >= 8:
                table_text += f" | {action_seq[i, 7]:7.3f}"
            table_text += "\n"
        
        if n_action_steps > 8:
            table_text += "... (more)\n"
        
        ax6.text(0.02, 0.98, table_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.axis('off')
        ax6.set_title('Prediction Values')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[可视化] 图像已保存: {save_path}")
        
        plt.show()
    
    def compare_with_training(self, training_zarr_path: str, save_path: str = None):
        """将推理轨迹与训练数据进行对比
        
        Args:
            training_zarr_path: 训练数据 zarr 路径
            save_path: 保存路径（可选）
        """
        try:
            import zarr
            training_root = zarr.open(training_zarr_path, mode='r')
            training_actions = training_root['data']['action'][:]
            
            fig = plt.figure(figsize=(16, 10))
            fig.suptitle(f'Inference Trajectory vs Training Data Distribution\n{Path(self.log_path).name}', fontsize=14)
            
            # 提取推理中的第一个预测动作
            inference_first_actions = []
            for action_seq in self.actions:
                if len(action_seq) > 0:
                    inference_first_actions.append(action_seq[0])
            inference_first_actions = np.array(inference_first_actions)
            
            # X, Y, Z 分布对比
            for i, label in enumerate(['X', 'Y', 'Z']):
                ax = fig.add_subplot(2, 3, i + 1)
                
                # 训练数据分布
                ax.hist(training_actions[:, i], bins=50, density=True, alpha=0.5,
                       label='Training Data', color='blue')
                
                # 推理数据分布
                if len(inference_first_actions) > 0 and i < inference_first_actions.shape[1]:
                    ax.hist(inference_first_actions[:, i], bins=30, density=True, alpha=0.5,
                           label='Inference Trajectory', color='red')
                
                ax.set_xlabel(label)
                ax.set_ylabel('Density')
                ax.set_title(f'{label} Distribution Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # 3D 对比
            ax4 = fig.add_subplot(2, 3, 4, projection='3d')
            
            # 训练数据采样
            sample_idx = np.random.choice(len(training_actions), 
                                         min(5000, len(training_actions)), replace=False)
            ax4.scatter(training_actions[sample_idx, 0], 
                       training_actions[sample_idx, 1],
                       training_actions[sample_idx, 2],
                       c='blue', alpha=0.1, s=1, label='Training Data')
            
            # 推理轨迹
            if len(inference_first_actions) > 0:
                ax4.plot(inference_first_actions[:, 0],
                        inference_first_actions[:, 1],
                        inference_first_actions[:, 2],
                        'r-', linewidth=2, label='Inference Trajectory')
            
            ax4.set_xlabel('X')
            ax4.set_ylabel('Y')
            ax4.set_zlabel('Z')
            ax4.set_title('3D Space Comparison')
            ax4.legend()
            
            # 状态轨迹
            ax5 = fig.add_subplot(2, 3, 5)
            if len(self.states) > 0:
                ax5.plot(self.states[:, 0], 'r-', label='state_x', linewidth=2)
                ax5.plot(self.states[:, 1], 'g-', label='state_y', linewidth=2)
                ax5.plot(self.states[:, 2], 'b-', label='state_z', linewidth=2)
            ax5.set_xlabel('Inference Step')
            ax5.set_ylabel('Position')
            ax5.set_title('Inference Process State Changes')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 统计信息
            ax6 = fig.add_subplot(2, 3, 6)
            stats_text = "Data Comparison:\n\n"
            stats_text += f"Training Data:\n"
            stats_text += f"  Samples: {len(training_actions)}\n"
            for i, label in enumerate(['x', 'y', 'z']):
                stats_text += f"  {label}: [{training_actions[:, i].min():.3f}, {training_actions[:, i].max():.3f}]\n"
            
            stats_text += f"\nInference Trajectory:\n"
            stats_text += f"  Steps: {len(inference_first_actions)}\n"
            if len(inference_first_actions) > 0:
                for i, label in enumerate(['x', 'y', 'z']):
                    stats_text += f"  {label}: [{inference_first_actions[:, i].min():.3f}, {inference_first_actions[:, i].max():.3f}]\n"
            
            ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax6.axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"[可视化] 图像已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"[错误] 加载训练数据失败: {e}")
    
    def interactive_browse(self):
        """交互式浏览模式"""
        print("\n" + "="*60)
        print("交互式浏览模式")
        print("="*60)
        print(f"总步数: {len(self.steps)}")
        print("命令:")
        print("  s           - 显示状态轨迹")
        print("  a           - 显示动作预测概览")
        print("  c           - 分析动作一致性")
        print("  <数字>      - 查看指定步骤的动作详情")
        print("  t <zarr>    - 与训练数据对比")
        print("  q           - 退出")
        print()
        
        while True:
            try:
                cmd = input("输入命令: ").strip()
                
                if cmd.lower() == 'q':
                    print("退出浏览模式")
                    break
                elif cmd.lower() == 's':
                    self.visualize_state_trajectory()
                elif cmd.lower() == 'a':
                    self.visualize_action_predictions()
                elif cmd.lower() == 'c':
                    self.visualize_action_consistency()
                elif cmd.lower().startswith('t '):
                    zarr_path = cmd[2:].strip()
                    self.compare_with_training(zarr_path)
                elif cmd.isdigit():
                    step_idx = int(cmd)
                    self.visualize_action_sequence_detail(step_idx)
                else:
                    print(f"[错误] 未知命令: {cmd}")
                    
            except KeyboardInterrupt:
                print("\n退出浏览模式")
                break
            except Exception as e:
                print(f"[错误] {e}")


def find_latest_log(log_dir: str) -> str:
    """找到最新的推理日志"""
    log_path = Path(log_dir)
    log_files = list(log_path.glob("inference_log_*.json"))
    
    if not log_files:
        raise FileNotFoundError(f"在 {log_dir} 中没有找到推理日志")
    
    # 按修改时间排序
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(log_files[0])


def main():
    parser = argparse.ArgumentParser(description='推理日志可视化工具')
    parser.add_argument('--log_path', type=str, default=None,
                       help='推理日志文件路径')
    parser.add_argument('--log_dir', type=str, default=None,
                       help='推理日志目录（将加载最新的日志）')
    parser.add_argument('--state', action='store_true',
                       help='显示状态轨迹')
    parser.add_argument('--action', action='store_true',
                       help='显示动作预测')
    parser.add_argument('--consistency', action='store_true',
                       help='分析动作一致性')
    parser.add_argument('--step', type=int, default=None,
                       help='查看指定步骤的动作详情')
    parser.add_argument('--compare_zarr', type=str, default=None,
                       help='与训练数据对比的 zarr 路径')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式浏览模式')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存图像的目录')
    
    args = parser.parse_args()
    
    # 确定日志路径
    if args.log_path:
        log_path = args.log_path
    elif args.log_dir:
        log_path = find_latest_log(args.log_dir)
        print(f"[可视化] 使用最新日志: {log_path}")
    else:
        # 默认使用 server/log 目录
        default_log_dir = Path(__file__).parent.parent / "server" / "log"
        if default_log_dir.exists():
            log_path = find_latest_log(str(default_log_dir))
            print(f"[可视化] 使用最新日志: {log_path}")
        else:
            print("[错误] 请指定 --log_path 或 --log_dir")
            return
    
    visualizer = InferenceLogVisualizer(log_path)
    
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
    
    if args.interactive:
        visualizer.interactive_browse()
    elif args.state:
        save_path = str(save_dir / 'state_trajectory.png') if args.save_dir else None
        visualizer.visualize_state_trajectory(save_path=save_path)
    elif args.action:
        save_path = str(save_dir / 'action_predictions.png') if args.save_dir else None
        visualizer.visualize_action_predictions(save_path=save_path)
    elif args.consistency:
        save_path = str(save_dir / 'action_consistency.png') if args.save_dir else None
        visualizer.visualize_action_consistency(save_path=save_path)
    elif args.step is not None:
        save_path = str(save_dir / f'step_{args.step}_detail.png') if args.save_dir else None
        visualizer.visualize_action_sequence_detail(args.step, save_path=save_path)
    elif args.compare_zarr:
        save_path = str(save_dir / 'compare_training.png') if args.save_dir else None
        visualizer.compare_with_training(args.compare_zarr, save_path=save_path)
    else:
        # 默认显示所有
        visualizer.visualize_state_trajectory()
        visualizer.visualize_action_predictions()
        visualizer.visualize_action_consistency()


if __name__ == '__main__':
    main()

