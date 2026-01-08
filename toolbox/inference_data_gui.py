import streamlit as st
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys

# --- 路径配置 ---
sys.path.insert(0, str(Path(__file__).parent.parent))

# 尝试配置字体
try:
    from toolbox.mpl_fonts import setup_matplotlib_fonts
    setup_matplotlib_fonts(verbose=False)
except Exception:
    pass

class InferenceGUI:
    def __init__(self, log_path):
        self.log_path = log_path
        self.valid = False
        self.load_data()

    def load_data(self):
        try:
            with open(self.log_path, 'r') as f:
                self.log_data = json.load(f)
            
            self.steps = self.log_data.get('steps', [])
            if not self.steps:
                st.error("日志文件为空或格式错误")
                return

            # 提取数据
            self.states = []
            self.actions = [] # 预测的动作序列
            self.timestamps = []
            
            for step in self.steps:
                # 状态 (实际发生的事)
                state = step.get('input', {}).get('state', [])
                self.states.append(state)
                
                # 时间戳
                timestamp = step.get('input', {}).get('timestamp', 0)
                self.timestamps.append(timestamp)
                
                # 动作 (模型预测的未来)
                action_data = step.get('action', {})
                action_values = action_data.get('values', []) # 通常是 (T_pred, Dim)
                self.actions.append(action_values)
            
            self.states = np.array(self.states)
            self.timestamps = np.array(self.timestamps)
            # actions 是列表的列表，因为每次预测长度可能不同，或者为了效率保持 list
            
            self.valid = True
            self.state_dim = self.states.shape[1] if len(self.states) > 0 else 0
            
        except Exception as e:
            st.error(f"加载失败: {e}")
            self.valid = False

    def plot_replay_frame(self, step_idx):
        """核心功能：绘制某一帧的‘过去’与‘未来’"""
        if not self.valid: return

        # 1. 获取数据
        # 历史轨迹 (0 -> current)
        history_traj = self.states[:step_idx+1]
        current_state = self.states[step_idx]
        
        # 预测轨迹 (current -> future)
        pred_traj = np.array(self.actions[step_idx])
        
        # 2. 创建画布
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig)

        # --- 子图 1: 3D 空间轨迹 (上帝视角) ---
        ax3d = fig.add_subplot(gs[:, 0], projection='3d')
        
        # A. 画历史 (灰色)
        if len(history_traj) > 1:
            ax3d.plot(history_traj[:, 0], history_traj[:, 1], history_traj[:, 2], 
                     'k-', alpha=0.3, linewidth=1, label='History (Actual)')
        
        # B. 画当前点 (蓝色大点)
        ax3d.scatter(current_state[0], current_state[1], current_state[2], 
                    c='b', s=100, label='Current', zorder=10)
        
        # C. 画预测 (红色虚线)
        if len(pred_traj) > 0:
            # 预测轨迹通常是绝对坐标，如果它是相对坐标，这里需要额外处理。
            # 假设日志记录的是绝对坐标（常见情况）
            ax3d.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], 
                     'r--', linewidth=2, label='Prediction (Plan)')
            ax3d.scatter(pred_traj[-1, 0], pred_traj[-1, 1], pred_traj[-1, 2], 
                        c='r', s=50, marker='x')

        ax3d.set_title(f"Step {step_idx}: 3D 空间轨迹", fontsize=12)
        ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')
        ax3d.legend(loc='upper left', fontsize=9)
        
        # 设置一致的视角范围，避免画面抖动
        # 以整个数据集的范围为基准
        margin = 0.1
        x_min, x_max = self.states[:,0].min(), self.states[:,0].max()
        y_min, y_max = self.states[:,1].min(), self.states[:,1].max()
        z_min, z_max = self.states[:,2].min(), self.states[:,2].max()
        ax3d.set_xlim(x_min-margin, x_max+margin)
        ax3d.set_ylim(y_min-margin, y_max+margin)
        ax3d.set_zlim(z_min-margin, z_max+margin)

        # --- 子图 2: XYZ 时间曲线 (展开视角) ---
        ax2d = fig.add_subplot(gs[0, 1])
        
        # 定义显示窗口：显示过去 50 步 + 未来预测
        window_start = max(0, step_idx - 50)
        hist_steps = np.arange(window_start, step_idx + 1)
        hist_data = self.states[window_start:step_idx + 1]
        
        # 预测的时间轴 (紧接在当前步之后)
        pred_steps = np.arange(step_idx, step_idx + len(pred_traj))
        
        colors = ['r', 'g', 'b']
        labels = ['X', 'Y', 'Z']
        
        for i in range(3): # 只画 XYZ
            if i >= self.state_dim: break
            # 历史实线
            ax2d.plot(hist_steps, hist_data[:, i], color=colors[i], alpha=0.4, linestyle='-')
            # 当前点
            ax2d.scatter(step_idx, current_state[i], color=colors[i], s=30)
            # 预测虚线
            if len(pred_traj) > 0:
                ax2d.plot(pred_steps, pred_traj[:, i], color=colors[i], linestyle='--', linewidth=1.5, label=f'{labels[i]} Pred')

        ax2d.set_title("XYZ 随时间变化 (实线=历史, 虚线=预测)", fontsize=10)
        ax2d.axvline(x=step_idx, color='k', linestyle=':', alpha=0.5)
        ax2d.grid(True, alpha=0.3)
        
        # --- 子图 3: 夹爪/其他维度 ---
        ax_btm = fig.add_subplot(gs[1, 1])
        if self.state_dim >= 8: # 假设第8维是夹爪
            gripper_idx = 7
            ax_btm.plot(hist_steps, hist_data[:, gripper_idx], 'k-', alpha=0.6, label='Gripper Hist')
            if len(pred_traj) > 0:
                ax_btm.plot(pred_steps, pred_traj[:, gripper_idx], 'r--', label='Gripper Pred')
            ax_btm.set_title("夹爪状态 (Gripper)", fontsize=10)
            ax_btm.set_ylim(-0.1, 1.1)
            ax_btm.grid(True, alpha=0.3)
        else:
            # 如果没有夹爪，显示四元数的第一维或者留空
            ax_btm.text(0.5, 0.5, "无夹爪数据", ha='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def plot_consistency_analysis(self):
        """分析预测的一致性（抖动）"""
        if len(self.actions) < 2:
            st.warning("数据不足以进行一致性分析")
            return

        # 计算抖动：第 T 步预测的动作[0] vs 第 T+1 步预测的动作[0] (或实际执行的差异)
        # 这里我们计算：模型在 Step T 计划要去的位置，和它在 Step T+1 真正去的位置的差异，
        # 以及模型在 Step T 对 T+1 的预测，和 Step T+1 对 T+1 的预测的差异。
        
        jitter_metrics = []
        for i in range(len(self.actions) - 1):
            curr_pred = np.array(self.actions[i])
            next_pred = np.array(self.actions[i+1])
            
            if len(curr_pred) > 1 and len(next_pred) > 0:
                # 比较: Step T 预测的 "下一刻" (index 1) vs Step T+1 预测的 "当前" (index 0)
                # 理论上这两个应该很接近
                diff = np.linalg.norm(curr_pred[1, :3] - next_pred[0, :3])
                jitter_metrics.append(diff)
            else:
                jitter_metrics.append(0.0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 1. 抖动曲线
        ax1.plot(jitter_metrics, color='purple', alpha=0.7)
        ax1.set_title("预测抖动 (Prediction Jitter)", fontsize=12)
        ax1.set_ylabel("位移偏差 (m)")
        ax1.text(0, np.max(jitter_metrics)*0.9, "数值越低越平滑\n表示模型意图稳定", bbox=dict(facecolor='white', alpha=0.8))
        ax1.grid(True, alpha=0.3)

        # 2. 推理耗时 (如果有时间戳)
        if len(self.timestamps) > 1:
            latencies = np.diff(self.timestamps) * 1000 # 转毫秒
            ax2.hist(latencies, bins=30, color='teal', alpha=0.7)
            ax2.axvline(np.mean(latencies), color='r', linestyle='--', label=f'Mean: {np.mean(latencies):.1f}ms')
            ax2.set_title("推理延迟分布 (Inference Latency)", fontsize=12)
            ax2.set_xlabel("耗时 (ms)")
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "无时间戳数据", ha='center')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    def compare_with_training(self, zarr_path):
        """对比训练集分布"""
        try:
            import zarr
            root = zarr.open(zarr_path, mode='r')
            train_actions = root['data']['action'][:]
            
            # 提取推理的所有首个预测动作
            inf_actions = []
            for a in self.actions:
                if len(a) > 0: inf_actions.append(a[0])
            inf_actions = np.array(inf_actions)

            st.write("### 分布对比")
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            labels = ['X', 'Y', 'Z']
            
            for i in range(3):
                ax = axes[i]
                # 训练集
                ax.hist(train_actions[:, i], bins=50, density=True, alpha=0.4, color='blue', label='Train')
                # 推理集
                if len(inf_actions) > 0:
                    ax.hist(inf_actions[:, i], bins=30, density=True, alpha=0.6, color='red', label='Inference')
                ax.set_title(f"{labels[i]} 轴分布")
                if i == 0: ax.legend()
            
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"读取训练集失败: {e}")

# --- 界面布局 ---

st.set_page_config(layout="wide", page_title="Inference Log Analyst")
st.sidebar.title("🧠 推理日志分析")

# 1. 自动寻找日志
default_log_dir = Path(__file__).parent.parent / "server" / "log"
log_files = []
if default_log_dir.exists():
    log_files = sorted(list(default_log_dir.glob("inference_log_*.json")), key=lambda x: x.stat().st_mtime, reverse=True)

# 2. 侧边栏文件选择
if log_files:
    selected_file = st.sidebar.selectbox("选择日志文件", log_files, format_func=lambda x: x.name)
    log_path = str(selected_file)
else:
    log_path = st.sidebar.text_input("输入日志文件路径", "inference_log.json")

# 3. 加载
if 'gui' not in st.session_state or st.session_state.log_path_cache != log_path:
    if Path(log_path).exists():
        st.session_state.gui = InferenceGUI(log_path)
        st.session_state.log_path_cache = log_path
    else:
        st.sidebar.warning("文件不存在")

# 4. 主界面
if 'gui' in st.session_state and st.session_state.gui.valid:
    gui = st.session_state.gui
    
    # 顶部指标
    col1, col2, col3 = st.columns(3)
    col1.metric("总步数 (Steps)", len(gui.steps))
    if len(gui.timestamps) > 1:
        duration = gui.timestamps[-1] - gui.timestamps[0]
        col2.metric("总耗时 (Duration)", f"{duration:.1f} s")
        avg_freq = len(gui.steps) / duration if duration > 0 else 0
        col3.metric("平均频率 (Freq)", f"{avg_freq:.1f} Hz")
    
    # 标签页
    tab1, tab2, tab3 = st.tabs(["🕵️ 交互式回放 (Replay)", "📉 稳定性与延迟", "📊 训练集对比"])
    
    with tab1:
        # 交互滑块
        step_idx = st.slider("时间轴 (Step)", 0, len(gui.steps)-1, 0, key='replay_slider')
        
        # 显示当前步的详细数据
        gui.plot_replay_frame(step_idx)
        
        # 显示具体数值
        with st.expander("查看详细数值"):
            st.write("当前状态 (State):", gui.states[step_idx])
            st.write("预测动作 (Prediction):", np.array(gui.actions[step_idx]))

    with tab2:
        st.markdown("#### 预测一致性分析")
        st.caption("一致性衡量模型是否在每个时间步都做出类似的规划。如果抖动（Jitter）很大，说明模型在震荡。")
        gui.plot_consistency_analysis()

    with tab3:
        zarr_input = st.text_input("输入训练集 Zarr 路径以进行对比", 
                                  "/home/jikangye/workspace/baselines/vla-baselines/RealWorld-DP/data/demo_test.zarr")
        if st.button("开始对比"):
            if Path(zarr_input).exists():
                gui.compare_with_training(zarr_input)
            else:
                st.error("Zarr 文件不存在")
                
else:
    st.info("👈 请在左侧选择或输入有效的推理日志路径")