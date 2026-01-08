# Toolbox - Diffusion Policy 调试可视化工具

这个目录包含用于分析和调试 Diffusion Policy 真机部署问题的可视化工具。

## 字体问题解决

### 问题诊断

如果遇到中文字体缺失警告，按以下步骤排查：

**步骤 1: 检查系统是否已安装中文字体**
```bash
fc-list :lang=zh | head -5
```
如果看到 `WenQuanYi` 或 `Noto Sans CJK` 等字体，说明字体已安装。

**步骤 2: 清除 matplotlib 字体缓存（重要！）**
```bash
# 使用脚本清除缓存
bash toolbox/fix_font_cache.sh

# 或手动清除
rm -rf ~/.cache/matplotlib
rm -rf ~/.matplotlib
```

**步骤 3: 如果未安装字体，安装中文字体**
```bash
# 方法 1: 使用安装脚本（推荐）
bash toolbox/install_fonts.sh

# 方法 2: 手动安装（Ubuntu/Debian）
sudo apt-get update
sudo apt-get install -y fonts-wqy-microhei fonts-wqy-zenhei fonts-noto-cjk
```

**步骤 4: 重新运行可视化脚本**
代码会自动检测并使用中文字体。如果找到字体，会显示 `[字体] 使用字体: xxx`。

### 常见问题

**Q: 安装了字体但还是显示方块？**
A: 这是 matplotlib 缓存问题。运行 `bash toolbox/fix_font_cache.sh` 清除缓存。

**Q: 字体警告太多？**
A: 代码已自动抑制字体警告，不会影响使用。如果仍有警告，说明 matplotlib 缓存需要清除。

**Q: 不想安装字体可以吗？**
A: 可以。代码已配置警告抑制，即使没有中文字体也能正常运行，只是图表中的中文会显示为方块。

## 工具列表

### 1. `visualize_training_data.py` - 训练数据可视化

可视化训练集中的图像序列和动作轨迹。

**功能：**
- 浏览 episode 图像序列
- 可视化 action 动作变化（位置、姿态、夹爪）
- 分析整个数据集的动作分布
- 可视化 3D 工作空间
- 对比多个 episodes

**使用方法：**

```bash
# 查看指定 episode 的图像和动作
python toolbox/visualize_training_data.py \
    --zarr_path /path/to/data.zarr \
    --episode_idx 0

# 分析动作分布
python toolbox/visualize_training_data.py \
    --zarr_path /path/to/data.zarr \
    --action_analysis

# 显示 3D 工作空间
python toolbox/visualize_training_data.py \
    --zarr_path /path/to/data.zarr \
    --workspace

# 对比多个 episodes
python toolbox/visualize_training_data.py \
    --zarr_path /path/to/data.zarr \
    --compare 0 1 2

# 交互式浏览模式
python toolbox/visualize_training_data.py \
    --zarr_path /path/to/data.zarr \
    --interactive

# 保存图像
python toolbox/visualize_training_data.py \
    --zarr_path /path/to/data.zarr \
    --episode_idx 0 \
    --save_dir ./analysis_output
```

### 2. `visualize_inference_log.py` - 推理日志可视化

分析推理服务器产生的日志文件，用于诊断真机部署问题。

**功能：**
- 可视化推理过程中的状态轨迹（位置、姿态、夹爪）
- 显示模型预测的动作序列
- 分析连续预测之间的一致性
- 与训练数据进行对比分析

**使用方法：**

```bash
# 使用默认路径（自动加载 server/log 中最新的日志）
python toolbox/visualize_inference_log.py

# 指定日志文件
python toolbox/visualize_inference_log.py \
    --log_path /home/jikangye/workspace/baselines/vla-baselines/RealWorld-DP/server/log/inference_log_20260108_124729.json

# 显示状态轨迹
python toolbox/visualize_inference_log.py --state

# 显示动作预测概览
python toolbox/visualize_inference_log.py --action

# 分析动作一致性
python toolbox/visualize_inference_log.py --consistency

# 查看指定步骤的详细预测
python toolbox/visualize_inference_log.py --step 10

# 与训练数据对比
python toolbox/visualize_inference_log.py \
    --compare_zarr /path/to/training_data.zarr

# 交互式浏览模式
python toolbox/visualize_inference_log.py --interactive

# 保存图像
python toolbox/visualize_inference_log.py \
    --state \
    --save_dir ./analysis_output
```

## 常见调试场景

### 场景 1: 检查训练数据质量

```bash
# 查看数据集整体统计
python toolbox/visualize_training_data.py \
    --zarr_path /path/to/data.zarr \
    --action_analysis

# 浏览多个 episode 对比
python toolbox/visualize_training_data.py \
    --zarr_path /path/to/data.zarr \
    --compare 0 5 10 15
```

### 场景 2: 分析推理失败原因

```bash
# 查看推理轨迹
python toolbox/visualize_inference_log.py --state

# 检查预测一致性
python toolbox/visualize_inference_log.py --consistency

# 对比训练数据
python toolbox/visualize_inference_log.py \
    --compare_zarr /path/to/training_data.zarr
```

### 场景 3: 确认动作空间匹配

1. 检查训练数据的动作范围：
```bash
python toolbox/visualize_training_data.py \
    --zarr_path /path/to/data.zarr \
    --action_analysis
```

2. 检查推理输出是否在合理范围内：
```bash
python toolbox/visualize_inference_log.py --state
```

## 依赖

```
numpy
matplotlib
zarr
```

## 示例输出

### 训练数据动作分布
显示每个动作维度（x, y, z, qx, qy, qz, qw, gripper）的分布直方图，帮助了解数据集的覆盖范围。

### 推理轨迹分析
- 3D 轨迹可视化
- 时间序列分析（位置、姿态、夹爪）
- 推理频率统计

### 一致性分析
分析连续推理步之间预测的一致性，帮助发现可能的问题：
- 如果一致性差，可能是模型过拟合或训练数据不足
- 如果轨迹异常，可能是输入数据预处理有问题

