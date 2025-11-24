# CogAct数据训练指南

本文档说明如何使用转换后的CogAct数据训练Diffusion Policy。

## 📋 准备工作

### 1. 确保数据已转换

```bash
# 检查数据是否存在
ls /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_640x480.zarr
ls /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_clean_640x480.zarr
```

如果不存在，先运行转换脚本：
```bash
# 完整数据
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_640x480.zarr \
    --resolution 640 480

# Clean数据（从episode 65开始）
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_clean_640x480.zarr \
    --resolution 640 480 \
    --start-episode 65
```

## 🚀 开始训练

### 方式1：使用完整数据集（153 episodes）

```bash
./train_cogact.sh
```

或者手动运行：
```bash
conda activate robodiff
python train.py --config-name=train_cogact_robot
```

### 方式2：使用Clean数据集（episode 65+，约89 episodes）

```bash
./train_cogact_clean.sh
```

或者手动运行：
```bash
conda activate robodiff
python train.py --config-name=train_cogact_robot_clean
```

## 📁 配置文件说明

### 新增的配置文件

1. **任务配置**：
   - `diffusion_policy/config/task/cogact_robot_7d.yaml` - 完整数据集配置
   - `diffusion_policy/config/task/cogact_robot_7d_clean.yaml` - Clean数据集配置

2. **训练配置**：
   - `diffusion_policy/config/train_cogact_robot.yaml` - 完整数据训练
   - `diffusion_policy/config/train_cogact_robot_clean.yaml` - Clean数据训练

3. **训练脚本**：
   - `train_cogact.sh` - 完整数据训练脚本
   - `train_cogact_clean.sh` - Clean数据训练脚本

### 与旧配置的区别

| 特性 | 旧配置 (real_robot_7d) | 新配置 (cogact_robot_7d) |
|------|------------------------|--------------------------|
| **数据格式** | 视频 + zarr | 纯zarr（图像存储在内） |
| **图像字段** | `camera_0` | `image` |
| **Observation** | `robot_eef_pose` [7D] | `robot_eef_pose` [7D] + `robot_gripper_state` [1D] |
| **图像分辨率** | 1920x1080 → 320x180 | 1920x1080 → 640x480 |
| **Crop大小** | 162x288 (90% of 180x320) | 432x576 (90% of 480x640) |
| **数据路径** | `/home/kyji/storage_net/realworld_eval/realworld_data/1119/` | `/home/kyji/public/dataset/cogact/1124/` |

## ⚙️ 训练参数

### 模型架构
- **模型**: Diffusion Transformer Hybrid
- **层数**: 8 layers
- **注意力头**: 4 heads
- **嵌入维度**: 256
- **参数量**: ~20M

### 训练设置
- **Epochs**: 600
- **Batch size**: 64
- **Learning rate**: 1e-4 (cosine scheduler)
- **Warmup steps**: 500
- **EMA**: 启用
- **Delta action**: 启用（使用相对位姿）

### 数据设置
- **Horizon**: 16
- **n_obs_steps**: 2（观察2帧）
- **n_action_steps**: 8（预测8步action）
- **Val ratio**: 0.1（10%数据用于验证）

## 📊 监控训练

### WandB
训练日志会自动上传到WandB：
- 完整数据: project `diffusion_policy_cogact`
- Clean数据: project `diffusion_policy_cogact_clean`

### 本地日志
```bash
# 训练输出目录
data/outputs/[date]/[time]_train_diffusion_transformer_hybrid_cogact_robot_7d/

# 包含：
├── checkpoints/         # 模型检查点
├── media/              # 验证可视化
└── logs/               # 训练日志
```

## 🔧 自定义配置

### 修改数据路径
编辑 `diffusion_policy/config/task/cogact_robot_7d.yaml`:
```yaml
dataset_path: /path/to/your/data.zarr
```

### 调整训练参数
编辑 `diffusion_policy/config/train_cogact_robot.yaml`:
```yaml
training:
  num_epochs: 1000        # 增加训练轮数
  
dataloader:
  batch_size: 32          # 减小batch size（如果显存不足）
  
optimizer:
  learning_rate: 5.0e-5   # 调整学习率
```

### 修改图像分辨率
如果需要使用不同分辨率：
1. 重新转换数据（例如 320x240）
2. 修改配置文件：
```yaml
image_shape: [3, 240, 320]  # [C, H, W]
dataset_path: /path/to/320x240.zarr

policy:
  crop_shape: [216, 288]  # 90% of 240x320
```

## ❓ 常见问题

### Q: 应该使用完整数据还是Clean数据？
**A**: 
- **完整数据**: 如果所有episodes质量都较好，使用完整数据可以获得更多训练样本
- **Clean数据**: 如果前面的episodes包含调试数据或质量较差，使用Clean数据效果更好

### Q: 训练多久能看到效果？
**A**: 
- 通常在50-100 epoch后可以看到初步效果
- 200-300 epoch后模型基本收敛
- 建议训练完整的600 epoch以获得最佳性能

### Q: 显存不足怎么办？
**A**: 修改配置文件：
```yaml
dataloader:
  batch_size: 32          # 从64降到32
  num_workers: 4          # 从8降到4
  
policy:
  n_layer: 6              # 从8层降到6层
  n_emb: 128              # 从256降到128
```

### Q: 如何恢复中断的训练？
**A**: 训练脚本默认启用了resume功能：
```yaml
training:
  resume: True
```
只需重新运行相同的训练命令，会自动从最后的checkpoint继续。

### Q: 能否同时训练多个模型？
**A**: 可以，使用不同的配置：
```bash
# Terminal 1: 训练完整数据
python train.py --config-name=train_cogact_robot

# Terminal 2: 训练clean数据
python train.py --config-name=train_cogact_robot_clean training.device=cuda:1
```

## 📈 下一步

训练完成后：
1. 查看WandB日志分析训练曲线
2. 使用最佳checkpoint进行推理测试
3. 在真实机器人上部署和验证

## 🔗 相关文档

- `scripts/COGACT_CONVERSION_README.md` - 数据转换详细说明
- `scripts/CONVERT_CLEAN_DATA.md` - Clean数据转换指南
- `train_realworld.sh` - 旧版训练脚本（参考）
