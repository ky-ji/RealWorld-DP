# CogAct数据集转换指南

## 数据结构分析

### CogAct原始数据格式

```
trajectories/
├── episode_0001/
│   ├── data.pkl          # pickle文件，包含机器人状态和动作数据
│   ├── images/           # RGB图像序列
│   │   ├── frame_0000.jpg
│   │   ├── frame_0001.jpg
│   │   └── ...
│   └── meta.json         # episode元数据
├── episode_0002/
└── ...
```

#### data.pkl内容
- `robot_eef_pose`: (N, 7) float32 - 机器人末端执行器位姿 [x, y, z, qx, qy, qz, qw]
- `robot_gripper`: (N,) int64 - 夹爪状态 (0或1)
- `action`: (N, 7) float32 - 动作命令 [x, y, z, qx, qy, qz, qw]
- `action_gripper`: (N,) int64 - 夹爪动作
- `timestamp`: (N,) float64 - 时间戳
- `image_index`: (N,) int64 - 图像帧索引

#### meta.json内容
```json
{
  "episode_id": 1,
  "start_time": "2025-11-23T20:26:04.081794",
  "duration": 15.199429750442505,
  "n_steps": 129,
  "n_images": 129,
  "data_shapes": {...}
}
```

### Diffusion Policy训练数据格式

转换后的zarr ReplayBuffer结构：

```
output.zarr/
├── data/
│   ├── image                  # (N, H, W, 3) uint8 - RGB图像
│   ├── robot_eef_pose         # (N, 7) float32 - 机器人位姿观测
│   ├── action                 # (N, 7) float32 - 动作
│   ├── robot_gripper_state    # (N, 1) float32 - 夹爪状态
│   └── timestamp              # (N,) float64 - 时间戳
└── meta/
    └── episode_ends           # (n_episodes,) int64 - 每个episode结束索引
```

## 使用方法

### 1. 数据检查

首先检查原始数据结构：

```bash
conda activate robodiff
cd /home/kyji/storage_net/realworld_eval/diffusion_policy

python scripts/inspect_cogact_data.py
```

### 2. 数据转换

#### 基本用法

转换所有数据（保持原始图像分辨率）：

```bash
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /home/kyji/public/dataset/cogact/1124/diffusion_policy_data.zarr
```

#### 调整图像分辨率

如果原始图像太大，可以调整分辨率：

```bash
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_640x480.zarr \
    --resolution 640 480
```

#### 测试转换

只转换前N个episodes进行测试：

```bash
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /tmp/test_cogact.zarr \
    --max-episodes 3
```

#### 转换部分数据

如果需要转换特定范围的episodes（例如clean数据集）：

```bash
# 从episode 65开始转换到最后
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_clean.zarr \
    --resolution 640 480 \
    --start-episode 65

# 转换episode 65到episode 100
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_range.zarr \
    --resolution 640 480 \
    --start-episode 65 \
    --max-episodes 36  # 从65开始的36个episodes (65-100)
```

#### 压缩选项

选择压缩算法和级别：

```bash
# 使用Blosc压缩（推荐，默认，稳定快速）
python scripts/convert_cogact_to_zarr.py \
    --input /path/to/trajectories \
    --output /path/to/output.zarr \
    --compressor blosc \
    --compression-level 5

# 使用JPEG2000压缩（压缩比高，但可能有兼容性问题）
python scripts/convert_cogact_to_zarr.py \
    --input /path/to/trajectories \
    --output /path/to/output.zarr \
    --compressor jpeg2k \
    --compression-level 50
```

### 3. 验证转换结果

```bash
python scripts/verify_converted_data.py \
    --zarr-path /home/kyji/public/dataset/cogact/1124/diffusion_policy_data.zarr
```

这个脚本会：
- 显示数据集统计信息
- 打印数据结构
- 保存样例图像到 `sample_images.png`
- 保存动作轨迹图到 `action_trajectory.png`

### 4. 在训练中使用

转换后的数据可以直接用于diffusion policy训练。需要创建相应的配置文件：

```yaml
# config/task/your_task.yaml
name: cogact_robot
shape_meta:
  obs:
    image:
      shape: [3, 1080, 1920]  # 或调整后的分辨率 [3, 480, 640]
      type: rgb
    robot_eef_pose:
      shape: [7]  # [x, y, z, qx, qy, qz, qw]
      type: low_dim
    robot_gripper_state:
      shape: [1]
      type: low_dim
  action:
    shape: [7]  # [x, y, z, qx, qy, qz, qw]

dataset:
  _target_: diffusion_policy.dataset.real_pusht_image_dataset.RealPushTImageDataset
  dataset_path: /home/kyji/public/dataset/cogact/1124/diffusion_policy_data.zarr
  horizon: 16
  pad_before: 1
  pad_after: 7
  n_obs_steps: 2
  use_cache: False
  val_ratio: 0.1
```

## 脚本说明

### `inspect_cogact_data.py`
检查原始CogAct数据格式，显示：
- Episode数量
- 数据结构和形状
- 数值范围统计

### `convert_cogact_to_zarr.py`
主要转换脚本，功能：
- 读取CogAct格式数据
- 转换为zarr ReplayBuffer格式
- 支持图像缩放
- 支持多种压缩算法
- 显示进度条

### `verify_converted_data.py`
验证转换结果，功能：
- 检查数据完整性
- 显示统计信息
- 可视化样例图像和轨迹
- 确认与diffusion policy兼容

## 注意事项

### 图像分辨率
- 原始分辨率：1920x1080（较大，训练慢）
- 建议分辨率：640x480 或 320x240（训练快，精度略降）
- 根据显存和训练速度权衡选择

### 存储空间
- 原始图像：每个episode约30-40MB
- 转换后（jpeg2k压缩）：类似大小
- 转换后（无压缩）：约3-5倍大小
- 153个episodes总计约5-8GB

### 内存使用
- 转换过程需要加载单个episode的所有图像
- 建议至少8GB可用内存
- 如遇内存不足，可分批转换

### 数据质量检查
转换前建议检查：
1. 所有episode的图像数量是否匹配n_steps
2. 动作数据范围是否合理
3. 是否有损坏的图像文件
4. timestamp是否连续

## 常见问题

### Q: 输出目录已存在报错？
A: 脚本会自动检测并删除已存在的输出目录，无需手动清理。如果中途中断转换，直接重新运行即可。

### Q: 转换很慢怎么办？
A: 
- 使用 `--resolution` 降低图像分辨率
- 使用 `blosc` 压缩替代 `jpeg2k`
- 确保在SSD上操作

### Q: 内存不足？
A: 
- 使用 `--max-episodes` 分批转换
- 降低图像分辨率
- 关闭其他占用内存的程序

### Q: 如何合并多个zarr文件？
A: 需要编写额外脚本，或在转换时一次性转换所有数据

### Q: 数据增强？
A: 数据增强在训练时动态进行，不在转换阶段

## 数据集统计（153 episodes）

- 总步数：~15,000-20,000步
- 平均episode长度：~100-130步
- 平均episode时长：~12-15秒
- 图像尺寸：1920x1080x3
- 动作空间：7D (xyz + quaternion)

## 下一步

1. **完整转换**：转换所有153个episodes
2. **配置文件**：创建训练配置文件
3. **训练测试**：用小批量数据测试训练流程
4. **超参数调优**：调整horizon, n_obs_steps等参数
5. **模型评估**：在实际机器人上评估训练后的策略
