# Color Jitter 数据增强模块使用指南

## 概述

`ColorJitterRandomizer` 是一个插件式的数据增强模块，用于模拟不同的光照条件。这对于实机部署（Real-world）非常重要，因为测试时的光照很难和训练集完全一致。

## 功能特性

- **随机调整亮度（Brightness）**：模拟不同光照强度
- **随机调整对比度（Contrast）**：增强模型对不同对比度的鲁棒性
- **随机调整饱和度（Saturation）**：模拟不同色彩环境
- **随机调整色相（Hue）**：增加色彩多样性

## 使用方法

### 1. 在配置文件中使用（推荐）

#### 方式一：使用默认参数

在 `train_assembly_chocolate_ddp.yaml` 中添加：

```yaml
policy:
  _target_: diffusion_policy.policy.diffusion_transformer_hybrid_image_policy.DiffusionTransformerHybridImagePolicy
  
  # ... 其他配置 ...
  
  # 启用颜色抖动（使用默认参数）
  color_jitter: True
```

#### 方式二：自定义参数

```yaml
policy:
  _target_: diffusion_policy.policy.diffusion_transformer_hybrid_image_policy.DiffusionTransformerHybridImagePolicy
  
  # ... 其他配置 ...
  
  # 自定义颜色抖动参数
  color_jitter:
    brightness: 0.2      # 亮度调整范围 [0.8, 1.2]
    contrast: 0.2        # 对比度调整范围 [0.8, 1.2]
    saturation: 0.2      # 饱和度调整范围 [0.8, 1.2]
    hue: 0.1             # 色相调整范围 [-0.1, 0.1]
    p: 1.0               # 应用概率（1.0 = 总是应用）
```

#### 方式三：使用范围元组

```yaml
policy:
  color_jitter:
    brightness: [0.9, 1.1]   # 自定义范围
    contrast: [0.9, 1.1]
    saturation: [0.9, 1.1]
    hue: [-0.05, 0.05]
    p: 0.8                    # 80% 概率应用
```

### 2. 在代码中直接使用

```python
from diffusion_policy.model.vision.color_jitter_randomizer import ColorJitterRandomizer

# 使用默认参数
color_jitter = ColorJitterRandomizer()

# 自定义参数
color_jitter = ColorJitterRandomizer(
    brightness=0.2,
    contrast=0.2,
    saturation=0.2,
    hue=0.1,
    p=1.0
)

# 应用到图像
# 假设 images 是 [B, C, H, W] 格式，值在 [0, 1] 范围
augmented_images = color_jitter(images)
```

### 3. 与 MultiImageObsEncoder 集成

```python
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder

obs_encoder = MultiImageObsEncoder(
    shape_meta=shape_meta,
    rgb_model=rgb_model,
    crop_shape=(162, 288),
    random_crop=True,
    color_jitter={
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
)
```

## 参数说明

### ColorJitterRandomizer 参数

- `brightness` (float or tuple): 亮度调整范围
  - float: 范围 [max(0, 1 - brightness), 1 + brightness]
  - tuple: 自定义范围 (min, max)
  - 默认: 0.2

- `contrast` (float or tuple): 对比度调整范围
  - float: 范围 [max(0, 1 - contrast), 1 + contrast]
  - tuple: 自定义范围 (min, max)
  - 默认: 0.2

- `saturation` (float or tuple): 饱和度调整范围
  - float: 范围 [max(0, 1 - saturation), 1 + saturation]
  - tuple: 自定义范围 (min, max)
  - 默认: 0.2

- `hue` (float or tuple): 色相调整范围
  - float: 范围 [-hue, hue]
  - tuple: 自定义范围 (min, max)
  - 默认: 0.1

- `p` (float): 应用变换的概率
  - 范围: [0.0, 1.0]
  - 默认: 1.0

## 行为说明

- **训练模式** (`model.training = True`): 应用随机颜色抖动
- **评估模式** (`model.training = False`): 返回原始图像（不应用变换）

这确保了：
- 训练时增加数据多样性
- 评估时保持一致性

## 示例配置

### 完整配置示例

```yaml
# diffusion_policy/config/train_assembly_chocolate_ddp.yaml

policy:
  _target_: diffusion_policy.policy.diffusion_transformer_hybrid_image_policy.DiffusionTransformerHybridImagePolicy
  
  shape_meta: ${shape_meta}
  
  noise_scheduler:
    _target_: diffusers.schedulers.scheduling_ddpm.DDPMScheduler
    num_train_timesteps: 100
    beta_start: 0.0001
    beta_end: 0.02
    beta_schedule: squaredcos_cap_v2
    clip_sample: True
    prediction_type: epsilon
    variance_type: fixed_small
  
  horizon: ${horizon}
  n_action_steps: ${eval:'${n_action_steps}+${n_latency_steps}'}
  n_obs_steps: ${n_obs_steps}
  num_inference_steps: 100
  
  # Crop settings
  crop_shape: [162, 288]
  obs_encoder_group_norm: True
  eval_fixed_crop: True
  
  # Color Jitter Augmentation (新增)
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
    p: 1.0
  
  # ... 其他配置 ...
```

## 注意事项

1. **输入格式**: 图像必须是 `[B, C, H, W]` 格式，值在 `[0, 1]` 范围（float32）
2. **GPU 兼容**: 使用 PyTorch 的随机数生成器，完全兼容 GPU 训练
3. **多进程安全**: 每个进程独立生成随机参数，适合 DDP 训练
4. **性能**: 变换在 GPU 上执行，对训练速度影响很小

## 推荐参数

根据任务类型，推荐以下参数：

### 轻度增强（适合光照变化较小的场景）
```yaml
color_jitter:
  brightness: 0.1
  contrast: 0.1
  saturation: 0.1
  hue: 0.05
```

### 中度增强（默认推荐）
```yaml
color_jitter:
  brightness: 0.2
  contrast: 0.2
  saturation: 0.2
  hue: 0.1
```

### 强度增强（适合光照变化很大的场景）
```yaml
color_jitter:
  brightness: 0.3
  contrast: 0.3
  saturation: 0.3
  hue: 0.15
```

## 故障排除

如果遇到问题，请检查：

1. 图像格式是否正确（`[B, C, H, W]`，值在 `[0, 1]`）
2. 模型是否处于训练模式（`model.train()`）
3. 参数范围是否合理（brightness/contrast/saturation 应该 > 0，hue 应该在合理范围内）

