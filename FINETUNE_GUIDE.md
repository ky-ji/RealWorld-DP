# 从 7D Action 模型微调到 8D Action 模型指南

## 背景

由于修改了 action shape（从 7D pose 改为 8D pose+gripper），需要从已有的 7D 模型继承参数并微调。

## 方法对比

### 方法 1：完全重新训练（不推荐）
- ❌ 训练时间长（600 epochs）
- ❌ 浪费已有的训练成果
- ✅ 最干净，无兼容性问题

### 方法 2：部分加载 + 微调（推荐）⭐
- ✅ 继承视觉编码器和大部分 Transformer 参数
- ✅ 只重新初始化 action 输出层
- ✅ 训练更快（可能只需 100-200 epochs）
- ✅ 利用已有的视觉特征提取能力

## 使用步骤

### 步骤 1：转换数据（包含 gripper）

```bash
cd /home/kyji/storage_net/realworld_eval/diffusion_policy

python scripts/convert_cogact_to_zarr.py \
    --input_dir /home/kyji/public/dataset/cogact/1124/trajectories \
    --output_zarr /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full_320x180_8d.zarr \
    --target_resolution 320 180
```

### 步骤 2：适配旧模型 checkpoint

```bash
python scripts/finetune_from_7d_to_8d.py \
    --old_ckpt data/outputs/2025.11.24/04.23.47_train_diffusion_transformer_hybrid_cogact_robot_7d/checkpoints/epoch=0550-train_loss=0.017.ckpt \
    --new_ckpt data/pretrained/cogact_7d_to_8d_init.ckpt
```

这个脚本会：
- 加载旧模型的所有参数
- 排除 action 输出层（因为维度不匹配）
- 保存适配后的 checkpoint

### 步骤 3：修改训练配置

编辑 `diffusion_policy/config/train_cogact_robot.yaml`：

```yaml
# 更新数据集路径
task:
  dataset_path: /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full_320x180_8d.zarr

# 训练配置
training:
  resume: False  # 不自动恢复
  num_epochs: 200  # 减少 epoch（因为是微调）
  checkpoint_every: 10
  
# 添加预训练模型路径（在训练脚本中手动加载）
pretrained:
  path: data/pretrained/cogact_7d_to_8d_init.ckpt
  load_weights: True
```

### 步骤 4：修改训练脚本加载预训练权重

创建 `train_cogact_finetune.sh`：

```bash
#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 运行训练，使用预训练权重
python train.py \
    --config-name=train_cogact_robot \
    training.resume=False \
    training.num_epochs=200 \
    task.dataset_path=/home/kyji/public/dataset/cogact/1124/diffusion_policy_data_full_320x180_8d.zarr
```

### 步骤 5：在代码中加载预训练权重

修改 `diffusion_policy/workspace/train_diffusion_transformer_hybrid_workspace.py`，在 `__init__` 方法中添加：

```python
def __init__(self, cfg: OmegaConf):
    super().__init__(cfg)
    
    # ... 现有代码 ...
    
    # 加载预训练权重（如果指定）
    if 'pretrained' in cfg and cfg.pretrained.get('load_weights', False):
        pretrained_path = cfg.pretrained.path
        print(f"加载预训练权重: {pretrained_path}")
        
        payload = torch.load(open(pretrained_path, 'rb'), pickle_module=dill)
        
        # 只加载 state_dicts
        if 'state_dicts' in payload:
            missing_keys, unexpected_keys = self.model.load_state_dict(
                payload['state_dicts'], strict=False
            )
            print(f"  缺失的键: {len(missing_keys)}")
            print(f"  意外的键: {len(unexpected_keys)}")
            if len(missing_keys) > 0:
                print(f"  缺失键示例: {missing_keys[:5]}")
        
        print("✓ 预训练权重加载完成")
```

### 步骤 6：开始微调

```bash
bash train_cogact_finetune.sh
```

## 预期效果

### 训练速度
- **完全重新训练**：600 epochs × 约 5 分钟/epoch = 50 小时
- **微调**：200 epochs × 约 5 分钟/epoch = 17 小时（节省 66%）

### 性能
- 前 50 epochs：快速收敛（利用预训练特征）
- 50-150 epochs：学习夹爪控制策略
- 150-200 epochs：精细调优

### 监控指标
- `train_loss`：应该从较低值开始（~0.02），而不是从头开始（~0.5）
- `val_loss`：快速下降到合理范围
- 夹爪预测准确率：在验证集上检查

## 验证方法

训练完成后，检查推理日志中的夹爪值：

```python
# 查看推理日志
import json
with open('mmalb_dp/log/inference_log_xxx.json') as f:
    log = json.load(f)

# 检查夹爪值是否变化
grippers = [step['action']['values'][0][-1] for step in log['steps']]
print(f"夹爪唯一值: {set(grippers)}")
print(f"夹爪变化次数: {sum(1 for i in range(len(grippers)-1) if grippers[i] != grippers[i+1])}")
```

如果夹爪值有 0 和 1 的变化，说明模型学会了控制夹爪！

## 故障排除

### 问题 1：维度不匹配错误
```
RuntimeError: size mismatch for model.output_proj.weight
```
**解决**：确保运行了步骤 2 的适配脚本，排除了输出层。

### 问题 2：loss 不下降
**可能原因**：
- 学习率太高/太低
- 数据集路径错误
- 预训练权重未正确加载

**解决**：检查日志中是否有 "加载预训练权重" 的消息。

### 问题 3：夹爪始终是 1.0
**可能原因**：
- 数据转换时未包含 action_gripper
- 模型未学习到夹爪控制

**解决**：检查转换后的 zarr 文件中 action 的 shape 是否为 (N, 8)。
