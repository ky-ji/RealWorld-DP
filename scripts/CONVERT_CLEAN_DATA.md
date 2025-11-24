# 转换Clean数据集指南

## 场景说明

当你需要从特定episode开始转换数据时（例如前面的episodes质量不好，只想用后面的"clean"数据进行训练），使用`--start-episode`参数。

## 快速使用

### 1. 转换完整Clean数据（从episode 65到最后）

```bash
conda activate robodiff
cd /home/kyji/storage_net/realworld_eval/diffusion_policy

python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_clean_640x480.zarr \
    --resolution 640 480 \
    --start-episode 65
```

**说明**：
- 总episodes：153
- Clean数据：episode_0065 到 episode_0152（共89个episodes）
- 预计转换时间：约12-15分钟

### 2. 测试转换（先转换3个episodes验证）

```bash
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /tmp/test_clean.zarr \
    --resolution 640 480 \
    --start-episode 65 \
    --max-episodes 3
```

### 3. 验证转换结果

```bash
python scripts/verify_converted_data.py \
    --zarr-path /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_clean_640x480.zarr
```

## 参数详解

### `--start-episode N`
- 从episode N开始转换
- 例：`--start-episode 65` 会转换 episode_0065, episode_0066, ..., episode_0152
- 可以与其他参数组合使用

### 组合使用示例

```bash
# 转换episode 65-100（共36个）
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /home/kyji/public/dataset/cogact/1124/clean_subset.zarr \
    --start-episode 65 \
    --max-episodes 36 \
    --resolution 640 480

# 转换episode 100到最后（原始分辨率）
python scripts/convert_cogact_to_zarr.py \
    --input /home/kyji/public/dataset/cogact/1124/trajectories \
    --output /home/kyji/public/dataset/cogact/1124/clean_final.zarr \
    --start-episode 100
```

## 数据集对比

| 数据集 | Episodes | 命令 | 用途 |
|-------|---------|------|------|
| 完整数据 | episode_0001-0152 (153个) | 不加`--start-episode` | 使用所有数据训练 |
| Clean数据 | episode_0065-0152 (89个) | `--start-episode 65` | 只用高质量数据训练 |

## 常见问题

### Q: 如何确定从哪个episode开始？
A: 
1. 先运行`inspect_cogact_data.py`查看前面几个episodes的质量
2. 或者根据采集时的记录确定
3. 通常前面的episodes可能包含调试、测试数据

### Q: 会跳过有问题的episodes吗？
A: 是的，脚本会自动检测并跳过：
- 图像数量不足的episodes
- 数据损坏的episodes
- 转换结束时会报告所有被跳过的episodes

### Q: Clean数据和完整数据可以同时保存吗？
A: 可以，只要使用不同的输出路径：
```bash
# 完整数据
python scripts/convert_cogact_to_zarr.py \
    --input /path/to/trajectories \
    --output /path/to/full_data.zarr \
    --resolution 640 480

# Clean数据  
python scripts/convert_cogact_to_zarr.py \
    --input /path/to/trajectories \
    --output /path/to/clean_data.zarr \
    --resolution 640 480 \
    --start-episode 65
```

## 下一步

转换完成后，更新训练配置文件中的`dataset_path`：

```yaml
# example_cogact_config.yaml
task:
  # 使用clean数据训练
  dataset_path: /home/kyji/public/dataset/cogact/1124/diffusion_policy_data_clean_640x480.zarr
```

然后开始训练！
