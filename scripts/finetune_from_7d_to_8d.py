#!/usr/bin/env python3
"""
从 7D action 模型微调到 8D action 模型
加载旧模型的大部分参数，只重新初始化 action 输出层
"""

import torch
import dill
from pathlib import Path

def load_and_adapt_checkpoint(
    old_ckpt_path: str,
    new_ckpt_path: str,
    exclude_keys: list = None
):
    """
    加载旧的 7D checkpoint 并适配到 8D 模型
    
    Args:
        old_ckpt_path: 旧的 7D 模型 checkpoint 路径
        new_ckpt_path: 保存适配后的 checkpoint 路径
        exclude_keys: 要排除的键（将被重新初始化）
    """
    if exclude_keys is None:
        # 排除与 action 维度相关的层
        exclude_keys = [
            'model.model.output_proj.weight',
            'model.model.output_proj.bias',
            'ema_model.model.output_proj.weight',
            'ema_model.model.output_proj.bias',
        ]
    
    print(f"加载旧模型: {old_ckpt_path}")
    old_payload = torch.load(open(old_ckpt_path, 'rb'), pickle_module=dill)
    
    # 获取 state_dict
    old_state_dict = old_payload.get('state_dicts', {})
    
    # 过滤掉需要排除的键
    filtered_state_dict = {}
    excluded_count = 0
    
    for key, value in old_state_dict.items():
        should_exclude = False
        for exclude_pattern in exclude_keys:
            if exclude_pattern in key:
                should_exclude = True
                excluded_count += 1
                print(f"  排除: {key} (shape: {value.shape if hasattr(value, 'shape') else 'N/A'})")
                break
        
        if not should_exclude:
            filtered_state_dict[key] = value
    
    print(f"\n总共加载 {len(filtered_state_dict)} 个参数")
    print(f"排除 {excluded_count} 个参数（将重新初始化）")
    
    # 创建新的 payload
    new_payload = {
        'cfg': old_payload['cfg'],
        'state_dicts': filtered_state_dict,
        'pickles': old_payload.get('pickles', {}),
    }
    
    # 更新配置中的 action shape
    if 'task' in new_payload['cfg'] and 'shape_meta' in new_payload['cfg']['task']:
        if 'action' in new_payload['cfg']['task']['shape_meta']:
            old_shape = new_payload['cfg']['task']['shape_meta']['action']['shape']
            new_payload['cfg']['task']['shape_meta']['action']['shape'] = [8]
            print(f"\n更新 action shape: {old_shape} -> [8]")
    
    # 保存新的 checkpoint
    new_ckpt_path = Path(new_ckpt_path)
    new_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存适配后的模型: {new_ckpt_path}")
    torch.save(new_payload, new_ckpt_path.open('wb'), pickle_module=dill)
    print("✓ 完成！")
    
    return new_ckpt_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_ckpt', type=str, required=True,
                      help='旧的 7D 模型 checkpoint 路径')
    parser.add_argument('--new_ckpt', type=str, required=True,
                      help='保存适配后的 checkpoint 路径')
    parser.add_argument('--exclude_keys', nargs='+', default=None,
                      help='要排除的键（可选）')
    
    args = parser.parse_args()
    
    load_and_adapt_checkpoint(
        old_ckpt_path=args.old_ckpt,
        new_ckpt_path=args.new_ckpt,
        exclude_keys=args.exclude_keys
    )
