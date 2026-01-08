"""
Multi-GPU DDP Training Entry Point
支持多卡分布式训练的入口脚本

Usage:
# 8-GPU training
torchrun --nproc_per_node=8 train_ddp.py --config-name=train_assembly_chocolate_ddp

# 4-GPU training
torchrun --nproc_per_node=4 train_ddp.py --config-name=train_assembly_chocolate_ddp

# Single GPU training (for testing)
python train_ddp.py --config-name=train_assembly_chocolate_ddp
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # 检查是否在分布式环境中
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if local_rank != -1:
        print(f"[Rank {os.environ.get('RANK', 0)}] 启动DDP训练进程 (local_rank={local_rank}, world_size={world_size})")
    else:
        print("启动单卡训练")
    
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()




