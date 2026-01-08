"""
Multi-GPU DDP Training Workspace for Diffusion Transformer
支持多卡分布式训练
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import pathlib
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionTransformerHybridWorkspaceDDP(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        
        # 初始化分布式环境
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        
        # 只在主进程打印
        self.is_main_process = (self.rank == 0)
        
        if self.world_size > 1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(self.local_rank)
            if self.is_main_process:
                print(f"初始化DDP: world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank}")

        # set seed (每个进程使用不同的seed以确保数据增强的多样性)
        seed = cfg.training.seed + self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionTransformerHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionTransformerHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        device = torch.device(f'cuda:{self.local_rank}')

        # resume training (只在主进程检查)
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                if self.is_main_process:
                    print(f"从checkpoint恢复训练: {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
        else:
            # 如果不是 resume，检查是否有预训练权重需要加载
            pretrained_path = pathlib.Path("data/pretrained/cogact_7d_to_8d_init.ckpt")
            if pretrained_path.is_file() and self.is_main_process:
                print(f"[训练] 加载预训练权重: {pretrained_path}")
                try:
                    import dill
                    payload = torch.load(open(pretrained_path, 'rb'), pickle_module=dill)
                    
                    if 'state_dicts' in payload:
                        missing_keys, unexpected_keys = self.model.load_state_dict(
                            payload['state_dicts'], strict=False
                        )
                        print(f"[训练]   缺失的键: {len(missing_keys)} (将随机初始化)")
                        print(f"[训练]   意外的键: {len(unexpected_keys)}")
                        if len(missing_keys) > 0 and len(missing_keys) <= 5:
                            print(f"[训练]   缺失键示例: {missing_keys}")
                        
                        # 同步到 EMA 模型
                        if cfg.training.use_ema and self.ema_model is not None:
                            self.ema_model.load_state_dict(payload['state_dicts'], strict=False)
                            print(f"[训练]   EMA 模型也已加载预训练权重")
                    
                    print("[训练] ✓ 预训练权重加载完成")
                except Exception as e:
                    print(f"[训练] ⚠ 加载预训练权重失败: {e}")
                    print("[训练] 将从头开始训练")

        # 同步所有进程，确保权重加载完成
        if self.world_size > 1:
            dist.barrier()

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        
        # 创建分布式采样器
        train_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=cfg.dataloader.shuffle,
                seed=cfg.training.seed
            )
            # 使用分布式采样器时，不要在DataLoader中设置shuffle
            dataloader_cfg = OmegaConf.to_container(cfg.dataloader, resolve=True)
            dataloader_cfg['shuffle'] = False
            train_dataloader = DataLoader(dataset, sampler=train_sampler, **dataloader_cfg)
        else:
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
        
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_sampler = None
        if self.world_size > 1:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
            val_dataloader_cfg = OmegaConf.to_container(cfg.val_dataloader, resolve=True)
            val_dataloader_cfg['shuffle'] = False
            val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **val_dataloader_cfg)
        else:
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # device transfer (在包装DDP之前)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        
        # 包装模型为DDP
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,  # 如果有未使用的参数，设为True
                broadcast_buffers=True
            )
            # EMA模型不需要包装为DDP
        
        optimizer_to(self.optimizer, device)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env (只在主进程运行rollout)
        env_runner: BaseImageRunner = None
        if self.is_main_process:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        # configure logging (只在主进程)
        wandb_run = None
        if self.is_main_process:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                    "world_size": self.world_size,
                }
            )

        # configure checkpoint (只在主进程)
        topk_manager = None
        if self.is_main_process:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )
        
        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger = None
        
        try:
            if self.is_main_process:
                json_logger = JsonLogger(log_path)
                json_logger.__enter__()
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                
                # 设置epoch用于DistributedSampler
                if train_sampler is not None:
                    train_sampler.set_epoch(self.epoch)
                
                # ========= train for this epoch ==========
                train_losses = list()
                
                # 只在主进程显示进度条
                if self.is_main_process:
                    tepoch = tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                            leave=False, mininterval=cfg.training.tqdm_interval_sec)
                else:
                    tepoch = train_dataloader
                
                for batch_idx, batch in enumerate(tepoch):
                    # device transfer
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # compute loss
                    # 从DDP模型中获取原始模型
                    model_for_loss = self.model.module if self.world_size > 1 else self.model
                    raw_loss = model_for_loss.compute_loss(batch)
                    loss = raw_loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    # step optimizer
                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    # update ema (只在主进程更新)
                    if cfg.training.use_ema and self.is_main_process:
                        # 从DDP模型中获取原始模型
                        model_for_ema = self.model.module if self.world_size > 1 else self.model
                        ema.step(model_for_ema)

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    if self.is_main_process and hasattr(tepoch, 'set_postfix'):
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    train_losses.append(raw_loss_cpu)
                    
                    if self.is_main_process:
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            if json_logger is not None:
                                json_logger.log(step_log)

                    self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

                # 同步所有进程的训练loss
                if self.world_size > 1:
                    train_losses_tensor = torch.tensor(train_losses, device=device)
                    dist.all_reduce(train_losses_tensor, op=dist.ReduceOp.SUM)
                    train_losses = (train_losses_tensor / self.world_size).cpu().numpy().tolist()

                # at the end of each epoch
                # replace train_loss with epoch average
                if self.is_main_process:
                    train_loss = np.mean(train_losses)
                    step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                # 获取用于评估的模型
                if self.world_size > 1:
                    policy = self.model.module if not cfg.training.use_ema else self.ema_model
                else:
                    policy = self.model if not cfg.training.use_ema else self.ema_model
                policy.eval()

                # run rollout (只在主进程)
                if self.is_main_process and (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        if self.is_main_process:
                            val_tepoch = tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec)
                        else:
                            val_tepoch = val_dataloader
                            
                        for batch_idx, batch in enumerate(val_tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            
                            # 使用原始模型计算loss（不是policy）
                            model_for_loss = self.model.module if self.world_size > 1 else self.model
                            loss = model_for_loss.compute_loss(batch)
                            val_losses.append(loss)
                            if (cfg.training.max_val_steps is not None) \
                                and batch_idx >= (cfg.training.max_val_steps-1):
                                break
                        
                        # 同步验证loss
                        if len(val_losses) > 0:
                            val_losses_tensor = torch.stack(val_losses)
                            if self.world_size > 1:
                                dist.all_reduce(val_losses_tensor, op=dist.ReduceOp.SUM)
                                val_losses_tensor = val_losses_tensor / self.world_size
                            
                            if self.is_main_process:
                                val_loss = torch.mean(val_losses_tensor).item()
                                # log epoch average validation loss
                                step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch (只在主进程)
                if self.is_main_process and (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint (只在主进程保存)
                if self.is_main_process and (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                if self.is_main_process:
                    wandb_run.log(step_log, step=self.global_step)
                    if json_logger is not None:
                        json_logger.log(step_log)
                
                self.epoch += 1
                
                # 同步所有进程
                if self.world_size > 1:
                    dist.barrier()
        
        finally:
            if json_logger is not None:
                json_logger.__exit__(None, None, None)
            if self.world_size > 1:
                dist.destroy_process_group()

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        """保存checkpoint时，需要从DDP模型中提取原始模型"""
        if not self.is_main_process:
            return
        
        # 临时保存原始model引用
        original_model = self.model
        
        # 如果是DDP模型，提取内部模型用于保存
        if self.world_size > 1 and isinstance(self.model, DDP):
            self.model = self.model.module
        
        # 调用父类的保存方法
        result = super().save_checkpoint(path, tag, exclude_keys, include_keys, use_thread)
        
        # 恢复DDP模型引用
        self.model = original_model
        
        return result


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionTransformerHybridWorkspaceDDP(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

