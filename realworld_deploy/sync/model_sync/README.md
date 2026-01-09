Upload:
```
tosutil mkdir tos://embodied-model/Diffusion_Policy/assembly_chocolate

tosutil cp /home/jikangye/workspace/baselines/vla-baselines/RealWorld-DP/data/outputs/2026.01.07/14.13.48_train_diffusion_transformer_hybrid_ddp_assembly_chocolate/checkpoints/epoch=0550-train_loss=0.057.ckpt tos://embodied-model/Diffusion_Policy/assembly_chocolate/

```

Download:

```

tosutil cp tos://embodied-model/Diffusion_Policy/assembly_chocolate/epoch=0550-train_loss=0.057.ckpt /home/yxlab/code/jikangye/RealWorld-DP/ckpts/

```