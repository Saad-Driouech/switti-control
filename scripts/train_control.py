"""
Training entry point for SwittiControlNet.

Analogous to train.py but:
  - Loads a pretrained Switti from HuggingFace
  - Wraps it in SwittiControlNet
  - Trains only spatial_encoder, control_blocks, zero_convs

Usage::

    torchrun --nproc_per_node=8 scripts/train_control.py \\
        --data_path /data/coco \\
        --ctrl_maps_dir /data/ctrl_maps \\
        --control_modalities canny \\
        --exp_name control_canny_phase1 \\
        --pretrained_switti yresearch/Switti \\
        --bs 32 --tblr 1e-4 --max_iters 100000
"""
import gc
import os
import sys
import time

# Make repo root importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader

import dist
from models.switti import SwittiHF
from models.vqvae import VQVAEHF
from models.clip import FrozenCLIPEmbedder
from models.basic_switti import AdaLNSelfCrossAttn
from models.control_switti import SwittiControlNet
from models.control_pipeline import SwittiControlPipeline
from trainer_control import SwittiControlTrainer
from utils import arg_util, misc
from utils.amp_sc import AmpOptimizer
from utils.fsdp import load_model_state, save_model_state
from utils.lr_control import filter_params, lr_wd_annealing
from utils.control_data import build_control_dataset, control_collate_fn
from utils.data_sampler import DistInfiniteBatchSampler


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def build_everything(args: arg_util.Args):
    # Tensorboard logger
    if dist.is_master():
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        tb_lg = misc.DistLogger(
            misc.TensorboardLogger(
                log_dir=args.tb_log_dir_path,
                filename_suffix=f'__{misc.time_str("%m%d_%H%M")}',
            ),
            verbose=True,
        )
        tb_lg.flush()
    else:
        tb_lg = misc.DistLogger(None, verbose=False)

    print(f"initial args:\n{str(args)}")

    # -------------------------------------------------------------------------
    # Build models
    # -------------------------------------------------------------------------
    pretrained_path = getattr(args, "pretrained_switti", "yresearch/Switti")
    device = dist.get_device()

    if dist.is_local_master():
        print(f"Loading pretrained Switti from {pretrained_path} ...")
    frozen_switti = SwittiHF.from_pretrained(pretrained_path).to(device)
    dist.barrier()

    vae_local = VQVAEHF.from_pretrained(
        getattr(args, "vae_ckpt", "yresearch/VQVAE-Switti"),
        reso=args.data_load_reso,
    ).to(device)

    num_modalities = getattr(args, "num_modalities", 5)
    control_net_wo_ddp = SwittiControlNet(
        frozen_switti=frozen_switti,
        num_modalities=num_modalities,
        use_gradient_checkpointing=getattr(args, "use_gradient_checkpointing", False),
    ).to(device)

    # Optionally resume from checkpoint
    control_ckpt = getattr(args, "control_ckpt", None)
    start_it = 0
    if control_ckpt and os.path.exists(control_ckpt):
        state = torch.load(control_ckpt, map_location="cpu")
        control_net_wo_ddp.load_state_dict(state, strict=False)
        print(f"Resumed control weights from {control_ckpt}")
    else:
        start_it = load_model_state(args, control_net_wo_ddp)

    count_p = lambda m: f"{sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6:.2f}M"
    print(f"[INIT] Trainable params: {count_p(control_net_wo_ddp)}")

    # Text encoders (frozen, CPU-offloaded during training)
    text_encoder = FrozenCLIPEmbedder(args.text_encoder_path, device=device)
    text_encoder_2 = FrozenCLIPEmbedder(args.text_encoder_2_path, device=device)

    pipe = SwittiControlPipeline(
        control_net=control_net_wo_ddp,
        vae=vae_local,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        device=device,
    )

    # -------------------------------------------------------------------------
    # FSDP / DDP wrapper — only wraps trainable sub-modules
    # -------------------------------------------------------------------------
    def _wrap_policy(module, recurse, **_etc):
        # Wrap each AdaLNSelfCrossAttn block individually for memory efficiency
        return recurse or isinstance(module, AdaLNSelfCrossAttn)

    # frozen_switti must NOT be sharded by FSDP: _frozen_block_forward accesses
    # its sub-layers directly (outside FSDP's managed forward), so parameters
    # must be full tensors on every rank, not flat 1-D shards.
    control_net: FSDP = (FSDP if dist.initialized() else NullDDP)(
        control_net_wo_ddp,
        ignored_modules=[control_net_wo_ddp.frozen_switti],
        auto_wrap_policy=_wrap_policy,
        device_id=dist.get_local_rank(),
        sharding_strategy=(
            ShardingStrategy.HYBRID_SHARD if args.use_fsdp else ShardingStrategy.NO_SHARD
        ),
        use_orig_params=True,
        forward_prefetch=True,
        limit_all_gathers=True,
    )

    # -------------------------------------------------------------------------
    # Optimizer — only trainable parameters
    # -------------------------------------------------------------------------
    names, paras, para_groups = filter_params(
        control_net,
        nowd_keys={"pos_embed", "pos_1LC", "pos_start", "lvl_embed", "gamma", "beta"},
    )
    optimizer = torch.optim.AdamW(
        params=para_groups,
        lr=args.tlr,
        weight_decay=0.0,
        betas=(args.adam_beta1, args.adam_beta2),
        fused=args.afuse if not args.use_fsdp else False,
    )
    control_optimizer = AmpOptimizer(
        mixed_precision=args.fp16,
        optimizer=optimizer,
        names=names,
        paras=paras,
        grad_clip=args.tclip,
    )
    del names, paras, para_groups

    # -------------------------------------------------------------------------
    # Dataset and dataloader
    # -------------------------------------------------------------------------
    modalities = getattr(args, "control_modalities", ["canny"])
    ctrl_maps_dir = getattr(args, "ctrl_maps_dir", None)

    dataset_train = build_control_dataset(
        data_path=args.data_path,
        final_reso=args.data_load_reso,
        modalities=modalities,
        ctrl_maps_dir=ctrl_maps_dir,
        mid_reso_factor=args.mid_reso,
    )
    ld_train = DataLoader(
        dataset=dataset_train,
        num_workers=args.workers,
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(),
        collate_fn=control_collate_fn,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train),
            glb_batch_size=args.glb_batch_size,
            same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True,
            fill_last=True,
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            start_it=start_it,
        ),
    )
    del dataset_train

    # -------------------------------------------------------------------------
    # Trainer
    # -------------------------------------------------------------------------
    trainer = SwittiControlTrainer(
        dataloader=ld_train,
        device=args.device,
        patch_nums=args.patch_nums,
        resos=args.resos,
        pipe=pipe,
        vae_local=vae_local,
        control_net_wo_ddp=control_net_wo_ddp,
        control_net=control_net,
        optimizer=control_optimizer,
        label_smooth=args.ls,
        args=args,
    )
    torch.cuda.empty_cache()
    return tb_lg, trainer, start_it


def main_training():
    torch.set_num_threads(32)
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    tb_lg, trainer, start_it = build_everything(args)
    dist.barrier()

    for cur_iter in range(start_it, args.max_iters):
        tb_lg.set_step(cur_iter)

        min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(
            args.sche,
            trainer.optimizer.optimizer,
            args.tlr,
            args.twd,
            args.twde,
            cur_iter,
            args.wp,
            args.max_iters,
            wp0=args.wp0,
            wpe=args.wpe,
            wp_start_it=start_it,
        )
        args.cur_lr, args.cur_wd = max_tlr, max_twd

        grad_norm, scale_log2 = trainer.train_step(g_it=cur_iter, tb_lg=tb_lg)

        tb_lg.update(head="Control_opt_lr/lr_min", sche_tlr=min_tlr)
        tb_lg.update(head="Control_opt_lr/lr_max", sche_tlr=max_tlr)
        tb_lg.update(head="Control_opt_grad/fp16", scale_log2=scale_log2)
        if args.tclip > 0:
            tb_lg.update(head="Control_opt_grad/grad", grad_norm=grad_norm)

        if cur_iter % args.save_iters == 0 and cur_iter > start_it:
            save_model_state(cur_iter, args, trainer.control_net)
            args.dump_log()
            tb_lg.flush()

    gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    args.remain_time, args.finish_time = "-", time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(time.time() - 60)
    )
    print(f"final args:\n\n{str(args)}")
    args.dump_log()
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()


if __name__ == "__main__":
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(
            sys.stderr, misc.SyncPrint
        ):
            sys.stdout.close(), sys.stderr.close()
