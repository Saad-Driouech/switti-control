import gc
import os
import sys
import time

import torch
from trainer import SwittiTrainer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import Iterable

import dist
from calculate_metrics import distributed_metrics_with_csv, to_PIL_image
from models import Switti, VQVAE, VQVAEHF, build_models
from models.basic_switti import AdaLNSelfCrossAttn
from utils import arg_util, control_metrics, misc
from utils.amp_sc import AmpOptimizer
from utils.fsdp import load_model_state, load_optimizer_state, save_model_state
from utils.lr_control import filter_params, lr_wd_annealing
from utils.data import build_dataset, coco_collate_fn
from utils.data_sampler import DistInfiniteBatchSampler
from utils.fid_score_in_memory import calculate_fid
from models.switti import SwittiHF

import math

def get_control_strength_schedule(current_iter, warmup_steps=10000):
    """
    Ramp control strength from 0→1 over warmup_steps using cosine schedule.
    
    Args:
        current_iter: Current training iteration
        warmup_steps: Steps to ramp up (default: 10000)
    
    Returns:
        Float in [0, 1]
    """
    if current_iter >= warmup_steps:
        return 1.0
    
    progress = current_iter / warmup_steps
    # Cosine schedule (smoother than linear)
    return 0.5 * (1 - math.cos(math.pi * progress))


DEFAULT_VAE_CKPT = "vae_ch160v4096z32.pth"

def _clean_name_for_matching(name: str) -> str:
    """
    Remove common wrapper prefixes introduced by DDP/FSDP so matching is simpler.
    Keep minimal transforms — this will handle common cases.
    """
    return (
        name
        .replace("_fsdp_wrapped_module.", "")
        .replace("_fully_sharded_module.", "")
        .replace("module.", "")
    )

def apply_control_only_freeze(
    model: torch.nn.Module,
    *,
    freeze_control_backbone: bool = True,
    verbose: bool = True,
) -> None:
    """
    Freeze all Switti params except a conservative set required for image-control training.

    This function:
      - Works when called on the unwrapped Switti (recommended), and will also
        tolerate being called on a wrapped FSDP/DDP module (it strips common prefixes).
      - Optionally freezes the control encoder backbone (according to `freeze_control_backbone`).
        The control-backbone parameter name(s) are matched using `control_backbone_name_tokens`.
      - Keeps projection heads (proj) trainable.

    Kept (trainable) by default:
      - Entire control_encoder (except backbone if freeze_control_backbone=True)
      - control projection layers (e.g. *.proj)
      - control fusion modules inside blocks:
          - .cross_attn_control
          - .cross_attention_control_norm
          - .attention_control_norm
          - .control_proj
      - crop_embed / crop_proj (if present)

    Everything else is frozen.
    """

    kept_cnt = frozen_cnt = 0
    kept_numel = frozen_numel = 0
    total = 0

    for name, p in model.named_parameters():
        total += 1
        clean = _clean_name_for_matching(name)

        # default: freeze
        keep = False

        # 1) Always keep the control_encoder's outer parameters (we may freeze inner backbone below)
        if clean.startswith("control_encoder"):
            keep = True

            if freeze_control_backbone and "control_encoder.encoder.encoder.backbone" in clean:
                keep = False  # freeze the backbone parameters
                # BUT allow projection heads to stay trainable if asked
                if ".proj" in clean or clean.endswith("proj"):
                    keep = True
        
        # 1.5) Keep learnable null control tokens
        if clean.startswith("null_control_tokens"):
            keep = True

        # 2) Allow specific control-fusion modules inside transformer blocks
        control_fusion_tokens = (
            ".cross_attn_control",
            ".cross_attention_control_norm",
            ".attention_control_norm",
            ".control_proj",
            ".control_gate",
        )
        if any(tok in clean for tok in control_fusion_tokens):
            keep = True

        # # 3) Keep crop condition modules (if present)
        # if clean.startswith("crop_embed") or clean.startswith("crop_proj") or clean.startswith("crop_cond_scales"):
        #     keep = True

        # # 4) Optionally keep text_pooler (default: freeze). If you want it trainable, uncomment below:
        # if clean.startswith("text_pooler"):
        #     keep = True

        # # 5) Keep Switti control head (if you added a specialized head for control)
        # if clean.startswith("head_nm") or clean.startswith("head"):
        #     keep = True

        # Set requires_grad
        p.requires_grad = bool(keep)
        if keep:
            kept_cnt += 1
            kept_numel += p.numel()
        else:
            frozen_cnt += 1
            frozen_numel += p.numel()

    if verbose:
        print(
            f"[apply_control_only_freeze] kept={kept_cnt} params ({kept_numel:,} elems), "
            f"frozen={frozen_cnt} params ({frozen_numel:,} elems), total={total}"
        )

def zero_init_control_layers(model):
    """
    Initialize control fusion so the residual contribution starts at exactly 0.

    - Additive fusion: zero `control_proj` (single linear, no chicken-and-egg).
    - Cross-attention fusion: zero `control_gate` (the tanh-bounded scalar gate).
      We deliberately DO NOT zero `cross_attn_control.proj` here, because that
      would make every learnable cross-attn parameter receive zero gradient at
      step 0 (chain rule: dL/d(to_q,to_kv,proj) ∝ gate · proj ≈ 0). The gate at 0
      already guarantees `Output = SWITTI(Input) + 0` while letting Q/K/V/proj
      receive non-zero gradients via the gate's gradient.
    """
    print("[INFO] Zero-initializing control fusion layers...")

    for name, module in model.named_modules():
        # 1. Cross-attention fusion: zero the scalar gate.
        if hasattr(module, "control_gate") and isinstance(module.control_gate, nn.Parameter):
            nn.init.zeros_(module.control_gate)
            print(f"  ✓ Zeroed {name}.control_gate")

        # 2. Additive fusion: zero the projection.
        if hasattr(module, "control_proj") and module.control_proj is not None:
            nn.init.zeros_(module.control_proj.weight)
            if module.control_proj.bias is not None:
                nn.init.zeros_(module.control_proj.bias)
            print(f"  ✓ Zeroed {name}.control_proj")

def build_everything(args: arg_util.Args):
    # create tensorboard logger
    tb_lg: misc.TensorboardLogger
    if dist.is_master():
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(
            misc.TensorboardLogger(
                log_dir=args.tb_log_dir_path,
                filename_suffix=f'__{misc.time_str("%m%d_%H%M")}',
            ),
            verbose=True,
        )
        tb_lg.flush()
    else:
        # noinspection PyTypeChecker
        tb_lg = misc.DistLogger(None, verbose=False)

    # log args
    print(f"initial args:\n{str(args)}")

    # build models
    vae_local, switti_wo_ddp, pipe = build_models(
        # VQVAE hyperparameters
        V=args.vqvae_vocab_size,
        Cvae=args.vqvae_channel_dim,
        ch=args.vqvae_n_channels,
        share_quant_resi=args.vqvae_share_quant_resi,
        # train hyperparameters
        device=dist.get_device(),
        patch_nums=args.patch_nums,
        depth=args.depth,
        attn_l2_norm=args.anorm,
        init_adaln=args.aln,
        init_adaln_gamma=args.alng,
        init_head=args.hd,
        init_std=args.ini,
        text_encoder_path=args.text_encoder_path,
        text_encoder_2_path=args.text_encoder_2_path,
        rope=args.rope,
        rope_theta=args.rope_theta,
        rope_size=args.rope_size,
        dpr=args.drop_path_rate,
        use_swiglu_ffn=args.use_swiglu_ffn,
        use_crop_cond=args.use_crop_cond,
        control_encoder_type=args.control_encoder_type,
        control_context_dim=args.control_context_dim,
        control_fusion=args.control_fusion,
        control_pretrained=args.control_pretrained,
        control_encoder_ckpt=args.control_encoder_ckpt,
        use_control_gate=args.use_control_gate,
    )

    if args.control_encoder_type:
        print(f"[CONTROL] Encoder type={args.control_encoder_type}, "
              f"fusion={args.control_fusion}, pretrained={args.control_pretrained}")
    
    # === Optional: Load Pretained Switti ===
    if getattr(args, "freeze_switti_backbone", False):
        if dist.is_master():
            print("[INFO] Loading pretrained SwittiHF weights from HF (yresearch/Switti)")

        # Load pretrained HF model (SwittiHF)
        pretrained_hf = SwittiHF.from_pretrained(args.switti_ckpt)

        # Copy HF weights into our training Switti instance
        missing, unexpected = switti_wo_ddp.load_state_dict(pretrained_hf.state_dict(), strict=False)

        if dist.is_master():
            print("[INFO] Loaded pretrained weights.")
            print("  Missing keys   :", len(missing))
            print("  Unexpected keys:", len(unexpected))

        del pretrained_hf
        dist.barrier()
    
    # Load VAE and Switti checkpoints
    if args.vae_ckpt is None:
        args.vae_ckpt = DEFAULT_VAE_CKPT
        if not os.path.exists(DEFAULT_VAE_CKPT) and dist.is_local_master():
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{DEFAULT_VAE_CKPT}')
        dist.barrier()
        vae_local.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu"), strict=True)
    else:
        vae_local = VQVAEHF.from_pretrained(args.vae_ckpt, reso=args.data_load_reso).to(dist.get_device())
        pipe.vae = vae_local

    start_it = load_model_state(args, switti_wo_ddp)
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    switti_wo_ddp: Switti = args.compile_model(switti_wo_ddp, args.tfast)
    if args.use_gradient_checkpointing:
        switti_wo_ddp.enable_gradient_checkpointing()

    if args.control_encoder_type is not None and start_it == 0:
        zero_init_control_layers(switti_wo_ddp)
    elif args.control_encoder_type is not None and start_it > 0:
        print(f"[INFO] Skipping zero-init (resuming from iteration {start_it})")

    print(f"[INIT] Switti model = {switti_wo_ddp}\n\n")
    count_p = lambda m: f"{sum(p.numel() for p in m.parameters())/1e6:.2f}"
    print(f"[INIT][#para] "
        + ", ".join([f"{k}={count_p(m)}"
        for k, m in (
            ("VAE", vae_local),
            ("VAE.enc", vae_local.encoder),
            ("VAE.dec", vae_local.decoder),
            ("VAE.quant", vae_local.quantize),
    )]))
    print(
        f"[INIT][#para] "
        + ", ".join([f"{k}={count_p(m)}" for k, m in (("Switti", switti_wo_ddp),)])
        + "\n\n"
    )

    # === optional: freeze Switti backbone (train only layers related to image conditioning) ===
    if args.freeze_switti_backbone:
        def debug_print_trainable(model):
            kept = []
            for n, p in model.named_parameters():
                if p.requires_grad:
                    kept.append(n)
            print("[trainable params sample]", kept[:200])
        
        apply_control_only_freeze(switti_wo_ddp, freeze_control_backbone=args.control_pretrained, verbose=dist.is_master())
        debug_print_trainable(switti_wo_ddp)

    # FSDP wrapper
    use_fsdp_now = dist.initialized() and args.use_fsdp
    switti: FSDP = (FSDP if use_fsdp_now else NullDDP)(
        switti_wo_ddp,
        auto_wrap_policy=lambda module, recurse, **_etc: recurse or isinstance(module, AdaLNSelfCrossAttn),
        device_id=dist.get_local_rank(),
        sharding_strategy=ShardingStrategy.HYBRID_SHARD if args.use_fsdp else ShardingStrategy.NO_SHARD, #FULL_SHARD,
        use_orig_params=True,
        forward_prefetch=True,
        limit_all_gathers=True,
    )
    # build optimizer
    names, paras, para_groups = filter_params(switti, nowd_keys={
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
    })

    # sanity: ensure that the number of params passed to the optimizer equals the number of trainable params in unwrapped switti
    num_trainable_from_filter = sum(p.numel() for p in paras)
    num_trainable_manual = sum(p.numel() for _, p in switti_wo_ddp.named_parameters() if p.requires_grad)
    if dist.is_master():
        print(f"[sanity] trainable params (filter)={num_trainable_from_filter:,}, (unwrapped)={num_trainable_manual:,}")
    assert num_trainable_from_filter == num_trainable_manual, \
        "Mismatch between freeze() and optimizer param collection! (Investigate requires_grad names)"

    optimizer = torch.optim.AdamW(
        params=para_groups,
        lr=args.tlr, weight_decay=0.0,
        betas=(args.adam_beta1, args.adam_beta2),
        fused=args.afuse if not args.use_fsdp else False,
    )

    switti_optimizer = AmpOptimizer(
        mixed_precision=args.fp16,
        optimizer=optimizer,
        names=names,
        paras=paras,
        grad_clip=args.tclip,
    )
    del names, paras, para_groups

    # Restore optimizer / AMP scaler / RNG state (must be after FSDP wrap +
    # optimizer construction). load_model_state above only loads weights.
    if start_it > 0:
        load_optimizer_state(args, switti, switti_optimizer)

    # build data
    print(f"[build PT data] ...\n")
    print(f"global bs={args.glb_batch_size}, local bs={args.batch_size}")
    dataset_train = build_dataset(
        args.data_path, final_reso=args.data_load_reso, hflip=args.hflip, mid_reso=args.mid_reso, control_types=args.control_types
    )
    ld_train = DataLoader(
        dataset=dataset_train, 
        num_workers=args.workers, 
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None, 
        pin_memory=True,
        generator=args.get_different_generator_for_each_rank(), # worker_init_fn=worker_init_fn,
        collate_fn=coco_collate_fn,
        batch_sampler=DistInfiniteBatchSampler(
            dataset_len=len(dataset_train), glb_batch_size=args.glb_batch_size, same_seed_for_all_ranks=args.same_seed_for_all_ranks,
            shuffle=True, fill_last=True, rank=dist.get_rank(), world_size=dist.get_world_size(), start_it=start_it,
        ),
    )
    del dataset_train

    # print("[Train] After data loader creation")

    if start_it > 0:
        print(f"[FIX] Resetting batch sampler start_it from {start_it} to 0 to avoid slow seeking")
        # The actual training iteration is tracked by the for loop (cur_iter),
        # not by the batch sampler, so this is safe
        ld_train.batch_sampler.start_it = 0

    # build trainer
    trainer = SwittiTrainer(
        dataloader=ld_train,
        device=args.device,
        patch_nums=args.patch_nums,
        resos=args.resos,
        pipe=pipe,
        vae_local=vae_local,
        switti_wo_ddp=switti_wo_ddp,
        switti=switti,
        optimizer=switti_optimizer,
        label_smooth=args.ls,
        args=args,
    )
    torch.cuda.empty_cache()

    return (tb_lg, trainer, start_it)


def main_training():
    torch.set_num_threads(32)
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    (tb_lg, trainer, start_it) = build_everything(args)
    dist.barrier()

    # train
    for cur_iter in range(start_it, args.max_iters):
        tb_lg.set_step(cur_iter)

        # get current lr, wd
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

        # Calculate control strength schedule
        if args.control_encoder_type is not None:
            control_strength = get_control_strength_schedule(
                current_iter=cur_iter,
                warmup_steps=args.control_warmup_steps
            )
        else:
            control_strength = 1.0

        # model forward-backward
        grad_norm, scale_log2 = trainer.train_step(g_it=cur_iter, tb_lg=tb_lg, control_strength=control_strength)

        tb_lg.update(head="AR_opt_lr/lr_min", sche_tlr=min_tlr)
        tb_lg.update(head="AR_opt_lr/lr_max", sche_tlr=max_tlr)
        tb_lg.update(head='AR_opt_wd/wd_max', sche_twd=max_twd)
        tb_lg.update(head='AR_opt_wd/wd_min', sche_twd=min_twd)
        tb_lg.update(head="AR_opt_grad/fp16", scale_log2=scale_log2)
        if args.tclip > 0:
            tb_lg.update(head="AR_opt_grad/grad", grad_norm=grad_norm)
            tb_lg.update(head="AR_opt_grad/grad", grad_clip=args.tclip)
        if args.control_encoder_type is not None:
            tb_lg.update(head="Train", control_strength=control_strength)

        if cur_iter % args.save_iters == 0 and cur_iter > start_it:
            save_model_state(cur_iter, args, trainer.switti, trainer.optimizer)
            # Calculate metrics
            trainer.pipe.switti.eval()
            for eval_set_name in ['coco', 'mjhq']:
                if eval_set_name == "coco":
                    eval_prompts_path = 'eval_prompts/coco.csv'
                    fid_stats_path = args.coco_ref_stats_path
                    control_images_path = os.path.join(args.data_path, "val_control")
                else:
                    eval_prompts_path = 'eval_prompts/mjhq.csv'
                    fid_stats_path = args.mjhq_ref_stats_path
                    control_images_path = None

                with FSDP.summon_full_params(trainer.switti, writeback=False):
                    local_images, local_pick_score, local_clip_score, local_image_reward, local_control_metrics = distributed_metrics_with_csv(
                        trainer.pipe,
                        eval_prompts_path,
                        control_images_path,
                        args,
                    )

                dist.allreduce(local_pick_score)
                pick_score = local_pick_score.item() / dist.get_world_size()

                dist.allreduce(local_clip_score)
                clip_score = local_clip_score.item() / dist.get_world_size()

                dist.allreduce(local_image_reward)
                image_reward = local_image_reward.item() / dist.get_world_size()

                control_metrics = {}
                for metric_name, metric_tensor in local_control_metrics.items():
                    dist.allreduce(metric_tensor)
                    control_metrics[metric_name] = metric_tensor.item() / dist.get_world_size()

                gathered_images = dist.allgather(local_images)
                images = [to_PIL_image(image) for image in gathered_images]

                if dist.is_master():
                    print("Evaluating FID score...")
                    fid_score = calculate_fid(
                        images, fid_stats_path, inception_path=args.inception_path
                    )

                    eval_metrics = {
                        "CLIP score": clip_score,
                        "FID": fid_score,
                        "Pickscore": pick_score,
                        "ImageReward": image_reward,
                    }
                    eval_metrics.update(control_metrics)
                    tb_lg.update(
                        head=f"{eval_set_name}_metrics_top_k={args.top_k}_top_p={args.top_p}_cfg={args.guidance}",
                        **eval_metrics,
                        step=cur_iter,
                    )

                del local_images, images, gathered_images 
                gc.collect(), torch.cuda.empty_cache()

                dist.barrier()
                print("Finished metrics calculation...")
                args.dump_log()
                tb_lg.flush()
            trainer.pipe.switti.train()

    gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    args.remain_time, args.finish_time = "-", time.strftime(
        "%Y-%m-%d %H:%M", time.localtime(time.time() - 60)
    )
    print(f"final args:\n\n{str(args)}")
    args.dump_log()
    tb_lg.flush()
    tb_lg.close()
    dist.barrier()



class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


if __name__ == "__main__":
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, misc.SyncPrint) and isinstance(
            sys.stderr, misc.SyncPrint
        ):
            sys.stdout.close(), sys.stderr.close()
