import math
import os

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType

import dist
from utils.misc import glob_with_latest_modified_first


def bcast_state_dict(state_dict):
    for data in state_dict.values():
        if isinstance(data, torch.Tensor):
            dist.broadcast(data, 0)
        elif isinstance(data, dict):
            bcast_state_dict(data)
        else:
            raise Exception(f"Unsupported type: {type(data)}")


def save_model_state(cur_iter: int, args, model: torch.nn.Module, amp_optimizer=None):
    """Save model weights, optimizer state, AMP scaler, RNG state, and training args.

    The optimizer / scaler / RNG state is required to truly resume training
    without resetting Adam moments and AMP scale (which causes a regression
    spike on every restart).
    """

    is_fsdp = isinstance(model, FSDP)
    optim = amp_optimizer.optimizer if amp_optimizer is not None else None

    # Save model state
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=False, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model_state_dict = model.state_dict()

        # Optimizer state must be gathered collectively from all ranks; only
        # rank 0 ends up with the full dict (rank0_only=True).
        optim_state_dict = None
        if optim is not None:
            if is_fsdp:
                optim_state_dict = FSDP.optim_state_dict(model, optim)
            else:
                optim_state_dict = optim.state_dict()

        if dist.is_master():
            os.makedirs(args.local_out_dir_path, exist_ok=True)
            model_save_path = os.path.join(
                args.local_out_dir_path, "model_state_dict.pt"
            )
            torch.save(model_state_dict, model_save_path)

            metadata = {"iter": cur_iter, "args": args.state_dict()}
            metadata_save_path = os.path.join(args.local_out_dir_path, "metadata.pt")
            torch.save(metadata, metadata_save_path)

            optim_payload = None
            if amp_optimizer is not None:
                optim_payload = {
                    "iter": cur_iter,
                    "optim": optim_state_dict,
                    "scaler": (
                        amp_optimizer.scaler.state_dict()
                        if amp_optimizer.scaler is not None
                        else None
                    ),
                    "rng": {
                        "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all(),
                    },
                }
                torch.save(
                    optim_payload,
                    os.path.join(args.local_out_dir_path, "optim_state_dict.pt"),
                )

            # Save global checkpoints
            if cur_iter % args.global_save_iters == 0:
                model_save_path = os.path.join(
                    args.local_out_dir_path, f"model_{cur_iter}_state_dict.pt"
                )
                torch.save(model_state_dict, model_save_path)

                metadata_save_path = os.path.join(
                    args.local_out_dir_path, f"metadata_{cur_iter}.pt"
                )
                torch.save(metadata, metadata_save_path)

                # Numbered optimizer snapshots are intentionally NOT written:
                # auto_resume only reads the rolling optim_state_dict.pt, and
                # numbered Adam states have no use case (~40 GB each).

            print(f"Saved model and optimizer state dicts to {args.local_out_dir_path}")


def load_model_state(args, model: torch.nn.Module) -> int:
    """Load model, optimizer state dict and metadata saved via save_training_state; update parameters in-place"""
    model_path = os.path.join(args.local_out_dir_path, f"model_state_dict.pt")
    metadata_path = os.path.join(args.local_out_dir_path, "metadata.pt")

    if not os.path.exists(model_path):
        start_iter = 0
        file = os.path.join(args.local_out_dir_path, "*.pt")
        all_ckpt = glob_with_latest_modified_first(file)
        if dist.is_master():
            print(f".pt files in {args.local_out_dir_path}: {all_ckpt}")
            print(f"[auto_resume failed] start training from scratch {start_iter}")
    else:
        model_state_dict = torch.load(model_path, map_location="cpu")
        metadata = torch.load(metadata_path, map_location="cpu")
        if args.resos[-1] != metadata["args"]["resos"][-1]:
            # rewrite registered buffers for a different target resolution
            L = sum([pn * pn for pn in args.patch_nums])
            C = args.depth * 64
            d = torch.cat(
                [torch.full((pn * pn,), i) for i, pn in enumerate(args.patch_nums)]
            ).view(1, L, 1)
            dT = d.transpose(1, 2)  # dT: 11L
            model_state_dict["lvl_1L"] = dT[:, 0].contiguous()
            attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(
                1, 1, L, L
            )
            model_state_dict["attn_bias_for_masking"] = attn_bias_for_masking

            if not args.rope:
                # absolute position embedding
                init_std = math.sqrt(1 / C / 3)
                pos_1LC = []
                for i, pn in enumerate(args.patch_nums):
                    pe = torch.empty(1, pn * pn, C)
                    nn.init.trunc_normal_(pe, mean=0, std=init_std)
                    pos_1LC.append(pe)
                pos_1LC = torch.cat(pos_1LC, dim=1)  # 1, L, C
                assert tuple(pos_1LC.shape) == (1, L, C)
                model_state_dict["pos_1LC"] = pos_1LC

        model.load_state_dict(model_state_dict)

        print(f"Loaded training state from {args.local_out_dir_path}: {metadata}")
        
        start_iter = metadata["iter"] + 1  # start from iter + 1 to avoid double evals
        print(f"[auto_resume success] resume from iteration {start_iter}")

    dist.barrier()
    bcast_state_dict(model.state_dict())
    start_iter_t = torch.tensor(start_iter, device=dist.get_device())
    dist.broadcast(start_iter_t, 0)
    start_iter = start_iter_t.item()
    dist.barrier()
    return start_iter


def load_optimizer_state(args, model: torch.nn.Module, amp_optimizer) -> None:
    """Load optimizer / AMP scaler / RNG state saved by save_model_state.

    Must be called after the optimizer is constructed and the model is wrapped
    in FSDP (matching the wrapping at save time). Without this, every resume
    starts Adam moments from zero and the loss spikes for thousands of iters.
    """
    optim_path = os.path.join(args.local_out_dir_path, "optim_state_dict.pt")
    if not os.path.exists(optim_path):
        if dist.is_master():
            print(f"[load_optimizer_state] no optim_state_dict.pt at {args.local_out_dir_path} — fresh optimizer")
        return

    is_fsdp = isinstance(model, FSDP)
    optim = amp_optimizer.optimizer

    # Rank 0 loads from disk, all ranks participate in the FSDP collective.
    if dist.is_master():
        payload = torch.load(optim_path, map_location="cpu", weights_only=False)
    else:
        payload = None

    full_optim_state = payload["optim"] if payload is not None else None
    if is_fsdp:
        sharded = FSDP.optim_state_dict_to_load(
            model=model,
            optim=optim,
            optim_state_dict=full_optim_state,
        )
        optim.load_state_dict(sharded)
    else:
        if full_optim_state is not None:
            optim.load_state_dict(full_optim_state)

    # Scaler + RNG: rank-0 has the saved state; broadcast scalar fields via
    # AMP scaler (small, no harm to load on every rank from disk).
    if payload is None and dist.is_master() is False:
        # other ranks read the small scaler/rng portion themselves
        payload = torch.load(optim_path, map_location="cpu", weights_only=False)

    if amp_optimizer.scaler is not None and payload.get("scaler") is not None:
        try:
            amp_optimizer.scaler.load_state_dict(payload["scaler"])
        except Exception as e:
            print(f"[load_optimizer_state] scaler load failed: {e}")

    rng = payload.get("rng")
    if rng is not None:
        try:
            torch.set_rng_state(rng["torch"])
            cuda_rng = rng["cuda"]
            num_dev = torch.cuda.device_count()
            if isinstance(cuda_rng, list) and len(cuda_rng) == num_dev:
                torch.cuda.set_rng_state_all(cuda_rng)
            elif isinstance(cuda_rng, list) and len(cuda_rng) > 0:
                torch.cuda.set_rng_state(cuda_rng[dist.get_local_rank() % len(cuda_rng)])
        except Exception as e:
            print(f"[load_optimizer_state] rng restore failed: {e}")

    if dist.is_master():
        print(f"[load_optimizer_state] restored optimizer/scaler/rng from iter {payload.get('iter')}")
    dist.barrier()
