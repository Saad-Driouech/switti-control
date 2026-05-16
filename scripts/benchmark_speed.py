"""
Inference speed benchmark.

Measures end-to-end wall-clock latency (text encode → generate → decode)
for a configurable list of models:

  • SWITTI baseline / Approach 1 / Approach 2  (this repo)
  • ControlNet + SD 1.5   (diffusers, optional)
  • ControlNet + SDXL     (diffusers, optional)

Each model is run with batch_size=1, fp16, on the current CUDA device.
Timing uses torch.cuda.Event for accurate GPU measurement.
GPU peak-memory is also recorded.

Usage (single GPU):
    python scripts/benchmark_speed.py --config scripts/benchmark_speed_config.yaml \\
        --out_dir /path/to/bench_out/

Output files:
    bench_out/results.csv   – one row per model
    bench_out/results.json  – same, as JSON list
"""

import argparse
import gc
import json
import os
import re
import sys
import time

import numpy as np
import torch
import yaml
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_models, VQVAEHF
from models.switti import SwittiHF
from utils.arg_util import RESOLUTION_PATCH_NUMS_MAPPING


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _cuda_time_ms(fn, n_warmup: int, n_runs: int) -> tuple[float, float]:
    """
    Returns (mean_ms, std_ms) over `n_runs` measured with CUDA events.
    `fn` must be a no-arg callable; it must NOT return anything large
    (we discard the output to avoid measuring transfer overhead).
    """
    torch.cuda.synchronize()

    for _ in range(n_warmup):
        with torch.no_grad():
            fn()
        torch.cuda.synchronize()

    times_ms = []
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(n_runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_ev.record()
        with torch.no_grad():
            fn()
        end_ev.record()
        torch.cuda.synchronize()
        times_ms.append(start_ev.elapsed_time(end_ev))

    return float(np.mean(times_ms)), float(np.std(times_ms))


def _peak_mem_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1024 ** 3


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def _strip_fsdp_prefix(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        k = k.replace("_fsdp_wrapped_module.", "")
        k = k.replace("_fully_sharded_module.", "")
        k = re.sub(r"^module\.", "", k)
        out[k] = v
    return out


def _build_switti_pipe(run: dict, cfg: dict):
    """Load a SWITTI-based pipeline (baseline, Approach 1, or Approach 2)."""
    reso = cfg.get("reso", 512)
    patch_nums = tuple(
        int(x) for x in RESOLUTION_PATCH_NUMS_MAPPING[reso].split("_")
    )
    device = torch.device("cuda")

    vae_local, switti, pipe = build_models(
        device=device,
        patch_nums=patch_nums,
        depth=cfg.get("depth", 30),
        use_swiglu_ffn=True,
        use_crop_cond=True,
        control_encoder_type=run.get("control_encoder_type"),
        control_context_dim=run.get("control_context_dim", 384),
        control_fusion=run.get("control_fusion", "cross"),
        control_pretrained=run.get("control_pretrained", True),
        control_encoder_ckpt=run.get("control_encoder_ckpt"),
    )

    base = SwittiHF.from_pretrained(cfg.get("pretrained_switti", "yresearch/Switti"))
    switti.load_state_dict(base.state_dict(), strict=False)
    del base

    vae_id = cfg.get("vae_ckpt", "yresearch/VQVAE-Switti")
    vae_local = VQVAEHF.from_pretrained(vae_id, reso=reso).to(device)
    pipe.vae = vae_local

    ckpt = run.get("ckpt")
    if ckpt:
        sd = _strip_fsdp_prefix(torch.load(ckpt, map_location="cpu"))
        switti.load_state_dict(sd, strict=False)

    switti.eval().to(device)
    vae_local.eval()

    # bfloat16: same memory as fp16 but float32 exponent range avoids inf/nan in logits
    switti.to(torch.bfloat16)
    vae_local.to(torch.bfloat16)

    return pipe


def _build_diffusers_sd15_controlnet(run: dict):
    """
    ControlNet + SD 1.5 via diffusers.
    run keys: model_id, controlnet_id, num_inference_steps, scheduler
    """
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        DDIMScheduler,
        UniPCMultistepScheduler,
    )

    controlnet_id = run.get("controlnet_id", "lllyasviel/sd-controlnet-canny")
    model_id = run.get("model_id", "runwayml/stable-diffusion-v1-5")

    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    scheduler_name = run.get("scheduler", "unipc")
    if scheduler_name == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()
    return pipe


def _build_diffusers_sdxl_controlnet(run: dict):
    """
    ControlNet + SDXL via diffusers.
    run keys: model_id, controlnet_id, num_inference_steps
    """
    from diffusers import (
        StableDiffusionXLControlNetPipeline,
        ControlNetModel,
        AutoencoderKL,
    )

    controlnet_id = run.get(
        "controlnet_id", "diffusers/controlnet-canny-sdxl-1.0"
    )
    model_id = run.get("model_id", "stabilityai/stable-diffusion-xl-base-1.0")

    controlnet = ControlNetModel.from_pretrained(
        controlnet_id, torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        model_id,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to("cuda")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()
    return pipe


# ---------------------------------------------------------------------------
# Per-type benchmark runner
# ---------------------------------------------------------------------------

def _bench_switti(run: dict, cfg: dict, prompt: str,
                  control_img_tensor, n_warmup: int, n_runs: int) -> dict:
    pipe = _build_switti_pipe(run, cfg)
    reso = cfg.get("reso", 512)
    guidance = cfg.get("guidance", 6.0)
    top_k = cfg.get("top_k", 400)
    top_p = cfg.get("top_p", 0.95)
    control_end_si = cfg.get("control_end_si", 8)

    has_control = run.get("control_encoder_type") is not None

    if has_control and control_img_tensor is not None:
        modality = run.get("modality", "canny")
        control_dict = {modality: control_img_tensor.cuda().to(torch.bfloat16).unsqueeze(0)}
    else:
        control_dict = None

    def _generate():
        with torch.autocast("cuda"):
            pipe(
                prompt=prompt,
                cfg=guidance,
                top_k=top_k,
                top_p=top_p,
                return_pil=False,
                control_dict=control_dict,
                control_end_si=control_end_si,
            )

    torch.cuda.reset_peak_memory_stats()
    mean_ms, std_ms = _cuda_time_ms(_generate, n_warmup, n_runs)
    peak_gb = _peak_mem_gb()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "peak_mem_gb": peak_gb,
    }


def _build_switti_v2_pipe(run: dict, cfg: dict):
    """Load an Approach-2 pipeline (SwittiControlNet + SwittiControlPipeline)."""
    from models.switti import SwittiHF
    from models.control_switti import SwittiControlNet
    from models.control_pipeline import SwittiControlPipeline
    from models.clip import FrozenCLIPEmbedder

    reso = cfg.get("reso", 512)
    device = torch.device("cuda")

    frozen_switti = SwittiHF.from_pretrained(
        cfg.get("pretrained_switti", "yresearch/Switti")
    ).to(device)

    control_net = SwittiControlNet(
        frozen_switti=frozen_switti,
        num_modalities=run.get("num_modalities", 7),
    ).to(device)

    ckpt = run.get("ckpt")
    if ckpt:
        sd = _strip_fsdp_prefix(torch.load(ckpt, map_location="cpu"))
        control_net.load_state_dict(sd, strict=False)

    vae_id = cfg.get("vae_ckpt", "yresearch/VQVAE-Switti")
    vae_local = VQVAEHF.from_pretrained(vae_id, reso=reso).to(device)

    text_encoder = FrozenCLIPEmbedder(
        cfg.get("text_encoder_path", "openai/clip-vit-large-patch14"), device=device
    )
    text_encoder_2 = FrozenCLIPEmbedder(
        cfg.get("text_encoder_2_path", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"),
        device=device,
    )

    pipe = SwittiControlPipeline(
        control_net=control_net,
        vae=vae_local,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        device=device,
    )

    control_net.eval().half()
    vae_local.eval().half()
    return pipe


def _bench_switti_v2(run: dict, cfg: dict, prompt: str,
                     control_img_tensor, n_warmup: int, n_runs: int) -> dict:
    pipe = _build_switti_v2_pipe(run, cfg)
    guidance = cfg.get("guidance", 6.0)
    top_k = cfg.get("top_k", 400)
    top_p = cfg.get("top_p", 0.95)
    modality = run.get("modality", "canny")

    ctrl_pil = Image.fromarray(
        (control_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )

    def _generate():
        with torch.autocast("cuda"):
            pipe(
                prompt=prompt,
                ctrl_image=ctrl_pil,
                modality=modality,
                cfg=guidance,
                top_k=top_k,
                top_p=top_p,
                return_pil=False,
            )

    torch.cuda.reset_peak_memory_stats()
    mean_ms, std_ms = _cuda_time_ms(_generate, n_warmup, n_runs)
    peak_gb = _peak_mem_gb()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "peak_mem_gb": peak_gb,
    }


def _bench_diffusers_sd15(run: dict, cfg: dict, prompt: str,
                           control_pil: Image.Image,
                           n_warmup: int, n_runs: int) -> dict:
    pipe = _build_diffusers_sd15_controlnet(run)
    steps = run.get("num_inference_steps", 20)
    reso = cfg.get("reso", 512)
    ctrl = control_pil.resize((reso, reso))

    def _generate():
        pipe(
            prompt=prompt,
            image=ctrl,
            num_inference_steps=steps,
            height=reso,
            width=reso,
        )

    torch.cuda.reset_peak_memory_stats()
    mean_ms, std_ms = _cuda_time_ms(_generate, n_warmup, n_runs)
    peak_gb = _peak_mem_gb()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "peak_mem_gb": peak_gb,
    }


def _bench_diffusers_sdxl(run: dict, cfg: dict, prompt: str,
                           control_pil: Image.Image,
                           n_warmup: int, n_runs: int) -> dict:
    pipe = _build_diffusers_sdxl_controlnet(run)
    steps = run.get("num_inference_steps", 20)
    reso = cfg.get("reso", 512)  # SDXL natively 1024; user can override
    ctrl = control_pil.resize((reso, reso))

    def _generate():
        pipe(
            prompt=prompt,
            image=ctrl,
            num_inference_steps=steps,
            height=reso,
            width=reso,
        )

    torch.cuda.reset_peak_memory_stats()
    mean_ms, std_ms = _cuda_time_ms(_generate, n_warmup, n_runs)
    peak_gb = _peak_mem_gb()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "peak_mem_gb": peak_gb,
    }


# ---------------------------------------------------------------------------
# Custom / external model support
# ---------------------------------------------------------------------------

def _bench_custom(run: dict, cfg: dict, prompt: str,
                  control_img_tensor, n_warmup: int, n_runs: int) -> dict:
    """
    Generic adapter for any external model (ControlAR, LlamaGen+Control, VAR, …).

    The run entry must specify:
        wrapper: /path/to/my_wrapper.py

    The wrapper module must expose two callables:

        def load_pipe(run_cfg: dict) -> object:
            # Build and return the model / pipeline object.
            # Called once; the return value is passed to `generate` below.

        def generate(pipe, prompt: str, ctrl_pil: PIL.Image.Image,
                     run_cfg: dict) -> None:
            # Run one forward pass. Return value is ignored.
            # Must NOT accumulate output tensors across calls.

    The wrapper is free to import any external library (e.g. the ControlAR
    repo), as long as those packages are on PYTHONPATH before calling this
    script.  A minimal ControlAR wrapper is provided in
        scripts/wrappers/controlar_wrapper.py
    """
    import importlib.util

    wrapper_path = run.get("wrapper")
    if not wrapper_path or not os.path.exists(wrapper_path):
        raise FileNotFoundError(
            f"'wrapper' path not found or not set: {wrapper_path!r}\n"
            "Each 'custom' run must have a 'wrapper' key pointing to a "
            "Python file that exposes load_pipe() and generate()."
        )

    spec = importlib.util.spec_from_file_location("_bench_wrapper", wrapper_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "load_pipe") or not hasattr(mod, "generate"):
        raise AttributeError(
            f"{wrapper_path} must define both load_pipe(run_cfg) "
            "and generate(pipe, prompt, ctrl_pil, run_cfg)."
        )

    reso = cfg.get("reso", 512)
    ctrl_pil = Image.fromarray(
        (control_img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ).resize((reso, reso))

    pipe = mod.load_pipe(run)

    def _generate():
        mod.generate(pipe, prompt, ctrl_pil, run)

    torch.cuda.reset_peak_memory_stats()
    mean_ms, std_ms = _cuda_time_ms(_generate, n_warmup, n_runs)
    peak_gb = _peak_mem_gb()

    if hasattr(mod, "cleanup"):
        mod.cleanup(pipe)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "peak_mem_gb": peak_gb,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DISPATCH = {
    "switti": _bench_switti,
    "switti_v2": _bench_switti_v2,
    "controlnet_sd15": _bench_diffusers_sd15,
    "controlnet_sdxl": _bench_diffusers_sdxl,
    "custom": _bench_custom,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", required=True)
    args_cli = ap.parse_args()

    with open(args_cli.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args_cli.out_dir, exist_ok=True)

    n_warmup = cfg.get("n_warmup", 5)
    n_runs = cfg.get("n_runs", 50)
    prompt = cfg.get("prompt", "A photo of a dog in a park.")
    reso = cfg.get("reso", 512)

    # Dummy control image (uniform grey)
    ctrl_pil = Image.fromarray(
        np.full((reso, reso, 3), 128, dtype=np.uint8)
    )
    ctrl_tensor = torch.from_numpy(
        np.array(ctrl_pil, dtype=np.float32) / 255.0
    ).permute(2, 0, 1)  # (3, H, W) in [0, 1]

    device_name = torch.cuda.get_device_name(0)
    print(f"Device : {device_name}")
    print(f"Warmup : {n_warmup}  |  Runs: {n_runs}")
    print(f"Prompt : {prompt!r}")

    results = []

    for run in cfg["runs"]:
        name = run["name"]
        run_type = run.get("type", "switti")
        print(f"\n{'='*60}")
        print(f"  {name}  [{run_type}]")
        print(f"{'='*60}")

        try:
            bench_fn = _DISPATCH.get(run_type)
            if bench_fn is None:
                print(f"  [skip] unknown type '{run_type}'")
                continue
            # switti/custom take a tensor; diffusers take a PIL
            if run_type in ("switti", "switti_v2", "custom"):
                timing = bench_fn(run, cfg, prompt, ctrl_tensor, n_warmup, n_runs)
            else:
                timing = bench_fn(run, cfg, prompt, ctrl_pil, n_warmup, n_runs)

        except Exception as e:
            print(f"  [ERROR] {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

        mean_s = timing["mean_ms"] / 1000.0
        std_s = timing["std_ms"] / 1000.0
        throughput = 1.0 / mean_s

        row = {
            "name": name,
            "type": run_type,
            "modality": run.get("modality"),
            "mean_latency_s": round(mean_s, 4),
            "std_latency_s": round(std_s, 4),
            "throughput_img_per_s": round(throughput, 4),
            "peak_gpu_mem_gb": round(timing["peak_mem_gb"], 3),
            "n_warmup": n_warmup,
            "n_runs": n_runs,
            "device": device_name,
            "resolution": reso,
        }
        if run_type in ("controlnet_sd15", "controlnet_sdxl"):
            row["num_inference_steps"] = run.get("num_inference_steps", 20)

        results.append(row)
        print(
            f"  latency : {mean_s:.3f} ± {std_s:.3f} s/image"
            f"  ({throughput:.2f} img/s)"
        )
        print(f"  peak mem: {timing['peak_mem_gb']:.2f} GB")

    # Write outputs
    json_path = os.path.join(args_cli.out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    csv_path = os.path.join(args_cli.out_dir, "results.csv")
    try:
        import pandas as pd
        pd.DataFrame(results).to_csv(csv_path, index=False)
        print(f"\n[done] {csv_path}  ({len(results)} rows)")
    except ImportError:
        import csv
        with open(csv_path, "w", newline="") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"\n[done] {csv_path}  ({len(results)} rows)")

    # Print summary table
    print(f"\n{'Model':<45} {'Latency (s)':<14} {'Img/s':<10} {'Mem (GB)'}")
    print("-" * 85)
    for r in results:
        steps_note = f" @{r['num_inference_steps']}steps" if "num_inference_steps" in r else ""
        print(
            f"{r['name'] + steps_note:<45}"
            f"{r['mean_latency_s']:.3f} ± {r['std_latency_s']:.3f}  "
            f"{r['throughput_img_per_s']:<10.2f}"
            f"{r['peak_gpu_mem_gb']:.2f}"
        )


if __name__ == "__main__":
    main()
