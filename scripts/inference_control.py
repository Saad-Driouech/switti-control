"""
inference_control.py — Generate images with a (optionally control-conditioned) Switti model.

Usage examples:
    # Text-to-image (base pretrained model):
    python scripts/inference_control.py \
        --prompts "a dog on the beach" "a cat in space" \
        --output_dir outputs/

    # With a fine-tuned checkpoint, text-to-image:
    python scripts/inference_control.py \
        --prompts "a dog on the beach" \
        --ckpt /path/to/model_state_dict.pt \
        --output_dir outputs/

    # With fine-tuned checkpoint + control images:
    python scripts/inference_control.py \
        --prompts_file prompts.txt \
        --control_dir /path/to/control_images \
        --control_type depth \
        --ckpt /path/to/model_state_dict.pt \
        --control_encoder_type vit \
        --control_encoder_ckpt vit_base_patch14_dinov2 \
        --control_context_dim 384 \
        --output_dir outputs/

Control images directory:
    Must contain images named 0.png, 1.png, ... (one per prompt, in order).
    Missing images are silently skipped (that prompt runs without control).

Prompts file:
    Plain text file, one prompt per line. Blank lines and lines starting with
    '#' are ignored.
"""

import argparse
import os
import re
import sys

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

# Add project root to sys.path so imports work when called from any cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models import build_models, VQVAEHF
from models.switti import SwittiHF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_fsdp_prefix(state_dict: dict) -> dict:
    """Remove FSDP / DDP wrapper prefixes from state-dict keys."""
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace("_fsdp_wrapped_module.", "")
        k = k.replace("_fully_sharded_module.", "")
        k = re.sub(r"^module\.", "", k)
        cleaned[k] = v
    return cleaned


def load_prompts(prompts: list[str] | None, prompts_file: str | None) -> list[str]:
    if prompts_file is not None:
        with open(prompts_file) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        return lines
    if prompts:
        return list(prompts)
    raise ValueError("Provide --prompts or --prompts_file")


def load_control_image(path: str, reso: int) -> torch.Tensor:
    """Load a control image, resize/crop to reso, normalise to [-1, 1]. Returns (3, H, W)."""
    img = Image.open(path).convert("RGB")
    mid = round(1.125 * reso)
    img = TF.resize(img, mid, interpolation=InterpolationMode.LANCZOS)
    img = TF.center_crop(img, (reso, reso))
    t = TF.to_tensor(img)               # [0, 1]
    t = t * 2.0 - 1.0                   # [-1, 1]
    return t


def save_image(tensor: torch.Tensor, path: str):
    """Save a (3, H, W) tensor in [0, 1] as PNG."""
    img = TF.to_pil_image(tensor.clamp(0, 1).cpu().float())
    img.save(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Switti control inference")

    # --- Input ---
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prompts", nargs="+", help="One or more text prompts")
    g.add_argument("--prompts_file", help="Path to a text file with one prompt per line")

    p.add_argument("--control_dir", default=None,
                   help="Directory of control images named 0.png, 1.png, ...")
    p.add_argument("--control_type", default=None,
                   help="Control modality: canny | depth | gray | hed | seg | normal")

    # --- Model ---
    p.add_argument("--ckpt", default=None,
                   help="Path to model_state_dict.pt (fine-tuned checkpoint). "
                        "If omitted, uses base pretrained Switti from HuggingFace.")
    p.add_argument("--pretrained_switti", default="yresearch/Switti",
                   help="HuggingFace model ID for the base Switti weights (default: yresearch/Switti)")
    p.add_argument("--vae_ckpt", default="yresearch/VQVAE-Switti",
                   help="HuggingFace model ID or local path for the VAE")
    p.add_argument("--reso", type=int, default=512, choices=[256, 512, 1024],
                   help="Output resolution (default: 512)")
    p.add_argument("--depth", type=int, default=30)

    # Control encoder (only needed when --ckpt has control weights)
    p.add_argument("--control_encoder_type", default=None, choices=["cnn", "vit"],
                   help="Control encoder architecture (required if checkpoint has control weights)")
    p.add_argument("--control_encoder_ckpt", default=None,
                   help="timm model name for pretrained ViT backbone (e.g. vit_base_patch14_dinov2)")
    p.add_argument("--control_context_dim", type=int, default=384)
    p.add_argument("--control_fusion", default=None, choices=["cross", "add"])
    p.add_argument("--control_pretrained", action="store_true", default=True)

    # --- Generation ---
    p.add_argument("--num_images", type=int, default=3,
                   help="Number of images to generate per prompt (default: 3)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cfg", type=float, default=6.0)
    p.add_argument("--top_k", type=int, default=400)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--more_smooth", action="store_true", default=False)
    p.add_argument("--control_end_si", type=int, default=8,
                   help="Stop applying control after this scale index (default: 8)")
    p.add_argument("--ctrl_strength", type=float, default=1.0,
                   help="Multiply control image by this factor (default: 1.0)")

    # --- Output ---
    p.add_argument("--output_dir", required=True, help="Directory to save generated images")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Build model
    # -----------------------------------------------------------------------
    print(f"[INFO] Building model (depth={args.depth}, reso={args.reso}) ...")
    device = torch.device(args.device)

    from utils.arg_util import RESOLUTION_PATCH_NUMS_MAPPING
    patch_nums = tuple(int(x) for x in RESOLUTION_PATCH_NUMS_MAPPING[args.reso].split("_"))

    vae_local, switti, pipe = build_models(
        device=device,
        patch_nums=patch_nums,
        depth=args.depth,
        use_swiglu_ffn=True,
        use_crop_cond=True,
        control_encoder_type=args.control_encoder_type,
        control_context_dim=args.control_context_dim,
        control_fusion=args.control_fusion,
        control_pretrained=args.control_pretrained,
        control_encoder_ckpt=args.control_encoder_ckpt,
    )

    # -----------------------------------------------------------------------
    # 2. Load pretrained base weights
    # -----------------------------------------------------------------------
    print(f"[INFO] Loading base Switti weights from '{args.pretrained_switti}' ...")
    pretrained_hf = SwittiHF.from_pretrained(args.pretrained_switti)
    missing, unexpected = switti.load_state_dict(pretrained_hf.state_dict(), strict=False)
    print(f"  Base weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
    del pretrained_hf

    # Load VAE
    vae_local = VQVAEHF.from_pretrained(args.vae_ckpt, reso=args.reso).to(device)
    pipe.vae = vae_local

    # -----------------------------------------------------------------------
    # 3. Load fine-tuned checkpoint (overwrites base weights that are present)
    # -----------------------------------------------------------------------
    if args.ckpt is not None:
        print(f"[INFO] Loading checkpoint from '{args.ckpt}' ...")
        state_dict = torch.load(args.ckpt, map_location="cpu")
        state_dict = _strip_fsdp_prefix(state_dict)
        missing, unexpected = switti.load_state_dict(state_dict, strict=False)
        print(f"  Checkpoint loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")

    switti.eval()
    vae_local.eval()

    # -----------------------------------------------------------------------
    # 4. Load prompts
    # -----------------------------------------------------------------------
    prompts = load_prompts(args.prompts, args.prompts_file)
    print(f"[INFO] {len(prompts)} prompt(s) loaded")

    # -----------------------------------------------------------------------
    # 5. Generate
    # -----------------------------------------------------------------------
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\n[{prompt_idx + 1}/{len(prompts)}] '{prompt[:80]}'")

        # Load control image if available
        control_dict = None
        if args.control_dir is not None and args.control_type is not None:
            ctrl_path = os.path.join(args.control_dir, f"{prompt_idx}.png")
            if os.path.exists(ctrl_path):
                ctrl_tensor = load_control_image(ctrl_path, args.reso) * args.ctrl_strength
                control_dict = {args.control_type: ctrl_tensor.unsqueeze(0).to(device)}
                print(f"  Control image: {ctrl_path}")

                # Save control image to output dir for reference
                ctrl_save = os.path.join(args.output_dir, f"prompt_{prompt_idx:04d}_control.png")
                save_image((ctrl_tensor * 0.5 + 0.5), ctrl_save)
            else:
                print(f"  No control image found at {ctrl_path}, running without control")

        # Generate num_images images with different seeds
        for img_idx in range(args.num_images):
            seed = args.seed + img_idx
            with torch.inference_mode():
                out = pipe(
                    prompt=[prompt],
                    seed=seed,
                    cfg=args.cfg,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    more_smooth=args.more_smooth,
                    return_pil=False,
                    control_dict=control_dict,
                    control_end_si=args.control_end_si,
                )
            # out: (1, 3, H, W) in [0, 1]
            save_path = os.path.join(args.output_dir, f"prompt_{prompt_idx:04d}_{img_idx}.png")
            save_image(out[0], save_path)
            print(f"  [{img_idx + 1}/{args.num_images}] seed={seed} → {save_path}")

    print(f"\n[INFO] Done. Images saved to '{args.output_dir}'")


if __name__ == "__main__":
    main()
