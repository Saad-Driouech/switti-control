"""
Inference pipeline for SwittiControlNet.

Extends SwittiPipeline with scale-wise spatial control injection.
Control is applied only to the conditional CFG branch; the null branch
uses the learned null modality embedding to match training-time control dropout.
"""
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from models.pipeline import SwittiPipeline, TRAIN_IMAGE_SIZE
from models.switti import get_crop_condition
from models.control_switti import SwittiControlNet, MODALITY_IDS
from models.vqvae import VQVAEHF
from models.clip import FrozenCLIPEmbedder
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng


class SwittiControlPipeline(SwittiPipeline):
    """
    Inference pipeline for SwittiControlNet.

    Usage::

        pipe = SwittiControlPipeline.from_pretrained(
            "yresearch/Switti",
            control_ckpt="path/to/control.pth",
        )
        imgs = pipe(
            prompt="a cat",
            ctrl_image=pil_image,
            modality="canny",
            ctrl_strength=1.0,
        )
    """

    def __init__(
        self,
        control_net: SwittiControlNet,
        vae,
        text_encoder,
        text_encoder_2,
        device,
        dtype=torch.float32,
    ):
        # Initialise the parent with the frozen Switti inside control_net
        super().__init__(
            switti=control_net.frozen_switti,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            device=device,
            dtype=dtype,
        )
        self.control_net = control_net.to(dtype)
        self.control_net.eval()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        control_ckpt: Optional[str] = None,
        torch_dtype=torch.bfloat16,
        device="cuda",
        reso=1024,
        num_modalities=7,
    ):
        from models.switti import SwittiHF
        from models.control_switti import SwittiControlNet

        frozen_switti = SwittiHF.from_pretrained(pretrained_model_name_or_path).to(device)
        vae = VQVAEHF.from_pretrained(cls.vae_path, reso=reso).to(device)
        text_encoder = FrozenCLIPEmbedder(cls.text_encoder_path, device=device)
        text_encoder_2 = FrozenCLIPEmbedder(cls.text_encoder_2_path, device=device)

        control_net = SwittiControlNet(frozen_switti, num_modalities=num_modalities)
        if control_ckpt is not None:
            state = torch.load(control_ckpt, map_location="cpu")
            control_net.load_state_dict(state, strict=False)
        control_net = control_net.to(device)

        return cls(control_net, vae, text_encoder, text_encoder_2, device, torch_dtype)

    @contextmanager
    def _kv_caching(self, enable_ar: bool):
        """Enable KV caching for both frozen and control blocks."""
        switti = self.switti
        ctrl_blocks = self.control_net.control_blocks
        for b in switti.blocks:
            b.attn.kv_caching(enable_ar)
            b.cross_attn.kv_caching(True)
        for b in ctrl_blocks:
            b.attn.kv_caching(enable_ar)
            b.cross_attn.kv_caching(True)
        try:
            yield
        finally:
            for b in switti.blocks:
                b.attn.kv_caching(False)
                b.cross_attn.kv_caching(False)
            for b in ctrl_blocks:
                b.attn.kv_caching(False)
                b.cross_attn.kv_caching(False)

    @staticmethod
    def _preprocess_ctrl_image(ctrl_image, device, dtype, size=512):
        """Convert a PIL image or (B,3,H,W) tensor to a normalised tensor."""
        if isinstance(ctrl_image, Image.Image):
            ctrl_image = [ctrl_image]
        if isinstance(ctrl_image, (list, tuple)) and isinstance(ctrl_image[0], Image.Image):
            to_tensor = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            ctrl_image = torch.stack([to_tensor(img) for img in ctrl_image])
        ctrl_image = ctrl_image.to(device=device, dtype=dtype)
        return ctrl_image

    @torch.inference_mode()
    def __call__(
        self,
        prompt,
        ctrl_image=None,
        modality: str = "canny",
        ctrl_strength: float = 1.0,
        null_prompt: str = "",
        seed: Optional[int] = None,
        cfg: float = 6.0,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = False,
        return_pil: bool = True,
        smooth_start_si: int = 0,
        turn_off_cfg_start_si: int = 10,
        turn_on_cfg_start_si: int = 0,
        last_scale_temp: Optional[float] = None,
    ):
        """
        :param prompt: text prompt or list of prompts
        :param ctrl_image: PIL Image, list of PIL Images, or (B,3,H,W) tensor in [-1,1]
        :param modality: one of 'canny', 'depth', 'seg', 'normal', 'hed'
        :param ctrl_strength: multiplier on the control signal (default 1.0)
        """
        assert not self.switti.training
        switti = self.switti
        control_net = self.control_net
        vae = self.vae
        vae_quant = vae.quantize

        if seed is None:
            rng = None
        else:
            switti.rng.manual_seed(seed)
            rng = switti.rng

        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)
        B = context.shape[0] // 2

        # ctrl_image=None → T2I mode: zero control signal
        if ctrl_image is None:
            ctrl_image = torch.zeros(B, 3, TRAIN_IMAGE_SIZE[0], TRAIN_IMAGE_SIZE[1])
            ctrl_strength = 0.0

        cond_vector = switti.text_pooler(cond_vector)

        if switti.use_crop_cond:
            crop_coords = get_crop_condition(
                2 * B * [TRAIN_IMAGE_SIZE[0]],
                2 * B * [TRAIN_IMAGE_SIZE[1]],
            ).to(cond_vector.device)
            crop_embed = switti.crop_embed(crop_coords.view(-1)).reshape(2 * B, switti.D)
            crop_cond = switti.crop_proj(crop_embed)
        else:
            crop_cond = None

        sos = cond_BD = cond_vector
        lvl_pos = switti.lvl_embed(switti.lvl_1L)
        if not switti.rope:
            lvl_pos += switti.pos_1LC

        next_token_map = (
            sos.unsqueeze(1)
            + switti.pos_start.expand(2 * B, switti.first_l, -1)
            + lvl_pos[:, : switti.first_l]
        )
        cur_L = 0
        f_hat = sos.new_zeros(B, switti.Cvae, switti.patch_nums[-1], switti.patch_nums[-1])

        # Pre-process control image and extract backbone features once
        ctrl_image_t = self._preprocess_ctrl_image(
            ctrl_image, device=self.device, dtype=cond_vector.dtype
        )
        ctrl_feat = control_net.spatial_encoder.extract_features(ctrl_image_t)  # (B, 128, h, w)
        modality_id = MODALITY_IDS.get(modality, 0)
        modality_ids = torch.full((B,), modality_id, dtype=torch.long, device=self.device)
        mod_embed = control_net.spatial_encoder.modality_embed(modality_ids)  # (B, 128)

        with self._kv_caching(enable_ar=switti.use_ar):
            for si, pn in enumerate(switti.patch_nums):
                ratio = si / switti.num_stages_minus_1
                x_BLC = next_token_map

                if switti.rope:
                    freqs_cis = switti.freqs_cis[:, cur_L : cur_L + pn * pn]
                else:
                    freqs_cis = switti.freqs_cis

                if si >= turn_off_cfg_start_si:
                    apply_smooth = False
                    x_BLC = x_BLC[:B]
                    context = context[:B]
                    context_attn_bias = context_attn_bias[:B]
                    freqs_cis = freqs_cis[:B]
                    cond_BD = cond_BD[:B]
                    if crop_cond is not None:
                        crop_cond = crop_cond[:B]
                    # Trim KV caches in frozen blocks
                    for b in switti.blocks:
                        if b.attn.caching and b.attn.cached_k is not None:
                            b.attn.cached_k = b.attn.cached_k[:B]
                            b.attn.cached_v = b.attn.cached_v[:B]
                        if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                            b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                            b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
                    # Trim KV caches in control blocks
                    for b in control_net.control_blocks:
                        if b.attn.caching and b.attn.cached_k is not None:
                            b.attn.cached_k = b.attn.cached_k[:B]
                            b.attn.cached_v = b.attn.cached_v[:B]
                        if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                            b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                            b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
                else:
                    apply_smooth = more_smooth

                cur_B = x_BLC.shape[0]

                # --- Build scale-wise ctrl tokens ---
                f = F.adaptive_avg_pool2d(ctrl_feat, (pn, pn))           # (B, 128, pn, pn)
                f = f + mod_embed[:, :, None, None]
                ctrl_tokens_s = control_net.spatial_encoder.proj(
                    f.permute(0, 2, 3, 1).reshape(B, pn * pn, 128)
                )  # (B, pn², C)

                if cur_B == 2 * B:
                    # Null branch uses the learned null modality embedding (last
                    # row of the table), matching control dropout during training.
                    null_idx = control_net.spatial_encoder.modality_embed.num_embeddings - 1
                    null_ids = torch.full(
                        (B,), null_idx, dtype=torch.long, device=self.device
                    )
                    null_mod_embed = control_net.spatial_encoder.modality_embed(null_ids)
                    f_null = F.adaptive_avg_pool2d(ctrl_feat, (pn, pn))
                    f_null = f_null + null_mod_embed[:, :, None, None]
                    ctrl_tokens_null = control_net.spatial_encoder.proj(
                        f_null.permute(0, 2, 3, 1).reshape(B, pn * pn, 128)
                    )
                    ctrl_tokens_full = torch.cat(
                        [ctrl_tokens_s, ctrl_tokens_null], dim=0
                    )  # (2B, pn², C)
                else:
                    ctrl_tokens_full = ctrl_tokens_s  # (B, pn², C)

                # --- Parallel block loop ---
                x_frozen = x_BLC
                x_ctrl = x_BLC + ctrl_tokens_full

                for frz_blk, ctrl_blk, zero_conv in zip(
                    switti.blocks,
                    control_net.control_blocks,
                    control_net.zero_convs,
                ):
                    x_ctrl = ctrl_blk(
                        x=x_ctrl,
                        cond_BD=cond_BD,
                        attn_bias=None,
                        context=context,
                        context_attn_bias=context_attn_bias,
                        freqs_cis=freqs_cis,
                        crop_cond=crop_cond,
                    )
                    ctrl_signal = zero_conv(x_ctrl)

                    x_frozen = control_net._frozen_block_forward(
                        frz_blk=frz_blk,
                        x=x_frozen,
                        ctrl_signal=ctrl_signal,
                        cond_BD=cond_BD,
                        attn_bias=None,
                        prompt_embeds=context,
                        prompt_attn_bias=context_attn_bias,
                        freqs_cis=freqs_cis,
                        crop_cond=crop_cond,
                        ctrl_strength=ctrl_strength,
                    )

                cur_L += pn * pn
                logits_BlV = switti.get_logits(x_frozen, cond_BD)

                # Classifier-free guidance
                if si < turn_on_cfg_start_si:
                    logits_BlV = logits_BlV[:B]
                elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
                    t = cfg * ratio
                    # Upcast to float32: fp16 logits can overflow (max 65504),
                    # and 0 * Inf = NaN when t=0 at si=0.
                    orig_dtype = logits_BlV.dtype
                    cond_f32 = logits_BlV[:B].float()
                    uncond_f32 = logits_BlV[B:].float()
                    logits_BlV = ((1 + t) * cond_f32 - t * uncond_f32).to(orig_dtype)
                elif last_scale_temp is not None:
                    logits_BlV = logits_BlV / last_scale_temp

                if apply_smooth and si >= smooth_start_si:
                    gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                    idx_Bl = gumbel_softmax_with_rng(
                        logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                    )
                    h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
                else:
                    idx_Bl = sample_with_top_k_top_p_(
                        logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
                    )[:, :, 0]
                    h_BChw = vae_quant.embedding(idx_Bl)

                h_BChw = h_BChw.transpose_(1, 2).reshape(B, switti.Cvae, pn, pn)
                f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                    si, len(switti.patch_nums), f_hat, h_BChw
                )
                if si != switti.num_stages_minus_1:
                    next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
                    next_token_map = (
                        switti.word_embed(next_token_map)
                        + lvl_pos[:, cur_L : cur_L + switti.patch_nums[si + 1] ** 2]
                    )
                    # Double batch for CFG
                    next_token_map = next_token_map.repeat(2, 1, 1)

        img = vae.fhat_to_img(f_hat).add(1).mul(0.5)
        if return_pil:
            img = self.to_image(img)
        return img
