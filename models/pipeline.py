import torch
from torchvision.transforms import ToPILImage
from PIL.Image import Image as PILImage
import numpy as np
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from models.vqvae import VQVAEHF
from models.clip import FrozenCLIPEmbedder
from models.switti import SwittiHF, get_crop_condition
from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng


TRAIN_IMAGE_SIZE = (512, 512)

class SwittiPipeline:
    vae_path = "yresearch/VQVAE-Switti"
    text_encoder_path = "openai/clip-vit-large-patch14"
    text_encoder_2_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

    def __init__(self, switti, vae, text_encoder, text_encoder_2,
                 device, dtype=torch.float32,
                 ):
        self.switti = switti.to(dtype)
        self.vae = vae.to(dtype)
        self.text_encoder = text_encoder.to(dtype)
        self.text_encoder_2 = text_encoder_2.to(dtype)

        self.switti.eval()
        self.vae.eval()

        self.device = device

        param = next(self.switti.parameters()).to(self.device)
        self.model_dtype = param.dtype

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        torch_dtype=torch.bfloat16,
                        device="cuda",
                        reso=1024,
                        ):
        switti = SwittiHF.from_pretrained(pretrained_model_name_or_path).to(device)
        vae = VQVAEHF.from_pretrained(cls.vae_path, reso=reso).to(device)
        text_encoder = FrozenCLIPEmbedder(cls.text_encoder_path, device=device)
        text_encoder_2 = FrozenCLIPEmbedder(cls.text_encoder_2_path, device=device)

        return cls(switti, vae, text_encoder, text_encoder_2, device, torch_dtype)

    @staticmethod
    def to_image(tensor):
        return [ToPILImage()(
            (255 * img.cpu().detach()).to(torch.uint8))
        for img in tensor]

    def _encode_prompt(self, prompt: str | list[str]):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        encodings = [
            self.text_encoder.encode(prompt),
            self.text_encoder_2.encode(prompt),
        ]
        prompt_embeds = torch.concat(
            [encoding.last_hidden_state for encoding in encodings], dim=-1
        )
        pooled_prompt_embeds = encodings[-1].pooler_output
        attn_bias = encodings[-1].attn_bias

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    def encode_prompt(
        self,
        prompt: str | list[str],
        null_prompt: str = "",
        encode_null: bool = True,
    ):
        prompt_embeds, pooled_prompt_embeds, attn_bias = self._encode_prompt(prompt)
        if encode_null:
            B, L, hidden_dim = prompt_embeds.shape
            pooled_dim = pooled_prompt_embeds.shape[1]

            null_embeds, null_pooled_embeds, null_attn_bias = self._encode_prompt(null_prompt)
            
            null_embeds = null_embeds[:, :L].expand(B, L, hidden_dim).to(prompt_embeds.device)
            null_pooled_embeds = null_pooled_embeds.expand(B, pooled_dim).to(pooled_prompt_embeds.device)
            null_attn_bias = null_attn_bias[:, :L].expand(B, L).to(attn_bias.device)

            prompt_embeds = torch.cat([prompt_embeds, null_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, null_pooled_embeds], dim=0)
            attn_bias = torch.cat([attn_bias, null_attn_bias], dim=0)

        return prompt_embeds, pooled_prompt_embeds, attn_bias

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str | list[str],
        null_prompt: str = "",
        seed: int | None = None,
        cfg: float = 6.,
        top_k: int = 400,
        top_p: float = 0.95,
        more_smooth: bool = False,
        return_pil: bool = True,
        smooth_start_si: int = 0,
        turn_off_cfg_start_si: int = 10,
        turn_on_cfg_start_si: int = 0,
        last_scale_temp: None | float = None,
        control_dict: dict[str, torch.Tensor] | None = None,
        control_end_si: int = 8,
    ) -> torch.Tensor | list[PILImage]:
        """
        only used for inference, on autoregressive mode
        :param prompt: text prompt to generate an image
        :param null_prompt: negative prompt for CFG
        :param seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: sampling using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if return_pil: list of PIL Images, else: torch.tensor (B, 3, H, W) in [0, 1]
        """
        assert not self.switti.training
        switti = self.switti
        vae = self.vae
        vae_quant = self.vae.quantize
        if seed is None:
            rng = None
        else:
            switti.rng.manual_seed(seed)
            rng = switti.rng

        context, cond_vector, context_attn_bias = self.encode_prompt(prompt, null_prompt)

        B = context.shape[0] // 2

        # --------------------------------------------------
        # Control Image Encoding (multi-scale)
        # --------------------------------------------------
        control_contexts_per_scale = None  # {scale_num: {ctrl_type: (B, scale², D)}}

        if control_dict is not None and hasattr(self.switti, "control_encoder"):
            control_contexts_per_scale = {}

            for ctrl_type, ctrl_input in control_dict.items():
                if ctrl_input is None:
                    continue

                # ---- Handle list/batch of tensors ----
                if isinstance(ctrl_input, (list, tuple)):
                    ctrl_list = []
                    for item in ctrl_input:
                        if isinstance(item, torch.Tensor):
                            # Already preprocessed
                            if item.ndim == 3:
                                item = item.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
                        else:
                            raise TypeError(f"Expected preprocessed tensor, got {type(item)}")
                        ctrl_list.append(item)
                    ctrl_batch = torch.cat(ctrl_list, dim=0).to(self.device, dtype=self.model_dtype)
                
                # ---- Handle single tensor ----
                elif isinstance(ctrl_input, torch.Tensor):
                    ctrl_batch = ctrl_input
                    if ctrl_batch.ndim == 3:
                        ctrl_batch = ctrl_batch.unsqueeze(0)
                    ctrl_batch = ctrl_batch.to(self.device, dtype=self.model_dtype)
                
                else:
                    raise TypeError(f"Control '{ctrl_type}' must be Tensor or list[Tensor], got {type(ctrl_input)}")

                # ---- Validation ----
                if ctrl_input.ndim != 4:
                    raise ValueError(
                        f"Control '{ctrl_type}' must be 4D [B, C, H, W] after conversion, "
                        f"got shape {ctrl_input.shape}"
                    )

                # ---- HARD ASSERTION (critical safety check) ----
                assert len(ctrl_input) == B, (
                    f"Control '{ctrl_type}' batch size ({len(ctrl_input)}) "
                    f"does not match number of prompts ({B}). "
                    f"Got {len(ctrl_input)} control images for {B} prompts."
                )

                # Get multi-scale control features
                ctrl_ms = self.switti.control_encoder(ctrl_batch) # ctrl_ms = {1: (B,1,D), 2: (B,4,D), ..., 32: (B,1024,D)}

                # Store per-scale for use in generation loop
                for scale, tokens in ctrl_ms.items():
                    if scale not in control_contexts_per_scale:
                        control_contexts_per_scale[scale] = {}
                    
                    # ---- Build unconditional branch using learnable null ----
                    null_tokens = self.switti.null_control_tokens[str(scale)]
                    null_tokens = null_tokens.expand(B, -1, -1)
                    
                    # Stack: [conditional, unconditional]
                    tokens_cfg = torch.cat([tokens, null_tokens], dim=0)
                    
                    control_contexts_per_scale[scale][ctrl_type] = tokens_cfg

        cond_vector = switti.text_pooler(cond_vector)

        if switti.use_crop_cond:
            crop_coords = get_crop_condition(2 * B * [TRAIN_IMAGE_SIZE[0]],
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

        for b in switti.blocks:
            b.attn.kv_caching(switti.use_ar) # Use KV caching if switti is in the AR mode 
            b.cross_attn.kv_caching(True)
            if b.cross_attn_control is not None:
                b.cross_attn_control.kv_caching(switti.use_ar)

        for si, pn in enumerate(switti.patch_nums):  # si: i-th segment
            ratio = si / switti.num_stages_minus_1
            x_BLC = next_token_map

            if switti.rope:
                freqs_cis = switti.freqs_cis[:, cur_L : cur_L + pn * pn]
            else:
                freqs_cis = switti.freqs_cis

            # Get control tokens for THIS scale only
            scale_control_contexts = None
            if control_contexts_per_scale is not None and pn in control_contexts_per_scale and si < control_end_si:  # drop control after this scale
                scale_control_contexts = control_contexts_per_scale[pn]
                # scale_control_contexts = {ctrl_type: (2B, pn², D)} for this specific scale

            if scale_control_contexts is not None:
                for k, v in scale_control_contexts.items():
                    assert v.shape[1] == pn * pn, f"Scale {pn}: expected {pn*pn} tokens, got {v.shape[1]}"

            if si >= turn_off_cfg_start_si:
                apply_smooth = False
                x_BLC = x_BLC[:B]
                context = context[:B]
                context_attn_bias = context_attn_bias[:B]
                freqs_cis = freqs_cis[:B]
                cond_BD = cond_BD[:B]
                if crop_cond is not None:
                    crop_cond = crop_cond[:B]
                if scale_control_contexts is not None:
                    scale_control_contexts = {
                        k: v[:B] for k, v in scale_control_contexts.items()
                    }
                for b in switti.blocks:
                    if b.attn.caching and b.attn.cached_k is not None:
                        b.attn.cached_k = b.attn.cached_k[:B]
                        b.attn.cached_v = b.attn.cached_v[:B]
                    if b.cross_attn.caching and b.cross_attn.cached_k is not None:
                        b.cross_attn.cached_k = b.cross_attn.cached_k[:B]
                        b.cross_attn.cached_v = b.cross_attn.cached_v[:B]
                    if b.cross_attn_control is not None and b.cross_attn_control.caching and b.cross_attn_control.cached_k is not None:
                        b.cross_attn_control.cached_k = b.cross_attn_control.cached_k[:B]
                        b.cross_attn_control.cached_v = b.cross_attn_control.cached_v[:B]
            else:
                apply_smooth = more_smooth

            for block in switti.blocks:
                x_BLC = block(
                    x=x_BLC,
                    cond_BD=cond_BD,
                    attn_bias=None,
                    context=context,
                    context_attn_bias=context_attn_bias,
                    control_contexts=scale_control_contexts,
                    control_context_attn_biases=None,
                    freqs_cis=freqs_cis,
                    crop_cond=crop_cond,
                )
            cur_L += pn * pn

            logits_BlV = switti.get_logits(x_BLC, cond_BD)

            # Guidance
            if si < turn_on_cfg_start_si:
                logits_BlV = logits_BlV[:B]
            elif si >= turn_on_cfg_start_si and si < turn_off_cfg_start_si:
                t = cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            elif last_scale_temp is not None:
                logits_BlV = logits_BlV / last_scale_temp

            if apply_smooth and si >= smooth_start_si:
                # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                idx_Bl = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng,
                )
                h_BChw = idx_Bl @ vae_quant.embedding.weight.unsqueeze(0)
            else:
                # default nucleus sampling
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1,
                )[:, :, 0]
                h_BChw = vae_quant.embedding(idx_Bl)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, switti.Cvae, pn, pn)
            f_hat, next_token_map = vae_quant.get_next_autoregressive_input(
                    si, len(switti.patch_nums), f_hat, h_BChw,
            )
            if si != switti.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view(B, switti.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    switti.word_embed(next_token_map)
                    + lvl_pos[:, cur_L : cur_L + switti.patch_nums[si + 1] ** 2]
                )
                # double the batch sizes due to CFG
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in switti.blocks:
            b.attn.kv_caching(False)
            b.cross_attn.kv_caching(False)
            if b.cross_attn_control is not None:
                b.cross_attn_control.kv_caching(False)

        # de-normalize, from [-1, 1] to [0, 1]
        img = vae.fhat_to_img(f_hat).add(1).mul(0.5)
        if return_pil:
            img = self.to_image(img)

        return img
