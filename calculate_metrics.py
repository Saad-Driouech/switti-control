import os

import ImageReward
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from transformers import AutoModel, AutoProcessor

from PIL import Image
from tqdm.auto import tqdm
import dist

from utils.control_metrics import calculate_control_metrics


@torch.no_grad()
def calc_pick_or_clip_scores(model, image_inputs, text_inputs, batch_size=50):
    assert len(image_inputs) == len(text_inputs["input_ids"])
    assert len(text_inputs.keys()) == 2

    scores = torch.zeros(len(image_inputs))
    for i in range(0, len(image_inputs), batch_size):
        image_batch = image_inputs[i : i + batch_size]
        text_batch = {
            "input_ids": text_inputs["input_ids"][i : i + batch_size],
            "attention_mask": text_inputs["attention_mask"][i : i + batch_size],
        }
        # embed
        with torch.amp.autocast('cuda'):
            image_embs = model.get_image_features(image_batch)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        with torch.amp.autocast('cuda'):
            text_embs = model.get_text_features(**text_batch)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        # score
        scores[i : i + batch_size] = (text_embs * image_embs).sum(-1)
    return scores.cpu()


@torch.no_grad()
def calculate_image_reward_score(
    images,
    prompts,
    device="cuda",
    batch_size=50,
    image_reward_path="ImageReward-v1.0",
):
    model = ImageReward.load(image_reward_path, device=device).eval()

    scores = []
    for i in range(0, len(prompts), batch_size):
        # text encode
        with torch.amp.autocast("cuda"):
            text_input = model.blip.tokenizer(
                prompts[i: i + batch_size],
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(device)

            processed_images = torch.stack(
                [
                    model.preprocess(image).to(device)
                    for image in images[i: i + batch_size]
                ]
            )
            image_embeds = model.blip.visual_encoder(processed_images)

            # text encode cross attention with image
            image_atts = torch.ones(
                image_embeds.size()[:-1], dtype=torch.long
            ).to(device)
            text_output = model.blip.text_encoder(
                text_input.input_ids,
                attention_mask=text_input.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        txt_features = text_output.last_hidden_state[:, 0].float()  # (feature_dim)
        rewards = model.mlp(txt_features)
        rewards = (rewards - model.mean) / model.std

        scores.extend(rewards[:, 0].tolist())

    return np.mean(scores)


@torch.no_grad()
def calculate_scores(
    images,
    prompts,
    device="cuda",
    clip_model_name_or_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    pickscore_model_name_or_path="yuvalkirstain/PickScore_v1",
    image_reward_path=None,
):
    processor = AutoProcessor.from_pretrained(clip_model_name_or_path)
    clip_model = AutoModel.from_pretrained(clip_model_name_or_path).eval().to(device)
    pickscore_model = (
        AutoModel.from_pretrained(pickscore_model_name_or_path).eval().to(device)
    )

    image_inputs = processor(
        images=images,
        return_tensors="pt",
    )[
        "pixel_values"
    ].to(device)

    text_inputs = processor(
        text=prompts,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    print("Evaluating PickScore...")
    pick_score = calc_pick_or_clip_scores(
        pickscore_model, image_inputs, text_inputs
    ).mean()

    print("Evaluating CLIP ViT-H-14 score...")
    clip_score = calc_pick_or_clip_scores(
        clip_model, image_inputs, text_inputs
    ).mean()

    # Free CLIP + PickScore before loading ImageReward to avoid OOM on 32 GB GPUs
    del clip_model, pickscore_model, processor, image_inputs, text_inputs
    import gc; gc.collect()
    torch.cuda.empty_cache()

    print("Evaluating ImageReward...")
    image_reward = calculate_image_reward_score(
        images,
        prompts,
        device,
        image_reward_path=image_reward_path,
    )
    image_reward = torch.full_like(clip_score, image_reward)

    return pick_score, clip_score, image_reward


@torch.no_grad()
def distributed_metrics_with_csv(
    pipe,
    csv_path,
    args,
    control_path: str = None,
    val_subset: str = "",
    control_modality: str = None,
    preview_save_dir: str = None,
):
    """
    Args:
        control_path:     root of ctrl map dirs (e.g. $LOCAL_DATA/ctrl_maps).
                          When set, generates in control mode using
                          <control_path>/<control_modality>/<val_subset>/<filename>.
        val_subset:       subdirectory for val images (default 'val2014').
        control_modality: modality string, e.g. 'canny' (required when control_path set).
    Returns:
        (local_images, local_pick_score, local_clip_score,
         local_image_reward, local_control_metric_tensors)
    """
    pipe.switti.eval()
    max_count = args.metrics_max_count
    rank_caption_batches, rank_filename_batches = prepare_prompts(csv_path, args.eval_batch_size, max_count)
    assert max_count % (args.eval_batch_size * dist.get_world_size()) == 0

    ctrl_transform = transforms.Compose([
        transforms.Resize((args.data_load_reso, args.data_load_reso)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]) if control_path is not None else None

    local_images, local_prompts = [], []
    all_ctrl_tensors = []

    for captions_batch, filenames_batch in tqdm(
        zip(rank_caption_batches, rank_filename_batches),
        unit="batch",
        disable=(dist.get_rank() != 0),
    ):
        texts = [str(caption) for caption in captions_batch
                 for _ in range(args.num_images_for_metrics)]

        ctrl_tensor_batch = None
        if control_path is not None and control_modality is not None:
            raw_ctrl = []
            for fname in filenames_batch:
                fname = str(fname)
                fname_png = fname.replace(".jpg", ".png")
                if fname in ("None", "nan", ""):
                    t = torch.zeros(3, args.data_load_reso, args.data_load_reso)
                else:
                    ctrl_fp = os.path.join(control_path, control_modality, val_subset, fname_png)
                    if os.path.exists(ctrl_fp):
                        try:
                            t = ctrl_transform(Image.open(ctrl_fp).convert("RGB"))
                        except Exception as e:
                            print(f"[Warning] {ctrl_fp}: {e}")
                            t = torch.zeros(3, args.data_load_reso, args.data_load_reso)
                    else:
                        t = torch.zeros(3, args.data_load_reso, args.data_load_reso)
                for _ in range(args.num_images_for_metrics):
                    raw_ctrl.append(t)
            ctrl_tensor_batch = torch.stack(raw_ctrl, dim=0)
            all_ctrl_tensors.extend(raw_ctrl)

        if ctrl_tensor_batch is not None:
            image_tensors = pipe(
                prompt=texts,
                ctrl_image=ctrl_tensor_batch,
                modality=control_modality,
                seed=args.seed,
                cfg=args.guidance,
                top_k=args.top_k,
                top_p=args.top_p,
                more_smooth=False,
                return_pil=False,
            )
        else:
            image_tensors = pipe(
                prompt=texts,
                seed=args.seed,
                cfg=args.guidance,
                top_k=args.top_k,
                top_p=args.top_p,
                more_smooth=False,
                return_pil=False,
            )

        local_images.extend(image_tensors)
        local_prompts.extend(texts)

        if preview_save_dir is not None and dist.is_master():
            os.makedirs(preview_save_dir, exist_ok=True)
            offset = len(local_images) - len(image_tensors)
            for j, t in enumerate(image_tensors):
                to_PIL_image(t).save(
                    os.path.join(preview_save_dir, f"{offset + j:04d}.jpg"), quality=90
                )

    local_images = torch.stack(local_images).cuda()
    pil_images = [to_PIL_image(image) for image in local_images.clone()]

    local_pick_score, local_clip_score, local_image_reward = calculate_scores(
        pil_images,
        local_prompts,
        device=dist.get_device(),
        clip_model_name_or_path=args.clip_model_name_or_path,
        pickscore_model_name_or_path=args.pickscore_model_name_or_path,
        image_reward_path=args.image_reward_path,
    )

    # Control-specific metrics
    local_control_metric_tensors = {}
    if control_path is not None and control_modality is not None and all_ctrl_tensors:
        ctrl_metrics = calculate_control_metrics(
            pil_images,
            all_ctrl_tensors,
            control_modality,
            device=dist.get_device(),
        )
        local_control_metric_tensors = {
            k: torch.tensor(v).cuda() for k, v in ctrl_metrics.items()
        }

    if dist.initialized():
        dist.barrier()
    return local_images, local_pick_score, local_clip_score, local_image_reward, local_control_metric_tensors


def save_images(images, prompts, save_path):
    for i, image in enumerate(images):
        image.save(os.path.join(save_path, f"{i:04d}.jpg"))
    if prompts:
        with open(os.path.join(save_path, "prompts.txt"), "w") as f:
            f.writelines("\n".join(prompts))


def prepare_prompts(prompts_path, batch_size=1, max_count=None):
    assert max_count % dist.get_world_size() == 0
    df = pd.read_csv(prompts_path)
    captions = df["captions"].astype(str).tolist()
    filenames = df["file_name"].astype(str).tolist() if "file_name" in df.columns else [None] * len(df)

    if max_count is not None:
        captions = captions[:max_count]
        filenames = filenames[:max_count]

    num_batches = (
        (len(captions) - 1) // (batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    caption_batches = np.array_split(np.array(captions), num_batches)
    filename_batches = np.array_split(np.array(filenames), num_batches)
    rank_caption_batches = caption_batches[dist.get_rank() :: dist.get_world_size()]
    rank_filename_batches = filename_batches[dist.get_rank() :: dist.get_world_size()]
    return rank_caption_batches, rank_filename_batches


def to_PIL_image(image_tensor):
    # [c, h, w] -> [h, w, c]
    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.tensor(image_tensor)
    img = (image_tensor.permute(1, 2, 0) * 255).cpu().numpy()
    return Image.fromarray(img.astype(np.uint8))
