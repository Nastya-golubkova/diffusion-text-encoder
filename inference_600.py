import torch
from torch import nn
import os
import torch
import numpy as np
from pathlib import Path
from diffusers import DiffusionPipeline
from typing import Literal
import hpsv2
import matplotlib_inline
import matplotlib.pyplot as plt
from PIL import Image
from transformers import T5EncoderModel, T5Tokenizer
from torch.utils.data import Dataset, DataLoader, random_split
import lpips
import torchvision.transforms as transforms
import gc
import torchvision
from tqdm.auto import tqdm
import neptune
from diffusers.utils.torch_utils import randn_tensor
import inspect
from diffusers.pipelines.deepfloyd_if import IFPipeline
from diffusers import DiffusionPipeline
from typing import Union, Optional, List, Dict, Callable
from diffusers.pipelines.deepfloyd_if import IFPipelineOutput
from train_600 import TrainCoefs

"""
Inference model with 600 coefficients from checkpoint
"""

size = "M"

pipe = DiffusionPipeline.from_pretrained(
    f"DeepFloyd/IF-I-{size}-v1.0",
    watermarker=None,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
    dtype=torch.float16,
    variant="fp16",
    device_map="balanced",
)
scheduler = pipe.scheduler
unet = pipe.unet
text_encoder = pipe.text_encoder
final_layer_norm = text_encoder.encoder.final_layer_norm
generator = torch.Generator().manual_seed(0)
del text_encoder
gc.collect()
torch.cuda.empty_cache()

model = TrainCoefs(pipe, final_layer_norm)
device=torch.device("cuda")
model.to(device)

best_model_name = "best_model_name.pt"
path_to_images = "eval_exp_"

model_dict = torch.load(best_model_name, map_location="cuda")
model.load_state_dict(model_dict)

model.eval()
batch_size = 32
path_to_pos_emb = "POS_val"

neg_embed = []
for i in range(1, 25):
    path_to_prompt = os.path.join("NEG_val", f"{i}")
    embed = torch.load(f"{path_to_prompt}.pt").float()
    neg_embed.append(embed)

neg_embed = torch.stack(neg_embed, dim=0)
max_val = neg_embed.abs().max()
scale = 0.99 * 65500 / (max_val + 1e-6)
neg_embed = (neg_embed * scale).to(device, dtype=torch.float16)
neg_embed = neg_embed.repeat(batch_size, 1, 1, 1)


@torch.no_grad
def get_embed_with_coef(hidden_states, coefficients):
    hidden_states = hidden_states.to(dtype=torch.float32)
    hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
    hidden_states = hidden_states.to(dtype=torch.float16)
    
    weighted = hidden_states * torch.nn.functional.softmax(coefficients, dim=-1).to(hidden_states.dtype).view(1, 24, 1, 1)
    weighted_sum = weighted.sum(dim=1)
    
    return weighted_sum

@torch.no_grad
def prepare_intermediate_images(batch_size, num_channels, height, width, dtype, generator):
    generator = generator.manual_seed(0)
    intermediate_images = randn_tensor(
        (batch_size, num_channels, height, width),
        generator=generator,
        dtype=dtype
    )
    intermediate_images = intermediate_images * scheduler.init_noise_sigma
    return intermediate_images

@torch.no_grad
def prepare_extra_step_kwargs(generator, eta):
    extra_step_kwargs = {}
    if "eta" in set(inspect.signature(scheduler.step).parameters.keys()):
        extra_step_kwargs["eta"] = eta
    if "generator" in set(inspect.signature(scheduler.step).parameters.keys()):
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

@torch.no_grad
def custom_pipe(
        prompt_embeds, 
        negative_prompt_embeds,
        guidance_scale=7.0,
        num_inference_steps=25,
        num_images_per_prompt=1,
        eta=0.0,
        cross_attention_kwargs=None,
):
    negative_prompt_embeds = negative_prompt_embeds.to(unet.device)
    batch_size = prompt_embeds.shape[0]
    do_classifier_free_guidance = guidance_scale > 1.0
    
    scheduler.set_timesteps(num_inference_steps, device=prompt_embeds.device)
    timesteps = scheduler.timesteps
    if hasattr(scheduler, "set_begin_index"):
        scheduler.set_begin_index(0)

    height = unet.config.sample_size
    width = unet.config.sample_size

    intermediate_images = prepare_intermediate_images(
        batch_size * num_images_per_prompt,
        unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        generator
    )

    extra_step_kwargs = prepare_extra_step_kwargs(generator, eta)
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
    outputs = prompt_embeds
    neg_outputs = negative_prompt_embeds
    for i, t in enumerate(timesteps):

        prompt_embeds = get_embed_with_coef(outputs, model.coefficients[i])
        prompt_embeds = final_layer_norm(prompt_embeds)

        negative_prompt_embeds = get_embed_with_coef(neg_outputs, model.coefficients[i])
        negative_prompt_embeds = final_layer_norm(negative_prompt_embeds)
        
        
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        model_input = (
            torch.cat([intermediate_images] * 2) if do_classifier_free_guidance else intermediate_images
        )

        model_input = scheduler.scale_model_input(model_input, t)

        device = torch.device("cuda")
        model_input = model_input.to(device=device, dtype=torch.float16)
        prompt_embeds = prompt_embeds.to(device=device, dtype=torch.float16)
        noise_pred = unet(
            model_input,
            t.to(device=device, dtype=torch.float16),
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]
        

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_uncond, _ = noise_pred_uncond.split(model_input.shape[1], dim=1)
            noise_pred_text, predicted_variance = noise_pred_text.split(model_input.shape[1], dim=1)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        if scheduler.config.variance_type not in ["learned", "learned_range"]:
            noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)
        t = t.to(noise_pred.device)
        intermediate_images = intermediate_images.to(noise_pred.device)
        
        intermediate_images = scheduler.step(
            noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
        )[0]
    
    image = intermediate_images
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    image = [Image.fromarray(i) for i in image]


    return image


def generate_image_from_emb(pos_emb, neg_emb):
        
    device = torch.device("cuda")
    pos_emb = pos_emb.to(device, dtype=torch.float32)
    neg_emb = neg_emb.to(device, dtype=torch.float32)
    
    images = custom_pipe(
        prompt_embeds=pos_emb, 
        negative_prompt_embeds=neg_emb, 
        num_inference_steps=25
    )
    return images


def generate(outputs, neg_embeddings):
    outputs = outputs.to(dtype=torch.float16)
    pos_embedding = outputs
    images = generate_image_from_emb(pos_embedding, neg_embeddings)

    return images



for style in ["anime", "concept-art", "paintings", "photo"]:
    save_dir = os.path.join(path_to_images, style)
    os.makedirs(save_dir, exist_ok=True)
    for batch_start_index in range(0, len(os.listdir(os.path.join(path_to_pos_emb, style))), batch_size):
        promt_batch = []
        if f"{batch_start_index + 100:05d}.jpg" in os.listdir(os.path.join(path_to_images, style)):
            continue
        
        
        if batch_start_index + 100 + batch_size - 1 >= 800:
            batch_size = len(os.listdir(os.path.join(path_to_pos_emb, style))) - batch_start_index
            neg_embed = neg_embed[:batch_size, :, :, :]

        for batch_iter in range(batch_size):
            
            filename = f"{batch_start_index + 100 + batch_iter :05d}"
            one_prompt_embed = []
            for i in range(1, 25):
                path_to_prompt = os.path.join(path_to_pos_emb, style, filename, f"{i}")
                embed = torch.load(f"{path_to_prompt}.pt").float()
                one_prompt_embed.append(embed)
            
            one_prompt_embed = torch.stack(one_prompt_embed, dim=0)
            promt_batch.append(one_prompt_embed)
        
        promt_batch = torch.stack(promt_batch, dim=0)
        max_val = promt_batch.abs().max()
        scale = 0.99 * 65500 / (max_val + 1e-6)
        promt_batch = (promt_batch * scale).to(device, dtype=torch.float16)
        images = generate(promt_batch, neg_embed)

        for num_image, image in enumerate(images):
            image_filename = f"{batch_start_index + 100 + num_image :05d}.jpg"
            image.save(os.path.join(save_dir, image_filename))



            
        
