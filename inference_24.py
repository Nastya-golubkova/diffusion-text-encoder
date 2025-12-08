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
from train_lpips_only import TrainCoefs

"""
Inference model with 24 coefficients from checkpoint
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


text_encoder = pipe.text_encoder
final_layer_norm = text_encoder.encoder.final_layer_norm
    
del text_encoder
gc.collect()
torch.cuda.empty_cache()

def get_embed_with_coef(hidden_states, model):
    hidden_states = hidden_states.to(dtype=torch.float32)
    hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
    hidden_states = hidden_states.to(dtype=torch.float16)
    weighted = hidden_states * torch.nn.functional.softmax(model.coefficients, dim=-1).to(hidden_states.dtype).view(1, 24, 1, 1)
    weighted_sum = weighted.sum(dim=1)
    return weighted_sum

def generate_image_from_emb(pipe, pos_emb, neg_emb):
    device = torch.device("cuda")
    pos_emb = pos_emb.to(device, dtype=torch.float32)
    neg_emb = neg_emb.to(device, dtype=torch.float32)
    images = pipe(prompt_embeds=pos_emb, negative_prompt_embeds=neg_emb).images
    return images

def generate(outputs, neg_embedding, model, pipe):
  outputs = outputs.to(dtype=torch.float32)
  pos_embedding = get_embed_with_coef(outputs, model)
  neg_embedding = get_embed_with_coef(neg_embedding, model)
  pos_embedding = final_layer_norm(pos_embedding)
  neg_embedding = final_layer_norm(neg_embedding)
  images = generate_image_from_emb(pipe, pos_embedding, neg_embedding)

  return images


model = TrainCoefs(pipe, final_layer_norm)
device=torch.device("cuda")
model.to(device)

best_model_name = "best_model_.pt"
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
        images = generate(promt_batch, neg_embed, model, pipe)

        for num_image, image in enumerate(images):
            image_filename = f"{batch_start_index + 100 + num_image :05d}.jpg"
            image.save(os.path.join(save_dir, image_filename))

            
        
