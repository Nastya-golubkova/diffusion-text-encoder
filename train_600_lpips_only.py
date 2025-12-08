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
import clip
from tqdm.auto import tqdm
import neptune
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.deepfloyd_if import IFPipeline
from diffusers import DiffusionPipeline
from typing import Union, Optional, List, Dict, Callable
from diffusers.pipelines.deepfloyd_if import IFPipelineOutput


class NormalizeTensor:
    def __call__(self, t):
        return t * 2 - 1
        
class CustomDataset(Dataset):
    """
    Dataset with columns 0-23 which are embeddings from blocks 0-23 of T5,
    column 'target', which is an image of XL model
    and column 'prompt' which is a prompt.
    """
    def __init__(self, start_num_samples=0, end_num_samples=100 , root_dir="POS_train", target_dir="deepFloyd/setup_1_XL"):
        
        self.root_dir = Path(root_dir)
        self.target_dir = Path(target_dir)
        self.styles = ["anime", "concept-art", "paintings", "photo"]
        self.num_samples = end_num_samples - start_num_samples
        self.num_columns = 24
        self.start_num_samples = start_num_samples
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            NormalizeTensor()
        ])
        self.samples = []
        for style in self.styles:
            for sample_idx in range(self.num_samples):
                self.samples.append({
                    'style': style,
                    'sample_idx': start_num_samples + sample_idx
                })
        all_prompts = hpsv2.benchmark_prompts('all')
        self.prompts = {key: value[start_num_samples : end_num_samples] for key, value in all_prompts.items()}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        style = sample_info['style']
        sample_idx = sample_info['sample_idx']
        input_tensors = []
        for j in range(1, self.num_columns + 1):
            file_path = os.path.join(self.root_dir, style, f"{sample_idx:05d}", f"{j}.pt")
            tensor = torch.load(file_path).float()
            tensor.requires_grad = False
            input_tensors.append(tensor)
        
        prompt = self.prompts[style][sample_idx - self.start_num_samples]
        target_path = os.path.join(self.target_dir, style, f"{sample_idx:05d}.jpg")
        image = Image.open(target_path).convert('RGB')
        target_tensor = self.transform(image)

        input_tensor = torch.stack(input_tensors, dim=0)
        return input_tensor, target_tensor, prompt
        
        
        prompt = self.prompts[style][sample_idx]
        target_path = os.path.join(self.target_dir, style, f"{sample_idx:05d}.jpg")
        image = Image.open(target_path).convert('RGB')
        target_tensor = self.transform(image)

        input_tensor = torch.stack(input_tensors, dim=0)
        return input_tensor, target_tensor, prompt


class TrainCoefs(nn.Module):
    """
    Learnable coefficients for embeddings. 
    All diffusion pipe is frozen. 
    """
    def __init__(self, pipe, final_layer_norm) -> None:
        super().__init__()

        self.unet = pipe.unet
        self.scheduler = pipe.scheduler
        self.progress_bar = pipe.progress_bar
        self.pipe = pipe
        self.num_inference_steps = 25

        for p in self.unet.parameters():
            p.data = p.data.half()
            p.requires_grad = False
        
        self.coefficients = nn.Parameter(torch.ones(size=(self.num_inference_steps, 24), dtype=torch.float32) * 1e-3)
        neg_embeddings_list = []
        for i in range(1, 25):
            neg = torch.load(f"NEG_train/{i}.pt").float()
            max_val = neg.abs().max()
            scale = 0.99 * 65500 / (max_val + 1e-6) 
            neg_scaled = (neg * scale).half()
            neg_embeddings_list.append(neg_scaled)

        self.neg_embeddings = nn.Parameter(
            torch.stack(neg_embeddings_list),
            requires_grad=False
        )

        self.generator = torch.Generator()
        self.final_layer_norm = final_layer_norm.half()

    def get_embed_with_coef(self, hidden_states, coefficients):
        hidden_states = hidden_states.to(dtype=torch.float32)
        hidden_states = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
        hidden_states = hidden_states.to(dtype=torch.float16)
        
        weighted = hidden_states * torch.nn.functional.softmax(coefficients, dim=-1).to(hidden_states.dtype).view(1, 24, 1, 1)
        weighted_sum = weighted.sum(dim=1)
        
        return weighted_sum

    def prepare_intermediate_images(self, batch_size, num_channels, height, width, dtype, generator):
        generator = generator.manual_seed(0)
        intermediate_images = randn_tensor(
            (batch_size, num_channels, height, width),
            generator=generator,
            dtype=dtype
        )
        intermediate_images = intermediate_images * self.scheduler.init_noise_sigma
        return intermediate_images

    def prepare_extra_step_kwargs(self, generator, eta):
        extra_step_kwargs = {}
        if "eta" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = eta
        if "generator" in set(inspect.signature(self.scheduler.step).parameters.keys()):
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def custom_pipe(
            self,
            prompt_embeds, 
            negative_prompt_embeds,
            guidance_scale=7.0,
            num_inference_steps=25,
            num_images_per_prompt=1,
            eta=0.0,
            cross_attention_kwargs=None,
    ):
        negative_prompt_embeds = negative_prompt_embeds.to(self.unet.device)
        batch_size = prompt_embeds.shape[0]
        do_classifier_free_guidance = guidance_scale > 1.0
        
        self.scheduler.set_timesteps(num_inference_steps, device=prompt_embeds.device)
        timesteps = self.scheduler.timesteps
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)

        height = self.unet.config.sample_size
        width = self.unet.config.sample_size

        intermediate_images = self.prepare_intermediate_images(
            batch_size * num_images_per_prompt,
            self.unet.config.in_channels,
            height,
            width,
            prompt_embeds.dtype,
            self.generator
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(self.generator, eta)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        outputs = prompt_embeds
        neg_outputs = negative_prompt_embeds
        for i, t in enumerate(timesteps):

            prompt_embeds = self.get_embed_with_coef(outputs, self.coefficients[i])
            prompt_embeds = self.final_layer_norm(prompt_embeds)

            negative_prompt_embeds = self.get_embed_with_coef(neg_outputs, self.coefficients[i])
            negative_prompt_embeds = self.final_layer_norm(negative_prompt_embeds)
            
            
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            model_input = (
                torch.cat([intermediate_images] * 2) if do_classifier_free_guidance else intermediate_images
            )

            model_input = self.scheduler.scale_model_input(model_input, t)
            device = torch.device("cuda")
            model_input = model_input.to(device=device, dtype=torch.float16)
            prompt_embeds = prompt_embeds.to(device=device, dtype=torch.float16)
            noise_pred = self.unet(
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

            if self.scheduler.config.variance_type not in ["learned", "learned_range"]:
                noise_pred, _ = noise_pred.split(model_input.shape[1], dim=1)
            t = t.to(noise_pred.device)
            intermediate_images = intermediate_images.to(noise_pred.device)
            
            intermediate_images = self.scheduler.step(
                noise_pred, t, intermediate_images, **extra_step_kwargs, return_dict=False
            )[0]

        return intermediate_images

   
    def generate_image_from_emb(self, pos_emb, neg_emb):
        
        device = torch.device("cuda")
        pos_emb = pos_emb.to(device, dtype=torch.float32)
        neg_emb = neg_emb.to(device, dtype=torch.float32)
       
        images = self.custom_pipe(
            prompt_embeds=pos_emb, 
            negative_prompt_embeds=neg_emb, 
            num_inference_steps=self.num_inference_steps
        )
        return images

    def forward(self, outputs):
        
        outputs = outputs.to(dtype=torch.float16)
        pos_embedding = outputs
        neg_embedding = self.neg_embeddings.unsqueeze(0).expand(pos_embedding.shape[0], -1, -1, -1)
        images = self.generate_image_from_emb(pos_embedding, neg_embedding)

        return images

def normalize(im):
    im = im.permute(1, 2, 0)
    im = im.float()
    im_min, im_max = im.min(), im.max()
    im = (im - im_min) / (im_max - im_min)
    return im

def plot_for_two(output_image, target_image, prompt):
    output_image = normalize(output_image)
    target_image = normalize(target_image)
    fig, ax = plt.subplots(1, 2, figsize=(6, 6))
    
    fig.suptitle(prompt, fontsize=12, y=0.95)
    ax[0].imshow(output_image)
    ax[0].set_title('output_image')
    ax[0].axis('off')

    ax[1].imshow(target_image)
    ax[1].set_title('target_image')
    ax[1].axis('off')
    
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.close(fig)
    return fig

def log_coefficients_hist(run, model, steps=[0, 9, 24]):
    coeff_softmax = torch.nn.functional.softmax(model.coefficients, dim=-1)
    coeff_np = coeff_softmax.detach().cpu().numpy()

    for step_idx in steps:
        values = coeff_np[step_idx]
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(range(len(values)), values)
        ax.set_xlabel("coefficient")
        ax.set_ylabel("after softmax")
        ax.set_title(f"step {step_idx + 1}")
        ax.set_ylim(0, max(values) + 0.03)
        plt.tight_layout()

        run[f"coefficients/step_{step_idx + 1}"].append(fig)
        plt.close(fig)


if __name__=="__main__":
    
    train_dataset = CustomDataset(start_num_samples=0, end_num_samples=80)
    val_dataset = CustomDataset(start_num_samples=80, end_num_samples=100)
    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

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

    model = TrainCoefs(pipe, final_layer_norm)
    device = torch.device("cuda")
    model.to(device)
    EPOCH = 50
    accumulation_step = 16
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    lr = 1e-2
    optimizer = torch.optim.AdamW([model.coefficients], lr=lr)
   
    with_clipping_grad = True
    patience = EPOCH
    counter = 0
    total_steps = EPOCH * (len(train_loader) // accumulation_step)
    
    run = neptune.init_run(
        project=os.environ.get("PROJECT_NAME")
        api_token=os.environ.get("API_TOKEN")
    )
    best_model_filename = "best_model_600.pt"

    log_coefficients_hist(run=run, model=model)
    run["description"] = f"25 шагов, 25 * 24 коэффициентов, lpips"
    
    params = {
        "learning_rate": lr,
        "accumulation_step": accumulation_step,
        "optimizer": "AdamW",
        "loss": "lpips",
        "best_model_filename": best_model_filename,
        "initial_values": "\n".join(map(str, model.coefficients.tolist())),
        "with_clipping_grad": with_clipping_grad,
    }

    best_val_metric = 0.0
    run["model/parameters"] = params
    for epoch in range(epoch_start, EPOCH):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        loss_for_log = 0.0
        metric_hpsv2, target_metric_hpsv2 = [], []
        for idx, (x, y, prompt) in enumerate(tqdm(train_loader, desc=f"train, epoch {epoch}")):
            x = x.to(device)
            
            # scale чтобы перевести в float16
            max_val = x.abs().max()
            scale = 0.99 * 65500 / (max_val + 1e-6)
            x = (x * scale).to(device, dtype=torch.float16)
            y = y.to(device, dtype=torch.float16)
            output = model(x)
            os.makedirs("train_loss", exist_ok=True)

            # Сохранение картинок
            if idx % 100 == 0:
                to_pil = transforms.ToPILImage()
                output_img = ((output[0].detach().cpu() + 1) / 2).clamp(0,1)
                y_img = ((y[0].detach().cpu() + 1) / 2).clamp(0,1)
                run["pictures/pred"].append(plot_for_two(output_img, y_img, prompt[0]))
               
            batch_loss = loss_fn_vgg(output.float(), y.float()).mean()
            normalized_loss = batch_loss / accumulation_step
            normalized_loss.backward()
            
            epoch_loss += batch_loss.item()
            loss_for_log += batch_loss
            
            pred = ((output[0] + 1) / 2)
            target = ((y[0] + 1) / 2)
            pred = to_pil(pred)
            target = to_pil(target)

            for name, param in model.named_parameters():
                if name=="coefficients" and param.grad is None:
                    print(f"no gradient {name}")
            
            if (idx + 1) % accumulation_step == 0:
                run["train/loss"].append(loss_for_log / accumulation_step)
                loss_for_log = 0.0
                if with_clipping_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                log_coefficients_hist(run=run, model=model)
        
        if len(train_loader) % accumulation_step != 0:
            if with_clipping_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}, loss {avg_epoch_loss}")
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            weight_norm = 0.0
            grad_norm = 0.0
            for param in model.parameters():
                if param.requires_grad:
                    weight_norm += param.detach().norm(2).item() ** 2
                    if param.grad is not None:
                        grad_norm += param.grad.detach().norm(2).item() ** 2

            weight_norm = weight_norm ** 0.5
            grad_norm = grad_norm ** 0.5

            run["train/norm_weights"].append(weight_norm)
            run["train/norm_gradients"].append(grad_norm)
            metric_hpsv2 = []
            target_metric_hpsv2 = []
            for idx, (x, y, prompt) in enumerate(tqdm(val_loader, desc="val")):
                x = x.to(device)
                max_val = x.abs().max()
                scale = 0.99 * 65500 / (max_val + 1e-6)
                x = (x * scale).to(device, dtype=torch.float16)
                y = y.to(device, dtype=torch.float16)

                output = model(x)
                os.makedirs("val_loss", exist_ok=True)

                # Сохранение
                if idx == len(val_loader) - 1:
                    to_pil = transforms.ToPILImage()
                    output_img = ((output[0].detach().cpu() + 1) / 2).clamp(0,1)
                    y_img = ((y[0].detach().cpu() + 1) / 2).clamp(0,1)
                    run["pictures/validation_0"].append(plot_for_two(output_img, y_img, prompt[0]))
                    output_img = ((output[1].detach().cpu() + 1) / 2).clamp(0,1)
                    y_img = ((y[1].detach().cpu() + 1) / 2).clamp(0,1)
                    run["pictures/validation_1"].append(plot_for_two(output_img, y_img, prompt[1]))
                
                
                for index_prompt in range(len(prompt)):
                    pred = ((output[index_prompt] + 1) / 2)
                    target = ((y[index_prompt] + 1) / 2)
                    pred = to_pil(pred)
                    target = to_pil(target)
                    metric_hpsv2.append(hpsv2.score(pred, prompt[index_prompt], hps_version="v2.1")[0])
                    target_metric_hpsv2.append(hpsv2.score(target, prompt[index_prompt], hps_version="v2.1")[0])
                
                lpips_loss = loss_fn_vgg(output.float(), y.float()).mean()
                val_loss += lpips_loss.item()

            run["metric/output"].append(sum(metric_hpsv2) / len(metric_hpsv2))
            run["metric/target"].append(sum(target_metric_hpsv2) / len(target_metric_hpsv2))
            metric_hpsv2 = []
            target_metric_hpsv2 = []

        avg_val_loss = val_loss / len(val_loader)
        run["val_avg_loss"].append(avg_val_loss)

        if sum(metric_hpsv2) / len(metric_hpsv2) >= best_val_metric:
            counter = 0
            best_val_metric = sum(metric_hpsv2) / len(metric_hpsv2)
            
            with open(best_model_filename, "wb") as f:
                torch.save(model.state_dict(), f)
                f.flush()
                os.fsync(f.fileno()) 
            
        else:
            counter += 1
            print(f"counter is {counter}")
        
        if counter >= patience:
            print("early stopping")
            break
