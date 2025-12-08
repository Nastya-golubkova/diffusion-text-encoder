from typing import Literal
import torch
import os
from transformers import T5Tokenizer
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel
import hpsv2

model_name = "DeepFloyd/IF-I-M-v1.0"

pipe = DiffusionPipeline.from_pretrained(
    model_name,
    unet=None,
    dtype=torch.float16,
    variant="fp16",
    device_map="balanced",
)
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer
last_block = text_encoder.encoder.final_layer_norm

def get_outputs_for_prompt(prompt):
    prompt= [el.lower().strip() for el in prompt]
    tokenized_prompt = tokenizer(
        prompt,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    device0 = text_encoder.device

    input_ids = tokenized_prompt.input_ids.to(device0)
    attention_mask = tokenized_prompt.attention_mask.to(device0)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )

    return outputs

def save_neg_emb():
    prompt=""
    tokenized_prompt = tokenizer(
        prompt,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt"
    )
    device0 = text_encoder.device

    input_ids = tokenized_prompt.input_ids.to(device0)
    attention_mask = tokenized_prompt.attention_mask.to(device0)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True
    )

    for i, output in enumerate(outputs.hidden_states[:25]):
        path = os.path.join("NEG_val", f"{i}.pt")
        os.makedirs(os.path.join("NEG_val"), exist_ok=True)
        torch.save(output[0], path)
    return outputs


def save_embeddings(all_prompts):
    for style, prompts in all_prompts.items():
        
        save_dir = os.path.join("POS_val", style)
        os.makedirs(save_dir, exist_ok=True)

        for idx, prompt in enumerate(prompts):
            idx += 100
            outputs = get_outputs_for_prompt(prompt=[prompt])
            hidden_states = outputs.hidden_states[:25]
            
            sample_dir = os.path.join(save_dir, f"{idx:05d}")
            os.makedirs(sample_dir, exist_ok=True)

            for layer_idx, layer_tensor in enumerate(hidden_states):
                tensor_to_save = layer_tensor[0].detach()
                path = os.path.join(sample_dir, f"{layer_idx}.pt")
                torch.save(tensor_to_save, path)

if __name__=="__main__":
    all_prompts = hpsv2.benchmark_prompts('all')
    all_prompts = { k : v[100:] for k, v in all_prompts.items() }
    save_neg_emb()
    batch_size=1
    save_embeddings(all_prompts)
