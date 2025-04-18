from diffusers import DiffusionPipeline
import torch
import os
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel
import requests
from torch import cosine_similarity
import numpy as np

clip_i_scores=[]
dino_scores=[]
clip_t_scores=[]
#load models
def get_clip_embedding(image, clip_model, clip_processor, device="cuda:1"):
    clip_model = clip_model.to(device)
    
    inputs = clip_processor(images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_embed = clip_model.get_image_features(**inputs)
    return image_embed

def get_dino_embedding(image, dino_model, dino_processor, device="cuda:1"):
    dino_model = dino_model.to(device)
    inputs = dino_processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
        image_embed = outputs.last_hidden_state[:,0,:]  #[1, embed_dim]
    return image_embed

def get_clip_text_embedding(prompt, clip_model, clip_processor, device="cuda:1"):
    clip_model = clip_model.to(device)
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embed = clip_model.get_text_features(**inputs)  # [1, embed_dim]
    return text_embed
#compute cos
def compute_cosine_similarity(embed1, embed2):
    return cosine_similarity(embed1, embed2).item()

clip_model = CLIPModel.from_pretrained("models/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-large-patch14")
dino_processor = ViTImageProcessor.from_pretrained('models/dino-vits16')
dino_model = ViTModel.from_pretrained('models/dino-vits16')

seeds=[2025]

subject_names=[]
unique_tokens=[]
prompts_list=[]
with open("group_prompts/group6.json", "r") as f:
    data= json.load(f)
    prompt_items=data["subjects"]
for item in prompt_items:
    subject_names.append(item["subject_name"])
    unique_tokens.append(item["unique_token"])
    prompts_list.append(item["prompts"])

for i in range(len(subject_names)):
    subject_name=subject_names[i]
    unique_token=unique_tokens[i]
    real_image = Image.open(f'dataset/{subject_name}/00.jpg')
    real_clip_embedding = get_clip_embedding(real_image, clip_model, clip_processor)
    real_dino_embedding = get_dino_embedding(real_image, dino_model, dino_processor)
    pipeline = DiffusionPipeline.from_pretrained(f"models/{subject_name}", torch_dtype=torch.float16, use_safetensors=True).to("cuda:1")
    prompts=prompts_list[i]
    for prompt in prompts:
        prompt_clip_embedding = get_clip_text_embedding(prompt, clip_model, clip_processor)
        for seed in seeds:
            generator = torch.Generator("cuda:1").manual_seed(seed)
            generated_image = pipeline(prompt, width=1536,height=1536, num_inference_steps=50, generator=generator,guidance_scale=7.5).images[0]
            generated_clip_embedding = get_clip_embedding(generated_image, clip_model, clip_processor)
            generated_dino_embedding = get_dino_embedding(generated_image, dino_model, dino_processor)
            #compute scores
            clip_i_score = compute_cosine_similarity(real_clip_embedding, generated_clip_embedding)
            dino_score = compute_cosine_similarity(real_dino_embedding, generated_dino_embedding)
            clip_t_score = compute_cosine_similarity(prompt_clip_embedding, generated_clip_embedding)
            #appending
            clip_i_scores.append(clip_i_score)
            dino_scores.append(dino_score)
            clip_t_scores.append(clip_t_score)
            print(f'{subject_name} {unique_token} {prompt[-5:]} {seed} clip_i_score: {clip_i_score} dino_score: {dino_score} clip_t_score: {clip_t_score}')

clip_i_scores=np.array(clip_i_scores)
dino_scores=np.array(dino_scores)
clip_t_scores=np.array(clip_t_scores)
# Save the scores to .npy files
np.save('clip_scores_direct_infer_gp6.npy', clip_i_scores)
np.save('dino_scores_direct_infer_gp6.npy', dino_scores)
np.save('clip_t_scores_direct_infer_gp6.npy', clip_t_scores)



