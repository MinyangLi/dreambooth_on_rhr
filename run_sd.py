import torch, os
from diffusers import DDIMScheduler
from pipelines.pipeline_sd import SDPipeline
from diffusers.utils.torch_utils import randn_tensor
from utils.preprocess import latent2image, image2latent
from utils.main_tools import quantic_HW, quantic_cfg, quantic_step
import json
from PIL import Image
from transformers import CLIPProcessor, CLIPModel,ViTImageProcessor, ViTModel
import requests
from torch import cosine_similarity
import numpy as np
from configs import (
    It_base_path, num_inference_steps,
    N, cfg_min, cfg_max, M_cfg, T_min, T_max, M_T, res_min, res_max,
)


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
        image_embed = outputs.last_hidden_state[:, 0, :]  #[1, embed_dim]
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

clip_model = CLIPModel.from_pretrained("models/clip-vit-large-patch14",)
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


def main():
    # 1. init
    clip_i_scores=[]
    dino_scores=[]
    clip_t_scores=[]
    refresh_step_list = quantic_step(T_min, T_max, N, M_T)
    cfg_list = quantic_cfg(cfg_min, cfg_max, N, M_cfg)
    resolution_list = quantic_HW(res_min, res_max, N)
    os.makedirs(It_base_path, exist_ok=True)
    MEMORY = {}

    for i in range(len(subject_names)):
        subject_name=subject_names[i]
        unique_token=unique_tokens[i]
        real_image = Image.open(f'dataset/{subject_name}/00.jpg')
        # real_imgae2 = Image.open(f'dataset/{subject_name}/01.jpg')
        real_clip_embedding = get_clip_embedding(real_image, clip_model, clip_processor)
        real_dino_embedding = get_dino_embedding(real_image, dino_model, dino_processor)
        # real_dino_embedding2 = get_dino_embedding(real_imgae2, dino_model, dino_processor)
        # real_dino_score= compute_cosine_similarity(real_dino_embedding, real_dino_embedding2)
        # print(f'{subject_name} {unique_token} real_dino_score: {real_dino_score}')

        pipe = SDPipeline.from_pretrained(f"models/{subject_name}", torch_dtype=torch.float16).to("cuda:1")
        pipe.vae.enable_tiling()
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        prompts=prompts_list[i]
    
        for prompt in prompts:
            prompt_clip_embedding = get_clip_text_embedding(prompt, clip_model, clip_processor)
            for seed in seeds:
                generator = torch.Generator(device="cuda:1").manual_seed(seed)
                MEMORY.update({
                'predict_x0_list': [],
                })
                start_latent = randn_tensor((1, 4, res_min[0] // pipe.vae_scale_factor, res_min[1] // pipe.vae_scale_factor), dtype=pipe.dtype, device=pipe.device, generator=generator)
                for i in range(len(resolution_list)):
                    if i == 0:
                        predict_x0_latent_noisey = start_latent
                    else:
                        predict_x0_latent = image2latent(latent2image(MEMORY['predict_x0_list'][-1], pipe).resize(resolution_list[i]), pipe)
                        noise_ = randn_tensor(predict_x0_latent.shape, dtype=predict_x0_latent.dtype, device=predict_x0_latent.device, generator=generator)
                        predict_x0_latent_noisey = pipe.scheduler.add_noise(predict_x0_latent, noise_, pipe.scheduler.timesteps[len(MEMORY['predict_x0_list']) - num_inference_steps])
                    hr_output = pipe(
                        prompt=prompt,
                        generator=generator,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=cfg_list[i],
                        latents=predict_x0_latent_noisey,
                        MEMORY=MEMORY,
                        strength=(num_inference_steps - len(MEMORY['predict_x0_list'])) / num_inference_steps,
                        denoising_end=refresh_step_list[i + 1] / num_inference_steps,
                    )
                generated_image=hr_output.images[0]
                generated_clip_embedding = get_clip_embedding(generated_image, clip_model, clip_processor)
                generated_dino_embedding = get_dino_embedding(generated_image, dino_model, dino_processor)
                
                clip_i_score = compute_cosine_similarity(real_clip_embedding, generated_clip_embedding)
                dino_score = compute_cosine_similarity(real_dino_embedding, generated_dino_embedding)
                clip_t_score = compute_cosine_similarity(prompt_clip_embedding, generated_clip_embedding)

                clip_i_scores.append(clip_i_score)
                dino_scores.append(dino_score)
                clip_t_scores.append(clip_t_score)

                print(f'{subject_name} {unique_token} {prompt[-5:]} {seed} clip_i_score: {clip_i_score} dino_score: {dino_score} clip_t_score: {clip_t_score}')
                # np.save('rhr_gp1.npy', clip_i_scores)
                # np.save('dino_scores_rhr_gp1.npy', dino_scores)
    clip_i_scores = np.array(clip_i_scores)
    dino_scores = np.array(dino_scores)
    clip_t_scores = np.array(clip_t_scores)
    np.save('clip_i_scores_rhr_gp6.npy', clip_i_scores)
    np.save('dino_scores_rhr_gp6.npy', dino_scores)
    np.save('clip_t_scores_rhr_gp6.npy', clip_t_scores)




if __name__ == '__main__':
    main()



