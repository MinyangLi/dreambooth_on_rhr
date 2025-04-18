import os, random, cv2
from PIL import Image
from einops import rearrange
import torch, yaml
import numpy as np
from typing import List, Optional, Tuple, Union

@torch.no_grad()
def image2latent(image, pipe):
    pipe_original_dtype = pipe.dtype
    with torch.no_grad():
        image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            pipe = pipe.to(torch.bfloat16)
            image = torch.from_numpy(image).to(torch.bfloat16) / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(pipe.device)
            latents = pipe.vae.encode(image)['latent_dist'].mean
            latents = latents * pipe.vae.config.scaling_factor
            pipe = pipe.to(pipe_original_dtype)
    return latents.to(pipe_original_dtype)

@torch.no_grad()
def latent2image(latents, pipe, return_type='np'):
    latents_original_type = latents.dtype
    pipe_original_type = pipe.dtype
    latents = latents.to(torch.bfloat16)
    pipe = pipe.to(torch.bfloat16)
    latents = 1 / pipe.vae.config.scaling_factor * latents.detach()
    image = pipe.vae.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).to(torch.float32).numpy()[0]
        image = (image * 255).astype(np.uint8)
    pipe = pipe.to(pipe_original_type)
    latents = latents.to(latents_original_type)
    return Image.fromarray(image)