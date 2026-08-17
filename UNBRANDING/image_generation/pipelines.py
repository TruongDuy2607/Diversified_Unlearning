from typing import Literal, Optional

import torch
from diffusers import (FluxPipeline, StableDiffusion3Pipeline,
                       StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline)

ModelKey = Literal["sdxl", "sd14", "sd35-large", "flux1-schnell", "flux1-dev"]

DEFAULT_MODELS: dict[ModelKey, str] = {
    "sd14": "CompVis/stable-diffusion-v1-4",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd35-large": "stabilityai/stable-diffusion-3.5-large",
    "flux1-schnell": "black-forest-labs/FLUX.1-schnell",
    "flux1-dev": "black-forest-labs/FLUX.1-dev",
    "qwen-image": "Qwen/Qwen-Image"
}


def load_pipeline(
    key: ModelKey,
    model_id: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """
    Load and cache the appropriate Diffusers pipeline for a given model key.
    """

    model_id = (model_id or DEFAULT_MODELS[key]).strip()
    if key == "sdxl":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True, variant="fp16"
        )
    elif key == "flux1-schnell" or key == "flux1-dev":
        pipe = FluxPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True
        )
    elif key == "sd14":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True
        )
    elif key == "sd35-large":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True
        )
    elif key == "qwen-image":
        torch_dtype = torch.bfloat16
        #TODO
        max_memory = {
            0: "40GiB",
            1: "40GiB",
            "cpu": "80GiB"
        }
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map="balanced", max_memory=max_memory)
    else:
        raise ValueError(f"Unsupported model key: {key}")

    if key != "qwen-image":
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe
