from typing import Optional

import torch
from PIL import Image


def generate_one(
    pipe,
    prompt: str,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Generate a single image and save it to output_dir. Returns the saved path.
    """
    if seed is not None:
        generator = torch.Generator(
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).manual_seed(seed)
    else:
        generator = None

    kwargs = dict(prompt=prompt, negative_prompt=negative_prompt, generator=generator)

    image: Image.Image = pipe(**kwargs).images[0]

    return image
