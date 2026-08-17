import json
import os
import random
from pathlib import Path
from typing import List
import torch
import numpy as np


def ensure_output_dir(path: str | Path) -> Path:
    """
    Create the output directory if needed and return it as Path.
    """
    os.makedirs(path, exist_ok=True)
    return path


def read_data(source: str | Path, prompt_set: list) -> List[str]:
    """
    Read data from a UTF-8 text file; empty/commented lines (#) are ignored.
    """
    src = Path(source)
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    brand = data.get("brand", None)
    mapping = data.get("mapping", None)
    category = data.get("category", None)

    prompts = []
    for p in prompt_set:
        prompts.extend(data["prompt_sets"].get(p, []))

    return prompts, brand, mapping, category

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
