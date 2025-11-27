import numpy as np
import torch
import random
import pandas as pd
from PIL import Image
import argparse
import os
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
import abc
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
import copy
import ast
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from transformers import CLIPTokenizer, CLIPTextModel
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

def view_images(images, num_rows=3, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img


def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (batch_size, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt,
    num_inference_steps = 50,
    guidance_scale = 7.5,
    generator = None,
    latent = None,
    low_resource = False,
):
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in model.scheduler.timesteps:
        latents = diffusion_step(model, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image


def generate_for_text(ldm_stable, test_text, num_samples = 9, seed = 1231):
    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    images = text2image_ldm_stable(ldm_stable, [test_text]*num_samples, latent=None, num_inference_steps=50, guidance_scale=7.5, generator=g, low_resource=False)
    return view_images(images)


def get_token_embedding(text):
    inputs = tokenizer(text, truncation=True, max_length=77, return_length=True,
                                    return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    input_ids = inputs["input_ids"]
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    attention_mask = inputs["attention_mask"]
    with torch.no_grad():
        token_embeddings = text_model.text_model.embeddings.token_embedding(input_ids)
        position_ids = torch.arange(77).expand((1, -1))
        position_ids = position_ids[:, :input_ids.shape[-1]]
        position_embeddings = text_model.text_model.embeddings.position_embedding(position_ids)
        token_embeddings += position_embeddings
    return token_embeddings, input_shape, input_ids, attention_mask

def get_text_embedding(token_embedding, input_shape, input_ids):
    causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, token_embedding.dtype, device=token_embedding.device
        )
    with torch.no_grad():
        encoder_outputs = text_model.text_model.encoder(
            token_embedding,
            attention_mask=None,
            causal_attention_mask=causal_attention_mask
        )
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = text_model.text_model.final_layer_norm(last_hidden_state)
        if text_model.text_model.eos_token_id == 2:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == text_model.text_model.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]
    outputs = BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
    z = outputs.last_hidden_state
    return z

def edit_model(ldm_stable, old_text_, new_text_, retain_text_, context_list, add=False, layers_to_edit=None, lamb=0.1, erase_scale=0.1, preserve_scale=0.1, with_to_k=True, technique='tensor', concept_type=''):
    ### collect all the cross attns modules
    max_bias_diff = 0.05
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [copy.deepcopy(l.to_v) for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [copy.deepcopy(l.to_k) for l in ca_layers]

    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    ### check the layers to edit (by default it is None; one can specify)
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb
        
    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text
        if n_t == '':
            n_t = ' '
        new_texts.append(n_t)
    if retain_text_ is None:
        ret_texts = ['']
        retain = False
    else:
        ret_texts = retain_text_
        retain = True
    print(ret_texts)
    print(old_texts, new_texts)
    
    ######################## START ERASING ###################################
    for layer_num in range(len(projection_matrices)):
        print(layer_num)
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue

        #### prepare input k* and v*
        with torch.no_grad():
            #mat1 = \lambda W + \sum{v k^T}
            mat1 = lamb * projection_matrices[layer_num].weight

            #mat2 = \lambda I + \sum{k k^T}
            mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1], device=projection_matrices[layer_num].weight.device)
            
            for cnt, t in enumerate(zip(old_texts, new_texts, context_list)):
                old_text = t[0]  # Prompt (e.g., "Mario")
                new_text = t[1]  # target_concept (e.g., "a plumber")
                context = t[2]   # Context (e.g., "man")
                
                # Create the texts for embedding
                texts = [old_text, new_text, context]
                print(texts)
                
                token_embeddings, input_shape, input_ids, attention_mask = get_token_embedding(texts)
                
                # Apply mixup: 0.999 × Prompt + 0.001 × Context
                old_token_emb = token_embeddings[0]  # Prompt embedding
                new_token_emb = token_embeddings[1]  # target_concept embedding
                context_token_emb = token_embeddings[2]  # Context embedding
                
                # Mixup: 0.999 × Prompt + 0.001 × Context
                mixed_token_emb = old_token_emb * 0.999 + context_token_emb * 0.001
                
                # Replace the original prompt embedding with the mixed version
                token_embeddings = torch.stack([mixed_token_emb, new_token_emb, context_token_emb])
                
                text_embeddings = get_text_embedding(token_embeddings, input_shape, input_ids)
                text_embeddings = text_embeddings.to(ldm_stable.device)
                
                final_token_idx = attention_mask[0].sum().item() - 2
                final_token_idx_new = attention_mask[1].sum().item() - 2
                farthest = max([final_token_idx_new, final_token_idx])
                
                old_emb = text_embeddings[0]  # This is now the mixed embedding
                old_emb = old_emb[final_token_idx:len(old_emb) - max(0, farthest - final_token_idx)]
                new_emb = text_embeddings[1]  # target_concept embedding
                new_emb = new_emb[final_token_idx_new:len(new_emb) - max(0, farthest - final_token_idx_new)]
                
                context = old_emb.detach()  # Use mixed embedding as context
                
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        if technique == 'tensor':
                            o_embs = layer(old_emb).detach()
                            u = o_embs
                            u = u / u.norm()
                            
                            new_embs = layer(new_emb).detach()
                            new_emb_proj = (u * new_embs).sum()
                            
                            target = new_embs - (new_emb_proj) * u 
                            values.append(target.detach()) 
                        elif technique == 'replace':
                            values.append(layer(new_emb).detach())
                        else:
                            values.append(layer(new_emb).detach())
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += erase_scale * for_mat1
                mat2 += erase_scale * for_mat2
            
            # Use target_concept for preservation
            for old_text, new_text in zip(ret_texts, ret_texts):
                print(old_text, new_text)
                text_input = ldm_stable.tokenizer(
                    [old_text, new_text],
                    padding="max_length",
                    max_length=ldm_stable.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                old_emb, new_emb = text_embeddings
                context = old_emb.detach()
                values = []
                with torch.no_grad():
                    for layer in projection_matrices:
                        values.append(layer(new_emb[:]).detach())
                context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                mat1 += preserve_scale * for_mat1
                mat2 += preserve_scale * for_mat2
                #update projection matrix
            projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return ldm_stable

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='TrainUCE',
                    description='Finetuning stable diffusion to debias the concepts')
    parser.add_argument('--prompt_csv', help='prompt csv corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--technique', help='technique to erase (either replace or tensor)', type=str, required=False, default='replace')
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='0')
    parser.add_argument('--base', help='base version for stable diffusion', type=str, required=False, default='1.4')
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=None)
    parser.add_argument('--preserve_number', help='number of preserve concepts', type=int, required=False, default=None)
    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=0.05)
    parser.add_argument('--concept_type', help='type of concept being erased', type=str, default='')
    parser.add_argument('--add_prompts', help='option to add additional prompts', type=bool, required=False, default=False)
    parser.add_argument('--info', help='info to add to model name', type=str, required=False, default='')
    parser.add_argument('--name', type=str, required=False, default='uce-diverse')
    parser.add_argument('--level', type=str, required=False, default="")

    args = parser.parse_args()
    level=args.level
    model_name = args.name
    technique = args.technique
    device = f'cuda:{args.device}'
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    add_prompts = args.add_prompts
    preserve_number = args.preserve_number
    concept_type = args.concept_type
    print_text = ''

    old_texts = []
    additional_prompts = []
    
    df = pd.read_csv(args.prompt_csv)
    prompt_csv = df["prompt"].tolist()
    context_list = df["context"].tolist()  # Get context values
    target_concept = df["target_concept"].tolist()  # Get target_concept values

    additional_prompts.extend(prompt_csv)
    
    if not add_prompts:
        additional_prompts = []
    
    old_texts = prompt_csv
    new_texts = target_concept  # Use target_concept for mapping
    
    assert len(new_texts) == len(old_texts)
    assert len(context_list) == len(old_texts)
    
    # Use target_concept for retain_texts
    # retain_texts = target_concept
    retain_texts = context_list
    
    if len(retain_texts) >= 1:
        print_text += f'-preserve_true'     
    else:
        print_text += f'-preserve_false'
    
    if preserve_scale is None:
        preserve_scale = max(0.1, 1/len(retain_texts))
    
    sd14 = "CompVis/stable-diffusion-v1-4"
    sd21 = 'stabilityai/stable-diffusion-2-1-base'

    if args.base == '1.4':
        model_version = sd14
    elif args.base == '2.1':
        model_version = sd21
    else:
        model_version = sd14
    
    print_text += f"-sd_{args.base.replace('.','_')}" 
    print_text += f"-method_{technique}" 
    print_text += f"-info_{args.info}"
    print_text = print_text.lower()
    
    print(print_text)
    print('old texts', old_texts)
    print('new texts', new_texts)
    print('context_list', context_list)
    print('retain texts', retain_texts)

    ldm_stable = StableDiffusionPipeline.from_pretrained(model_version, cache_dir='./cache').to(device)
    ldm_stable = edit_model(
        ldm_stable=ldm_stable, 
        old_text_=old_texts, 
        new_text_=new_texts, 
        add=False, 
        retain_text_=retain_texts,
        context_list=context_list,  # Pass context list
        lamb=0.5, 
        erase_scale=erase_scale, 
        preserve_scale=preserve_scale, 
        technique=technique, 
        concept_type=concept_type
    )
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('info', exist_ok=True)
    os.makedirs(f'models/{model_name}-{level}', exist_ok=True)

    torch.save(ldm_stable.unet.state_dict(), f'models/{model_name}-{level}/{model_name}-{level}.pt')