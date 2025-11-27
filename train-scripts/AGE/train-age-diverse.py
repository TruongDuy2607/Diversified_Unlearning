from omegaconf import OmegaConf
import torch
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
from einops import rearrange
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import einops
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import random
import glob
import re
import shutil
import pdb
import argparse
from convertModels import savemodelDiffusers
from PIL import Image
from torch.autograd import Variable
from utils_exp import str2bool
from utils_alg import load_img, moving_average, plot_loss, get_models, save_to_dict
from gen_embedding_matrix import learn_k_means_from_input_embedding, learn_k_means_from_output, save_embedding_matrix, search_closest_tokens, retrieve_embedding_token

##### MODIFY DIVERSE ######
import pandas as pd
def get_diverse_prompts(csv_path, objects_mode=False):
    """
    Load diverse prompts from CSV file
    Returns list of prompts, keywords (concepts to erase), and contexts
    
    If objects_mode is True:
    - Extract context from prompt by removing the keyword
    - Context is the remaining part after removing keyword from prompt
    """
    df = pd.read_csv(csv_path)
    prompts = df['prompt'].tolist()
    
    if 'keyword' in df.columns:
        keywords = df['keyword'].tolist()
    elif 'concept' in df.columns:
        keywords = df['concept'].tolist()
    else:
        raise ValueError("CSV must have either 'keyword' or 'concept' column")
    
    if objects_mode:
        # Extract context by removing keyword from prompt
        contexts = []
        for prompt, keyword in zip(prompts, keywords):
            # Remove keyword from prompt to get context
            # Handle case-insensitive matching
            context = prompt.replace(keyword, "").strip()
            # Clean up extra spaces
            context = ' '.join(context.split())
            contexts.append(context)
    else:
        # Use existing context column if available, otherwise use empty context
        if 'context' in df.columns:
            contexts = df['context'].tolist()
        else:
            contexts = [""] * len(prompts)
    
    return prompts, keywords, contexts

###########################

# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


@torch.no_grad()
def sample_model(model, sampler, c, h, w, ddim_steps, scale, ddim_eta, start_code=None, n_samples=1,t_start=-1,log_every_t=None,till_T=None,verbose=True):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(S=ddim_steps,
                                     conditioning=c,
                                     batch_size=n_samples,
                                     shape=shape,
                                     verbose=False,
                                     x_T=start_code,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta,
                                     verbose_iter = verbose,
                                     t_start=t_start,
                                     log_every_t = log_t,
                                     till_T = till_T
                                    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim

def train_age_diverse(prompt, train_method, start_guidance, negative_guidance, iterations, lr, config_path, ckpt_path, diffusers_config_path, devices, prompts, setting_name, seperator=None, image_size=512, ddim_steps=50, args=None):
    '''
    Function to train diffusion models to erase concepts from model weights using Diversified-AGE
    
    This implements the Diversified-AGE formula using diverse prompts from CSV file:
    min_{θ'} E_{c_e ∈ E, c ∈ C} max_{c_t ∈ C} [||ε_θ'(c + c_e) - ε_θ(c + c_t)||_2^2 + λ||ε_θ'(c + c_t) - ε_θ(c + c_t)||_2^2]
    
    Where c_e and c_t are diverse prompts containing the concept in different contexts from the CSV file.

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Mario").
    train_method : str
        The parameters to train for erasure (noxattn, xattan).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    '''
    # PROMPT CLEANING
    word_print = prompt.replace(' ','')

    # Load diverse prompts from CSV
    diverse_prompts, diverse_keywords, diverse_contexts = get_diverse_prompts(csv_path=prompts, objects_mode=args.objects)
    
    # If mixing keywords per-iteration, we still need the full df for sampling later
    # Otherwise, filter by current prompt (keyword)
    if not getattr(args, 'mix_keywords', False):
        filtered_indices = [i for i, keyword in enumerate(diverse_keywords) if keyword == prompt]
        filtered_prompts = [diverse_prompts[i] for i in filtered_indices]
        filtered_contexts = [diverse_contexts[i] for i in filtered_indices]
    else:
        filtered_prompts = diverse_prompts
        filtered_contexts = diverse_contexts
        filtered_keywords = diverse_keywords
    
    print(f'Found {len(filtered_prompts)} diverse prompts for concept "{prompt}"')
    print('Sample prompts:', filtered_prompts[:5])
    print('Sample contexts:', filtered_contexts[:5])
    
    # Ensure we have diverse prompts
    if len(filtered_prompts) == 0:
        raise ValueError(f"No prompts found for the provided CSV (check columns prompt, keyword, context)")

    # Simplify concept handling - no need for get_prompt()
    # The concept to erase is directly from the command line argument
    # For preserved concepts, we'll use an empty space as default
    preserved = ' '

    if seperator is not None:
        erased_words = prompt.split(seperator)
        erased_words = [word.strip() for word in erased_words]
        preserved_words = [preserved] * len(erased_words)
    else:
        erased_words = [prompt]
        preserved_words = [preserved]
    
    print('to be erased:', erased_words)
    print('to be preserved:', preserved_words)
    preserved_words.append('')

    ddim_eta = 0

    model_orig, _, model, sampler = get_models(config_path, ckpt_path, devices)

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                print(name)
                parameters.append(param)
        # train only qkv layers in x attention layers
        if train_method == 'xattn_matching':
            if 'attn2' in name and ('to_q' in name or 'to_k' in name or 'to_v' in name):
                print(name)
                parameters.append(param)
                # return_nodes[name] = name
        # train all layers
        if train_method == 'full':
            print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                print(name)
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    print(name)
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    print(name)
                    parameters.append(param)
    
    def decode_and_save_image(model_orig, z, path):
        x = model_orig.decode_first_stage(z)
        x = torch.clamp((x + 1.0)/2.0, min=0.0, max=1.0)
        x = rearrange(x, 'b c h w -> b h w c')
        image = Image.fromarray((x[0].cpu().numpy()*255).astype(np.uint8))
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(path)
        plt.close()

    model.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda cond, s, code, t: sample_model(model, sampler,
                                                                 cond, image_size, image_size, ddim_steps, s, ddim_eta,
                                                                 start_code=code, till_T=t, verbose=False)

    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history_dict = {}

    name = f'age-diverse-{setting_name}'
    models_path = args.models_path
    os.makedirs(f'evaluation_folder/{name}', exist_ok=True)
    os.makedirs(f'invest_folder/{name}', exist_ok=True)
    os.makedirs(f'{models_path}/{name}', exist_ok=True)

    pbar = tqdm(range(args.pgd_num_steps*iterations))

    def create_prompt(word):
        prompt = f'{word}'
        emb = model.get_learned_conditioning([prompt])
        init = emb
        return init

    fixed_start_code = torch.randn((1, 4, 64, 64)).to(devices[0])    

    # create a matrix of embeddings for the entire vocabulary
    if not os.path.exists('models/embedding_matrix_dict_EN3K.pt'):
        save_embedding_matrix(model, model_name='SD-v1-4', save_mode='dict', vocab='EN3K')

    if not os.path.exists('models/embedding_matrix_array_EN3K.pt'):
        save_embedding_matrix(model, model_name='SD-v1-4', save_mode='array', vocab='EN3K')
    
    # initialize the preserved set for each erased concept
    tokens_embedding = []
    all_sim_dict = dict()
    for word in erased_words:
        top_k_tokens, sorted_sim_dict = search_closest_tokens(word, model, k=args.gumbel_k_closest, sim='l2', model_name='SD-v1-4', ignore_special_tokens=args.ignore_special_tokens, vocab=args.vocab)
        tokens_embedding.extend(top_k_tokens)
        all_sim_dict[word] = {key:sorted_sim_dict[key] for key in top_k_tokens}

    if args.gumbel_num_centers > 0:
        assert args.gumbel_num_centers % len(erased_words) == 0, 'Number of centers should be divisible by number of erased words'
    preserved_dict = dict()

    for word in erased_words:
        temp = learn_k_means_from_input_embedding(sim_dict=all_sim_dict[word], num_centers=args.gumbel_num_centers)
        preserved_dict[word] = temp

    history_dict = save_to_dict(preserved_dict, f'preserved_set_0', history_dict)

    # Create context-specific preserved matrices
    print('Creating context-specific preserved matrices')
    context_preserved_matrices = {}
    weight_pi_dict = {}
    
    for context in filtered_contexts:
        context_matrix_list = []
        
        # For each token in preserved set, create embedding with context
        for word in erased_words:
            preserved_set = preserved_dict[word]
            for token in preserved_set:
                if args.objects:
                    # In objects mode: get token embedding and add context embedding
                    token_emb = model.get_learned_conditioning([token])
                    if context.strip():  # Only add context if it's not empty
                        context_emb = model.get_learned_conditioning([context])
                        # Add token and context embeddings
                        combined_emb = token_emb + context_emb
                    else:
                        combined_emb = token_emb
                    context_matrix_list.append(combined_emb.flatten())
                else:
                    # Original mode: create context-specific prompt with preserved token
                    context_prompt = f"{token} {context}"
                    emb = model.get_learned_conditioning([context_prompt])
                    # Make sure we flatten to a vector
                    context_matrix_list.append(emb.flatten())
        
        # Stack embeddings into a matrix - each row is an embedding vector
        preserved_matrix = torch.stack(context_matrix_list)
        
        # Print shape information for debugging
        print(f"Context: {context}, Matrix shape: {preserved_matrix.shape}")
        
        # Store the matrix
        context_preserved_matrices[context] = preserved_matrix
        
        # Initialize weight_pi for this context
        n = preserved_matrix.shape[0]  # Number of preserved tokens
        weight_pi = torch.zeros((1, n), device=devices[0], dtype=torch.float32)
        weight_pi = weight_pi + 1 / n  # Initialize with uniform probability
        weight_pi = Variable(weight_pi, requires_grad=True)
        weight_pi_dict[context] = weight_pi
    
    print(f'Created preserved matrices for {len(context_preserved_matrices)} contexts')
    history_dict = save_to_dict(weight_pi_dict, f'one_hot_dict_0', history_dict)

    # optimizer for all pi vectors
    opt_weight_pi = torch.optim.Adam([weight_pi for weight_pi in weight_pi_dict.values()], lr=args.gumbel_lr)

    """
    Gumbel-Softmax function
        if `hard` is 1, then it is one-hot, if `hard` is 0, then it is a new soft version, which takes the top-k highest values and normalize them to 1
    """
    def gumbel_softmax(logits, temperature=args.gumbel_temp, hard=args.gumbel_hard, eps=1e-10, k=args.gumbel_topk):
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        y = logits + gumbel
        y = torch.nn.functional.softmax(y / temperature, dim=-1)
        if hard == 1:
            y_hard = torch.zeros_like(logits)
            y_hard.scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y
        elif hard == 0:
            top_k_values, _ = torch.topk(y, k, dim=-1)
            top_k_mask = y >= top_k_values[..., -1].unsqueeze(-1)
            y = y * top_k_mask.float()
            y = y / y.sum(dim=-1, keepdim=True)
        return y

    # Prepare keyword index for mix mode
    if getattr(args, 'mix_keywords', False):
        # Build mapping: keyword -> indices
        from collections import defaultdict
        kw_to_indices = defaultdict(list)
        for idx, kw in enumerate(filtered_keywords):
            if isinstance(kw, str) and len(kw) > 0:
                kw_to_indices[kw].append(idx)
        unique_kws = list(kw_to_indices.keys())
        if args.max_keywords is not None and args.max_keywords > 0:
            unique_kws = unique_kws[:args.max_keywords]

    for i in pbar:
        # Sample keyword and prompt/context per-iteration if mix mode
        if getattr(args, 'mix_keywords', False):
            kw = random.choice(unique_kws)
            idx = random.choice(kw_to_indices[kw])
            context = diverse_contexts[idx]
            full_prompt = diverse_prompts[idx]
            # Also update word_print for logging
            word_print = kw.replace(' ', '')
            print(f"[iter {i}] keyword: {kw}")
        else:
            # Randomly select a context and corresponding prompt (single keyword mode)
            context_idx = random.randint(0, len(filtered_contexts) - 1)
            context = filtered_contexts[context_idx]
            full_prompt = filtered_prompts[context_idx]
        
        # Reset gradients
        opt.zero_grad()
        model.zero_grad()
        model_orig.zero_grad()
        opt_weight_pi.zero_grad()
        
        # Get embeddings for the full prompt (c + c_e)
        emb_c_e = model.get_learned_conditioning([full_prompt])  # e.g., "Mario smiling"
        
        # Get embedding for context with concept replacement (c + c_t)
        preserved_matrix = context_preserved_matrices[context]
        
        # Debug information
        print(f"Debug - weight_pi shape: {weight_pi_dict[context].shape}")
        print(f"Debug - preserved_matrix shape: {preserved_matrix.shape}")
        
        # Ensure weight_pi has the right shape for matrix multiplication
        # weight_pi should be [1, n] and preserved_matrix should be [n, 77*768]
        # where n is the number of preserved tokens
        gumbel_weights = gumbel_softmax(weight_pi_dict[context])
        
        # Ensure the matrix multiplication is done correctly
        emb_c_t = torch.matmul(gumbel_weights, preserved_matrix)
        emb_c_t = torch.reshape(emb_c_t, (1, 77, 768))
        
        # Clone emb_c_t for reference
        emb_0 = emb_c_t.clone().detach()
        
        # Select random time step
        t_enc = torch.randint(ddim_steps, (1,), device=devices[0])
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc)/ddim_steps)*1000)
        og_num_lim = round((int(t_enc+1)/ddim_steps)*1000)

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=devices[0])

        start_code = torch.randn((1, 4, 64, 64)).to(devices[0])

        with torch.no_grad():
            # generate images with the concept and replacement
            z_c_e = quick_sample_till_t(emb_c_e.to(devices[0]), start_guidance, start_code, int(t_enc))
            z_c_t = quick_sample_till_t(emb_c_t.to(devices[0]), start_guidance, start_code, int(t_enc))

            # get conditional and unconditional scores from frozen model
            eps_0_org = model_orig.apply_model(z_c_e.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_0.to(devices[1]))
            eps_e_org = model_orig.apply_model(z_c_e.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_c_e.to(devices[1]))
            eps_t_org = model_orig.apply_model(z_c_t.to(devices[1]), t_enc_ddpm.to(devices[1]), emb_c_t.to(devices[1]))

        # get conditional score from the model being trained
        eps_e = model.apply_model(z_c_e.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_c_e.to(devices[0]))
        eps_t = model.apply_model(z_c_t.to(devices[0]), t_enc_ddpm.to(devices[0]), emb_c_t.to(devices[0]))

        eps_0_org.requires_grad = False
        eps_e_org.requires_grad = False
        eps_t_org.requires_grad = False

        # using DDIM inversion to project the x_t to x_0
        # check that the alphas is in descending order
        assert torch.all(sampler.ddim_alphas[:-1] >= sampler.ddim_alphas[1:])
        alpha_bar_t = sampler.ddim_alphas[int(t_enc)]
        eps_e_pred = (z_c_e - torch.sqrt(1 - alpha_bar_t) * eps_e) / torch.sqrt(alpha_bar_t)
        eps_t_pred = (z_c_t - torch.sqrt(1 - alpha_bar_t) * eps_t) / torch.sqrt(alpha_bar_t)

        eps_e_org_pred = (z_c_e - torch.sqrt(1 - alpha_bar_t) * eps_e_org) / torch.sqrt(alpha_bar_t)
        eps_0_org_pred = (z_c_e - torch.sqrt(1 - alpha_bar_t) * eps_0_org) / torch.sqrt(alpha_bar_t)
        eps_t_org_pred = (z_c_t - torch.sqrt(1 - alpha_bar_t) * eps_t_org) / torch.sqrt(alpha_bar_t)

        if i % args.pgd_num_steps == 0:
            # optimize the model (min_θ')
            loss = 0
            # L1 = ||ε_θ'(c + c_e) - ε_θ(c + c_t)||_2^2
            loss += criteria(eps_e_pred.to(devices[0]), eps_0_org_pred.to(devices[0]) - (negative_guidance * (eps_e_org_pred.to(devices[0]) - eps_0_org_pred.to(devices[0]))))
            # L2 = λ||ε_θ'(c + c_t) - ε_θ(c + c_t)||_2^2
            loss += args.lamda * criteria(eps_t_pred.to(devices[0]), eps_t_org_pred.to(devices[0]))

            # update weights to erase the concept
            loss.backward()
            losses.append(loss.item())
            pbar.set_postfix({"loss": loss.item(), "context": context})
            history_dict = save_to_dict(loss.item(), 'loss', history_dict)
            opt.step()
        else:
            # update the weight_pi vector (max_c_t∈C)
            opt.zero_grad()
            opt_weight_pi.zero_grad()
            model.zero_grad()
            model_orig.zero_grad()
            
            loss = 0 
            # Negate loss to maximize instead of minimize
            loss -= criteria(eps_e_pred.to(devices[0]), eps_0_org_pred.to(devices[0]))
            loss -= args.lamda * criteria(eps_t_pred.to(devices[0]), eps_t_org_pred.to(devices[0]))
            
            loss.backward()
            opt_weight_pi.step()
            
            # Save the current state of weight_pi
            word = erased_words[0]  # Assuming single concept erasure
            preserved_set = preserved_dict[word]
            
            # Find the token with highest weight for this context
            context_weights = weight_pi_dict[context]
            token_idx = torch.argmax(context_weights, dim=1).item()
            token_idx = token_idx % len(preserved_set)  # Handle case where we have multiple words
            best_token = preserved_set[token_idx]
            
            history_dict = save_to_dict([context, context_weights.cpu().detach().numpy(), i, best_token], 'weight_pi', history_dict)

        if i % (args.save_freq) == 0:
            with torch.no_grad():
                # Evaluate on a few contexts
                eval_contexts = random.sample(filtered_contexts, min(3, len(filtered_contexts)))
                for eval_context in eval_contexts:
                    # Get the original prompt with this context
                    eval_idx = filtered_contexts.index(eval_context)
                    eval_prompt = filtered_prompts[eval_idx]
                    
                    # Get embedding for the original prompt
                    emb_orig = model.get_learned_conditioning([eval_prompt])
                    
                    # Get embedding for the context with best replacement
                    preserved_matrix = context_preserved_matrices[eval_context]
                    gumbel_weights = gumbel_softmax(weight_pi_dict[eval_context])
                    emb_replaced = torch.matmul(gumbel_weights, preserved_matrix)
                    emb_replaced = torch.reshape(emb_replaced, (1, 77, 768))
                    
                    # Generate images
                    z_orig = quick_sample_till_t(emb_orig.to(devices[0]), start_guidance, fixed_start_code, int(ddim_steps))
                    z_replaced = quick_sample_till_t(emb_replaced.to(devices[0]), start_guidance, fixed_start_code, int(ddim_steps))
                    
                    # Save images
                    decode_and_save_image(model_orig, z_orig, path=f'evaluation_folder/{name}/orig_{i}_{eval_context}.png')
                    decode_and_save_image(model_orig, z_replaced, path=f'evaluation_folder/{name}/replaced_{i}_{eval_context}.png')

        if i % 100 == 0:
            # Save history and model periodically
            save_history(losses, name, word_print, models_path=models_path)
            torch.save(history_dict, f'invest_folder/{name}/history_dict_{i}.pt')
            
            # Save context-specific best replacements
            context_replacements = {}
            for ctx in filtered_contexts:
                ctx_weights = weight_pi_dict[ctx]
                token_idx = torch.argmax(ctx_weights, dim=1).item() % len(preserved_set)
                best_token = preserved_set[token_idx]
                context_replacements[ctx] = best_token
            
            with open(f'invest_folder/{name}/context_replacements_{i}.txt', 'w') as f:
                for ctx, token in context_replacements.items():
                    f.write(f"{ctx}: {token}\n")

    model.eval()

    save_model(model, name, None, models_path=models_path, save_compvis=True, save_diffusers=True, compvis_config_file=config_path, diffusers_config_file=diffusers_config_path)
    save_history(losses, name, word_print, models_path=models_path)
    
def save_model(model, name, num, models_path, compvis_config_file=None, diffusers_config_file=None, device='cpu', save_compvis=True, save_diffusers=True):
    folder_path = f'{models_path}/{name}'
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f'{folder_path}/{name}-epoch_{num}.pt'
    else:
        path = f'{folder_path}/{name}.pt'

    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        print('Saving Model in Diffusers Format')
        savemodelDiffusers(name, compvis_config_file, diffusers_config_file, device=device)

def save_history(losses, name, word_print, models_path):
    folder_path = f'{models_path}/{name}'
    os.makedirs(folder_path, exist_ok=True)
    with open(f'{folder_path}/loss.txt', 'w') as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f'{folder_path}/loss.png', word_print, n=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Finetuning stable diffusion model to erase concepts using Diversified-AGE')
    parser.add_argument('--prompt', help='prompt corresponding to concept to erase', type=str, required=True)
    parser.add_argument('--prompt_path', help='path to CSV file with diverse prompts', type=str, required=True)
    parser.add_argument('--train_method', help='method of training', type=str, required=True)
    parser.add_argument('--start_guidance', help='guidance of start image used to train', type=float, required=False, default=3)
    parser.add_argument('--negative_guidance', help='guidance of negative training used to train', type=float, required=False, default=1)
    parser.add_argument('--iterations', help='iterations used to train', type=int, required=False, default=1000)
    parser.add_argument('--lr', help='learning rate used to train', type=float, required=False, default=1e-5)
    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/stable-diffusion/v1-inference.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt')
    parser.add_argument('--diffusers_config_path', help='diffusers unet config json path', type=str, required=False, default='diffusers_unet_config.json')
    parser.add_argument('--devices', help='cuda devices to train on', type=str, required=False, default='0,0')
    parser.add_argument('--seperator', help='separator if you want to train bunch of erased_words separately', type=str, required=False, default=None)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=512)
    parser.add_argument('--ddim_steps', help='ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--info', help='info to add to model name', type=str, required=False, default='')
    parser.add_argument('--save_freq', help='frequency to save data, per iteration', type=int, required=False, default=10)
    parser.add_argument('--models_path', help='method of prompting', type=str, required=True, default='models')
    parser.add_argument('--erase_all_keywords', action='store_true', help='erase all unique keywords in the diverse CSV', required=False, default=False)
    parser.add_argument('--max_keywords', type=int, help='maximum number of keywords to erase (when erase_all_keywords)', required=False, default=10)
    parser.add_argument('--mix_keywords', action='store_true', help='randomly mix concepts: each iteration samples a random keyword from CSV', required=False, default=False)

    parser.add_argument('--gumbel_lr', help='learning rate for prompt', type=float, required=False, default=1e-3)
    parser.add_argument('--gumbel_temp', help='temperature for gumbel softmax', type=float, required=False, default=2)
    parser.add_argument('--gumbel_hard', help='hard for gumbel softmax, 0: soft, 1: hard', type=int, required=False, default=0, choices=[0,1])
    parser.add_argument('--gumbel_num_centers', help='number of centers for kmeans, if <= 0 then do not apply kmeans', type=int, required=False, default=100)
    parser.add_argument('--gumbel_update', help='update frequency for preserved set, if <= 0 then do not update', type=int, required=False, default=100)
    parser.add_argument('--gumbel_time_step', help='time step for the starting point to estimate epsilon', type=int, required=False, default=0)
    parser.add_argument('--gumbel_multi_steps', help='multi steps for calculating the output', type=int, required=False, default=2)
    parser.add_argument('--gumbel_k_closest', help='number of closest tokens to consider', type=int, required=False, default=1000)
    parser.add_argument('--gumbel_topk', help='number of top-k values in the soft gumbel softmax to be considered', type=int, required=False, default=5)
    parser.add_argument('--ignore_special_tokens', help='ignore special tokens in the embedding matrix', type=str2bool, required=False, default=True)
    parser.add_argument('--vocab', help='vocab', type=str, required=False, default='EN3K')
    parser.add_argument('--pgd_num_steps', help='number of step to optimize adversarial concepts', type=int, required=False, default=2)
    parser.add_argument('--lamda', help='lambda for the loss function', type=float, required=False, default=1)
    parser.add_argument('--objects', action='store_true', help='use objects mode: extract context from prompt and combine token+context embeddings', required=False, default=False)
    parser.add_argument('--name', help='Name of the setting', type=str, required=False, default='celeb')
    args = parser.parse_args()
    
    prompt = args.prompt
    prompt_path = args.prompt_path
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    name = args.name

    if args.erase_all_keywords:
        # iterate over unique keywords from the CSV (up to max_keywords)
        import pandas as _pd
        df_all = _pd.read_csv(prompt_path)
        unique_keywords = [kw for kw in df_all['keyword'].dropna().unique().tolist()]
        if args.max_keywords is not None and args.max_keywords > 0:
            unique_keywords = unique_keywords[:args.max_keywords]
        print(f"Erasing {len(unique_keywords)} concepts: {unique_keywords}")
        for kw in unique_keywords:
            print(f"\n===== Training erase for concept: {kw} =====")
            train_age_diverse(prompt=kw, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, prompts=prompt_path, setting_name=name, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, args=args)
    else:
        train_age_diverse(prompt=prompt, train_method=train_method, start_guidance=start_guidance, negative_guidance=negative_guidance, iterations=iterations, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=diffusers_config_path, devices=devices, prompts=prompt_path, setting_name=name, seperator=seperator, image_size=image_size, ddim_steps=ddim_steps, args=args)