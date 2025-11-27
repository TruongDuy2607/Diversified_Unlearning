import os
import argparse
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
from tqdm import tqdm
import lpips
import torchvision.transforms as transforms

def load_image_for_clip(image_path, preprocess):
    """Load and preprocess an image for CLIP."""
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return preprocess(image)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def image_loader_lpips(image_name):
    """Load and preprocess an image for LPIPS following lpips_eval.py"""
    imsize = 512
    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])

    image = Image.open(image_name)
    # Ensure the image is RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = loader(image).unsqueeze(0)
    image = (image-0.5)*2
    return image.to(torch.float)

def evaluate_retain(args):
    # Load CLIP model
    print(f"Loading CLIP model: {args.clip_model}...")
    device = torch.device(args.device)
    clip_model, preprocess = clip.load(args.clip_model, device=device)
    clip_model.eval()
    
    # Load LPIPS model
    print("Loading LPIPS model...")
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    # Read prompts CSV
    print(f"Reading prompts CSV from {args.prompts_csv}...")
    df_prompts = pd.read_csv(args.prompts_csv)
    
    # Create a copy to store results
    df_results = df_prompts.copy()
    df_results['clip_sim'] = None
    df_results['lpips_sim'] = None
    # Add new columns for CLIP-t metrics
    df_results['clip_t_sd'] = None
    df_results['clip_t_retain'] = None
    
    # Get unique concepts
    # unique_concepts = df_prompts['label_str'].unique()
    unique_concepts = df_prompts['prompt'].unique()
    print(f"Found {len(unique_concepts)} unique concepts: {unique_concepts}")
    
    # Dictionary to store concept statistics
    concept_stats = {concept: {'clip_sim': [], 'lpips_sim': [],
                              'clip_t_sd': [], 'clip_t_retain': []} 
                    for concept in unique_concepts}
    
    # Process each concept using its own images
    for concept in unique_concepts:
        print(f"\nEvaluating retain capability for concept: {concept}")
        
        # Filter for the current concept being evaluated
        # eval_df = df_prompts[df_prompts['label_str'] == concept]
        eval_df = df_prompts[df_prompts['prompt'] == concept]
        
        # Process each row in the filtered dataset
        for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Processing {concept}"):
            # target_concept = str(row['label_str'])
            target_concept = str(row['prompt'])
            prompt_id = str(row['case_number'])
            prompt_text = str(row['prompt'])
            seed = str(row['evaluation_seed'])
            
            # Construct image filename
            img_filename = f"{prompt_id}_0.png"
            
            # Paths to images from different models
            sd_path = os.path.join(args.sd_folder, img_filename)
            retain_path = os.path.join(args.retain_folder, img_filename)
            
            # Skip if any of the files doesn't exist
            if not all(os.path.exists(p) for p in [sd_path, retain_path]):
                print(f"Skipping {img_filename} - one or more files not found")
                continue
            
            # Calculate CLIP similarity
            try:
                # Load images for CLIP
                sd_img_clip = load_image_for_clip(sd_path, preprocess)
                retain_img_clip = load_image_for_clip(retain_path, preprocess)
                
                # Skip if any image failed to load
                if sd_img_clip is None or retain_img_clip is None:
                    print(f"Skipping {img_filename} due to image loading error")
                    continue
                
                # Move to device
                sd_img_clip = sd_img_clip.unsqueeze(0).to(device)
                retain_img_clip = retain_img_clip.unsqueeze(0).to(device)
                
                # Extract features with CLIP - following clip_eval.py
                with torch.no_grad():
                    # Extract image features
                    sd_features = clip_model.encode_image(sd_img_clip)
                    retain_features = clip_model.encode_image(retain_img_clip)
                    
                    # Normalize image features
                    sd_features = sd_features / sd_features.norm(dim=1, keepdim=True)
                    retain_features = retain_features / retain_features.norm(dim=1, keepdim=True)
                    
                    # Calculate image-to-image cosine similarity
                    clip_sim_score = (sd_features @ retain_features.T).item()
                    
                    # Encode the text prompt
                    text_tokens = clip.tokenize([prompt_text]).to(device)
                    text_features = clip_model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=1, keepdim=True)
                    
                    # Calculate text-to-image cosine similarity
                    clip_t_sd_score = (text_features @ sd_features.T).item()
                    clip_t_retain_score = (text_features @ retain_features.T).item()
                
                # Store CLIP image-to-image scores
                df_results.loc[idx, 'clip_sim'] = clip_sim_score
                
                # Store CLIP text-to-image scores
                df_results.loc[idx, 'clip_t_sd'] = clip_t_sd_score
                df_results.loc[idx, 'clip_t_retain'] = clip_t_retain_score
                
                # Add to concept stats
                concept_stats[target_concept]['clip_sim'].append(clip_sim_score)
                concept_stats[target_concept]['clip_t_sd'].append(clip_t_sd_score)
                concept_stats[target_concept]['clip_t_retain'].append(clip_t_retain_score)
                
            except Exception as e:
                print(f"Error calculating CLIP similarity for {img_filename}: {e}")
                continue  # Skip LPIPS calculation if CLIP fails
            
            # Calculate LPIPS score - following lpips_eval.py
            try:
                # Load images for LPIPS
                sd_img_lpips = image_loader_lpips(sd_path).to(device)
                retain_img_lpips = image_loader_lpips(retain_path).to(device)
                
                # Calculate LPIPS scores
                lpips_sim_score = loss_fn_alex(sd_img_lpips, retain_img_lpips).item()
                
                # Store LPIPS scores
                df_results.loc[idx, 'lpips_sim'] = lpips_sim_score
                
                # Add to concept stats
                concept_stats[target_concept]['lpips_sim'].append(lpips_sim_score)
                
            except Exception as e:
                print(f"Error calculating LPIPS score for {img_filename}: {e}")
    
    # Save results to CSV
    df_results.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    
    # Calculate and save summary statistics
    with open(args.output_txt, 'w') as f:
        f.write("Retain Capability Evaluation Results\n")
        f.write("==================================\n\n")
        
        f.write("Per-Concept Statistics:\n")
        f.write("---------------------\n")
        
        # Calculate overall statistics
        all_clip_sim = []
        all_lpips_sim = []
        all_clip_t_sd = []
        all_clip_t_retain = []
        
        # Table header for concept statistics
        f.write("\n+" + "-"*70 + "+\n")
        f.write(f"| {'Concept':<15} | {'CLIP':<8} | {'LPIPS':<8} | {'CLIP-t SD':<8} | {'CLIP-t Ret':<8} | {'Samples':<7} |\n")
        f.write("+" + "-"*70 + "+\n")
        
        for concept in unique_concepts:
            # Skip concepts with no data
            if not concept_stats[concept]['clip_sim']:
                continue
                
            clip_sim_avg = np.mean(concept_stats[concept]['clip_sim'])
            lpips_sim_avg = np.mean(concept_stats[concept]['lpips_sim'])
            clip_t_sd_avg = np.mean(concept_stats[concept]['clip_t_sd'])
            clip_t_retain_avg = np.mean(concept_stats[concept]['clip_t_retain'])
            sample_count = len(concept_stats[concept]['clip_sim'])
            
            all_clip_sim.extend(concept_stats[concept]['clip_sim'])
            all_lpips_sim.extend(concept_stats[concept]['lpips_sim'])
            all_clip_t_sd.extend(concept_stats[concept]['clip_t_sd'])
            all_clip_t_retain.extend(concept_stats[concept]['clip_t_retain'])
            
            # Write table row
            f.write(f"| {concept:<15} | {clip_sim_avg:<8.4f} | {lpips_sim_avg:<8.4f} | {clip_t_sd_avg:<8.4f} | {clip_t_retain_avg:<8.4f} | {sample_count:<7} |\n")
        
        # Table footer
        f.write("+" + "-"*70 + "+\n")
        
        # Overall statistics
        overall_clip_sim = np.mean(all_clip_sim)
        overall_lpips_sim = np.mean(all_lpips_sim)
        overall_clip_t_sd = np.mean(all_clip_t_sd)
        overall_clip_t_retain = np.mean(all_clip_t_retain)
        total_samples = len(all_clip_sim)
        
        f.write(f"| {'OVERALL':<15} | {overall_clip_sim:<8.4f} | {overall_lpips_sim:<8.4f} | {overall_clip_t_sd:<8.4f} | {overall_clip_t_retain:<8.4f} | {total_samples:<7} |\n")
        f.write("+" + "-"*70 + "+\n")
        
        # Add interpretation notes
        f.write("\nNotes:\n")
        f.write("- Higher CLIP similarity (closer to 1.0) indicates better retention of content\n")
        f.write("- Lower LPIPS score (closer to 0.0) indicates better retention of visual details\n")
        f.write("- CLIP-t measures text-to-image similarity using the text prompts\n")
    
    print(f"Summary statistics saved to {args.output_txt}")
    
    # Print overall results to console
    print("\nOverall Retain Capability Results:")
    print(f"CLIP:         {overall_clip_sim:.4f}")
    print(f"LPIPS:        {overall_lpips_sim:.4f}")
    print(f"CLIP-t SD:    {overall_clip_t_sd:.4f}")
    print(f"CLIP-t Retain: {overall_clip_t_retain:.4f}")
    print(f"Total samples: {total_samples}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Retain Capability Evaluator',
        description='Evaluate retain capability of concept editing models compared to SD')
    
    parser.add_argument('--sd_folder', help='path to SD image folder', 
                        type=str, default="results/esd_multi-level/SD-v1-4/retaining_all-celebs")
    parser.add_argument('--retain_folder', help='path to edited model image folder', 
                        type=str, default="k-images/retain/retain_henry_cavill")
    parser.add_argument('--prompts_csv', help='path to prompts CSV file', 
                        type=str, default="diverse_prompts/celebs/multi-level/retaining_prompts/retaining_all-celebs.csv")
    parser.add_argument('--output_csv', help='path to save results CSV', 
                        type=str, default="results/retain_evaluation.csv")
    parser.add_argument('--output_txt', help='path to save summary TXT', 
                        type=str, default="results/retain_evaluation.txt")
    parser.add_argument('--clip_model', type=str, default='ViT-B/32',
                        help='CLIP model to use (e.g., ViT-B/32, ViT-L/14)')
    parser.add_argument('--device', type=str, default='cuda:4' if torch.cuda.is_available() else 'cpu',
                        help='Device to run evaluation on (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_txt), exist_ok=True)
    
    evaluate_retain(args) 
