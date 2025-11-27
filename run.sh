# Train ESD
CUDA_VISIBLE_DEVICES=0 python3 train-scripts/ESD/train-esd.py --train_method "xattn" --ckpt_path "models/erase/sd-v1-4-full-ema.ckpt" --diffusers_config_path "models/erase/config.json" --prompt "henry cavill" --config_path "configs/stable-diffusion/v1-inference.yaml" --info "none"

# Train ESD diverse
# Level-1
CUDA_VISIBLE_DEVICES=0 python3 train-scripts/ESD/train-esd-diverse.py --prompt_csv "diverse_prompts/celebs/multi-level/training_prompts/level1_henry-cavill.csv" --seperator "," --train_method "xattn" --ckpt_path "models/erase/sd-v1-4-full-ema.ckpt" --diffusers_config_path "models/erase/config.json" --prompt "henry cavill" --config_path "configs/stable-diffusion/v1-inference.yaml" --info "none" --level "level-1"

# Train UCE
CUDA_VISIBLE_DEVICES=0 python3 train-scripts/UCE/train-uce.py --prompt 'mario' --guided_concepts 'plumber' --concept_type 'object' --name 'uce'

# Train UCE Diverse (Token mixup)
CUDA_VISIBLE_DEVICE=0 python3 train-scripts/UCE/train-uce-diverse.py --prompt_csv diverse_prompts/characters/training_prompts/token-05-uce-training-mario.csv --name 'uce-diverse' --level 'character'

# Train AP
CUDA_VISIBLE_DEVICES=0 python3 train-scripts/AP/train-ap.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'henry cavill' \
    --seperator ',' \
    --train_method 'xattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path 'models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path 'models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 1000 \
    --info 'normal_gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --neutral_prompt "a man" \
    --name 'henry-cavill'

# Train AP Diverse
CUDA_VISIBLE_DEVICES=0 python3 train-scripts/AP/train-ap-diverse.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'henry cavill' \
    --prompt
    --seperator ',' \
    --train_method 'xattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path 'models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path 'models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 1000 \
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --name 'henry-cavill'
    
# Train AGE
CUDA_VISIBLE_DEVICES=0 python3 train-scripts/AGE/train-age.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'henry cavill' \
    --train_method 'xattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path 'models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path 'models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 1000 \
    --vocab "EN3K" \
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --name 'henry-cavill'

# Train AGE diverse
CUDA_VISIBLE_DEVICES=0 python3 train-scripts/AGE/train-age-diverse.py \
    --save_freq 50 \
    --models_path=models \
    --prompt 'Henry Cavill' \
    --prompt_path 'diverse_prompts/celebs/multi-level/training_prompts/level1_henry-cavill.csv' \
    --seperator ',' \
    --train_method 'xattn' \
    --config_path 'configs/stable-diffusion/v1-inference.yaml' \
    --ckpt_path 'models/erase/sd-v1-4-full-ema.ckpt' \
    --diffusers_config_path 'models/erase/config.json' \
    --lr 1e-5 \
    --gumbel_lr 1e-2 \
    --gumbel_temp 2 \
    --gumbel_hard 1 \
    --gumbel_num_centers 100 \
    --gumbel_update -1 \
    --gumbel_time_step 0 \
    --gumbel_multi_steps 2 \
    --gumbel_k_closest 1000 \
    --info 'gumbel_lr_1e-2_temp_2_hard_1_num_100_update_-1_timestep_0_multi_2_kclosest_1000' \
    --name 'henry-cavill'

