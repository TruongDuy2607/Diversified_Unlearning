export CUDA_VISIBLE_DEVICES=0

# export CONCEPT="human_20_prompts_2"
# CUDA_VISIBLE_DEVICES=1 python src/train_ace_diverse.py \
#   --prompt_csv "data/concept_csv/$CONCEPT.csv" \
#   --surrogate '' \
#   --train_method 'full' \
#   --devices '0' \
#   --iterations 1500 \
#   --change_step_rate 1 \
#   --lr 0.001 \
#   --negative_guidance 3 \
#   --surrogate_guidance 3 \
#   --ddim_steps 30 \
#   --anchor_prompt_path "data/concept_text/nudity_anchor_concept.txt" \
#   --anchor_batch_size 2 \
#   --pl_weight 0.8 \
#   --null_weight 0.99 \
#   --is_train_null \
#   --with_prior_preservation \
#   --no_certain_sur 

export CSV="coco_30k"
export ADD_NAME="_256"
export OUTPUT_DIR="evaluation-outputs/$CSV$ADD_NAME"
export MODEL_NAME="ACE_lora_human_20_prompts_clothes-sc_-ng_3.0-iter_1500-lr_0.001-lora-prior_2_tr_null_True_nc_False_no_cer_sur_True_tensor_False_nw_0.99_pl_0.8_sg_new_3.0_is_sc_clip_False"
accelerate launch --config_file config.yaml src/generate_images_lora.py \
  --model_name "${MODEL_NAME}" \
  --prompts_path "data/concept_csv/$CSV.csv" \
  --generate_concept_path "data/concept_text/nudity_anchor_concept.txt"\
  --save_path="$OUTPUT_DIR" \
  --image_size 256 \
  --ddim_steps 30 \
  --num_samples 1 \
  --multipliers 1 \
  --lora_rank 4 \
  --is_lora \
  --lora_name "ACE_lora_human_20_prompts_clothes-sc_-ng_3.0-iter_1500-lr_0.001-lora-prior_2_tr_null_True_nc_False_no_cer_sur_True_tensor_False_nw_0.99_pl_0.8_sg_new_3.0_is_sc_clip_False"