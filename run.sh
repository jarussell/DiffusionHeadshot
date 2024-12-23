#!/usr/bin/env bash

accelerate launch --config_file=accelerate.yaml \
  train_dreambooth_lora_flux_miniature.py \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --data_df_path="embeddings.parquet" \
  --output_dir="russell_family_lora_flux_nf4" \
  --mixed_precision="bf16" \
  --use_8bit_adam \
  --weighting_scheme="none" \
  --resolution=1024 \
  --train_batch_size=1 \
  --repeats=1 \
  --learning_rate=1e-4 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --cache_latents \
  --rank=4 \
  --max_train_steps=700 \
  --seed="0"
