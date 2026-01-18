#!/bin/bash

model_name_or_path=Efficient-Large-Model/Fast_dLLM_v2_7B
dataset_path=data/alpaca/train_conversation
output_dir=output_models/finetune_fast_dLLM_7B-test
deepspeed_args="--master_port=11000"
conversation_template=fast_dllm_v2
export CUDA_HOME=/lustre/fsw/portfolios/nvr/users/chengyuew/anaconda3/envs/lmflow

trust_remote_code=1

latest_checkpoint=""
if [ -d "${output_dir}" ]; then
    latest_checkpoint=$(find "${output_dir}" -name "checkpoint-*" -type d | sort -V | tail -1)
    if [ -n "${latest_checkpoint}" ]; then
        echo "Found latest checkpoint: ${latest_checkpoint}"
    else
        echo "No checkpoint found in ${output_dir}"
        latest_checkpoint=""
    fi
else
    echo "Output directory ${output_dir} does not exist, training from scratch"
    latest_checkpoint=""
fi

resume_arg=""
if [ -n "${latest_checkpoint}" ]; then
    resume_arg="--resume_from_checkpoint ${latest_checkpoint}"
fi

cmd="deepspeed ${deepspeed_args} \
  train_scripts/finetune.py \
    --model_name_or_path ${model_name_or_path} \
    --trust_remote_code ${trust_remote_code} \
    --dataset_path ${dataset_path} \
    --output_dir ${output_dir} \
    ${resume_arg} \
    --conversation_template ${conversation_template} \
    --num_train_epochs 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant_with_warmup \
    --warmup_ratio 0.03 \
    --disable_group_texts 0 \
    --block_size 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --deepspeed configs/ds_config_zero2_no_offload.json \
    --bf16 \
    --run_name finetune \
    --validation_split_percentage 0 \
    --logging_steps 1 \
    --do_train \
    --ddp_timeout 72000 \
    --save_steps 1000 \
    --dataloader_num_workers 8 \
    --preprocessing_num_workers 32 \
    --save_total_limit 10 \
    --gradient_checkpointing 1 

echo $cmd
eval $cmd
