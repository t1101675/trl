#!/bin/bash

export OMP_NUM_THREADS=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ -z $CUDA_VISIBLE_DEVICES ]]; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    gpu_count=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
echo "Available GPU count: $gpu_count"

# ============================================================
# Paths — local pre-downloaded models and data
# ============================================================
STUDENT_MODEL="pretrained_models/Qwen2.5-1.5B"
TEACHER_MODEL="pretrained_models/Qwen2.5-7B-Instruct"
DATASET_PATH="dataset/OpenR1-Math-220k"

GROUP_NAME=kd/openr1-math-220k/qwen2.5-1.5B-instruct/lr5e-6_l4096_bs256
JOB_TYPE=train
RUN_NAME=${JOB_TYPE}/${GROUP_NAME}
OUTPUT_DIR=results/${RUN_NAME}

# Tensorboard logging directory (override with TB_LOG_DIR env var)
TB_LOG_DIR=${TB_LOG_DIR:-${OUTPUT_DIR}/tb_logs}

BATCH_SIZE=256
MICRO_BATCH_SIZE=2
GRAD_ACC=$((BATCH_SIZE / MICRO_BATCH_SIZE / gpu_count))

read -r -d '' cmd <<EOF
torchrun --nproc_per_node $gpu_count \
trl/scripts/kd.py \
    --model_name_or_path $STUDENT_MODEL \
    --dtype bfloat16 \
    --teacher_model_name_or_path $TEACHER_MODEL \
    --teacher_dtype bfloat16 \
    --dataset_name $DATASET_PATH \
    --kd_alpha 0.5 \
    --kd_temperature 1.0 \
    --learning_rate 5.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps 100 \
    --num_train_epochs 1 \
    --per_device_train_batch_size $MICRO_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --gradient_checkpointing \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 500 \
    --output_dir ${OUTPUT_DIR} \
    --attn_implementation flash_attention_2 \
    --resume_from_checkpoint True \
    --report_to tensorboard \
    --logging_dir ${TB_LOG_DIR} \
    --logging_steps 1 \
    --logging_first_step true \
    --dataset_num_proc 64 \
    --max_seq_length 4096 \
    --deepspeed trl/accelerate_configs/zero1_ds.json
EOF

echo ${cmd}
bash -c "${cmd}"
