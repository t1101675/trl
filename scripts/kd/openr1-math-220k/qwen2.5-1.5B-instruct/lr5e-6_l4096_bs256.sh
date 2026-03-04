#!/bin/bash

if [ -n "$SLURM_JOB_ID" ] && [ ! -t 0 ]; then
    # on nv cluster
    nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
    nodes_array=($nodes)
    head_node=${nodes_array[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

    export LOGLEVEL=INFO

    export PATH="$HOME/workspace/anaconda3/envs/trl/bin:$PATH"
    CODEDIR="$HOME/workspace/code/trl"
    cd $CODEDIR
else
    SLURM_NNODES=1
    head_node_ip=localhost
fi

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
STUDENT_MODEL="pretrained_models/Qwen_Qwen2.5-1.5B-Instruct"
TEACHER_MODEL="pretrained_models/Qwen_Qwen2.5-7B-Instruct"
DATASET_PATH="dataset/llm_rl/OpenR1-Math-220k"

GROUP_NAME=kd/openr1-math-220k/qwen2.5-1.5B-instruct/lr5e-6_l4096_bs256
JOB_TYPE=train
RUN_NAME=${JOB_TYPE}/${GROUP_NAME}
OUTPUT_DIR=results/${RUN_NAME}

# Tensorboard logging directory (override with TB_LOG_DIR env var)
TB_LOG_DIR=${TB_LOG_DIR:-${OUTPUT_DIR}/tb_logs}

NUM_PROCESS=$((gpu_count * SLURM_NNODES))

BATCH_SIZE=256
MICRO_BATCH_SIZE=2
GRAD_ACC=$((BATCH_SIZE / MICRO_BATCH_SIZE / NUM_PROCESS))

read -r -d '' cmd <<EOF
accelerate launch --config_file=trl/accelerate_configs/zero1.yaml --num_processes $NUM_PROCESS \
--num_machines $SLURM_NNODES --rdzv_backend c10d --main_process_ip $head_node_ip --main_process_port 29500 \
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
    --max_seq_length 4096
EOF

if [ -n "$SLURM_JOB_ID" ] && [ ! -t 0 ]; then
    echo ${cmd}
    srun bash -c "${cmd}"
else
    echo ${cmd}
    bash -c "${cmd}"
fi
