torchrun --nproc_per_node=1 --master_port=12345 train.py \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --teacher_model "Qwen/Qwen2.5-1.5B-Instruct" \
    --dataset "nvidia/Nemotron-Post-Training-Dataset-v2" \
    --split "chat" \
    --trainer "gkd"