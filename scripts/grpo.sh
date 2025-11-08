torchrun --nproc_per_node=1 --master_port=12345 grpo.py \
    --model "Qwen/Qwen2.5-0.5B-Instruct" \
    --dataset "trl-lib/tldr" \