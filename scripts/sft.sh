torchrun --nproc_per_node=1 --master_port=12345 train.py \
    --model "Qwen/Qwen2.5-0.5B" \
    --dataset "trl-lib/Capybara" \
    --trainer "sft"