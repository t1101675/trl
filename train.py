from trl import (
    SFTTrainer,
    GRPOTrainer,
    GKDTrainer,
)
from datasets import load_dataset
from argparse import ArgumentParser

from transformers import AutoTokenizer


def init_sft_trainer(model_name, dataset):
    trainer = SFTTrainer(
        model=model_name,
        train_dataset=dataset,
    )
    return trainer


def init_grpo_trainer(model_name, dataset, reward_func):
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=reward_func,
        train_dataset=dataset,
    )
    return trainer


def init_gkd_trainer(model_name, teacher_model_name, dataset):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    trainer = GKDTrainer(
        model=model_name,
        processing_class=tokenizer,
        teacher_model=teacher_model_name,
        train_dataset=dataset,
    )
    return trainer


def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]


def main():
    parser = ArgumentParser()
    parser.add_argument("--trainer", type=str, default="sft")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="nvidia/Nemotron-Post-Training-Dataset-v2")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split=args.split)

    if args.trainer == "sft":
        trainer = init_sft_trainer(args.model, dataset)
    elif args.trainer == "grpo":
        trainer = init_grpo_trainer(args.model, dataset, reward_num_unique_chars)
    elif args.trainer == "gkd":
        trainer = init_gkd_trainer(args.model, args.teacher_model, dataset)
    else:
        raise ValueError(f"Unknown trainer type: {args.trainer}")

    trainer.train()
    


if __name__ == "__main__":
    main()