from trl import GRPOTrainer
from datasets import load_dataset
from argparse import ArgumentParser


# Dummy reward function: count the number of unique characters in the completions
def reward_num_unique_chars(completions, **kwargs):
    return [len(set(c)) for c in completions]


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", type=str, default="trl-lib/tldr")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split="train")

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_num_unique_chars,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()