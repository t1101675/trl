from trl import SFTTrainer
from datasets import load_dataset
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset", type=str, default="trl-lib/Capybara")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split="train")

    trainer = SFTTrainer(
        model=args.model,
        train_dataset=dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()