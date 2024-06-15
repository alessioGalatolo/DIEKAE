# Eval plain PLM on test sets
import argparse
from functools import partial
from os import path
from transformers import Trainer, TrainingArguments
from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer
import torch
from hf_datasets import KnowledgeDataset, pad_longest


def main():
    parser = argparse.ArgumentParser(description="Train memory LLaMA with LoRA")
    parser.add_argument(
        "--data-path", default="./data", help="Location of the data directory"
    )
    args = parser.parse_args()

    DATASETS_NAMES = ["cmu_dog", "curio", "dream", "quasar_t", "wow"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", device_map=device, torch_dtype=torch.bfloat16
    )
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = None
    tokenizer.pad_token_id = 0

    print("\n", "#" * 10, "Evaluating with knowledge to PLM", "#" * 10, "\n")
    total_time = 0.0
    total_loss = 0.0
    for dataset in DATASETS_NAMES:
        data_test = KnowledgeDataset(
            path.join(args.data_path, dataset),
            tokenizer,
            "test",
            limit_number_of_samples=1000
        )

        data_collator = partial(
            pad_longest, pad_token_id=tokenizer.pad_token_id, max_tokens=1024
        )

        trainer = Trainer(
            model=model,
            eval_dataset=data_test,
            data_collator=data_collator,
            args=TrainingArguments(report_to="none", output_dir="tmp_trainer")
        )
        result = trainer.evaluate()
        total_time += result["eval_runtime"]
        print(f"Evaluation results for {dataset}:\n", result)
        print("Perplexity: ", 2 ** result["eval_loss"], "\n\n")
        total_loss += result["eval_loss"] / len(DATASETS_NAMES)
    print("\nAverage loss: ", total_loss)
    print("Average perplexity: ", 2 ** total_loss, "\n\n")
    print(f"Total time taken: {total_time:.2f} seconds\n")
    total_time = 0.0

    print("\n", "#" * 10, "Evaluating WITHOUT knowledge to PLM", "#" * 10, "\n")
    for dataset in DATASETS_NAMES:
        data_test = KnowledgeDataset(
            no_knowledge=True,
            data_path=path.join(args.data_path, dataset),
            tokenizer=tokenizer,
            split="test",
            limit_number_of_samples=1000
        )

        data_collator = partial(
            pad_longest, pad_token_id=tokenizer.pad_token_id, max_tokens=1024
        )

        trainer = Trainer(
            model=model,
            eval_dataset=data_test,
            data_collator=data_collator,
            args=TrainingArguments(report_to="none", output_dir="tmp_trainer")
        )
        result = trainer.evaluate()
        total_time += result["eval_runtime"]
        print(f"Evaluation results for {dataset}:\n", result)
        print("Perplexity: ", 2 ** result["eval_loss"], "\n\n")
        total_loss += result["eval_loss"] / len(DATASETS_NAMES)
    print("\nAverage loss: ", total_loss)
    print("Average perplexity: ", 2 ** total_loss, "\n\n")
    print(f"Total time taken: {total_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
