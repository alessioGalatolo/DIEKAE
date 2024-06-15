# Eval LoRA on test sets
import argparse
from functools import partial
from os import environ, path
import numpy as np
from sympy import limit
from transformers import Trainer, TrainingArguments
from transformers.models.llama import LlamaForSequenceClassification, LlamaTokenizer
import torch
from torch.utils.data import ConcatDataset
from peft.peft_model import PeftModel

from lora_datasets import ICRDataset, KnowledgeDataset, pad_longest


def main():
    parser = argparse.ArgumentParser(description="Train memory LLaMA with LoRA")
    parser.add_argument(
        "--data-path", default="./data", help="Location of the data directory"
    )
    args = parser.parse_args()

    DATASETS_NAMES = ["folio", "proofwriter"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LlamaForSequenceClassification.from_pretrained(
        "meta-llama/Llama-2-7b-hf", device_map=device, torch_dtype=torch.bfloat16, label2id=ICRDataset.token_to_number, id2label=ICRDataset.number_to_token, num_labels=len(ICRDataset.token_to_number)
    )
    model = PeftModel.from_pretrained(model, "checkpoints/lora/icr", torch_dtype=torch.bfloat16)
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = None
    tokenizer.pad_token_id = 0

    print("\n", "#" * 10, "Evaluating with knowledge to LoRA", "#" * 10, "\n")
    total_time = 0.0
    for dataset in DATASETS_NAMES:
        data_test = ICRDataset(
            path.join(args.data_path, dataset),
            tokenizer,
            "val",  # folio doesn't have a test set
            limit_number_of_samples=1000
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": (predictions == labels).sum() / len(labels)}

        trainer = Trainer(
            model=model,
            eval_dataset=data_test,
            # data_collator=data_collator,
            compute_metrics=compute_metrics,
            args=TrainingArguments(report_to="none", output_dir="tmp_trainer", per_device_eval_batch_size=1)
        )

        result = trainer.evaluate()
        total_time += result["eval_runtime"]
        print(f"Evaluation results for {dataset}:\n", result)
    print(f"Total time taken: {total_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
