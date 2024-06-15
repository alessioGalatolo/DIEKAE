# Eval LoRA on test sets
import argparse
from functools import partial
from os import environ, path
import numpy as np
from sympy import false
from transformers import Trainer, TrainingArguments
from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer
import torch
from tqdm import tqdm

from hf_datasets import ICRDataset


def main():
    parser = argparse.ArgumentParser(description="Train memory LLaMA with LoRA")
    parser.add_argument(
        "--data-path", default="./data", help="Location of the data directory"
    )
    args = parser.parse_args()

    DATASETS_NAMES = ["folio", "proofwriter"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", device_map=device, torch_dtype=torch.bfloat16
    )
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = None
    tokenizer.pad_token_id = 0

    print("\n", "#" * 10, "Evaluating with knowledge to PLM", "#" * 10, "\n")
    total_time = 0.0
    for dataset in DATASETS_NAMES:
        data_test = ICRDataset(
            path.join(args.data_path, dataset),
            tokenizer,
            "val",  # folio doesn't have a test set
            limit_number_of_samples=1000
        )

        un_tok = 853
        true_token = 5852
        false_token = 7700
        count = 0
        with torch.no_grad():
            with torch.inference_mode():
                for step, data in enumerate(tqdm(data_test)):
                    if step == 1000:
                        break
                    # This code below is working but it's really ugly. Note to self: change the code, do not forget.
                    logits = model.forward(torch.tensor([data["input_ids"]], dtype=torch.long, device="cuda"), return_dict=True)["logits"]
                    p = logits[0][-1]
                    if p[un_tok] > p[false_token] and p[un_tok] > p[true_token]:
                        if data["labels"] == un_tok:
                            count += 1
                    elif p[true_token] > p[false_token]:
                        if data["labels"] == true_token:
                            count += 1
                    else:
                        if data["labels"] == false_token:
                            count += 1

        print(f"Evaluation results for {dataset}:\n", count / len(data_test))
    print(f"Total time taken: {total_time:.2f} seconds\n")

    print("\n", "#" * 10, "Evaluating without knowledge to PLM", "#" * 10, "\n")
    total_time = 0.0
    for dataset in DATASETS_NAMES:
        data_test = ICRDataset(
            path.join(args.data_path, dataset),
            tokenizer,
            "val",  # folio doesn't have a test set
            no_knowledge=True,
            limit_number_of_samples=1000
        )

        un_tok = 853
        true_token = 5852
        false_token = 7700
        count = 0
        with torch.no_grad():
            with torch.inference_mode():
                for step, data in enumerate(tqdm(data_test)):
                    if step == 1000:
                        break
                    # This code below is working but it's really ugly. Note to self: change the code, do not forget.
                    logits = model.forward(torch.tensor([data["input_ids"]], dtype=torch.long, device="cuda"), return_dict=True)["logits"]
                    p = logits[0][-1]
                    if p[un_tok] > p[false_token] and p[un_tok] > p[true_token]:
                        if data["labels"] == un_tok:
                            count += 1
                    elif p[true_token] > p[false_token]:
                        if data["labels"] == true_token:
                            count += 1
                    else:
                        if data["labels"] == false_token:
                            count += 1
        print(f"Evaluation results for {dataset}:\n", count / len(data_test))

    print(f"Total time taken: {total_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
