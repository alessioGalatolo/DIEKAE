import argparse
from os import environ, path
from peft.peft_model import PeftModel
from tqdm import tqdm
from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer
import torch
from lora_datasets import EditDatasetTest


def main():
    parser = argparse.ArgumentParser(description="Train memory LLaMA with LoRA")
    parser.add_argument(
        "--data-path", default="./data", help="Location of the data directory"
    )
    args = parser.parse_args()

    DATASETS_NAMES = ["counterfact"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", device_map=device, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, "checkpoints/lora/edit", torch_dtype=torch.bfloat16)
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = None
    tokenizer.pad_token_id = 0

    model.to(device)
    model.eval()
    with torch.no_grad():

        for dataset in DATASETS_NAMES:
            data_test = EditDatasetTest(
                path.join(args.data_path, dataset),
                tokenizer,
                "train",
            )

            correct_efficacy = 0
            correct_paraphrase = 0
            number_paraphrases = 0
            correct_neighborhoods = 0
            number_neighborhood = 0
            pbar = tqdm(data_test)
            for step, (src, wanted_token, original_token, paraphrases, neighborhoods) in enumerate(pbar):
                logits = model.forward(
                    src.to(device),
                    return_dict=True
                )["logits"]

                # just like memit
                if logits[:, -1][0, wanted_token] > logits[:, -1][0, original_token]:
                    correct_efficacy += 1
                for paraphrase in paraphrases:
                    number_paraphrases += 1

                    logits = model.forward(
                        paraphrase.to(device)
                    )["logits"]
                    if logits[:, -1][0, wanted_token] > logits[:, -1][0, original_token]:
                        correct_paraphrase += 1
                for neighborhood in neighborhoods:
                    number_neighborhood += 1
                    logits = model.forward(
                        neighborhood.to(device)
                    )["logits"]
                    if logits[:, -1][0, wanted_token] < logits[:, -1][0, original_token]:
                        correct_neighborhoods += 1
                pbar.set_description("Efficacy: {:.2f} Paraphrase: {:.2f} Neighborhood: {:.2f}".format(
                    correct_efficacy / (step+1) * 100,
                    correct_paraphrase / number_paraphrases * 100,
                    correct_neighborhoods / number_neighborhood * 100
                ))

            print("Efficacy: ", correct_efficacy / (step+1) * 100)
            print("Paraphrase: ", correct_paraphrase / number_paraphrases * 100)
            print("Neighborhood: ", correct_neighborhoods / number_neighborhood * 100)


if __name__ == "__main__":
    main()
