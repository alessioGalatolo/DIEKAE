import argparse
from functools import partial
from os import path
from time import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
)

from datasets import EditDatasetTest, KnowledgeDataset
from encoder import Encoder, ModelArgs
from trainers import FineTuner
from model import CausalLlama4Encoders
from helpers import pad_longest


def main():
    parser = argparse.ArgumentParser(description="Train memory LLaMA")
    parser.add_argument(
        "--data-path", default="./data", help="Location of the data directory"
    )
    parser.add_argument(
        "--model-name",
        default="meta-llama/Llama-2-7b-hf",
        help="Model to use as decoder (LLM)",
        type=str,
    )
    parser.add_argument("--encoder-layers", nargs="+", default=list(range(3, 9)))
    parser.add_argument(
        "--enc-dim", type=int, help="Hidden dim of encoder", default=128
    )
    parser.add_argument(
        "--enc-n-layers", type=int, help="Number of encoder layers", default=4
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        dest="lr",
        help="The maximum learning rate",
    )
    parser.add_argument("--batch-size", type=int, help="Maximum batch size")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument(
        "--output", "-o", default="./checkpoints/encoder", help="Checkpoints output dir"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume previous run, put checkpoint path here",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        default="",
        help="Location of pretrained weights to load into the encoders",
    )
    parser.add_argument("--wandb", action="store_true", help="Use W&B")
    parser.add_argument("--debug", action="store_true", help="Debug mode, use CPU")
    args = parser.parse_args()

    config = {
        "model_name": args.model_name,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "virtual_batch_size": 64,
        "epochs": args.epochs,
        "enc_dim": args.enc_dim,
        "enc_n_layers": args.enc_n_layers,
        "encoder_layers": args.encoder_layers,
        "enc_seq_len": 4096,
    }

    DATASETS_NAMES = ["counterfact"]

    if args.resume:
        resume_dict = torch.load(args.resume, map_location="cpu")
        config = resume_dict["config"]
        print("Resuming previous run, using config:")
        print(config)

    torch.manual_seed(42)

    # get model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"  # easy model swap
    model = CausalLlama4Encoders.from_pretrained(
        model_name, device_map="cuda", use_cache=False, torch_dtype=torch.bfloat16
    )
    config["enc_seq_len"] = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = None
    tokenizer.pad_token_id = 0

    model.config.use_cache = False

    # setup encoder(s)
    encoders = [
        (
            Encoder(
                ModelArgs(
                    dim=config["enc_dim"],
                    n_layers=config["enc_n_layers"],
                    max_seq_len=model.config.max_position_embeddings,  # no reason in giving more
                ),
                model.get_input_embeddings(),
            )
            if i in config["encoder_layers"]
            else None
        )
        for i in range(model.config.num_hidden_layers)
    ]

    # load pretrained weights
    if args.resume:
        print("Resuming previous run, loading state dicts")
        for i, encoder in enumerate(filter(None, encoders)):
            encoder.load_state_dict(resume_dict[f"encoder{i}"])
        print("Loaded and ready!")

    # get training data
    total_loss = 0.0
    total_time = 0.0
    model.eval()
    for encoder in filter(None, encoders):
        encoder.eval().to(dtype=torch.bfloat16, device="cuda")
    with torch.inference_mode():
        for dataset_name in DATASETS_NAMES:
            data_test = EditDatasetTest(
                path.join(args.data_path, dataset_name), tokenizer, "train"
            )

            correct_efficacy = 0
            correct_paraphrase = 0
            number_paraphrases = 0
            correct_neighborhoods = 0
            number_neighborhood = 0
            pbar = tqdm(data_test)
            for step, (src, wanted_token, original_token, paraphrases, neighborhoods) in enumerate(pbar):
                enc_outs = []
                for encoder in encoders:
                    if encoder is None:
                        enc_outs.append(None)
                        continue
                    enc_out = encoder(src.to("cuda"))
                    enc_outs.append(enc_out)
                logits = model.forward(
                    src.to("cuda"),
                    encoded_knowledge=enc_outs,
                    return_dict=True
                )["logits"]

                # just like memit
                if logits[:, -1][0, wanted_token] > logits[:, -1][0, original_token]:
                    correct_efficacy += 1
                for paraphrase in paraphrases:
                    number_paraphrases += 1
                    enc_outs = []
                    for encoder in encoders:
                        if encoder is None:
                            enc_outs.append(None)
                            continue
                        enc_out = encoder(paraphrase.to("cuda"))
                        enc_outs.append(enc_out)
                    logits = model.forward(
                        paraphrase.to("cuda"),
                        encoded_knowledge=enc_outs,
                        return_dict=True
                    )["logits"]
                    if logits[:, -1][0, wanted_token] > logits[:, -1][0, original_token]:
                        correct_paraphrase += 1
                for neighborhood in neighborhoods:
                    number_neighborhood += 1
                    enc_outs = []
                    for encoder in encoders:
                        if encoder is None:
                            enc_outs.append(None)
                            continue
                        enc_out = encoder(neighborhood.to("cuda"))
                        enc_outs.append(enc_out)
                    logits = model.forward(
                        neighborhood.to("cuda"),
                        encoded_knowledge=enc_outs,
                        return_dict=True
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
    print("\nAverage loss: ", total_loss)
    print("Average perplexity: ", 2 ** total_loss, "\n\n")
    print(f"Total time taken: {total_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
