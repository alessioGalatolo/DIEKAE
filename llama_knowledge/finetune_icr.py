import argparse
from functools import partial
from os import path

import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding
)

from datasets import ICRDataset
from encoder import Encoder, ModelArgs
from trainers import FineTunerICR
from model import ClassificationLlama4Encoders
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
    parser.add_argument("--encoder-layers", nargs="+", type=int, default=list(range(3, 9)))
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
        "batch_size": 1,  # forced because I'm lazy
        "virtual_batch_size": 64,
        "epochs": args.epochs,
        "enc_dim": args.enc_dim,
        "enc_n_layers": args.enc_n_layers,
        "encoder_layers": args.encoder_layers,
        "enc_seq_len": 4096,
    }

    DATASETS_NAMES = ["folio", "proofwriter"]

    if args.resume:
        resume_dict = torch.load(args.resume, map_location="cpu")
        config = resume_dict["config"]
        print("Resuming previous run, using config:")
        print(config)

    grad_accum_steps = (
        2 if args.debug else config["virtual_batch_size"] // config["batch_size"]
    )

    torch.manual_seed(42)

    # get model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"  # easy model swap
    model = ClassificationLlama4Encoders.from_pretrained(
        model_name, device_map="cpu", use_cache=False, label2id=ICRDataset.token_to_number, id2label=ICRDataset.number_to_token, num_labels=len(ICRDataset.token_to_number)
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
    if args.pretrained_weights:
        pretrained_weights = torch.load(args.pretrained_weights, map_location="cpu")
        for i, encoder in enumerate(encoders):
            if encoder is not None:
                # NOTE: pretrained weights are for all layers 0-31 but this training may only use a few of them
                encoder.load_state_dict(pretrained_weights[f"encoder{i}"])

    # get training data
    data_train = []
    data_eval = []
    for dataset_name in DATASETS_NAMES:
        data_train.append(
            ICRDataset(
                path.join(args.data_path, dataset_name), tokenizer, "train"
            )
        )
        data_eval.append(
            ICRDataset(path.join(args.data_path, dataset_name), tokenizer, "val")
        )

    data_train = ConcatDataset(data_train)
    data_eval = ConcatDataset(data_eval)
    data_train = DataLoader(
        data_train,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer, max_length=1024, padding=False),
    )
    data_eval = DataLoader(
        data_eval,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=DataCollatorWithPadding(tokenizer, max_length=1024, padding=False),
    )

    # freeze llama
    for name, param in model.named_parameters():
        if "score" in name:
            param.requires_grad = True
        param.requires_grad = False

    # prepare encoders
    for encoder in filter(None, encoders):
        for name, param in encoder.named_parameters():
            if "embed" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    print(
        "Model trainable parameters: ",
        sum(
            p.numel()
            for encoder in filter(None, encoders)
            for p in encoder.parameters()
            if p.requires_grad
        ),
    )

    ##############
    # Finetuning #
    ##############
    FineTunerICR(
        model,
        encoders,
        data_train,
        data_eval,
        grad_accum_steps,
        lr=config["lr"],
        epochs=config["epochs"],
        resume=args.resume,
        output=args.output,
        use_cpu=args.debug,
        use_wandb=args.wandb,
        config=config,
    ).train()


if __name__ == "__main__":
    main()
