import argparse
from functools import partial
from os import path

import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import (
    AutoTokenizer,
)

from datasets import KnowledgeDataset
from encoder import Encoder, ModelArgs
from trainers import Trainer
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

    DATASETS_NAMES = ["cmu_dog", "curio", "dream", "nat_ques_short", "quasar_t", "wow"]

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
    model = CausalLlama4Encoders.from_pretrained(
        model_name, device_map="cpu", torch_dtype=torch.bfloat16
    )
    config["enc_seq_len"] = model.config.max_position_embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = None
    tokenizer.pad_token_id = 0

    # make sure the model is correctly loaded and generates sensical output
    # everything is in the cpu so it's slow af
    print(
        "Model loaded, testing generation:"
        + tokenizer.decode(
            model.generate(
                **tokenizer(["My name is Teven and I am"], return_tensors="pt"),
                max_new_tokens=10
            )[0]
        )
    )
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

    # get training data
    data_train = []
    data_eval = []
    for dataset_name in DATASETS_NAMES:
        data_train.append(
            KnowledgeDataset(
                path.join(args.data_path, dataset_name), tokenizer, "train"
            )
        )
        data_eval.append(
            KnowledgeDataset(path.join(args.data_path, dataset_name), tokenizer, "val")
        )
    data_train = ConcatDataset(data_train)
    data_eval = ConcatDataset(data_eval)
    data_train = DataLoader(
        data_train,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=partial(
            pad_longest, pad_token_id=tokenizer.pad_token_id, max_tokens=1024
        ),  # OOM on 40GB without max_tokens, dunno the limit
    )
    data_eval = DataLoader(
        data_eval,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=partial(
            pad_longest, pad_token_id=tokenizer.pad_token_id, max_tokens=1024
        ),
    )

    # freeze llama
    for name, param in model.named_parameters():
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

    ############
    # TRAINING #
    ############
    Trainer(
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
