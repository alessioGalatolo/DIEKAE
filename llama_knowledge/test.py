import argparse
from functools import partial
from os import path
from time import time

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
)

from datasets import KnowledgeDataset
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

    DATASETS_NAMES = ["cmu_dog", "curio", "dream", "quasar_t", "wow"]

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
        model_name, device_map="cpu", use_cache=False, torch_dtype=torch.bfloat16
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

    # load pretrained weights
    if args.pretrained_weights:
        pretrained_weights = torch.load(args.pretrained_weights, map_location="cpu")
        for i, encoder in enumerate(encoders):
            if encoder is not None:
                # NOTE: pretrained weights are for all layers 0-31 but this training may only use a few of them
                encoder.load_state_dict(pretrained_weights[f"encoder{i}"])

    # get training data
    total_loss = 0.0
    total_time = 0.0
    for dataset_name in DATASETS_NAMES:
        data_test = KnowledgeDataset(
            path.join(args.data_path, dataset_name), tokenizer, "test", limit_number_of_samples=1000
        )

        data_test = DataLoader(
            data_test,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=partial(
                pad_longest, pad_token_id=tokenizer.pad_token_id, max_tokens=1024
            ),
        )

        # relying on the class but only evaluating
        trainer = FineTuner(
            model,
            encoders,
            None,
            data_test,
            grad_accum_steps,
            lr=config["lr"],
            epochs=config["epochs"],
            resume=args.resume,
            output=args.output,
            use_cpu=args.debug,
            use_wandb=args.wandb,
            config=config,
        )
        t0 = time()
        result = trainer.evaluate()
        torch.cuda.synchronize()
        total_time += time() - t0
        print(f"Evaluation results for {dataset_name}:\n")
        print("CE loss: ", result)
        print("Perplexity: ", 2 ** result, "\n\n")
        total_loss += result / len(DATASETS_NAMES)
    print("\nAverage loss: ", total_loss)
    print("Average perplexity: ", 2 ** total_loss, "\n\n")
    print(f"Total time taken: {total_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
