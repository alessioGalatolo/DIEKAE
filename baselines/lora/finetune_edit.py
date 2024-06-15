import argparse
from functools import partial
from os import environ, path
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments
from transformers.models.llama import LlamaForCausalLM, LlamaTokenizer
import torch
from torch.utils.data import ConcatDataset
from lora_datasets import EditDataset, pad_longest
from transformers import EarlyStoppingCallback


def main():
    parser = argparse.ArgumentParser(description="Train memory LLaMA with LoRA")
    parser.add_argument(
        "--data-path", default="./data", help="Location of the data directory"
    )
    parser.add_argument(
        "--output", "-o", default="./out", help="Checkpoints output dir"
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        type=float,
        dest="lr",
        help="The maximum learning rate",
    )
    parser.add_argument("--batch-size", type=int, help="Maximum batch size")
    parser.add_argument(
        "--virtual-batch-size",
        type=int,
        default=64,
        help="Virtual batch size, achieved through gradient accoumulation",
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--wandb", action="store_true", help="Use W&B")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug and use cpu")

    """
    r values to obtain similar number of trainable parameters:

    all layers -> r=128
    converged layers -> r=116
    best layers -> r=56
    memit layers -> r=24
    """
    parser.add_argument("-r", type=int, default=24, help="Rank for lora")
    args = parser.parse_args()
    DATASETS_NAMES = ["counterfact"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "LOCAL_RANK" in environ:
        device = int(environ["LOCAL_RANK"])
    if args.debug:
        device = "cpu"
    lora_config = LoraConfig(
        r=args.r,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=32,
        lora_dropout=0.05,
    )
    base_model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", device_map=device
    )
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = None
    tokenizer.pad_token_id = 0

    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    lora_model.train(True)

    data_train = []
    data_eval = []
    for dataset_name in DATASETS_NAMES:
        data_train.append(
            EditDataset(
                path.join(args.data_path, dataset_name),
                tokenizer,
                "train",
            )
        )
        data_eval.append(
            EditDataset(
                path.join(args.data_path, dataset_name),
                tokenizer,
                "train",
            )
        )
    dataset = ConcatDataset(data_train)
    eval = ConcatDataset(data_eval)

    data_collator = partial(
        pad_longest, pad_token_id=tokenizer.pad_token_id, max_tokens=1024
    )

    trainer = Trainer(
        model=lora_model,
        train_dataset=dataset,
        data_collator=data_collator,
        eval_dataset=eval,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        args=TrainingArguments(
            greater_is_better=False,  # needed for early stopping (lower loss is better)
            use_cpu=args.debug,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.virtual_batch_size // args.batch_size,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            bf16=True,
            fp16=False,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=250,
            save_steps=250,
            output_dir=args.output,
            save_total_limit=3,
            load_best_model_at_end=True,
            group_by_length=False,
            report_to="wandb" if args.wandb else "none",
            run_name=(
                (environ["RUN_NAME"] if "RUN_NAME" in environ else None)
                if args.wandb
                else None
            ),
        ),
    )
    trainer.train()
    lora_model.save_pretrained(path.join(args.output, f"llama2-lora-edit-r{args.r}"))


if __name__ == "__main__":
    main()
