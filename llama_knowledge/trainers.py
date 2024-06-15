from datetime import datetime
import importlib
from os import environ, makedirs, path
from shutil import rmtree
from typing import List

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import LlamaPreTrainedModel, get_cosine_schedule_with_warmup
from accelerate import Accelerator

from encoder import Encoder, EncodersMoK
from helpers import zero_rows, shift_rows, EarlyStopping
import warnings


class Trainer:
    def __init__(
        self,
        model: LlamaPreTrainedModel,
        encoders: List[Encoder | None],
        data_train: DataLoader,
        data_eval: DataLoader = None,
        grad_accum_steps: int = 1,
        lr: float = 1e-4,
        epochs: int = 5,
        resume: str = None,
        output: str = "./checkpoints",
        use_cpu: bool = False,
        use_wandb: bool = False,
        config: dict = dict(),
    ) -> None:
        warnings.warn("This class implements a different early stopper compared to the others. Training won't stop but a checkpoint will be saved.")
        self.accelerator = Accelerator(
            gradient_accumulation_steps=grad_accum_steps,
            mixed_precision="bf16",
            cpu=use_cpu,
        )
        self.model = model
        self.encoders = encoders
        self.epochs = epochs
        self.output = output
        if self.accelerator.is_main_process:
            self.early_stopper = EarlyStopping(patience=5)

        self._create_optimizers(lr)
        self._create_schedulers(
            data_train.batch_size * grad_accum_steps, len(data_train)
        )
        self.last_checkpoints = (
            []
        )  # used to delete old checkpoints while run is continuing
        self.epoch_0 = 0
        self.step_0 = 0
        self.train_step = 0
        self.eval_step = 0
        self.data_train, self.data_eval = self.accelerator.prepare(
            data_train, data_eval
        )
        self.optimizers = [
            self.accelerator.prepare(optimizer) for optimizer in self.optimizers
        ]
        self.schedulers = [
            self.accelerator.prepare(scheduler) for scheduler in self.schedulers
        ]
        self.model = self.accelerator.prepare(self.model)
        self.encoders = [
            self.accelerator.prepare(encoder) if encoder is not None else None
            for encoder in encoders
        ]
        if resume:
            print("Resuming previous run, loading state dicts")
            resume_dict = torch.load(resume, map_location="cpu")
            for i, encoder in enumerate(filter(None, encoders)):
                self.schedulers[i].load_state_dict(resume_dict[f"scheduler{i}"])
                self.optimizers[i].load_state_dict(resume_dict[f"optimizer{i}"])
                encoder.load_state_dict(resume_dict[f"encoder{i}"])

            self.epoch_0 = resume_dict["epoch"]
            self.step_0 = resume_dict["step"]
            print("Loaded and ready!")
        self.wandb = use_wandb and self.accelerator.is_main_process

        if self.wandb:
            self.wandb = importlib.import_module("wandb")

            self.wandb.init(
                project="memory-LLaMA-V2",
                name=environ["RUN_NAME"] if "RUN_NAME" in environ else None,
                config=config,
            )
            self.config = self.wandb.config
            self.wandb.define_metric("Train/Step")
            self.wandb.define_metric("Train/*", step_metric="Train/Step")
            self.wandb.define_metric("Val/Step")
            self.wandb.define_metric("Val/*", step_metric="Val/Step")

            for encoder in filter(None, encoders):
                self.wandb.watch(encoder, log_freq=50)

    def _create_optimizers(self, lr):
        self.optimizers = [
            AdamW(
                params=list(filter(lambda p: p.requires_grad, encoder.parameters())),
                lr=lr,
                betas=(0.9, 0.95),
            )
            for encoder in filter(None, self.encoders)
        ]

    def _create_schedulers(self, virtual_batch_size, train_len):
        self.schedulers = [
            get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=1400,
                num_training_steps=train_len * self.epochs // virtual_batch_size,
            )
            for optimizer in self.optimizers
        ]

    def train(self):
        for epoch in range(self.epoch_0, self.epochs):
            self.train_eval_epoch(epoch)
            self.save_checkpoint(epoch, 0)
        self.save_final()

    def train_eval_epoch(self, epoch):
        mse_loss = torch.nn.MSELoss(reduction="sum")
        self.model.eval()
        for encoder in filter(None, self.encoders):
            encoder.train()
        save_eval_interval_percent = 0.05  # 5% of the steps

        # after how many steps to do save/eval (but multiple of gradient accumulation steps)
        save_eval_interval = self.accelerator.gradient_accumulation_steps * round(
            save_eval_interval_percent
            * len(self.data_train)
            / self.accelerator.gradient_accumulation_steps
        )
        for step, batch in enumerate(tqdm(self.data_train, desc="Training epoch")):
            if epoch == self.epoch_0 and step < self.step_0:
                continue  # skip to the last step from previous runs
            with self.accelerator.accumulate(list(filter(None, self.encoders))):
                with torch.no_grad():
                    h_states = self.model.forward(
                        input_ids=batch["input_no_tgt"],
                        return_dict=False,  # FIXME: does not work with True (because I would need to change the return dict class)
                    )[-1]

                    knowledge_h_states = self.model.forward(
                        input_ids=batch["knowledge_ids"], return_dict=False
                    )[-1]

                losses = []
                shift = (batch["input_no_tgt"] != 0).sum(dim=1) - (
                    batch["knowledge_ids"] != 0
                ).sum(dim=1)
                for layer_id, encoder in enumerate(self.encoders):
                    if encoder is None:
                        continue
                    enc_out = encoder(batch["knowledge_ids"])
                    tgt_len = h_states[layer_id].size(1)
                    enc_out = zero_rows(shift_rows(enc_out, shift), shift)[:, :tgt_len]
                    with torch.no_grad():
                        enc_tgt = zero_rows(
                            shift_rows(knowledge_h_states[layer_id], shift), shift
                        ).to(torch.float32)
                        enc_tgt = enc_tgt[:, :tgt_len] - h_states[layer_id].to(
                            torch.float32
                        )
                    losses.append(
                        mse_loss(enc_out, enc_tgt) / (tgt_len * enc_out.size(0))
                    )

                for loss in losses:
                    self.accelerator.backward(loss)
                for optimizer, scheduler in zip(self.optimizers, self.schedulers):
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # log wandb maybe
                if (step + 1) % (
                    10 * self.accelerator.gradient_accumulation_steps
                ) == 0 and self.wandb:
                    wandb_log = {
                        "Epoch": epoch + step / len(self.data_train),
                        "Train/Step": self.train_step + 1,
                        "Train/Loss": sum(losses) / len(losses),
                        "Train/LR": self.optimizers[0].state_dict()["param_groups"][0]["lr"],
                    }
                    for loss in losses:
                        wandb_log[f"Train/Loss_{losses.index(loss)}"] = loss.item()
                    self.wandb.log(wandb_log)
                self.train_step += 1
            if (step + 1) % save_eval_interval == 0:
                self.partial_eval_epoch(save_eval_interval_percent)
                self.save_checkpoint(step=step, epoch=epoch)
                self.model.eval()
                for encoder in filter(None, self.encoders):
                    encoder.train()

    @torch.no_grad()
    def partial_eval_epoch(self, eval_steps_percent):
        if self.data_eval is None:
            return
        self.model.eval()
        for encoder in filter(None, self.encoders):
            encoder.eval()
        eval_steps_todo = int(eval_steps_percent * len(self.data_eval))
        total_loss = 0.0
        accumulated_loss = 0.0
        logging_steps = int(eval_steps_todo * 0.05) + 1

        eval_iterator = iter(self.data_eval)

        for step in tqdm(range(eval_steps_todo), desc="Validation epoch"):
            try:
                batch = next(eval_iterator)
            except StopIteration:
                break

            with torch.no_grad():
                enc_outs = []
                shift = (batch["input_ids"] != 0).sum(dim=1) - (
                    batch["knowledge_ids"] != 0
                ).sum(dim=1)
                for encoder in self.encoders:
                    if encoder is None:
                        enc_outs.append(None)
                        continue
                    # NOTE: If the encoder is no longer masked for language modelling then the tgt should be removed from knowledge_ids
                    enc_out = encoder(batch["knowledge_ids"])
                    enc_out = zero_rows(shift_rows(enc_out, shift), shift)[
                        :, : batch["labels"].size(1)
                    ]
                    enc_outs.append(enc_out)

                loss = self.model.forward(
                    input_ids=batch["input_ids"],
                    encoded_knowledge=enc_outs,
                    labels=batch["labels"],
                )[0]
                accumulated_loss += loss.item() / logging_steps
                if torch.isinf(loss).item() or torch.isnan(loss).item():
                    total_loss += total_loss / (step+1)
                else:
                    total_loss += loss.item() / len(self.data_eval)

                if (step + 1) % (logging_steps) == 0:
                    if self.wandb:
                        wandb_log = {
                            "Val/Step": self.eval_step / len(self.data_eval),
                            "Val/Loss": accumulated_loss,
                        }
                        self.wandb.log(wandb_log)
                    accumulated_loss = 0.0
            self.eval_step += 1
        if self.accelerator.is_main_process:
            self.early_stopper(total_loss / step)

    def save_checkpoint(self, epoch, step):
        if self.output is None or not self.accelerator.is_main_process:
            return
        now = datetime.now()
        now = now.strftime("%Y_%m_%d_%H_%M")  # descending order for sorting
        save_dict = {"epoch": epoch, "step": step, "config": dict(self.config)}
        for i, encoder in enumerate(filter(None, self.encoders)):
            save_dict[f"encoder{i}"] = self.accelerator.unwrap_model(
                encoder
            ).state_dict()
            save_dict[f"optimizer{i}"] = self.optimizers[i].state_dict()
            save_dict[f"scheduler{i}"] = self.schedulers[i].state_dict()

        path_no_filename = path.join(self.output, now)
        try:
            makedirs(path_no_filename)
            self.last_checkpoints.append(path_no_filename)
        except OSError:
            # dir already exists so skip appending to list
            pass

        with open(path.join(path_no_filename, "checkpoint.pth"), "wb") as f:
            torch.save(save_dict, f)
        if step == 0:
            save_dict["epoch"] = epoch + 1
            with open(
                path.join(self.output, f"checkpoint_epoch{epoch+1}.pth"), "wb"
            ) as f:
                torch.save(save_dict, f)
        if self.early_stopper.should_stop_training():
            save_dict = {"epoch": epoch, "step": step, "config": dict(self.config)}
            for i, encoder in enumerate(filter(None, self.encoders)):
                save_dict[f"encoder{i}"] = self.accelerator.unwrap_model(
                    encoder
                ).state_dict()
            with open(path.join(self.output, "converged_model.pth"), "wb") as f:
                torch.save(save_dict, f)
        if len(self.last_checkpoints) > 3:
            rmtree(self.last_checkpoints.pop(0))

    def save_final(self):
        if self.output is None or not self.accelerator.is_main_process:
            return
        save_dict = {"config": dict(self.config)}
        for i, encoder in enumerate(filter(None, self.encoders)):
            save_dict[f"encoder{i}"] = self.accelerator.unwrap_model(
                encoder
            ).state_dict()

        try:
            makedirs(self.output)
        except OSError:
            # dir already exists so skip appending to list
            pass

        with open(path.join(self.output, "final.pth"), "wb") as f:
            torch.save(save_dict, f)


class FineTuner:
    def __init__(
        self,
        model: LlamaPreTrainedModel,
        encoders: List[Encoder | None],
        data_train: DataLoader,
        data_eval: DataLoader = None,
        grad_accum_steps: int = 1,
        lr: float = 1e-4,
        epochs: int = 5,
        resume: str = None,
        output: str = "./checkpoints",
        use_cpu: bool = False,
        use_wandb: bool = False,
        config: dict = dict(),
    ) -> None:
        self.model = model
        self.encoders = encoders
        self.epochs = epochs
        self.output = output

        self.last_checkpoints = (
            []
        )  # used to delete old checkpoints while run is continuing
        self.epoch_0 = 0
        self.step_0 = 0
        self.train_step = 0
        self.eval_step = 0
        self.accelerator = Accelerator(
            gradient_accumulation_steps=grad_accum_steps,
            mixed_precision="bf16",
            cpu=use_cpu,
        )
        if self.accelerator.is_main_process:
            self.early_stopper = EarlyStopping(patience=5)
        if data_train is not None:
            self._create_optimizer(lr)
            self._create_scheduler(
                data_train.batch_size * grad_accum_steps, len(data_train)
            )
            self.data_train = self.accelerator.prepare(data_train)
            self.optimizer = self.accelerator.prepare(self.optimizer)
            self.scheduler = self.accelerator.prepare(self.scheduler)
            self.model = self.accelerator.prepare(self.model)
            self.encoders = [
                self.accelerator.prepare(encoder) if encoder is not None else None
                for encoder in encoders
            ]
        else:
            device = torch.device("cpu") if use_cpu else torch.device("cuda")
            self.model.to(device, dtype=torch.bfloat16)
            for encoder in filter(None, encoders):
                encoder.to(device, dtype=torch.bfloat16)
        if data_eval is not None:
            self.data_eval = self.accelerator.prepare(
                data_eval
            )
        if resume:
            print("Resuming previous run, loading state dicts")
            resume_dict = torch.load(resume, map_location="cpu")
            if data_train is not None:
                self.scheduler.load_state_dict(resume_dict["scheduler"])
                self.optimizer.load_state_dict(resume_dict["optimizer"])
                self.epoch_0 = resume_dict["epoch"]
                self.step_0 = resume_dict["step"]
            for i, encoder in enumerate(filter(None, encoders)):
                encoder.load_state_dict(resume_dict[f"encoder{i}"])
            print("Loaded and ready!")

        self.wandb = use_wandb and self.accelerator.is_main_process

        self.config = config
        if self.wandb:
            self.wandb = importlib.import_module("wandb")

            self.wandb.init(
                project="memory-LLaMA-V2",
                name=environ["RUN_NAME"] if "RUN_NAME" in environ else None,
                config=config,
            )
            self.config = self.wandb.config
            self.wandb.define_metric("Train/Step")
            self.wandb.define_metric("Train/*", step_metric="Train/Step")
            self.wandb.define_metric("Val/Step")
            self.wandb.define_metric("Val/*", step_metric="Val/Step")

            for encoder in filter(None, encoders):
                self.wandb.watch(encoder, log_freq=50)

    def _create_optimizer(self, lr):
        self.optimizer = AdamW(
            params=torch.nn.ModuleList(filter(None, self.encoders)).parameters(),
            lr=lr,
            betas=(0.9, 0.95),
        )

    def _create_scheduler(self, virtual_batch_size, train_len):
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=train_len * self.epochs // virtual_batch_size,
        )

    def train(self):
        for epoch in range(self.epoch_0, self.epochs):
            self.train_eval_epoch(epoch)
            if self.accelerator.is_main_process and self.early_stopper.shutdown:
                print("Early stopping")
                self.accelerator.end_training()
                return
            self.save_checkpoint(epoch, 0)
        self.save_final()

    # training epoch but also does eval inside every x steps
    def train_eval_epoch(self, epoch):
        self.model.eval()
        for encoder in filter(None, self.encoders):
            encoder.train()
        save_eval_interval_percent = 0.05  # 5% of the steps

        # after how many steps to do save/eval (but multiple of gradient accumulation steps)
        save_eval_interval = self.accelerator.gradient_accumulation_steps * round(
            save_eval_interval_percent
            * len(self.data_train)
            / self.accelerator.gradient_accumulation_steps
        )

        for step, batch in enumerate(tqdm(self.data_train, desc="Training epoch")):
            if epoch == self.epoch_0 and step < self.step_0:
                continue  # skip to the last step from previous runs
            with self.accelerator.accumulate(
                [self.model, *list(filter(None, self.encoders))]
            ):
                shift = (
                    (batch["input_ids"] != 0).sum(dim=1)
                    - (batch["knowledge_ids"] != 0).sum(dim=1)
                    - (batch["labels"] != 0).sum(dim=1)
                )
                enc_outs = []
                for layer_id, encoder in enumerate(self.encoders):
                    if encoder is None:
                        enc_outs.append(None)
                        continue
                    enc_out = encoder(batch["knowledge_ids"])
                    tgt_len = batch["labels"].size(1)
                    enc_out = zero_rows(shift_rows(enc_out, shift), shift)[:, :tgt_len]
                    enc_outs.append(enc_out)

                loss = self.model.forward(
                    input_ids=batch["input_ids"],
                    encoded_knowledge=enc_outs,
                    labels=batch["labels"],
                )[0]
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # log wandb maybe
                if (step + 1) % (
                    10 * self.accelerator.gradient_accumulation_steps
                ) == 0 and self.wandb:
                    wandb_log = {
                        "Epoch": epoch + step / len(self.data_train),
                        "Train/Step": self.train_step + 1,
                        "Train/Loss": loss.item(),
                        "Train/LR": self.optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                    self.wandb.log(wandb_log)
                self.train_step += 1
            if (step + 1) % save_eval_interval == 0:
                self.eval_epoch(save_eval_interval_percent)
                self.save_checkpoint(step=step, epoch=epoch)
                if self.accelerator.is_main_process and self.early_stopper.shutdown:
                    return
                self.model.eval()
                for encoder in filter(None, self.encoders):
                    encoder.train()

    @torch.no_grad()
    def eval_epoch(self, eval_steps_percent):
        if self.data_eval is None:
            return
        self.model.eval()
        for encoder in filter(None, self.encoders):
            encoder.eval()

        eval_steps_todo = int(eval_steps_percent * len(self.data_eval))
        total_loss = 0.0
        accumulated_loss = 0.0
        logging_steps = int(eval_steps_todo * 0.05) + 1

        for step, batch in enumerate(tqdm(self.data_eval, desc="Validation epoch")):

            with torch.no_grad():
                enc_outs = []
                shift = (
                    (batch["input_ids"] != 0).sum(dim=1)
                    - (batch["knowledge_ids"] != 0).sum(dim=1)
                    - (batch["labels"] != 0).sum(dim=1)
                )
                for encoder in self.encoders:
                    if encoder is None:
                        enc_outs.append(None)
                        continue
                    enc_out = encoder(batch["knowledge_ids"])
                    enc_out = zero_rows(shift_rows(enc_out, shift), shift)[
                        :, : batch["labels"].size(1)
                    ]
                    enc_outs.append(enc_out)

                loss = self.model.forward(
                    input_ids=batch["input_ids"],
                    encoded_knowledge=enc_outs,
                    labels=batch["labels"],
                )[0]

                accumulated_loss += loss.item() / logging_steps
                if torch.isinf(loss).item() or torch.isnan(loss).item():
                    total_loss += total_loss / (step+1)
                else:
                    total_loss += loss.item() / len(self.data_eval)

                if (step + 1) % (logging_steps) == 0:
                    if self.wandb:
                        wandb_log = {
                            "Val/Step": self.eval_step / len(self.data_eval),
                            "Val/Loss": accumulated_loss,
                        }
                        self.wandb.log(wandb_log)
                    accumulated_loss = 0.0
            self.eval_step += 1

        if self.accelerator.is_main_process:
            self.early_stopper(total_loss)
        return total_loss

    def evaluate(self):
        # TODO: could be nice to return other stats such as time, etc.
        return self.eval_epoch(0.05)

    def save_checkpoint(self, epoch, step):
        if self.output is None or not self.accelerator.is_main_process:
            return
        now = datetime.now()
        now = now.strftime("%Y_%m_%d_%H_%M")  # descending order for sorting
        save_dict = {
            "epoch": epoch,
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": dict(self.config),
        }
        for i, encoder in enumerate(filter(None, self.encoders)):
            save_dict[f"encoder{i}"] = self.accelerator.unwrap_model(
                encoder
            ).state_dict()

        path_no_filename = path.join(self.output, now)
        try:
            makedirs(path_no_filename)
            self.last_checkpoints.append(path_no_filename)
        except OSError:
            # dir already exists so skip appending to list
            pass

        with open(path.join(path_no_filename, "checkpoint.pth"), "wb") as f:
            torch.save(save_dict, f)
        if step == 0:
            save_dict["epoch"] = epoch + 1
            with open(
                path.join(self.output, f"checkpoint_epoch{epoch+1}.pth"), "wb"
            ) as f:
                torch.save(save_dict, f)
        if self.early_stopper.should_stop_training():
            save_dict.pop("optimizer")
            save_dict.pop("scheduler")
            with open(path.join(self.output, "converged_model.pth"), "wb") as f:
                torch.save(save_dict, f)
        if len(self.last_checkpoints) > 3:
            rmtree(self.last_checkpoints.pop(0))

    def save_final(self):
        if self.output is None or not self.accelerator.is_main_process:
            return
        save_dict = {"config": dict(self.config)}
        for i, encoder in enumerate(filter(None, self.encoders)):
            save_dict[f"encoder{i}"] = self.accelerator.unwrap_model(
                encoder
            ).state_dict()

        try:
            makedirs(self.output)
        except OSError:
            # dir already exists so skip appending to list
            pass

        with open(path.join(self.output, "final.pth"), "wb") as f:
            torch.save(save_dict, f)


class FineTunerICR:
    def __init__(
        self,
        model: LlamaPreTrainedModel,
        encoders: List[Encoder | None],
        data_train: DataLoader,
        data_eval: DataLoader = None,
        grad_accum_steps: int = 1,
        lr: float = 1e-4,
        epochs: int = 5,
        resume: str = None,
        output: str = "./checkpoints",
        use_cpu: bool = False,
        use_wandb: bool = False,
        config: dict = dict(),
    ) -> None:
        self.model = model
        self.encoders = encoders
        self.epochs = epochs
        self.output = output

        self.last_checkpoints = (
            []
        )  # used to delete old checkpoints while run is continuing
        self.epoch_0 = 0
        self.step_0 = 0
        self.train_step = 0
        self.eval_step = 0
        self.accelerator = Accelerator(
            gradient_accumulation_steps=grad_accum_steps,
            mixed_precision="bf16",
            cpu=use_cpu,
        )
        if self.accelerator.is_main_process:
            self.early_stopper = EarlyStopping(patience=5)
        if data_train is not None:
            self._create_optimizer(lr)
            self._create_scheduler(
                data_train.batch_size * grad_accum_steps, len(data_train)
            )
            self.data_train = self.accelerator.prepare(data_train)
            self.optimizer = self.accelerator.prepare(self.optimizer)
            self.scheduler = self.accelerator.prepare(self.scheduler)
            self.model = self.accelerator.prepare(self.model)
            self.encoders = [
                self.accelerator.prepare(encoder) if encoder is not None else None
                for encoder in encoders
            ]
        else:
            device = torch.device("cpu") if use_cpu else torch.device("cuda")
            self.model.to(device, dtype=torch.bfloat16)
            for encoder in filter(None, encoders):
                encoder.to(device, dtype=torch.bfloat16)
        if data_eval is not None:
            self.data_eval = self.accelerator.prepare(
                data_eval
            )
        if resume:
            print("Resuming previous run, loading state dicts")
            resume_dict = torch.load(resume, map_location="cpu")
            if data_train is not None:
                self.scheduler.load_state_dict(resume_dict["scheduler"])
                self.optimizer.load_state_dict(resume_dict["optimizer"])
            self.model.score.load_state_dict(resume_dict["cls_head"])
            for i, encoder in enumerate(filter(None, encoders)):
                encoder.load_state_dict(resume_dict[f"encoder{i}"])
            self.epoch_0 = resume_dict["epoch"]
            self.step_0 = resume_dict["step"]
            print("Loaded and ready!")

        self.wandb = use_wandb and self.accelerator.is_main_process

        self.config = config
        if self.wandb:
            self.wandb = importlib.import_module("wandb")

            self.wandb.init(
                project="memory-LLaMA-V2",
                name=environ["RUN_NAME"] if "RUN_NAME" in environ else None,
                config=config,
            )
            self.config = self.wandb.config
            self.wandb.define_metric("Train/Step")
            self.wandb.define_metric("Train/*", step_metric="Train/Step")
            self.wandb.define_metric("Val/Step")
            self.wandb.define_metric("Val/*", step_metric="Val/Step")

            for encoder in filter(None, encoders):
                self.wandb.watch(encoder, log_freq=50)

    def _create_optimizer(self, lr):
        modules = list(filter(None, self.encoders))
        modules.append(self.model.score)
        self.optimizer = AdamW(
            params=torch.nn.ModuleList(modules).parameters(),
            lr=lr,
            betas=(0.9, 0.95),
        )

    def _create_scheduler(self, virtual_batch_size, train_len):
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=train_len * self.epochs // virtual_batch_size,
        )

    def train(self):
        for epoch in range(self.epoch_0, self.epochs):
            self.train_eval_epoch(epoch)
            if self.accelerator.is_main_process and self.early_stopper.shutdown:
                print("Early stopping")
                self.accelerator.end_training()
                return
            self.save_checkpoint(epoch, 0)
        self.save_final()

    # training epoch but also does eval inside every x steps
    def train_eval_epoch(self, epoch):
        self.model.eval()
        for encoder in filter(None, self.encoders):
            encoder.train()
        save_eval_interval_percent = 0.05  # 5% of the steps

        # after how many steps to do save/eval (but multiple of gradient accumulation steps)
        save_eval_interval = self.accelerator.gradient_accumulation_steps * round(
            save_eval_interval_percent
            * len(self.data_train)
            / self.accelerator.gradient_accumulation_steps
        )

        for step, batch in enumerate(tqdm(self.data_train, desc="Training epoch")):
            if epoch == self.epoch_0 and step < self.step_0:
                continue  # skip to the last step from previous runs
            with self.accelerator.accumulate(
                [self.model, *list(filter(None, self.encoders))]
            ):
                shift = (
                    (batch["input_ids"] != 0).sum(dim=1)
                    - (batch["knowledge_ids"] != 0).sum(dim=1)
                )
                enc_outs = []
                for layer_id, encoder in enumerate(self.encoders):
                    if encoder is None:
                        enc_outs.append(None)
                        continue
                    enc_out = encoder(batch["knowledge_ids"])
                    enc_out = zero_rows(shift_rows(enc_out, shift), shift)[:, : batch["input_ids"].size(1)]
                    enc_outs.append(enc_out)

                loss = self.model.forward(
                    input_ids=batch["input_ids"],
                    encoded_knowledge=enc_outs,
                    labels=batch["labels"],
                )[0]
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # log wandb maybe
                if (step + 1) % (
                    10 * self.accelerator.gradient_accumulation_steps
                ) == 0 and self.wandb:
                    wandb_log = {
                        "Epoch": epoch + step / len(self.data_train),
                        "Train/Step": self.train_step + 1,
                        "Train/Loss": loss.item(),
                        "Train/LR": self.optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                    self.wandb.log(wandb_log)
                self.train_step += 1
            if (step + 1) % save_eval_interval == 0:
                self.eval_epoch(save_eval_interval_percent)
                self.save_checkpoint(step=step, epoch=epoch)
                if self.accelerator.is_main_process and self.early_stopper.shutdown:
                    return
                self.model.eval()
                for encoder in filter(None, self.encoders):
                    encoder.train()

    @torch.no_grad()
    def eval_epoch(self, eval_steps_percent):
        if self.data_eval is None:
            return
        self.model.eval()
        for encoder in filter(None, self.encoders):
            encoder.eval()

        eval_steps_todo = int(eval_steps_percent * len(self.data_eval))
        total_loss = 0.0
        accumulated_loss = 0.0
        logging_steps = int(eval_steps_todo * 0.05) + 1
        acc_clas_acc = 0.0
        total_acc = 0.0

        for step, batch in enumerate(self.data_eval):

            with torch.no_grad():
                enc_outs = []
                shift = (
                    (batch["input_ids"] != 0).sum(dim=1)
                    - (batch["knowledge_ids"] != 0).sum(dim=1)
                )
                for encoder in self.encoders:
                    if encoder is None:
                        enc_outs.append(None)
                        continue
                    enc_out = encoder(batch["knowledge_ids"])
                    enc_out = zero_rows(shift_rows(enc_out, shift), shift)[:, : batch["input_ids"].size(1)]
                    enc_outs.append(enc_out)

                loss, logits = self.model.forward(
                    input_ids=batch["input_ids"],
                    encoded_knowledge=enc_outs,
                    labels=batch["labels"],
                )[0:2]

                acc_clas_acc += ((logits.argmax(dim=-1) == batch["labels"]).sum().item() / batch["labels"].size(0)) / logging_steps
                total_acc += ((logits.argmax(dim=-1) == batch["labels"]).sum().item() / batch["labels"].size(0)) / len(self.data_eval)

                accumulated_loss += loss.item() / logging_steps
                if torch.isinf(loss).item() or torch.isnan(loss).item():
                    total_loss += total_loss / (step+1)
                else:
                    total_loss += loss.item() / len(self.data_eval)

                if (step + 1) % (logging_steps) == 0:
                    if self.wandb:
                        wandb_log = {
                            "Val/Step": self.eval_step / len(self.data_eval),
                            "Val/Loss": accumulated_loss,
                            "Val/Acc": acc_clas_acc,
                        }
                        self.wandb.log(wandb_log)
                    accumulated_loss = 0.0
            self.eval_step += 1

        if self.accelerator.is_main_process:
            self.early_stopper(total_loss)
        return total_acc

    def evaluate(self):
        # TODO: could be nice to return other stats such as time, etc.
        return self.eval_epoch(0.05)

    def save_checkpoint(self, epoch, step):
        if self.output is None or not self.accelerator.is_main_process:
            return
        now = datetime.now()
        now = now.strftime("%Y_%m_%d_%H_%M")  # descending order for sorting
        save_dict = {
            "epoch": epoch,
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": dict(self.config),
            "cls_head": self.model.score.state_dict(),
        }
        for i, encoder in enumerate(filter(None, self.encoders)):
            save_dict[f"encoder{i}"] = self.accelerator.unwrap_model(
                encoder
            ).state_dict()

        path_no_filename = path.join(self.output, now)
        try:
            makedirs(path_no_filename)
            self.last_checkpoints.append(path_no_filename)
        except OSError:
            # dir already exists so skip appending to list
            pass

        with open(path.join(path_no_filename, "checkpoint.pth"), "wb") as f:
            torch.save(save_dict, f)
        if step == 0:
            save_dict["epoch"] = epoch + 1
            with open(
                path.join(self.output, f"checkpoint_epoch{epoch+1}.pth"), "wb"
            ) as f:
                torch.save(save_dict, f)
        if self.early_stopper.should_stop_training():
            save_dict.pop("optimizer")
            save_dict.pop("scheduler")
            with open(path.join(self.output, "converged_model.pth"), "wb") as f:
                torch.save(save_dict, f)
        if len(self.last_checkpoints) > 3:
            rmtree(self.last_checkpoints.pop(0))

    def save_final(self):
        if self.output is None or not self.accelerator.is_main_process:
            return
        save_dict = {
            "config": dict(self.config),
            "cls_head": self.model.score.state_dict(),
        }
        for i, encoder in enumerate(filter(None, self.encoders)):
            save_dict[f"encoder{i}"] = self.accelerator.unwrap_model(
                encoder
            ).state_dict()

        try:
            makedirs(self.output)
        except OSError:
            # dir already exists so skip appending to list
            pass

        with open(path.join(self.output, "final.pth"), "wb") as f:
            torch.save(save_dict, f)


class FineTunerEdit:
    def __init__(
        self,
        model: LlamaPreTrainedModel,
        encoders: List[Encoder | None],
        data_train: DataLoader,
        data_eval: DataLoader = None,
        grad_accum_steps: int = 1,
        lr: float = 1e-4,
        epochs: int = 5,
        resume: str = None,
        output: str = "./checkpoints",
        use_cpu: bool = False,
        use_wandb: bool = False,
        config: dict = dict(),
    ) -> None:
        self.model = model
        self.encoders = encoders
        self.epochs = epochs
        self.output = output

        self.last_checkpoints = (
            []
        )  # used to delete old checkpoints while run is continuing
        self.epoch_0 = 0
        self.step_0 = 0
        self.train_step = 0
        self.eval_step = 0
        self.accelerator = Accelerator(
            gradient_accumulation_steps=grad_accum_steps,
            mixed_precision="bf16",
            cpu=use_cpu,
        )
        if self.accelerator.is_main_process:
            self.early_stopper = EarlyStopping(patience=5)
        if data_train is not None:
            self._create_optimizer(lr)
            self._create_scheduler(
                data_train.batch_size * grad_accum_steps, len(data_train)
            )
            self.data_train = self.accelerator.prepare(data_train)
            self.optimizer = self.accelerator.prepare(self.optimizer)
            self.scheduler = self.accelerator.prepare(self.scheduler)
            self.model = self.accelerator.prepare(self.model)
            self.encoders = [
                self.accelerator.prepare(encoder) if encoder is not None else None
                for encoder in encoders
            ]
        else:
            device = torch.device("cpu") if use_cpu else torch.device("cuda")
            self.model.to(device, dtype=torch.bfloat16)
            for encoder in filter(None, encoders):
                encoder.to(device, dtype=torch.bfloat16)
        if data_eval is not None:
            self.data_eval = self.accelerator.prepare(
                data_eval
            )
        if resume:
            print("Resuming previous run, loading state dicts")
            resume_dict = torch.load(resume, map_location="cpu")
            if data_train is not None:
                self.scheduler.load_state_dict(resume_dict["scheduler"])
                self.optimizer.load_state_dict(resume_dict["optimizer"])
            for i, encoder in enumerate(filter(None, encoders)):
                encoder.load_state_dict(resume_dict[f"encoder{i}"])
            self.epoch_0 = resume_dict["epoch"]
            self.step_0 = resume_dict["step"]
            print("Loaded and ready!")

        self.wandb = use_wandb and self.accelerator.is_main_process

        self.config = config
        if self.wandb:
            self.wandb = importlib.import_module("wandb")

            self.wandb.init(
                project="memory-LLaMA-V2",
                name=environ["RUN_NAME"] if "RUN_NAME" in environ else None,
                config=config,
            )
            self.config = self.wandb.config
            self.wandb.define_metric("Train/Step")
            self.wandb.define_metric("Train/*", step_metric="Train/Step")
            self.wandb.define_metric("Val/Step")
            self.wandb.define_metric("Val/*", step_metric="Val/Step")

            for encoder in filter(None, encoders):
                self.wandb.watch(encoder, log_freq=50)

    def _create_optimizer(self, lr):
        self.optimizer = AdamW(
            params=torch.nn.ModuleList(filter(None, self.encoders)).parameters(),
            lr=lr,
            betas=(0.9, 0.95),
        )

    def _create_scheduler(self, virtual_batch_size, train_len):
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=100,
            num_training_steps=train_len * self.epochs // virtual_batch_size,
        )

    def train(self):
        for epoch in range(self.epoch_0, self.epochs):
            self.train_eval_epoch(epoch)
            if self.accelerator.is_main_process and self.early_stopper.shutdown:
                print("Early stopping")
                self.accelerator.end_training()
                return
            self.save_checkpoint(epoch, 0)
        self.save_final()

    # training epoch but also does eval inside every x steps
    def train_eval_epoch(self, epoch):
        self.model.eval()
        for encoder in filter(None, self.encoders):
            encoder.train()
        save_eval_interval_percent = 0.05  # 5% of the steps

        # after how many steps to do save/eval (but multiple of gradient accumulation steps)
        save_eval_interval = self.accelerator.gradient_accumulation_steps * round(
            save_eval_interval_percent
            * len(self.data_train)
            / self.accelerator.gradient_accumulation_steps
        )

        for step, batch in enumerate(tqdm(self.data_train, desc="Training epoch")):
            if epoch == self.epoch_0 and step < self.step_0:
                continue  # skip to the last step from previous runs
            with self.accelerator.accumulate(
                [self.model, *list(filter(None, self.encoders))]
            ):
                enc_outs = []
                for layer_id, encoder in enumerate(self.encoders):
                    if encoder is None:
                        enc_outs.append(None)
                        continue
                    enc_out = encoder(batch["input_no_tgt"])
                    enc_outs.append(enc_out)

                loss = self.model.forward(
                    input_ids=batch["input_ids"],
                    encoded_knowledge=enc_outs,
                    labels=batch["labels"],
                )[0]
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # log wandb maybe
                if (step + 1) % (
                    10 * self.accelerator.gradient_accumulation_steps
                ) == 0 and self.wandb:
                    wandb_log = {
                        "Epoch": epoch + step / len(self.data_train),
                        "Train/Step": self.train_step + 1,
                        "Train/Loss": loss.item(),
                        "Train/LR": self.optimizer.state_dict()["param_groups"][0]["lr"],
                    }
                    self.wandb.log(wandb_log)
                self.train_step += 1
            if (step + 1) % save_eval_interval == 0:
                self.eval_epoch(save_eval_interval_percent)
                self.save_checkpoint(step=step, epoch=epoch)
                if self.accelerator.is_main_process and self.early_stopper.shutdown:
                    return
                self.model.eval()
                for encoder in filter(None, self.encoders):
                    encoder.train()

    @torch.no_grad()
    def eval_epoch(self, eval_steps_percent):
        if self.data_eval is None:
            return
        self.model.eval()
        for encoder in filter(None, self.encoders):
            encoder.eval()

        eval_steps_todo = int(eval_steps_percent * len(self.data_eval))
        total_loss = 0.0
        accumulated_loss = 0.0
        logging_steps = int(eval_steps_todo * 0.05) + 1

        for step, batch in enumerate(tqdm(self.data_eval, desc="Validation epoch")):

            with torch.no_grad():
                enc_outs = []
                for encoder in self.encoders:
                    if encoder is None:
                        enc_outs.append(None)
                        continue
                    enc_out = encoder(batch["input_no_tgt"])
                    enc_outs.append(enc_out)

                loss = self.model.forward(
                    input_ids=batch["input_ids"],
                    encoded_knowledge=enc_outs,
                    labels=batch["labels"],
                )[0]

                accumulated_loss += loss.item() / logging_steps
                if torch.isinf(loss).item() or torch.isnan(loss).item():
                    total_loss += total_loss / (step+1)
                else:
                    total_loss += loss.item() / len(self.data_eval)

                if (step + 1) % (logging_steps) == 0:
                    if self.wandb:
                        wandb_log = {
                            "Val/Step": self.eval_step / len(self.data_eval),
                            "Val/Loss": accumulated_loss,
                        }
                        self.wandb.log(wandb_log)
                    accumulated_loss = 0.0
            self.eval_step += 1

        if self.accelerator.is_main_process:
            self.early_stopper(total_loss)
        return total_loss

    def evaluate(self):
        # TODO: could be nice to return other stats such as time, etc.
        return self.eval_epoch(0.05)

    def save_checkpoint(self, epoch, step):
        if self.output is None or not self.accelerator.is_main_process:
            return
        now = datetime.now()
        now = now.strftime("%Y_%m_%d_%H_%M")  # descending order for sorting
        save_dict = {
            "epoch": epoch,
            "step": step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "config": dict(self.config),
        }
        for i, encoder in enumerate(filter(None, self.encoders)):
            save_dict[f"encoder{i}"] = self.accelerator.unwrap_model(
                encoder
            ).state_dict()

        path_no_filename = path.join(self.output, now)
        try:
            makedirs(path_no_filename)
            self.last_checkpoints.append(path_no_filename)
        except OSError:
            # dir already exists so skip appending to list
            pass

        with open(path.join(path_no_filename, "checkpoint.pth"), "wb") as f:
            torch.save(save_dict, f)
        if step == 0:
            save_dict["epoch"] = epoch + 1
            with open(
                path.join(self.output, f"checkpoint_epoch{epoch+1}.pth"), "wb"
            ) as f:
                torch.save(save_dict, f)
        if self.early_stopper.should_stop_training():
            save_dict.pop("optimizer")
            save_dict.pop("scheduler")
            with open(path.join(self.output, "converged_model.pth"), "wb") as f:
                torch.save(save_dict, f)
        if len(self.last_checkpoints) > 3:
            rmtree(self.last_checkpoints.pop(0))

    def save_final(self):
        if self.output is None or not self.accelerator.is_main_process:
            return
        save_dict = {"config": dict(self.config)}
        for i, encoder in enumerate(filter(None, self.encoders)):
            save_dict[f"encoder{i}"] = self.accelerator.unwrap_model(
                encoder
            ).state_dict()

        try:
            makedirs(self.output)
        except OSError:
            # dir already exists so skip appending to list
            pass

        with open(path.join(self.output, "final.pth"), "wb") as f:
            torch.save(save_dict, f)
