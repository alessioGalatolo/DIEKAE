from typing import Dict, List, Mapping

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def zero_rows(t, shift):
    # Step 1: Create a range tensor that goes from 0 to the maximum sequence length minus one
    indices = torch.arange(t.size(1)).unsqueeze(0).to(t.device)  # Unsqueeze to allow broadcasting

    # Step 2: Modify shift values for negative indices to count from the end
    adjusted_shift = torch.where(shift < 0, t.size(1) + shift, shift)

    # Step 3: Compare indices array with the adjusted shift tensor (broadcasted)
    # This creates a boolean mask where True values indicate the positions to be zeroed
    mask = indices >= adjusted_shift.unsqueeze(1)  # Unsqueeze shift to enable broadcasting across columns

    # Step 4: Use the mask to set the appropriate elements in t to 0
    t[mask] = 0
    return t


def shift_rows(t, shift):
    indices = torch.arange(t.size(1)).unsqueeze(0).repeat(t.size(0), 1).to(t.device)
    indices = (indices - shift.unsqueeze(1)) % t.size(1)
    return torch.gather(t, 1, indices.unsqueeze(2).repeat(1, 1, t.size(2)).to(t.device))


# partially borrowed from transformers lib
def pad_longest(batch: List[Dict[str, List[int]]], pad_token_id: int, max_tokens: int = None, pad_token_id_tgt: int = None):
    # If we have a list of dicts, let's convert it in a dict of lists
    # We do this to allow using this method as a collate_fn function in PyTorch Dataloader
    if isinstance(batch, (list, tuple)) and isinstance(batch[0], Mapping):
        batch = {key: [example[key] for example in batch] for key in batch[0].keys()}

    max_length = {
        key: max(len(inputs) for inputs in value) for key, value in batch.items()
    }
    if max_tokens is not None:
        max_length = {key: min(max_length[key], max_tokens) for key in max_length.keys()}

    for key, value in batch.items():
        for i in range(len(value)):
            if len(value[i]) < max_length[key]:
                if key == "labels" and pad_token_id_tgt is not None:
                    # pad with -100 to exclude from loss computation
                    value[i] += [pad_token_id_tgt] * (max_length[key] - len(value[i]))
                else:
                    value[i] += [pad_token_id] * (max_length[key] - len(value[i]))
            elif len(value[i]) > max_length[key]:
                value[i] = value[i][:max_length[key]]

    batch = [
        {key: torch.tensor(value[i], dtype=torch.long) for key, value in batch.items()}
        for i in range(len(batch[list(batch.keys())[0]]))
    ]
    return default_collate(batch)


# partially borrowed from transformers lib
# TODO: this class can be greatly improved and cleaned
class EarlyStopping:
    """

    Args:
        early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
        early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    """

    def __init__(
        self,
        patience: int = 1,
        threshold: float = 0.0,
        greater_is_better: bool = False,  # false for loss
    ):
        self.patience = patience
        self.threshold = threshold
        # patience_counter denotes the number of times validation metrics failed to improve.
        self.patience_counter = 0

        self.greater_is_better = greater_is_better
        self.best_metric = float("inf") if not greater_is_better else -float("inf")
        self.should_training_stop = False
        self.shutdown = False

    def check_metric_value(self, metric_value):
        print(f"Early stopper received value {metric_value}")
        operator = np.greater if self.greater_is_better else np.less
        if operator(metric_value, self.best_metric) and abs(metric_value - self.best_metric) > self.threshold:
            print(f"I've resetted the patentience counter because value was {metric_value} while best one is {self.best_metric}")
            self.patience_counter = 0
            self.best_metric = metric_value
        else:
            self.patience_counter += 1

    def __call__(self, metric_value):
        if self.shutdown or self.should_training_stop:
            return
        self.check_metric_value(metric_value)
        if self.patience_counter >= self.patience:
            print("Time to stop the training")
            self.should_training_stop = True

    def should_stop_training(self):
        if self.should_training_stop:
            self.shutdown = True
            return True
        return False
