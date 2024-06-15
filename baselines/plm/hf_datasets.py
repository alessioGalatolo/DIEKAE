# used to import from datasets in llama_knowledge/datasets.py

import sys
import os

# Get the current directory of the running script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate two levels up to reach the project root directory
project_root = os.path.join(current_dir, '..', '..')

# Add the project root directory to the sys.path
sys.path.append(project_root)

from baselines.lora.lora_datasets import EditDatasetTest, KnowledgeDataset, pad_longest, ICRDataset as ICRD  # noqa: E402


class ICRDataset(ICRD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        super_item = super().__getitem__(index)
        # cannot pass just index of label as plain PLM does not have a classification head
        labels = self.tokenizer.encode(self.number_to_token[super_item["labels"]])[1]
        return {"input_ids": super_item["input_ids"], "labels": labels}
