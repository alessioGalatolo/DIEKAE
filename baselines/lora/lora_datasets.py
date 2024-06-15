# used to import from datasets in llama_knowledge/datasets.py

import sys
import os

# Get the current directory of the running script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate two levels up to reach the project root directory
project_root = os.path.join(current_dir, '..', '..')

# Add the project root directory to the sys.path
sys.path.append(project_root)

from llama_knowledge.datasets import KnowledgeDataset as KD  # noqa: E402
from llama_knowledge.datasets import ICRDataset as ICRD  # noqa: E402
from llama_knowledge.datasets import EditDataset, EditDatasetTest  # noqa: E402
from llama_knowledge.helpers import pad_longest as pl  # noqa: E402


class KnowledgeDataset(KD):
    def __init__(self, *args, **kwargs):
        self.no_knowledge = kwargs.pop("no_knowledge", False)
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        super_item = super().__getitem__(index)
        # recieves input_no_tgt, input_ids, labels, knowledge_ids
        # where input_no_tgt = src
        # input_ids = src + tgt
        # labels = src + tgt where src is filled with 0s
        # knowledge_ids = knowledge + src (and not tgt)

        if self.no_knowledge:  # used to test PLM without given knowledge
            return {"input_ids": super_item.pop("input_ids"), "labels": super_item.pop("labels")}
        len_src_only = len(super_item.pop("input_no_tgt"))
        input_ids = super_item.pop("knowledge_ids")
        old_input_ids = super_item.pop("input_ids")
        input_ids += old_input_ids[len_src_only:]
        super_item["input_ids"] = input_ids
        labels = super_item.pop("labels")
        k_len = len(input_ids) - len(labels)

        labels = [0] * k_len + labels

        # because hf expects -100 to exclude value from loss computation
        labels = [-100 if label == 0 else label for label in labels]
        return {"input_ids": input_ids, "labels": labels}


class ICRDataset(ICRD):
    def __init__(self, *args, **kwargs):
        self.no_knowledge = kwargs.pop("no_knowledge", False)
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        super_item = super().__getitem__(index)
        # recieves input_ids, labels, knowledge_ids
        # input_ids = src
        # labels = tgt as a number
        # knowledge_ids = knowledge + src (and not tgt)

        if self.no_knowledge:  # used to test PLM without given knowledge
            return {"input_ids": super_item.pop("input_ids"), "labels": super_item.pop("labels")}

        return {"input_ids": super_item.pop("knowledge_ids"), "labels": super_item.pop("labels")}


def pad_longest(batch, pad_token_id, max_tokens=None):
    return pl(batch, pad_token_id, max_tokens, pad_token_id_tgt=-100)
