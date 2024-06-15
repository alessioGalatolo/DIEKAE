import json
from os import path
import torch
from torch.utils.data import Dataset
import urllib.request


class CounterfactTestDataset(Dataset):
    def __init__(self, data, tokenizer, max_tokens_dec, max_tokens_enc):
        raise NotImplementedError("Needs to be double checked")
        self.prompts = data["prompts"]
        self.ground_truth = data["ground_truth"]
        self.target_new = data["target_new"]
        self.paraphrases = data["paraphrases"]
        self.neighborhoods = data["neighborhoods"]

        self.max_tokens_dec = max_tokens_dec
        self.max_tokens_enc = max_tokens_enc
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        prompt = self.prompts[index]
        ground_truth = self.ground_truth[index]
        target_new = self.target_new[index]
        paraphrases = self.paraphrases[index]
        neighborhoods = self.neighborhoods[index]

        prompt = "[USR] " + prompt + " [SYS] "

        tgt = torch.tensor(self.tokenizer.encode(target_new, bos=True, eos=True), dtype=torch.long)
        src = torch.tensor(self.tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.long)
        original = torch.tensor(self.tokenizer.encode(ground_truth, bos=True, eos=True), dtype=torch.long)
        paraphrases = list(map(
            lambda x:
                torch.tensor(
                    self.tokenizer.encode("[USR] "+x+" [SYS] ", bos=True, eos=False),
                    dtype=torch.long),
            paraphrases
        ))
        neighborhoods = list(map(
            lambda x:
                torch.tensor(
                    self.tokenizer.encode("[USR] "+x+" [SYS] ", bos=True, eos=False),
                    dtype=torch.long),
            neighborhoods
        ))

        return src, tgt, original, paraphrases, neighborhoods


# generic knowledge dataset
class KnowledgeDataset(Dataset):
    def __init__(self, data_path, tokenizer, split, limit_number_of_samples=float("inf")):
        assert split in ["train", "val", "test"]
        self.counterfact = "counterfact" in data_path
        self.classification = "folio" in data_path or "proofwriter" in data_path
        if self.classification or self.counterfact:
            raise NotImplementedError("You should now use their own specific classes")

        with open(path.join(data_path, f"{split}.src"), "r") as f:
            self.src_file = f.readlines()
        with open(path.join(data_path, f"{split}.tgt"), "r") as f:
            self.tgt_file = f.readlines()
        with open(path.join(data_path, f"{split}.mem"), "r") as f:
            self.mem_file = f.readlines()

        assert len(self.src_file) == len(self.tgt_file) and len(self.tgt_file) == len(self.mem_file)

        self.tokenizer = tokenizer
        self.samples_limit = limit_number_of_samples

    def __len__(self):
        return min(self.samples_limit, len(self.src_file))

    def __getitem__(self, index):
        src = self.src_file[index].strip()
        tgt = self.tgt_file[index].strip()
        mem = self.mem_file[index].strip()

        tgt = src + " " + tgt
        tgt = self.tokenizer.encode(tgt)
        src_only = self.tokenizer.encode(src)
        src_only = src_only[:-1]  # remove eos
        src = tgt.copy()  # [:-1]
        tgt[: len(src_only)] = [0] * len(src_only)
        tgt = tgt  #[1:]  DO NOT shift as it is done internally by hf

        # "Imagine that {knowledge}" is taken from ICE, Cohen et al. 2024
        knowledge = self.tokenizer.encode("Imagine that {" + mem + "} ") + [self.tokenizer.eos_token_id]
        knowledge = knowledge + src_only

        return {"input_ids": src, "labels": tgt, "knowledge_ids": knowledge, "input_no_tgt": src_only}


class ICRDataset(Dataset):
    token_to_number = {
        "False": 0,  # False
        "True": 1,  # True
        "Unknown": 2,  # Unk
    }
    number_to_token = {
        0: "False",
        1: "True",
        2: "Unknown"
    }

    def __init__(self, data_path, tokenizer, split, limit_number_of_samples=float("inf")):
        assert split in ["train", "val", "test"]
        assert "folio" in data_path or "proofwriter" in data_path

        with open(path.join(data_path, f"{split}.src"), "r") as f:
            self.src_file = f.readlines()
        with open(path.join(data_path, f"{split}.tgt"), "r") as f:
            self.tgt_file = f.readlines()
        with open(path.join(data_path, f"{split}.mem"), "r") as f:
            self.mem_file = f.readlines()

        assert len(self.src_file) == len(self.tgt_file) and len(self.tgt_file) == len(self.mem_file)

        self.tokenizer = tokenizer
        self.samples_limit = limit_number_of_samples

    def __len__(self):
        return min(len(self.src_file), self.samples_limit)

    def __getitem__(self, index):
        src = self.src_file[index].strip()
        tgt = self.tgt_file[index].strip()
        mem = self.mem_file[index].strip()

        src_only = self.tokenizer.encode(src)
        src_only = src_only

        # "Imagine that {knowledge}" is taken from ICE, Cohen et al. 2024
        knowledge = self.tokenizer.encode("Imagine that {" + mem + "} ") + [self.tokenizer.eos_token_id]
        knowledge = knowledge + src_only

        tgt = ICRDataset.token_to_number[tgt]
        return {"input_ids": src_only, "labels": tgt, "knowledge_ids": knowledge}


class EditDataset(Dataset):
    def __init__(self, data_path, tokenizer, split):
        assert split in ["train", "val", "test"]

        with open(path.join(data_path, f"{split}.src"), "r") as f:
            self.src_file = f.readlines()
        with open(path.join(data_path, f"{split}.tgt"), "r") as f:
            self.tgt_file = f.readlines()
        with open(path.join(data_path, f"{split}.mem"), "r") as f:
            self.true_file = f.readlines()

        assert len(self.src_file) == len(self.tgt_file) and len(self.tgt_file) == len(self.true_file)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src_file)

    def __getitem__(self, index):
        src = self.src_file[index].strip()  # Beginning of sentece e.g. [USR] York Bowen passed away at
        tgt = self.tgt_file[index].strip()  # new completion to be inserted e.g. "Geneva"
        _ = self.true_file[index].strip()  # true completion to be replaced e.g. "London"

        src += " [SYS] "

        src_only = self.tokenizer.encode(src)
        src = src + " " + tgt
        tgt = self.tokenizer.encode(src)
        src = tgt.copy()

        tgt[: len(src_only)] = [0] * len(src_only)

        return {"input_ids": src, "labels": tgt, "input_no_tgt": src_only}


class EditDatasetTest(Dataset):
    def __init__(self, data_path, tokenizer, split):
        assert split in ["train", "val", "test"]

        self.prompts = []
        self.ground_truth = []
        self.target_new = []
        self.paraphrases = []
        self.neighborhoods = []

        # FIXME: it's downloading the dataset because during preprocessing, evaluation part is not used (TODO: fix preprocessing and remove this part)
        print("Downloading dataset...")
        urllib.request.urlretrieve("https://rome.baulab.info/data/dsets/counterfact.json", path.join(data_path, "facts.json"))
        print("Dataset downloaded")
        print("Parsing files...")
        with open(path.join(data_path, "facts.json"), "r") as raw_file:
            facts = json.load(raw_file)
        train = facts[: int(len(facts) * 0.85)]
        for dialog in train:
            self.prompts.append(dialog["requested_rewrite"]["prompt"].replace(
                "{}",
                dialog["requested_rewrite"]["subject"]
            ))
            self.ground_truth.append(dialog["requested_rewrite"]["target_true"]["str"])
            self.target_new.append(dialog["requested_rewrite"]["target_new"]["str"])
            self.paraphrases.append(dialog["paraphrase_prompts"])
            self.neighborhoods.append(dialog["neighborhood_prompts"])

        with open(path.join(data_path, f"{split}.src"), "r") as f:
            self.src_file = f.readlines()
        with open(path.join(data_path, f"{split}.tgt"), "r") as f:
            self.tgt_file = f.readlines()
        with open(path.join(data_path, f"{split}.mem"), "r") as f:
            self.true_file = f.readlines()

        assert len(self.src_file) == len(self.tgt_file) and len(self.tgt_file) == len(self.true_file)

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src_file)

    def __getitem__(self, index):
        prompt = self.prompts[index]
        ground_truth = self.ground_truth[index]
        target_new = self.target_new[index]
        paraphrases = self.paraphrases[index]
        neighborhoods = self.neighborhoods[index]

        prompt = "[USR] " + prompt + " [SYS] "

        tgt = self.tokenizer.encode(target_new)[1]  # remove bos
        src = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long)
        original = self.tokenizer.encode(ground_truth)[1]
        paraphrases = list(map(
            lambda x:
                torch.tensor(
                    [self.tokenizer.encode("[USR] " + x + " [SYS] ")],
                    dtype=torch.long),
            paraphrases
        ))
        neighborhoods = list(map(
            lambda x:
                torch.tensor(
                    [self.tokenizer.encode("[USR] "+x+" [SYS] ")],
                    dtype=torch.long),
            neighborhoods
        ))

        return src, tgt, original, paraphrases, neighborhoods
