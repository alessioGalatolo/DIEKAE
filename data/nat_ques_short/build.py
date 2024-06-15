import json
from os import path, remove, rmdir, rename
from shutil import rmtree
import subprocess
from glob import glob
import urllib.request

current_dir = path.abspath(path.dirname(__file__))

print("Downloading dataset...")
urllib.request.urlretrieve("https://huggingface.co/datasets/cjlovering/natural-questions-short/resolve/main/train.json?download=true", path.join(current_dir, "train.json"))
urllib.request.urlretrieve("https://huggingface.co/datasets/cjlovering/natural-questions-short/resolve/main/dev.json?download=true", path.join(current_dir, "val.json"))
print("Dataset downloaded")
print("Parsing files...")
for split in ["train", "val"]:
    with open(path.join(current_dir, f"{split}.src"), "w") as src, \
        open(path.join(current_dir, f"{split}.tgt"), "w") as tgt, \
            open(path.join(current_dir, f"{split}.mem"), "w") as mem:

        with open(path.join(current_dir, f"{split}.json"), "r") as raw_file:
            data = json.load(raw_file)
        for dialog in data:
            src.write(f"[USR] {dialog['questions'][0]['input_text']} [SYS] \n")
            tgt.write(dialog["answers"][0]["span_text"] + "\n")
            mem.write(dialog["contexts"] + "\n")

print("Removing temporary files...")
remove(path.join(current_dir, "train.json"))
remove(path.join(current_dir, "val.json"))
print("Done!")
