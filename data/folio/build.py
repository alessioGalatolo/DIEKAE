import json
from os import path, makedirs, remove
from shutil import rmtree
import subprocess
import urllib.request


current_dir = path.abspath(path.dirname(__file__))

print("Downloading dataset...")
urllib.request.urlretrieve("https://github.com/Yale-LILY/FOLIO/raw/main/data/v0.0/folio-train.jsonl", path.join(current_dir, "train.json"))
urllib.request.urlretrieve("https://github.com/Yale-LILY/FOLIO/raw/main/data/v0.0/folio-validation.jsonl", path.join(current_dir, "val.json"))
subprocess.run(["unzip", path.join(current_dir, "proofwriter.zip"), "-d", current_dir])
print("Dataset downloaded")


print("Parsing files...")
for split in ["train", "val"]:
    with open(path.join(current_dir, f"{split}.src"), "w") as src, \
        open(path.join(current_dir, f"{split}.tgt"), "w") as tgt, \
            open(path.join(current_dir, f"{split}.mem"), "w") as mem:

        with open(path.join(current_dir, split + ".json"), "r", encoding="utf-8") as raw_file:
            data = json.loads("[" + ",".join(raw_file.readlines()) + "]")
        for entry in data:
            mem.write(" ".join(entry["premises"]) + "\n")
            src.write(f"[USR] {entry['conclusion']} [SYS] " + "\n")
            tgt.write(str(entry["label"]) + "\n")

print("Removing temporary files...")
remove(path.join(current_dir, "train.json"))
remove(path.join(current_dir, "val.json"))
print("Done!")
