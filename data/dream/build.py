import urllib.request
import json
from os import path, remove
from glob import glob
from re import sub


current_dir = path.abspath(path.dirname(__file__))

print("Downloading dataset...")
urllib.request.urlretrieve("https://github.com/nlpdata/dream/raw/master/data/train.json", path.join(current_dir, "train.json"))
urllib.request.urlretrieve("https://github.com/nlpdata/dream/raw/master/data/dev.json", path.join(current_dir, "val.json"))
urllib.request.urlretrieve("https://github.com/nlpdata/dream/raw/master/data/test.json", path.join(current_dir, "test.json"))
print("Dataset downloaded")

print("Parsing files...")
for split in ["train", "val", "test"]:
    with open(path.join(current_dir, f"{split}.json"), "r", encoding="utf-8") as raw_file:
        data = json.load(raw_file)
    with open(path.join(current_dir, f"{split}.src"), "w") as src,\
        open(path.join(current_dir, f"{split}.tgt"), "w") as tgt,\
            open(path.join(current_dir, f"{split}.mem"), "w") as mem:
        for dialog in data:
            if "__" in dialog[1][0]["question"]:
                continue  # skipping false questions where they only mask the answer
            mem.write(" [SEP] ".join(map(lambda x: x.replace("W:", "Woman:", 1).replace("M:", "Man:", 1), dialog[0])) + "\n")
            src.write("[USR] " + dialog[1][0]["question"] + " [SYS] " + "\n")
            tgt.write(dialog[1][0]["answer"] + "\n")
print("Removing temporary files...")
for file in glob(path.join(current_dir, "*.json")):
    remove(file)
print("Done!")
