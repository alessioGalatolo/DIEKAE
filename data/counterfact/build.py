import urllib.request
import json
from os import path, remove
from collections import defaultdict
from glob import glob
from re import sub
from tqdm import tqdm


current_dir = path.abspath(path.dirname(__file__))

print("Downloading dataset...")
urllib.request.urlretrieve("https://rome.baulab.info/data/dsets/counterfact.json", path.join(current_dir, "facts.json"))
print("Dataset downloaded")

with open(path.join(current_dir, "facts.json"), "r") as raw_file:
    facts = json.load(raw_file)


print("Parsing files...")
train = facts[: int(len(facts) * 0.85)]
val = facts[int(len(facts) * 0.85): ]
for (split, data) in [("train", train), ("val", val)]:
    with open(path.join(current_dir, f"{split}.src"), "w") as src, \
        open(path.join(current_dir, f"{split}.tgt"), "w") as tgt, \
            open(path.join(current_dir, f"{split}.mem"), "w") as mem:

        for dialog in tqdm(data):
            src_text = dialog["requested_rewrite"]["prompt"].replace(
                "{}",
                dialog["requested_rewrite"]["subject"]
            )

            mem.write(dialog["requested_rewrite"]["target_true"]["str"] + "\n")
            src.write("[USR] " + src_text + "\n")
            tgt.write(dialog["requested_rewrite"]["target_new"]["str"] + "\n")
print("Removing temporary files...")
for file in glob(path.join(current_dir, "*.json")):
    remove(file)
print("Done!")
