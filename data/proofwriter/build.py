import json
from os import path, makedirs, remove
from shutil import rmtree
import subprocess
import urllib.request


current_dir = path.abspath(path.dirname(__file__))

print("Downloading dataset...")
urllib.request.urlretrieve("https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip", path.join(current_dir, "proofwriter.zip"))
subprocess.run(["unzip", path.join(current_dir, "proofwriter.zip"), "-d", current_dir])
print("Dataset downloaded")


print("Parsing files...")
for split, filename in zip(["train", "val", "test"], ["meta-train.jsonl", "meta-dev.jsonl", "meta-test.jsonl"]):
    with open(path.join(current_dir, f"{split}.src"), "w") as src, \
        open(path.join(current_dir, f"{split}.tgt"), "w") as tgt, \
            open(path.join(current_dir, f"{split}.mem"), "w") as mem:

        with open(path.join(current_dir, "proofwriter-dataset-V2020.12.3", "OWA", "depth-2", filename), "r", encoding="utf-8") as raw_file:
            data = json.loads("[" + ",".join(raw_file.readlines()) + "]")
        for entry in data:
            for question in entry["questions"].values():
                mem.write(entry["theory"] + "\n")
                src.write(f"[USR] {question['question']} [SYS] " + "\n")
                tgt.write(str(question["answer"]) + "\n")

print("Removing temporary files...")
remove(path.join(current_dir, "proofwriter.zip"))
rmtree(path.join(current_dir, "proofwriter-dataset-V2020.12.3"))
print("Done!")
