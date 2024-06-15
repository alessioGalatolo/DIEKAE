import json
from os import path, makedirs
from shutil import rmtree
import subprocess
import urllib.request


current_dir = path.abspath(path.dirname(__file__))


print("Downloading dataset...")
to_download = [f"questions/{split}_questions.json.gz" for split in ["train", "dev", "test"]]
to_download.extend([f"contexts/short/{split}_contexts.json.gz" for split in ["train", "dev", "test"]])
for file in to_download:
    makedirs(path.join(current_dir, "/".join(file.split("/")[:-1])), exist_ok=True)
    urllib.request.urlretrieve("http://curtis.ml.cmu.edu/datasets/quasar/quasar-t/" + file, path.join(current_dir, file))
    subprocess.run(["gunzip", path.join(current_dir, file)])
print("Dataset downloaded")

print("Parsing files...")
for split, file_split in zip(["train", "val", "test"], ["train", "dev", "test"]):
    with open(path.join(current_dir, f"{split}.src"), "w") as src, \
        open(path.join(current_dir, f"{split}.tgt"), "w") as tgt, \
            open(path.join(current_dir, f"{split}.mem"), "w") as mem:

        with open(path.join(current_dir, "questions", f"{file_split}_questions.json"), "r", encoding="utf-8") as raw_file:
            questions = json.loads("[" + ",".join(raw_file.readlines()) + "]")
        with open(path.join(current_dir, "contexts/short", f"{file_split}_contexts.json"), "r", encoding="utf-8") as raw_file:
            contexts = json.loads("[" + ",".join(raw_file.readlines()) + "]")
        for question, context in zip(questions, contexts):
            src.write("[USR] " + question["question"] + " [SYS] " + "\n")
            tgt.write(question["answer"] + "\n")
            # a couple of contexts as answer may not be there
            mem.write(" [SEP] ".join(map(lambda x: x[1], context["contexts"][:2])) + "\n")

print("Removing temporary files...")
rmtree(path.join(current_dir, "contexts"))
rmtree(path.join(current_dir, "questions"))
print("Done!")
