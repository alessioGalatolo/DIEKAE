import urllib.request
import json
from os import path, remove, rename
from shutil import rmtree
from collections import defaultdict
from glob import glob
from re import sub
import subprocess


# if true will filter out low quality convs
QUALITY_FILTER = True


current_dir = path.abspath(path.dirname(__file__))

print("Downloading dataset...")
urllib.request.urlretrieve("https://github.com/festvox/datasets-CMU_DoG/archive/master.zip", path.join(current_dir, "repo.zip"))
subprocess.run(["unzip", path.join(current_dir, "repo.zip"), "-d", current_dir])
print("Dataset downloaded")

print("Parsing files...")
wiki_data = {}
for file in glob(path.join(current_dir, "datasets-CMU_DoG-master", "WikiData", "*.json")):
    with open(file, "r") as raw_file:
        facts = json.load(raw_file)
        id = facts["wikiDocumentIdx"]
        wiki_data[id] = facts
        fact0 = ""
        for key, fact in facts["0"].items():
            joint_fact = " | ".join(fact) if isinstance(fact, list) else fact
            fact0 += f"{key}: " + joint_fact + " [SEP] "
        facts.pop("wikiDocumentIdx")
        facts["0"] = fact0

try:
    rename(
        path.join(current_dir, "datasets-CMU_DoG-master", "Conversations", "valid"),
        path.join(current_dir, "datasets-CMU_DoG-master", "Conversations", "val")
    )
except FileNotFoundError:
    pass

for split in ["train", "val", "test"]:
    with open(path.join(current_dir, f"{split}.src"), "w") as src, \
        open(path.join(current_dir, f"{split}.tgt"), "w") as tgt, \
            open(path.join(current_dir, f"{split}.mem"), "w") as mem:

        for file in glob(path.join(current_dir, "datasets-CMU_DoG-master", "Conversations", split, "*.json")):
            with open(file, "r", encoding="utf-8") as raw_file:
                data = json.load(raw_file)
            if data["status"] == 0:
                continue
            if QUALITY_FILTER and data["rating"] != 3:
                continue
            current_fact = wiki_data[data["wikiDocumentIdx"]]

            msgs2write = ""
            user1_is_sys = ""
            user2_is_sys = ""
            last_docIdx = 0
            for message in data["history"]:
                if message["docIdx"] != last_docIdx:
                    msgs2write = ""
                    user1_is_sys = ""
                    user2_is_sys = ""
                    last_docIdx = message["docIdx"]
                if len(data["whoSawDoc"]) > 1:
                    user1_is_sys += " [SYS] " if message["uid"] == "user1" else " [USR] "
                    user2_is_sys += " [SYS] " if message["uid"] == "user2" else " [USR] "
                    user2write = user1_is_sys if message["uid"] == "user1" else user2_is_sys
                    do_write = True
                else:
                    msgs2write += " [SYS] " if data["whoSawDoc"][0] == message["uid"] else " [USR] "
                    user2write = msgs2write
                    do_write = data["whoSawDoc"][0] == message["uid"]
                message["text"] = message["text"].replace("\n", " ")
                if do_write:
                    mem.write(current_fact[str(message["docIdx"])] + "\n")
                    src.write(user2write.strip() + "\n")
                    tgt.write(message["text"] + "\n")

                user1_is_sys += message["text"]
                user2_is_sys += message["text"]
                msgs2write += message["text"]

print("Removing temporary files...")
remove(path.join(current_dir, "repo.zip"))
rmtree(path.join(current_dir, "datasets-CMU_DoG-master"))
print("Done!")
