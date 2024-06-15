import urllib.request
import json
from os import path, remove
from collections import defaultdict
from glob import glob
from re import sub
from tqdm import tqdm

# CHANGE DATASET PARSING
filter_unused_facts = True  # if true will only write used facts in memory. WARNING: tgt=mem in this case
paraphrase_used_fact = True  # advised use if above is true. Paraphrase mem so that tgt != mem

if paraphrase_used_fact:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from os import environ

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "humarin/chatgpt_paraphraser_on_T5_base",
        cache_dir=environ.get("HF_CACHE_DIR", None)
    )
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")


def remove_refs(s):
    # removes from s appearances of < ref > ....
    s = sub(r"< ref (?:(?!< / ref >).)*< / ref >", "", s)
    s = sub(r"< ref [^\{]*\{\{ (?:(?!< / ref >).)*(\}\})?(< / ref >)?", "", s)  # FIXME: not perfect, in the middle should also avoid }} pattern
    s = sub(r"< ref [^\[]*\[[^\]]*\](< / ref >)?", "", s)
    s = sub("< ref >", "", s)
    return s


current_dir = path.abspath(path.dirname(__file__))

print("Downloading dataset...")
urllib.request.urlretrieve("https://github.com/facebookresearch/curiosity/raw/main/dialog_data/fact_db_links.json", path.join(current_dir, "facts.json"))
urllib.request.urlretrieve("https://github.com/facebookresearch/curiosity/raw/main/dialog_data/curiosity_dialogs.train.json", path.join(current_dir, "train.json"))
urllib.request.urlretrieve("https://github.com/facebookresearch/curiosity/raw/main/dialog_data/curiosity_dialogs.val.json", path.join(current_dir, "val.json"))
urllib.request.urlretrieve("https://github.com/facebookresearch/curiosity/raw/main/dialog_data/curiosity_dialogs.test.json", path.join(current_dir, "test.json"))
print("Dataset downloaded")

with open(path.join(current_dir, "facts.json"), "r") as raw_file:
    facts = json.load(raw_file)

facts = defaultdict(lambda: {"text": ""}, facts["linked_facts"])

print("Parsing files...")
for split in ["train", "val", "test"]:
    with open(path.join(current_dir, f"{split}.json"), "r", encoding="utf-8") as raw_file:
        data = json.load(raw_file)
    with open(path.join(current_dir, f"{split}.src"), "w", encoding="utf-8") as src, \
        open(path.join(current_dir, f"{split}.tgt"), "w", encoding="utf-8") as tgt, \
            open(path.join(current_dir, f"{split}.mem"), "w", encoding="utf-8") as mem:

        for dialog in tqdm(data["dialogs"]):
            msgs = dialog["messages"]
            msgs2write = ""
            for msg in msgs:
                msg["message"] = remove_refs(msg["message"])
                if msg["sender"] == "assistant":
                    used_facts = msg["facts"]
                    if filter_unused_facts:
                        used_facts = list(filter(lambda x: x["used"], msg["facts"]))

                    facts2write = filter(None, map(lambda x: facts[str(x["fid"])]["text"], used_facts))
                    facts2write = list(map(remove_refs, facts2write))
                    if not facts2write:
                        # don't have any memory, use tgt and paraphrase
                        facts2write = [msg["message"]]
                    if paraphrase_used_fact:
                        assert len(facts2write) == 1
                        batch = tokenizer(
                            f"paraphrase: {facts2write[0]}",
                            return_tensors="pt",
                            padding="longest"
                        )["input_ids"].cuda()
                        generated_ids = model.generate(batch, max_new_tokens=len(batch[0])+20)
                        facts2write = [tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]]
                    mem.write(" [SEP] ".join(facts2write) + "\n")
                    src.write(msgs2write + " [SYS] " + "\n")
                    tgt.write(msg["message"] + "\n")
                    msgs2write += " [SYS] " + msg["message"]
                else:
                    if msgs2write:
                        msgs2write += " "
                    msgs2write += "[USR] " + msg["message"]
print("Removing temporary files...")
for file in glob(path.join(current_dir, "*.json")):
    remove(file)
print("Done!")
