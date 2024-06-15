import json
from os import path, remove, rmdir, rename
import subprocess
from glob import glob


QUALITY_FILTER = True
paraphrase_used_fact = True  # Paraphrase mem if tgt == mem

if paraphrase_used_fact:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from os import environ

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "humarin/chatgpt_paraphraser_on_T5_base",
        cache_dir=environ.get("HF_CACHE_DIR", None)
    )
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")

current_dir = path.abspath(path.dirname(__file__))

print("Downloading dataset...")

# requires parlai (pip install parlai)
subprocess.run(["parlai", "display_data", "--task", "wizard_of_wikipedia", "--dp", current_dir])
for file in glob(path.join(current_dir, "wizard_of_wikipedia", "*")):
    rename(file, path.join(current_dir, file.split("/")[-1]))
print("Dataset downloaded")

print("Parsing files...")
for split, filename in zip(["train", "val", "test"], ["train", "valid_topic_split", "test_topic_split"]):
    with open(path.join(current_dir, f"{split}.src"), "w") as src, \
        open(path.join(current_dir, f"{split}.tgt"), "w") as tgt, \
            open(path.join(current_dir, f"{split}.mem"), "w") as mem:

        with open(path.join(current_dir, f"{filename}.json"), "r", encoding="utf-8") as raw_file:
            data = json.load(raw_file)
        for dialogs in data:
            if QUALITY_FILTER and dialogs["wizard_eval"] < 4:
                continue
            msgs2write = ""
            persona = dialogs["persona"]
            for dialog in dialogs["dialog"]:
                text = dialog["text"]
                if "wizard" in dialog["speaker"].lower():
                    if msgs2write:
                        msgs2write += " "
                    msgs2write += "[SYS] "
                    if "no_passages_used" in dialog["checked_sentence"]:
                        # No mem data available, use answer instead
                        mem_data = text
                        if paraphrase_used_fact:
                            batch = tokenizer(
                                f"paraphrase: {mem_data}",
                                return_tensors="pt",
                                padding="longest"
                            )["input_ids"].cuda()
                            generated_ids = model.generate(batch, max_new_tokens=len(batch[0]) + 20)
                            mem_data = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    else:
                        try:
                            mem_data = list(dialog["checked_sentence"].values())[0]  # FIXME: may not contain answer
                        except IndexError:
                            mem_data = list(dialog["checked_passage"].values())[0]
                    mem.write(" [SEP] ".join([persona, mem_data]) + "\n")
                    src.write(msgs2write + "\n")
                    tgt.write(text + "\n")
                else:
                    if msgs2write:
                        msgs2write += " "
                    msgs2write += "[USR] "
                msgs2write += text
print("Removing temporary files...")
for file in glob(path.join(current_dir, "*.json")):
    remove(file)
remove(path.join(current_dir, "wizard_of_wikipedia", ".built"))
rmdir(path.join(current_dir, "wizard_of_wikipedia"))
print("Done!")
