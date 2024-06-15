# DIEKAE: Difference Injection for Efficient Knowledge Augmentation and Editing of Large Language Models
This code is relative to the paper "DIEKAE: Difference Injection for Efficient Knowledge Augmentation and Editing of Large Language Models".

## Folder structure
Following is the folder structure of this repo alongside each folder description.
```bash
.
├── baselines                         # Includes the scripts used to collect the LoRA  and MEMIT baselines
│   ├── EasyEdit                      # A clone of the repository zjunlp/EasyEdit (see acknowledgements below)
│   ├── lora                          # Contains the scripts to train and evaluate our lora baselines
│   ├── sft                           # Contains the scripts to train and evaluate our sft baselines
│   └── plm                           # Contains the scripts to evaluate the plain plm baselines
├── data                              # Includes the scripts to download and preprocess all the datasets (see also 'build_all_dataset.sh')
├── llama_knowledge                   # Contains the scripts for our method
│   ├── encoder.py                    # Code for encoder (relies on LLaMA's implementation)
│   ├── model.py                      # Transformers 🤗 LLaMA model adapted to accept our encoders' output
│   ├── trainers.py                   # Contains trainer classes to train and finetune our method
│   ├── memory_usage.py               # Script to analyse the memory usage of the plain plm and our method
│   ├── finetune.py                   # Finetunes on all knowledge datasets
│   ├── train.py                      # Trains on all knowledge datasets
│   └── ...
├── build_all_datasets.sh             # Downloads and pre-processes the datasets
└── ...
```
In each method folder (e.g. `baselines/lora` or `llama_knowledge`), key scripts follow the same naming convention:
* `finetune.py`: finetunes the relative method on the knowledge datasets (cmu_dog, curio, dream, nat_ques, quasar_t, wow)
* `finetune_icr.py`: finetunes the relative method (while also adding a cls head) on the in-context reasoning datasets (proofwriter, folio)
* `finetune_edit.py`: finetunes the relative method on the knowledge editing datasets (counterfact)
* `test.py`: tests the relative method on the knowledge datasets. Returns loss, perplexity and running time (total and per dataset)
* `test_icr.py`: tests the relative method on the icr datasets. Returns classification accuracy. Note: in the case of `baselines/plm/test_icr.py` the LM head is used as no classification head is present.
* `test_edit.py`: tests the relavtive method on the knowledge editing datasets. Returns Efficacy (ES), Paraphrase (PS), Neighborhood (NS).

## Quick start
Experiments were made using Python 3.10.8

Requirements are in [requirements.txt](requirements.txt).

## Datasets
Following are the datasets used and their source. In each folder under [data/](data/) there is a script `build.py` to download and pre-process each dataset. The script [`build_all_datasets.sh`](build_all_datasets.sh) runs all the `build.py` for you.

## Acknowledgements
Part of this code was borrowed from [facebookresearch/llama](https://github.com/facebookresearch/llama/tree/main), [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit), [huggingface/transformers](https://github.com/huggingface/transformers/tree/v4.40.0), [hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI). Part of this code was inspired by [OpenGVLab/LLaMA-Adapter](https://github.com/OpenGVLab/LLaMA-Adapter), [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora).
