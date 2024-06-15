# used to import from datasets in llama_knowledge/datasets.py

import sys
import os

# Get the current directory of the running script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate two levels up to reach the project root directory
project_root = os.path.join(current_dir, '..', '..')

# Add the project root directory to the sys.path
sys.path.append(project_root)

from baselines.lora.lora_datasets import *
