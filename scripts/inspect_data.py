
import os
import sys
import warnings
import torch
import numpy as np
import pickle

# Suppress warnings
warnings.filterwarnings("ignore")

# Add paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "dreamer-datasets"))

import ENV
ENV.init_paths(project_name="DriveDreamer2")

from dreamer_datasets import load_dataset

DATA_PATH = "/DriveDreamer2/dreamer-data/v1.0-mini/cam_all_val/v0.0.2"
dataset = load_dataset(DATA_PATH)

print(f"Sample 0 'calib' keys: {dataset[0]['calib'].keys()}")
print(f"Sample 0 'cam2ego':\n{dataset[0]['calib']['cam2ego']}")
