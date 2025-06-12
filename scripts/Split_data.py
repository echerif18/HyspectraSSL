import sys
import os
import warnings
import subprocess


import subprocess
warnings.filterwarnings('ignore')  # ignore warnings, like ZeroDivision

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from src.utils_data import *

dataset_repo_url = "https://huggingface.co/datasets/Avatarr05/GreenHyperSpectra"

directory_path = os.path.join(project_root, "Splits")
directory_path_Ds = os.path.join(project_root, "Datasets")
directory_path_Ds_unlb = os.path.join(directory_path_Ds, "unlb")

import subprocess

# !sudo apt-get install git-lfs -y
# !git lfs install

# # Run Git LFS install
# subprocess.run(["git", "lfs", "install"], check=True)

# Clone the repo into your desired directory
subprocess.run([
    "git", "clone",
    dataset_repo_url,
    directory_path_Ds
], check=True)


num_splits = 20  # Number of output splits
chunk_size = 5000  # Tune based on your memory constraints


os.makedirs(directory_path, exist_ok=True)  # Create the output folder if it doesn't exist
split_csvs_with_proportions_sequential(directory_path_Ds_unlb, directory_path, num_splits, chunk_size)
# split_parquets_with_proportions_sequential(directory_path_Ds_unlb, directory_path, num_splits, chunk_size)
