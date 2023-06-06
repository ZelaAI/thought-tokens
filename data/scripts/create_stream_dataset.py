"""
This script loads a Huggingface Dataset, manually shards it, and uploads it to a destination repo.

For use with `data/stream_dataset.py`
"""
source_dataset_repo_id = "alexedw/minipile_recreation_tiny"
num_shards = 10
destination_dataset_repo_id = "ZelaAI/minipile_test"

import os
import math
from huggingface_hub import HfApi
import json
from datasets import load_dataset

print("Loading dataset...")
dataset_sample = load_dataset(source_dataset_repo_id, split="train")

dataset_length = len(dataset_sample)

config = {
    "num_shards": num_shards,
    "dataset_length": dataset_length,
}
os.makedirs("dataset_upload.ignore", exist_ok=True)
with open("dataset_upload.ignore/config.json", "w") as f:
    json.dump(config, f)

print("Sharding dataset...")
# split dataset into shards
shard_size = math.ceil(dataset_length / num_shards)
for shard_id in range(num_shards):
    start = shard_id * shard_size
    end = min((shard_id + 1) * shard_size, dataset_length)
    
    # Save as a parquet file
    dataset_sample.select(range(start, end)).to_parquet(f"dataset_upload.ignore/shard_{shard_id}.parquet")

api = HfApi()

print("Uploading to HuggingFace Hub...")
api.create_repo(repo_id=destination_dataset_repo_id, exist_ok=True, repo_type="dataset")
api.upload_folder(
    repo_id=destination_dataset_repo_id,
    folder_path="dataset_upload.ignore",
    repo_type="dataset",
)

print("Cleaning up....")
os.remove("dataset_upload.ignore/config.json")
for shard_id in range(num_shards):
    os.remove(f"dataset_upload.ignore/shard_{shard_id}.parquet")
os.rmdir("dataset_upload.ignore")

print("Done!")