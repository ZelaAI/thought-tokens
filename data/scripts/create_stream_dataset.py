"""
This script loads a Huggingface Dataset, manually shards it, and uploads it to a destination repo.

For use with `data/stream_dataset.py`
"""
source_datasets_repo_ids = ["ZelaAI/lj_speech_2048"]
num_shards = 10
destination_dataset_repo_id = "ZelaAI/lj_speech_2048_streamable"
create_validation_set = True
validation_set_size = 50


import os
import math
from huggingface_hub import HfApi
import json
from datasets import concatenate_datasets, load_dataset

print("Loading dataset...")
dataset = concatenate_datasets([load_dataset(v, split="train") for v in source_datasets_repo_ids])
dataset = dataset.shuffle(seed=42) # Can slow things down, comment out if dataset is singular and already shuffled

dataset_length = len(dataset)

os.makedirs("dataset_upload.ignore", exist_ok=True)

if create_validation_set:
    print("Creating validation set...")
    validation_set = dataset.select(range(dataset_length - validation_set_size, dataset_length))
    validation_set.to_parquet("dataset_upload.ignore/validation.parquet")
    dataset = dataset.select(range(dataset_length - validation_set_size))
    dataset_length = len(dataset)

config = {
    "num_shards": num_shards,
    "dataset_length": dataset_length,
}

with open("dataset_upload.ignore/config.json", "w") as f:
    json.dump(config, f)

print("Sharding dataset...")
# split dataset into shards
shard_size = math.ceil(dataset_length / num_shards)
for shard_id in range(num_shards):
    start = shard_id * shard_size
    end = min((shard_id + 1) * shard_size, dataset_length)
    
    # Save as a parquet file
    dataset.select(range(start, end)).to_parquet(f"dataset_upload.ignore/shard_{shard_id}.parquet")

api = HfApi()

print("Uploading to HuggingFace Hub...")
api.create_repo(repo_id=destination_dataset_repo_id, exist_ok=True, repo_type="dataset")


for i in range(num_shards//10):
    # break up into chunks to avoid timeouts
    allow_patterns = [f"shard_{shard_id}.parquet" for shard_id in range(i*10, (i+1)*10)] + ["config.json", "validation.parquet"]

    print(f"Uploading shards {i*10} to {(i+1)*10}...")

    api.upload_folder(
        repo_id=destination_dataset_repo_id,
        folder_path="dataset_upload.ignore",
        repo_type="dataset",
        allow_patterns=allow_patterns
    )


print("Cleaning up....")
os.remove("dataset_upload.ignore/config.json")
for shard_id in range(num_shards):
    os.remove(f"dataset_upload.ignore/shard_{shard_id}.parquet")
if validation_set:
    os.remove("dataset_upload.ignore/validation.parquet")
os.rmdir("dataset_upload.ignore")

print("Done!")