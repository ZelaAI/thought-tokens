from torch.utils.data import IterableDataset
from huggingface_hub import hf_hub_download
import pandas as pd
import torch
import math
import json


"""
This is a custom IterableDataset that manually loads shards from a HuggingFace Hub
repo in the above format.

We're using this 'special' dataset because the HuggingFace Datasets when
streamed don't quite have enough info to performantly 'skip' shards.
"""
class HuggingfaceStreamDataset(IterableDataset):
    def __init__(self, huggingface_name, skip_to=0):
        self.huggingface_name = huggingface_name
        self.global_index = skip_to

        # start by getting the config
        config_file = hf_hub_download(
            repo_id=self.huggingface_name,
            filename="config.json",
            repo_type="dataset",
        )
        
        with open(config_file, "r") as f:
            config = json.load(f)
        
        self.num_shards = config["num_shards"]
        self.dataset_length = config["dataset_length"]
        self.shard_size = math.ceil(self.dataset_length / self.num_shards)

        print(f"Loading from dataset with {self.dataset_length} examples, split into {self.num_shards} shards of size {self.shard_size}")
        self.shards = {}
        
        # prefetch the first shard
        self.load_shard(self.global_index // self.shard_size)

    def load_shard(self, shard_id):
        print(f"Loading shard {shard_id}")
        shard_file = hf_hub_download(
            repo_id=self.huggingface_name,
            filename=f"shard_{shard_id}.parquet",
            repo_type="dataset",
        )
        self.shards[shard_id] = pd.read_parquet(shard_file)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            offset = 0
            increase = 1
        else:
            offset = worker_info.id
            increase = worker_info.num_workers

        self.global_index += offset

        while self.global_index < self.dataset_length:
            shard_id = self.global_index // self.shard_size
            shard_offset = self.global_index % self.shard_size

            if shard_id not in self.shards:
                raise ValueError(f"Unexpected: Shard {shard_id} not loaded!")
            if shard_id + 1 not in self.shards and shard_id + 1 < self.num_shards:
                # ensure we prefetch the next shard to keep things fast
                self.load_shard(shard_id + 1)

            shard = self.shards[shard_id]
        
            yield shard.iloc[shard_offset]
            
            self.global_index += increase


stream_dataset = HuggingfaceStreamDataset("ZelaAI/minipile_test")