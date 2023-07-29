from torch.utils.data import IterableDataset
from huggingface_hub.file_download import hf_hub_download
import pandas as pd
import torch
import math
import json

from data.train_sequence import TrainSequence, AudioTrainSequence

"""
This is a custom IterableDataset that manually loads shards from a HuggingFace Hub
repo in the above format.

We're using this 'special' dataset because the HuggingFace Datasets when
streamed don't quite have enough info to performantly 'skip' shards.
"""
class HuggingfaceStreamDataset(IterableDataset):
    def __init__(self, huggingface_name, skip_to=0, audio=False):
        self.huggingface_name = huggingface_name
        self.skip_to = skip_to
        self.audio = audio
        
        # start by getting the config
        config_file = hf_hub_download(
            repo_id=self.huggingface_name,
            filename="config.json",
            repo_type="dataset",
        )
        assert config_file is not None, f"Couldn't find config.json in {self.huggingface_name}"
        
        with open(config_file, "r") as f:
            config = json.load(f)
        
        self.num_shards = config["num_shards"]
        self.dataset_length = config["dataset_length"]
        self.shard_size = math.ceil(self.dataset_length / self.num_shards)

        print(f"Loading from dataset with {self.dataset_length} examples, split into {self.num_shards} shards of size {self.shard_size}")
        self.shards = {}
        
        # prefetch the first shard
        self.load_shard(self.skip_to // self.shard_size)

    def load_shard(self, shard_id):
        print(f"Loading shard {shard_id}")
        shard_file = hf_hub_download(
            repo_id=self.huggingface_name,
            filename=f"shard_{shard_id}.parquet",
            repo_type="dataset",
        )
        assert shard_file is not None, f"Couldn't find shard_{shard_id}.parquet in {self.huggingface_name}"
        self.shards[shard_id] = pd.read_parquet(shard_file)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info() # type: ignore

        if worker_info is None:
            offset = 0
            increase = 1
        else:
            offset = worker_info.id
            increase = worker_info.num_workers

        global_index = offset + self.skip_to

        while global_index < self.dataset_length:
            shard_id = global_index // self.shard_size
            shard_offset = global_index % self.shard_size

            if shard_id not in self.shards:
                self.load_shard(shard_id)
            if worker_info.id == 0 and shard_id + 1 not in self.shards and shard_id + 1 < self.num_shards:
                # download the next shard to keep things fast
                self.load_shard(shard_id + 1)

            shard = self.shards[shard_id]
            
            if self.audio:
                text_tokens = torch.tensor(shard.iloc[shard_offset]['text_tokens'])
                audio_tokens_1 = torch.tensor(shard.iloc[shard_offset]['audio_tokens_1'])
                audio_tokens_2 = torch.tensor(shard.iloc[shard_offset]['audio_tokens_2'])
                yield AudioTrainSequence(text_tokens, audio_tokens_1, audio_tokens_2)
            else:
                tokens = torch.tensor(shard.iloc[shard_offset]['tokens'])
                yield TrainSequence(tokens)
        
            
            global_index += increase
