from dataclasses import dataclass
import random
from typing import List
import torch
from core.model import THOUGHT_TOKEN_ID
from data.sequence import Sequence

max_seq_len = 512

class TrainSequence(Sequence):
    """
    Basic train sequence class, just holds inputs and targets
    Does not support masking between sequences or anything fancy
    """
    def __init__(self, tokens):
        self.inputs = tokens[:-1]
        self.targets = tokens[1:]
        
        if len(self.inputs) < max_seq_len:
            self.inputs = torch.cat([self.inputs, torch.zeros(max_seq_len - len(self.inputs), dtype=torch.long)])
            self.targets = torch.cat([self.targets, -torch.ones(max_seq_len - len(self.targets), dtype=torch.long)])
        
        self.length = len(self.inputs)

    def __str__(self):
        return f"<TrainSequence length={self.length} inputs={self.inputs}, targets={self.targets}>"

    def get_attn_mask_bounds(self):
        top = torch.arange(self.length, dtype=torch.long)
        bottom = torch.ones(self.length, dtype=torch.long) * (self.length - 1)

        return top, bottom

    def add_thought_tokens(self, tokens):
        num_to_insert = 12
        short_tokens = tokens[:-num_to_insert]
        
        for i in range(num_to_insert):
            index = random.randint(0, len(short_tokens))
            short_tokens.insert(index, THOUGHT_TOKEN_ID)
        
        return short_tokens


@dataclass
class TrainBatch:
    """
    Simple train batch class.
    Assumes all inputs are of same length and are pre-packed into a single tensor.
    """
    inputs: torch.Tensor
    targets: torch.Tensor
    attn_mask_bound_top: torch.Tensor
    attn_mask_bound_bottom: torch.Tensor
    max_dense_tokens: int

    def to(self, device):
        if 'cpu' not in device:
            self.inputs = self.inputs.to(device, non_blocking=True)
            self.targets = self.targets.to(device, non_blocking=True)
            self.attn_mask_bound_top = self.attn_mask_bound_top.to(device, non_blocking=True)
            self.attn_mask_bound_bottom = self.attn_mask_bound_bottom.to(device, non_blocking=True)
        return self

    @classmethod
    def collate_fn(cls, sequences: List[TrainSequence]):
        inputs = torch.stack([seq.inputs for seq in sequences])
        targets = torch.stack([seq.targets for seq in sequences])
        
        tops, bottoms = zip(*[seq.get_attn_mask_bounds() for seq in sequences])
    
        attn_mask_bound_top = torch.stack(tops)
        attn_mask_bound_bottom = torch.stack(bottoms)
        
        max_dense_tokens = 0
        
        return TrainBatch(inputs, targets, attn_mask_bound_top, attn_mask_bound_bottom, max_dense_tokens)
