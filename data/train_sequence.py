from dataclasses import dataclass
import random
from typing import List
import torch
from data.sequence import Sequence

max_seq_len = 512

class TrainSequence(Sequence):
    """
    Basic train sequence class, just holds inputs and targets
    Does not support masking between sequences or anything fancy
    """
    def __init__(self, tokens, add_thought_tokens=True):
        self.inputs = tokens[:-1]
        self.targets = tokens[1:].clone()
        
        if len(self.inputs) < max_seq_len:
            self.inputs = torch.cat([self.inputs, torch.zeros(max_seq_len - len(self.inputs), dtype=torch.long)])
            self.targets = torch.cat([self.targets, -torch.ones(max_seq_len - len(self.targets), dtype=torch.long)])
        
        self.length = len(self.inputs)

    def __str__(self):
        return f"<TrainSequence length={self.length} inputs={self.inputs}, targets={self.targets}>"

@dataclass
class TrainBatch:
    """
    Simple train batch class.
    Assumes all inputs are of same length and are pre-packed into a single tensor.
    """
    inputs: torch.Tensor
    targets: torch.Tensor

    def to(self, device):
        if 'cpu' not in device:
            self.inputs = self.inputs.to(device, non_blocking=True)
            self.targets = self.targets.to(device, non_blocking=True)
        return self

    @classmethod
    def collate_fn(cls, sequences: List[TrainSequence]):
        inputs = torch.stack([seq.inputs for seq in sequences])
        targets = torch.stack([seq.targets for seq in sequences])
        
        return TrainBatch(inputs, targets)
