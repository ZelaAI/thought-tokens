from dataclasses import dataclass
import random
from typing import List
import torch
from data.sequence import Sequence

max_seq_len = 2048

def get_inputs_targets_from_tokens(tokens):
    inputs = tokens[:-1]
    targets = tokens[1:].clone()
    
    if len(inputs) < max_seq_len:
        inputs = torch.cat([inputs, torch.zeros(max_seq_len - len(inputs), dtype=torch.long)])
        targets = torch.cat([targets, -torch.ones(max_seq_len - len(targets), dtype=torch.long)])
    
    return inputs, targets


class TrainSequence(Sequence):
    """
    Basic train sequence class, just holds inputs and targets
    Does not support masking between sequences or anything fancy
    """
    def __init__(self, tokens):
        self.inputs, self.targets = get_inputs_targets_from_tokens(tokens)
        self.length = len(self.inputs)

    def __str__(self):
        return f"<TrainSequence length={self.length} inputs={list(self.inputs)}, targets={list(self.inputs)}>"

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

    def __hash__(self) -> int:
        # Used for testing purposes
        return hash((self.inputs, self.targets))
    
    @classmethod
    def collate_fn(cls, sequences: List[TrainSequence]):
        inputs = torch.stack([seq.inputs for seq in sequences])
        targets = torch.stack([seq.targets for seq in sequences])
        
        return TrainBatch(inputs, targets)

class AudioTrainSequence(Sequence):
    """
    Basic train sequence class, just holds inputs and targets
    Does not support masking between sequences or anything fancy
    """
    def __init__(self, text_tokens, audio_tokens_1, audio_tokens_2):
        self.inputs_text, self.targets_text = get_inputs_targets_from_tokens(text_tokens)
        self.inputs_audio_1, self.targets_audio_1 = get_inputs_targets_from_tokens(audio_tokens_1)
        self.inputs_audio_2, self.targets_audio_2 = get_inputs_targets_from_tokens(audio_tokens_2)
        
        self.length = len(self.inputs_text)

    def __str__(self):
        return f"<AudioTrainSequence length={self.length}>"

@dataclass
class AudioTrainBatch:
    """
    Simple train batch class.
    Assumes all inputs are of same length and are pre-packed into a single tensor.
    """
    inputs_text: torch.Tensor
    targets_text: torch.Tensor
    
    inputs_audio_1: torch.Tensor
    targets_audio_1: torch.Tensor
    
    inputs_audio_2: torch.Tensor
    targets_audio_2: torch.Tensor

    def to(self, device):
        if 'cpu' not in device:
            self.inputs_text = self.inputs_text.to(device, non_blocking=True)
            self.targets_text = self.targets_text.to(device, non_blocking=True)
            self.inputs_audio_1 = self.inputs_audio_1.to(device, non_blocking=True)
            self.targets_audio_1 = self.targets_audio_1.to(device, non_blocking=True)
            self.inputs_audio_2 = self.inputs_audio_2.to(device, non_blocking=True)
            self.targets_audio_2 = self.targets_audio_2.to(device, non_blocking=True)
            
        return self

    @classmethod
    def collate_fn(cls, sequences: List[AudioTrainSequence]):
        inputs_text = torch.stack([seq.inputs_text for seq in sequences])
        targets_text = torch.stack([seq.targets_text for seq in sequences])
        
        inputs_audio_1 = torch.stack([seq.inputs_audio_1 for seq in sequences])
        targets_audio_1 = torch.stack([seq.targets_audio_1 for seq in sequences])
        
        inputs_audio_2 = torch.stack([seq.inputs_audio_2 for seq in sequences])
        targets_audio_2 = torch.stack([seq.targets_audio_2 for seq in sequences])
        
        return AudioTrainBatch(
            inputs_text, targets_text,
            inputs_audio_1, targets_audio_1,
            inputs_audio_2, targets_audio_2
        )