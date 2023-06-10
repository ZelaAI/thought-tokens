import pytest
import torch
from data.train_sequence import TrainSequence, THOUGHT_TOKEN_ID, max_seq_len

def test_basic_inputs_and_targets():
    tokens = torch.arange(10)  # just an example, use your own tensor here
    sequence = TrainSequence(tokens, add_thought_tokens=False)

    assert torch.allclose(sequence.inputs[:9], torch.tensor([0,1,2,3,4,5,6,7,8]))
    assert torch.allclose(sequence.targets[:9], torch.tensor([1,2,3,4,5,6,7,8,9]))

def test_padding_works():
    tokens = torch.arange(max_seq_len - 2)  # tensor 2 elements short of max_seq_len
    sequence = TrainSequence(tokens, add_thought_tokens=False)

    # The padding should be zeros for inputs and -1 for targets
    assert torch.allclose(sequence.inputs[len(tokens):], torch.tensor([0,0]))
    assert torch.allclose(sequence.targets[len(tokens):], torch.tensor([-1,-1]))

def test_thought_tokens_added():
    tokens = torch.arange(100)
    sequence = TrainSequence(tokens)

    # Count how many thought tokens are in the inputs and targets
    num_thoughts_in_inputs = torch.sum(sequence.inputs == THOUGHT_TOKEN_ID)
    num_thoughts_in_targets = torch.sum(sequence.targets == THOUGHT_TOKEN_ID)

    # Check if the count of thought tokens are as expected
    assert num_thoughts_in_inputs == 12
    assert num_thoughts_in_targets == 0  # we should be masking them here

    print(sequence.inputs)
    