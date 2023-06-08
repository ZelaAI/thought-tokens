import pytest
from core.model import Tokenizer
from data.sequence import Sequence
import torch

@pytest.fixture
def prepare_data():
    t = Tokenizer()
    
    sequences = {
        'a': Sequence(0, context=t.encode("Question: 4+6= Answer:"), 
                      completions=[t.encode(' 10'), t.encode(' 12'), t.encode(' cat in the hat')],
                      max_new_tokens=0),
        'b': Sequence(0, context=t.encode("Question: Who is the CEO of SpaceX?\nAnswer: Elon Musk\nQuestion: Who is the president of the United States?\nAnswer:"), 
                      completions=[t.encode('')], 
                      max_new_tokens=3),
        'c': Sequence(0, context=t.encode("Question: Are you an AGI? Answer:"), 
                      completions=[t.encode(' Yes I am.')], 
                      max_new_tokens=0),
        'd': Sequence(0, context=t.encode('"Hello there..." said Obi-Wan Kenobi.'), 
                      completions=[t.encode(''), t.encode('')], 
                      max_new_tokens=15)
    }
    
    return sequences

def test_mask_positions(prepare_data):
    a = prepare_data['a']
    assert torch.allclose(a.get_pos_mask(), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 9, 10, 11]))

def test_inputs(prepare_data):
    a = prepare_data['a']
    assert torch.allclose(a.get_inputs(), torch.tensor([23433, 27, 577, 12, 23, 30, 37741, 27, 884, 1249, 5798, 275, 253, 7856]))

def test_target_positions(prepare_data):
    a = prepare_data['a']
    assert torch.allclose(a.get_target_pos_mask(), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 10, 11, 12, 13 ]))

def test_targets(prepare_data):
    a = prepare_data['a']
    c = prepare_data['c']
    assert torch.allclose(a.get_targets(False), torch.tensor([-1, -1, -1, -1, -1, -1, -1, 884, 1249, 5798, 275, 253, 7856, -1]))
    assert torch.allclose(c.get_targets(False), torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, 6279, 309, 717, 15, -1]))

def test_generate_positions(prepare_data):
    d = prepare_data['d']
    assert torch.allclose(d.get_generate_positions(), torch.tensor([12], dtype=torch.long))

def test_attn_mask_bounds(prepare_data):
    a = prepare_data['a']
    assert torch.allclose(a.get_attn_mask_bounds(False)[0], torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13]))
    assert torch.allclose(a.get_attn_mask_bounds(True)[0], torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))
    assert torch.allclose(a.get_attn_mask_bounds(True)[1], torch.tensor([13, 13, 13, 13, 13, 13, 13, 13, 8, 9, 13, 13, 13, 13]))
