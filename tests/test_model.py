# First we are going to test the model.
# Then we're going to rewrite against it in tinygrad.
import pytest
from core.model import GPTConfig, GPT, get_rotary_sin_cos, rotary_apply, CausalSelfAttention, MLP, DenseCapture, DenseInject, Block
from core.model_tiny import MLP as TinyMLP
import torch
from tinygrad.tensor import Tensor
import numpy as np

def test_mlp():
    batch_size, seq_length = 4, 64
    config = GPTConfig(
        n_embd=128,
        n_embd_proj=128*4,
        bias=True,
    )
    mlp = MLP(config)
    tiny_mlp = TinyMLP(config)
    
    # copy the weights from the original model to the tinygrad model    
    with torch.no_grad():
        tiny_mlp.c_fc.weight.assign(Tensor(mlp.c_fc.weight.numpy()))
        tiny_mlp.c_fc.bias.assign(Tensor(mlp.c_fc.bias.numpy()))
        tiny_mlp.c_proj.weight.assign(Tensor(mlp.c_proj.weight.numpy()))
        tiny_mlp.c_proj.bias.assign(Tensor(mlp.c_proj.bias.numpy()))

    x = torch.randn(batch_size, seq_length, config.n_embd)

    # Pass the input through the MLP
    output = mlp(x)
    output_tiny = tiny_mlp(Tensor(x.numpy()))

    # Check the output values
    np.testing.assert_allclose(output_tiny.numpy(), output.detach().numpy(), atol=5e-4, rtol=1e-5)
    
    
