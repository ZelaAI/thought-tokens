# First we are going to test the model.
# Then we're going to rewrite against it in tinygrad.
import pytest
from core.model import GPTConfig, GPT, get_rotary_sin_cos, rotary_apply, CausalSelfAttention, MLP, DenseCapture, DenseInject, Block, apply_rotary_mask
from core.model_tiny import MLP as TinyMLP, get_rotary_sin_cos as tiny_get_rotary_sin_cos, rotary_apply as tiny_rotary_apply, apply_rotary_mask as tiny_apply_rotary_mask, CausalSelfAttention as TinyCausalSelfAttention, Block as TinyBlock
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
    
def test_get_rotary_sin_cos():
    config = GPTConfig()
    max_seq_len = 1024
    
    sin, cos = get_rotary_sin_cos(max_seq_len, config)
    tiny_sin, tiny_cos = tiny_get_rotary_sin_cos(max_seq_len, config)

    np.testing.assert_allclose(sin.numpy(), tiny_sin.numpy(), atol=5e-4, rtol=1e-5)
    np.testing.assert_allclose(cos.numpy(), tiny_cos.numpy(), atol=5e-4, rtol=1e-5)    


def test_apply_rotary_mask():
    pos_mask = torch.arange(1024, dtype=torch.long).unsqueeze(0)
    pos_mask_tiny = Tensor(pos_mask.numpy())
    
    sin_cached, cos_cached = get_rotary_sin_cos(1024, GPTConfig())
    tiny_sin_cached, tiny_cos_cached = Tensor(sin_cached.numpy()), Tensor(cos_cached.numpy())
    
    sin, cos = apply_rotary_mask(pos_mask, sin_cached, cos_cached)
    tiny_sin, tiny_cos = tiny_apply_rotary_mask(pos_mask_tiny, tiny_sin_cached, tiny_cos_cached)
    
    np.testing.assert_allclose(sin.numpy(), tiny_sin.numpy(), atol=5e-4, rtol=1e-5)
    np.testing.assert_allclose(cos.numpy(), tiny_cos.numpy(), atol=5e-4, rtol=1e-5)

    
    
def test_rotary_apply():
    batch_size, n_head, seq_length, head_size, rotary_pct = 4, 8, 1024, 128, 0.25
    rotary_ndims = int(head_size * rotary_pct)
    sin_cached, cos_cached = get_rotary_sin_cos(seq_length, GPTConfig(n_embd=n_head*head_size, n_head=n_head, rotary_pct=rotary_pct))
    
    sin, cos = apply_rotary_mask(torch.arange(seq_length, dtype=torch.long).unsqueeze(0), sin_cached, cos_cached) 
    tiny_sin, tiny_cos = Tensor(sin.numpy()), Tensor(cos.numpy())
    
    x = torch.randn(batch_size, n_head, seq_length, head_size)
    tiny_x = Tensor(x.numpy())
    
    # Pass the input through the rotary apply
    output = rotary_apply(x, sin, cos, rotary_ndims)
    output_tiny = tiny_rotary_apply(tiny_x, tiny_sin, tiny_cos, rotary_ndims)
    
    # Check the output values
    np.testing.assert_allclose(output_tiny.numpy(), output.detach().numpy(), atol=5e-4, rtol=1e-5)


def test_casual_self_attention():
    batch_size, seq_length, n_embd = 4, 1024, 768

    config = GPTConfig(n_embd=n_embd)
    sin_cached, cos_cached = get_rotary_sin_cos(seq_length, config)
    sin, cos = apply_rotary_mask(torch.arange(seq_length, dtype=torch.long).unsqueeze(0), sin_cached, cos_cached) 
    tiny_sin, tiny_cos = Tensor(sin.numpy()), Tensor(cos.numpy())
    
    x = torch.randn(batch_size, seq_length, n_embd)
    tiny_x = Tensor(x.numpy())
    
    causal_self_attention = CausalSelfAttention(config)
    tiny_causal_self_attention = TinyCausalSelfAttention(config)

    # copy the weights from the original model to the tinygrad model
    with torch.no_grad():
        tiny_causal_self_attention.c_attn.weight.assign(Tensor(causal_self_attention.c_attn.weight.numpy()))
        tiny_causal_self_attention.c_attn.bias.assign(Tensor(causal_self_attention.c_attn.bias.numpy()))
        tiny_causal_self_attention.c_proj.weight.assign(Tensor(causal_self_attention.c_proj.weight.numpy()))
        tiny_causal_self_attention.c_proj.bias.assign(Tensor(causal_self_attention.c_proj.bias.numpy()))
        
    output = causal_self_attention(x, sin, cos).detach().numpy()
    output_tiny = tiny_causal_self_attention(tiny_x, tiny_sin, tiny_cos).numpy()
    np.testing.assert_allclose(output_tiny, output, atol=5e-4, rtol=1e-5)
    
def test_block():
    batch_size, seq_length, n_embd = 4, 1024, 768

    config = GPTConfig(n_embd=n_embd)
    sin_cached, cos_cached = get_rotary_sin_cos(seq_length, config)
    sin, cos = apply_rotary_mask(torch.arange(seq_length, dtype=torch.long).unsqueeze(0), sin_cached, cos_cached) 
    tiny_sin, tiny_cos = Tensor(sin.numpy()), Tensor(cos.numpy())
    
    x = torch.randn(batch_size, seq_length, n_embd)
    tiny_x = Tensor(x.numpy())
    
    block = Block(config)
    tiny_block = TinyBlock(config)

    # copy the weights from the original model to the tinygrad model
    with torch.no_grad():
        tiny_block.ln_1.weight.assign(Tensor(block.ln_1.weight.numpy()))
        tiny_block.ln_1.bias.assign(Tensor(block.ln_1.bias.numpy()))
        tiny_block.attn.c_attn.weight.assign(Tensor(block.attn.c_attn.weight.numpy()))
        tiny_block.attn.c_attn.bias.assign(Tensor(block.attn.c_attn.bias.numpy()))
        tiny_block.attn.c_proj.weight.assign(Tensor(block.attn.c_proj.weight.numpy()))
        tiny_block.attn.c_proj.bias.assign(Tensor(block.attn.c_proj.bias.numpy()))
        tiny_block.ln_2.weight.assign(Tensor(block.ln_2.weight.numpy()))
        tiny_block.ln_2.bias.assign(Tensor(block.ln_2.bias.numpy()))
        tiny_block.mlp.c_fc.weight.assign(Tensor(block.mlp.c_fc.weight.numpy()))
        tiny_block.mlp.c_fc.bias.assign(Tensor(block.mlp.c_fc.bias.numpy()))
        tiny_block.mlp.c_proj.weight.assign(Tensor(block.mlp.c_proj.weight.numpy()))
        tiny_block.mlp.c_proj.bias.assign(Tensor(block.mlp.c_proj.bias.numpy()))
        
    output = block(x, sin, cos, None).detach().numpy()
    output_tiny = tiny_block(tiny_x, tiny_sin, tiny_cos, None).numpy()
    np.testing.assert_allclose(output_tiny, output, atol=5e-4, rtol=1e-5)
