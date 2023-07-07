# First we are going to test the model.
# Then we're going to rewrite against it in tinygrad.
import pytest
from core.model import GPTConfig, GPT, get_rotary_sin_cos, rotary_apply, CausalSelfAttention, MLP, DenseCapture, DenseInject, Block, apply_rotary_mask
from core.model_tiny import MLP as TinyMLP, get_rotary_sin_cos as tiny_get_rotary_sin_cos, rotary_apply as tiny_rotary_apply, CausalSelfAttention as TinyCausalSelfAttention, Block as TinyBlock, GPT as TinyGPT
import torch
from tinygrad.tensor import Tensor
import numpy as np
from tinygrad.state import load_state_dict


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
    
    sin_cached, cos_cached = get_rotary_sin_cos(max_seq_len, config)
    sin, cos = apply_rotary_mask(torch.arange(max_seq_len, dtype=torch.long).unsqueeze(0), sin_cached, cos_cached)
    
    tiny_sin, tiny_cos = tiny_get_rotary_sin_cos(max_seq_len, config)

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

def test_gpt():
    batch_size, seq_length = 4, 64
    config = GPTConfig(block_size=seq_length)
    
    x = torch.randint(0, 1000, (batch_size, seq_length))
    tiny_x = Tensor(x.numpy())  

    gpt = GPT(config)
    tiny_gpt = TinyGPT(config)
    
    # copy the weights from the original model to the tinygrad model
    with torch.no_grad():
        tiny_gpt.ln_f.weight.assign(
            Tensor(gpt.transformer.ln_f.weight.numpy())
        )
        tiny_gpt.ln_f.bias.assign(Tensor(gpt.transformer.ln_f.bias.numpy()))
        tiny_gpt.wte.weight.assign(Tensor(gpt.transformer.wte.weight.numpy()))
        tiny_gpt.lm_head.weight.assign(Tensor(gpt.lm_head.weight.numpy()))
        
        for i in range(config.n_layer):
            tiny_gpt.h[i].ln_1.weight.assign(Tensor(gpt.transformer.h[i].ln_1.weight.numpy()))
            tiny_gpt.h[i].ln_1.bias.assign(Tensor(gpt.transformer.h[i].ln_1.bias.numpy()))
            tiny_gpt.h[i].attn.c_attn.weight.assign(Tensor(gpt.transformer.h[i].attn.c_attn.weight.numpy()))
            tiny_gpt.h[i].attn.c_attn.bias.assign(Tensor(gpt.transformer.h[i].attn.c_attn.bias.numpy()))
            tiny_gpt.h[i].attn.c_proj.weight.assign(Tensor(gpt.transformer.h[i].attn.c_proj.weight.numpy()))
            tiny_gpt.h[i].attn.c_proj.bias.assign(Tensor(gpt.transformer.h[i].attn.c_proj.bias.numpy()))
            tiny_gpt.h[i].ln_2.weight.assign(Tensor(gpt.transformer.h[i].ln_2.weight.numpy()))
            tiny_gpt.h[i].ln_2.bias.assign(Tensor(gpt.transformer.h[i].ln_2.bias.numpy()))
            tiny_gpt.h[i].mlp.c_fc.weight.assign(Tensor(gpt.transformer.h[i].mlp.c_fc.weight.numpy()))
            tiny_gpt.h[i].mlp.c_fc.bias.assign(Tensor(gpt.transformer.h[i].mlp.c_fc.bias.numpy()))
            tiny_gpt.h[i].mlp.c_proj.weight.assign(Tensor(gpt.transformer.h[i].mlp.c_proj.weight.numpy()))
            tiny_gpt.h[i].mlp.c_proj.bias.assign(Tensor(gpt.transformer.h[i].mlp.c_proj.bias.numpy()))

    output = gpt(x)[0].detach().numpy()
    output_tiny = tiny_gpt(tiny_x).numpy()
    
    np.testing.assert_allclose(output_tiny, output, atol=5e-3, rtol=5e-5)
    
def test_load_from_huggingface():
    batch_size, seq_length = 4, 64
    config = GPTConfig.from_pretrained("EleutherAI/pythia-70m")
    config.block_size = seq_length
    
    model = GPT(config)
    state_dict = GPT.state_dict_from_huggingface("EleutherAI/pythia-70m")
    model.load_state_dict(state_dict)

    tiny_model = TinyGPT(config)
    tiny_state_dict = TinyGPT.state_dict_from_huggingface("EleutherAI/pythia-70m")
    load_state_dict(tiny_model, tiny_state_dict, strict=False)
    
    x = torch.randint(1, 50000, (batch_size, seq_length))
    tiny_x = Tensor(x.numpy())
    
    output = model(x)[0].detach().numpy()
    output_tiny = tiny_model(tiny_x).numpy()

    np.testing.assert_allclose(output_tiny, output, atol=5e-2, rtol=5e-2)
        