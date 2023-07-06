from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, LayerNorm, Embedding
from core.model import GPTConfig
import math
import numpy as np

class MLP:
    def __init__(self, config: GPTConfig):
        self.c_fc = Linear(config.n_embd, config.n_embd_proj, bias=config.bias)
        self.c_proj = Linear(config.n_embd_proj, config.n_embd, bias=config.bias)

    def __call__(self, x):
        x = self.c_fc(x)
        x = x.gelu()
        x = self.c_proj(x)
        return x

def get_rotary_sin_cos(max_seq_len, config, base=10000):
    dim = int((config.n_embd // config.n_head) * config.rotary_pct)
    inv_freq = 1.0 / (base ** (Tensor.arange(start=0, stop=dim, step=2).float() / dim))
    
    t = Tensor.arange(max_seq_len)
    freqs = inv_freq.unsqueeze(0) * t.unsqueeze(1)
    
    emb = Tensor.cat(freqs, freqs, dim=-1)

    cos = emb.cos()
    sin = emb.sin()
    return sin, cos

def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return Tensor.cat(-x2, x1, dim=-1)    

def rotary_apply(t, sin, cos, rotary_ndims):
    # Split
    t_rot, t_pass = t[..., : rotary_ndims], t[..., rotary_ndims :]

    # Apply
    t_embed = (t_rot * cos) + (rotate_half(t_rot) * sin)
    # Re-join
    return Tensor.cat(t_embed, t_pass, dim=-1)    

def apply_rotary_mask(mask, sin_cached, cos_cached):
    # mask is (B, T)       # sin, cos are (T, D)        # output is (B, 1, T, D)
    sin = Tensor(sin_cached.numpy()[mask.numpy(), :]).unsqueeze(1)
    cos = Tensor(cos_cached.numpy()[mask.numpy(), :]).unsqueeze(1)
    
    return sin, cos






class CausalSelfAttention:
    def __init__(self, config):
        assert config.n_embd % config.n_head == 0
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
    
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        
        self.rotary_ndims = int((config.n_embd // config.n_head) * config.rotary_pct)
    
    def __call__(self, x, sin, cos, attn_mask=None):
        B, T, C = x.shape
        head_size = C // self.n_head
        
        qkv = self.c_attn(x)
        qkv = qkv.reshape(B, T, self.n_head, 3 * head_size)
        
        q, k, v = qkv[:, :, :, :head_size], qkv[:, :, :, head_size: 2 * head_size], qkv[:, :, :, 2 * head_size:]
        
        k = k.reshape(B, T, self.n_head, head_size).transpose(1, 2)
        q = q.reshape(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, head_size).transpose(1, 2)
        
        q = rotary_apply(q, sin, cos, self.rotary_ndims)
        k = rotary_apply(k, sin, cos, self.rotary_ndims)
                
        scores = q.matmul(k.transpose(-2, -1)) * (1.0 / math.sqrt(k.shape[-1]))
        
        if attn_mask is None:
            attn_mask = np.full((1, 1, T, T), float("-inf"), dtype=np.float32)
            attn_mask = np.triu(attn_mask, k=1)  # TODO: this is hard to do in tinygrad
            attn_mask = Tensor(attn_mask)
        
        scores = scores + attn_mask

        scores = scores.softmax(axis=-1)
        result = scores.matmul(v)
        
        result = result.transpose(1, 2).reshape(B, T, C)
        result = self.c_proj(result)
        return result


class Block:
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def __call__(self, x, sin, cos, attn_mask):
        # Parallelize attention and MLP layers
        # x = x + attn(ln1(x)) + mlp(ln2(x))
        x = self.attn(self.ln_1(x), sin, cos, attn_mask=attn_mask) + self.mlp(self.ln_2(x)) + x
        return x


class GPT:
    def __init__(self, config):
        
        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_eps),
        self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False)
        self.sin_cached, self.cos_cached = get_rotary_sin_cos(2048, config)
        
    def __call__(self, ids):
        x = self.wte(ids)
        
        sin, cos = apply_rotary_mask(Tensor.arange(x.shape[-1]), self.sin_cached, self.cos_cached)
        
        for block in self.h:
            x = block(x, sin, cos, attn_mask=None)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
        

