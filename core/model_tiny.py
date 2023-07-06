from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from core.model import GPTConfig


class MLP:
    def __init__(self, config: GPTConfig):
        self.c_fc = Linear(config.n_embd, config.n_embd_proj, bias=True)
        self.c_proj = Linear(config.n_embd_proj, config.n_embd, bias=True)

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
    return sin_cached[mask, :].unsqueeze(1), cos_cached[mask, :].unsqueeze(1)
