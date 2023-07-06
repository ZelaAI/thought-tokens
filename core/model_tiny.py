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
    