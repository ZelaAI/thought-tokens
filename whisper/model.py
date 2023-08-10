import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import whisper

@dataclass
class WhisperConfig:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int

def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = nn.Linear(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        
    def forward(self, x, xa=None, causal=False):
        q = self.query(x)
        k = self.key(x if xa is None else xa) # TODO: Cache these for `xa` 
        v = self.value(x if xa is None else xa)
        
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) # BS, n_head, seq_len, head_size
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal).permute(0, 2, 1, 3)
        
        return self.out(y.flatten(start_dim=2))

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_state * 4), nn.GELU(), nn.Linear(n_state * 4, n_state)
        )
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x, xa = None, mask = None):
        x = x + self.attn(self.attn_ln(x), causal=self.cross_attn is not None)
    
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa)
    
        return x + self.mlp(self.mlp_ln(x))

class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])
        self.ln_post = nn.LayerNorm(n_state)

    def forward(self, x):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        x = (x + self.positional_embedding)

        for block in self.blocks:
            x = block(x)

        return self.ln_post(x)

class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks = nn.ModuleList([ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_state)

    def forward(self, x, xa):
        x = (self.token_embedding(x) + self.positional_embedding[:x.shape[-1]])

        for block in self.blocks:
            x = block(x, xa)

        x = self.ln(x)
        logits = (x @ torch.transpose(self.token_embedding.weight, 0, 1))

        return logits

class Whisper(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.encoder = AudioEncoder(config.n_mels, config.n_audio_ctx, config.n_audio_state, config.n_audio_head, config.n_audio_layer)
        self.decoder = TextDecoder(config.n_vocab, config.n_text_ctx, config.n_text_state, config.n_text_head, config.n_text_layer)
    
    @classmethod
    def load_from_pretrained(cls, name: str):
        pretrained = whisper.load_model(name)
        config = WhisperConfig(**pretrained.dims.__dict__)
        
        model = cls(config)
        model.load_state_dict(pretrained.state_dict())
        return model
    
    def sample_top_p(self, logits, temperature, top_p):
        """
        Takes a list of logits returns a sampled token
        from the top-p distribution for each sequence where mask[x] == 1.
        """
        logits = logits / temperature
        sorted_logits, sorted_indices = torch.sort(F.softmax(logits, dim=-1), dim=-1, descending=True)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_keep = torch.cumsum(sorted_logits, dim=-1) <= top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_keep = torch.roll(sorted_indices_to_keep, shifts=1, dims=-1)
        sorted_indices_to_keep[..., 0] = 1  # Keep first token always

        mask_zero = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, sorted_indices_to_keep)
        logits = logits * mask_zero

        # Sample tokens only for active positions
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


    def generate(self, tokens, encoder_logits, max_new_tokens=40):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            logits = self.decoder(tokens, encoder_logits)
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :]
            
            # sample from the top-p distribution
            token_next = self.sample_top_p(logits, 0.1, 0.1)
            
            # append sampled index to the running sequence and continue
            tokens = torch.cat((tokens, token_next), dim=1)

        return tokens