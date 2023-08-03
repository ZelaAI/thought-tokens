"""
Uses Andrej Karpathy's nanoGPT and Huggingface/EleutherAI's GPTNeoX as starting points for this model implementation.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

def get_rotary_sin_cos(max_seq_len, config, base=10000):
    dim = int((config.n_embd // config.n_head) * config.rotary_pct)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    t = torch.arange(max_seq_len)
    freqs = inv_freq.unsqueeze(0) * t.unsqueeze(1)

    emb = torch.cat((freqs, freqs), dim=-1)
    
    cos = emb.cos()
    sin = emb.sin()
    return sin, cos

def apply_rotary_mask(mask, sin_cached, cos_cached):
    # mask is (B, T)       # sin, cos are (T, D)        # output is (B, 1, T, D)
    return sin_cached[mask, :].unsqueeze(1), cos_cached[mask, :].unsqueeze(1)

def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)    

def rotary_apply(t, sin, cos, rotary_ndims):
    # Split
    t_rot, t_pass = t[..., : rotary_ndims], t[..., rotary_ndims :]

    # Apply
    t_embed = (t_rot * cos) + (rotate_half(t_rot) * sin)
    # Re-join
    return torch.cat((t_embed, t_pass), dim=-1)    


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
                
        self.rotary_ndims = int((config.n_embd // config.n_head) * config.rotary_pct)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')# and self.dropout == 0.0
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")

    def forward(self, x, sin, cos, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        head_size = C // self.n_head
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        qkv = qkv.view(B, T, self.n_head, 3 * head_size)
        q, k, v = qkv.split(head_size, dim=3)

        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        
        q = rotary_apply(q, sin, cos, self.rotary_ndims)
        k = rotary_apply(k, sin, cos, self.rotary_ndims)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels -> currently disabled due to # https://github.com/pytorch/pytorch/issues/96099#issuecomment-1458609375
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=attn_mask is None)
        else:
            attn_mask = torch.ones(T, T, dtype=torch.bool).tril(diagonal=0) if attn_mask is None else attn_mask
            # Added  -1.0 to make it equivalent
            attn_mask = attn_mask.float().masked_fill(attn_mask == 0, -float('inf')) - 1 if attn_mask.dtype==torch.bool else attn_mask
            
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att + attn_mask
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)            

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.n_embd_proj, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd_proj, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, sin, cos, attn_mask):
        # Parallelize attention and MLP layers
        # x = x + attn(ln1(x)) + mlp(ln2(x))
        x = self.attn(self.ln_1(x), sin, cos, attn_mask=attn_mask) + self.mlp(self.ln_2(x)) + x
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    audio_vocab_size: int = 1030
    
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    n_embd_proj: int = 768 * 4 # intermediate dimensionality
    rotary_pct: int = 0.25 
    use_parallel_residual: bool = True
    layer_norm_eps = 1e-5
    origin: str = None
    
    def __str__(self):
        return (f"GPTConfig(block_size={self.block_size}, vocab_size={self.vocab_size}, "
                f"n_layer={self.n_layer}, n_head={self.n_head}, n_embd={self.n_embd}, "
                f"dropout={self.dropout}, bias={self.bias}"
                f"n_embd_proj={self.n_embd_proj}, rotary_pct={self.rotary_pct}, "
                f"use_parallel_residual={self.use_parallel_residual}, "
                f"layer_norm_eps={self.layer_norm_eps}, origin={self.origin})")
    
    @classmethod
    def from_pretrained(cls, model_type, revision="main"):
        from transformers import GPTNeoXConfig

        # init a huggingface/transformers model
        gpt_neox_config = GPTNeoXConfig.from_pretrained(model_type, revision=revision)
        
        config = GPTConfig(
            n_embd = gpt_neox_config.hidden_size,
            n_head = gpt_neox_config.num_attention_heads,
            n_layer = gpt_neox_config.num_hidden_layers,
            vocab_size = gpt_neox_config.vocab_size,
            block_size = gpt_neox_config.max_position_embeddings,
            bias = True,
            dropout= 0.0,
            
            n_embd_proj = gpt_neox_config.intermediate_size,
            rotary_pct = gpt_neox_config.rotary_pct,
            
            origin = f"{model_type}@{revision}"
        )

        return config

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # wte = nn.Embedding(config.vocab_size, config.n_embd),
            wte_audio_1 = nn.Embedding(config.audio_vocab_size, config.n_embd),
            wte_audio_2 = nn.Embedding(config.audio_vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps),
        ))
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.audio_head_1 = nn.Linear(config.n_embd, config.audio_vocab_size, bias=False)
        self.audio_head_2 = nn.Linear(config.n_embd, config.audio_vocab_size, bias=False)

        sin_cached, cos_cached = get_rotary_sin_cos(self.config.block_size, self.config)

        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
       
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, inputs_audio_1=None, inputs_audio_2=None, targets_audio_1=None, targets_audio_2=None):
        device = idx.device
        b, t = inputs_audio_1.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        
        pos_mask = torch.arange(t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        
        sin, cos = apply_rotary_mask(pos_mask, self.sin_cached, self.cos_cached) 

        # forward the GPT model itself
        # sum not average token embeddings for each modality
        # x = self.transformer.wte(idx) + 
        x = self.transformer.wte_audio_1(inputs_audio_1) + self.transformer.wte_audio_2(inputs_audio_2)
        x = self.transformer.drop(x)
        
        for block in self.transformer.h:
            x = block(x, sin, cos, attn_mask=None)
    
        x = self.transformer.ln_f(x)

        logits = None#self.lm_head(x)
        logits_audio_1 = self.audio_head_1(x)
        logits_audio_2 = self.audio_head_2(x)
               
        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) if targets is not None else None
        loss_audio_1 = F.cross_entropy(logits_audio_1.view(-1, logits_audio_1.size(-1)), targets_audio_1.view(-1), ignore_index=-1) if targets_audio_1 is not None else None
        loss_audio_2 = F.cross_entropy(logits_audio_2.view(-1, logits_audio_2.size(-1)), targets_audio_2.view(-1), ignore_index=-1) if targets_audio_2 is not None else None
        
        loss = loss_audio_1 + loss_audio_2 if loss_audio_1 is not None and loss_audio_2 is not None else None
        # if loss is not None and loss_audio_1 is not None and loss_audio_2 is not None:
        #     loss = loss + loss_audio_1 + loss_audio_2
        
        
        return logits, logits_audio_1, logits_audio_2, loss
    
    @classmethod
    def state_dict_from_huggingface(cls, huggingface_model_name, revision="main"):
        from transformers import GPTNeoXForCausalLM
        config = GPTConfig.from_pretrained(huggingface_model_name, revision=revision)
        sd_hf = GPTNeoXForCausalLM.from_pretrained(huggingface_model_name, revision=revision).state_dict()
        sd = {}
        
        sd_keys_hf = list(sd_hf.keys())
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attention.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attention.bias')] # same, just the mask (buffer)
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.rotary_emb.inv_freq')] # same, just the mask (buffer)

        def rename_key(k):
            k = k.replace('gpt_neox.layers', 'transformer.h')
            # Attention
            k = k.replace('.attention.dense.', '.attn.c_proj.')
            k = k.replace('.attention.query_key_value.', '.attn.c_attn.')
            # MLP
            k = k.replace('.mlp.dense_h_to_4h.', '.mlp.c_fc.')
            k = k.replace('.mlp.dense_4h_to_h.', '.mlp.c_proj.')
            # LayerNorm
            k = k.replace('.input_layernorm.', '.ln_1.')
            k = k.replace('.post_attention_layernorm.', '.ln_2.')
            # Embedding
            k = k.replace('gpt_neox.embed_in.', 'transformer.wte.')
            k = k.replace('gpt_neox.final_layer_norm.', 'transformer.ln_f.')
            k = k.replace('embed_out.', 'lm_head.')
            return k

        for k in sd_keys_hf:
            new_k = rename_key(k)
            sd[new_k] = sd_hf[k]
            
        return sd

    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Special parameters that require higher learning rate
        special_params_names = []
        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        special_params = [p for n, p in param_dict.items() if n in special_params_names]
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and n not in special_params_names]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and n not in special_params_names]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay, 'lr': learning_rate, 'name': 'decay_params'},
            {'params': nodecay_params, 'weight_decay': 0.0, 'lr': learning_rate, 'name': 'nodecay_params'},
            {'params': special_params, 'weight_decay': weight_decay, 'lr': learning_rate * 5, 'name': 'special_params'}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_special_params = sum(p.numel() for p in special_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        print(f"num special parameter tensors: {len(special_params)}, with {num_special_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt, T=None):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        
        if T is None:
            T = cfg.block_size
        
        L, H, Q = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        if torch.cuda.is_available():
            flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        else:
            flops_promised = 10e12
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, inputs_audio_1, inputs_audio_2, max_new_tokens, temperature=0.7, top_p=0.9):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """

        for _ in tqdm(range(max_new_tokens)):
            # if the sequence context is growing too long we must crop it at block_size
            logits_text, logits_audio_1, logits_audio_2, _ = self(idx, None, inputs_audio_1, inputs_audio_2)

            # pluck the logits at the final step and scale by desired temperature
            # logits_text = logits_text[:, -1, :]
            logits_audio_1 = logits_audio_1[:, -1, :]
            logits_audio_2 = logits_audio_2[:, -1, :]

            # sample from the top-p distribution
            # idx_next = self.sample_top_p(logits_text, temperature, top_p)
            idx_next_audio_1 = self.sample_top_p(logits_audio_1, temperature, top_p)
            idx_next_audio_2 = self.sample_top_p(logits_audio_2, temperature, top_p)
            # append sampled index to the running sequence and continue
            # idx = torch.cat((idx, idx_next), dim=1)
            inputs_audio_1 = torch.cat((inputs_audio_1, idx_next_audio_1), dim=1)
            inputs_audio_2 = torch.cat((inputs_audio_2, idx_next_audio_2), dim=1)

        return idx, inputs_audio_1, inputs_audio_2

    @torch.no_grad()
    def sample_top_p_selective(self, logits, positions, temperature, top_p):
        # Takes batches of logits and masks, returns batches of samples
        logits_flat = logits.view(-1, logits.size(-1))
        positions_expanded = positions.unsqueeze(-1).expand(-1, logits.size(-1))

        logits_gathered = logits_flat.gather(0, positions_expanded)

        tokens = self.sample_top_p(logits_gathered, temperature, top_p).view(-1)
        
        return tokens, -F.cross_entropy(logits_gathered, tokens, reduction='none')

    @torch.no_grad()
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

    @torch.no_grad()
    def loglikelihood(self, logits, targets):
        return -F.cross_entropy(logits, targets, ignore_index=-1)
    
    @torch.no_grad()
    def loglikelihood_selective(self, logits, targets, logits_positions):
        expanded_positions = logits_positions.unsqueeze(-1).expand_as(logits)
        logits_rearranged = torch.gather(logits, 1, expanded_positions)
        loglikelihoods = -F.cross_entropy(logits_rearranged.view(-1, logits.size(-1)), targets.view(-1), reduction='none', ignore_index=-1)
        return loglikelihoods.view(logits.size(0), -1)
