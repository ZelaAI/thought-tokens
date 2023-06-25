"""
Uses Andrej Karpathy's nanoGPT and Huggingface/EleutherAI's GPTNeoX as starting points for this model implementation.
"""

import math
import inspect
from dataclasses import dataclass
import functools

import torch
import torch.nn as nn
from transformers import GPTNeoXTokenizerFast
from torch.nn import functional as F

THOUGHT_TOKEN_ID = 50277

class Tokenizer:
    def __init__(self, name = 'EleutherAI/pythia-410m'):
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(name)

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.add_tokens(['<|dense|>'])
        
        self.dense_token_id = self.tokenizer.encode('<|dense|>')[0]
        
        assert THOUGHT_TOKEN_ID == self.dense_token_id
    
    @functools.lru_cache(maxsize=None)
    def _cached_encode(self, value):
        return self.tokenizer.encode(value)

    def encode(self, value):
        cached_result = self._cached_encode(value)
        return torch.tensor(cached_result, dtype=torch.long)
    
    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


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


def bounds_to_mask(top, bottom, max_seq_len=16):
    batch_size = top.size(0)
    range_tensor = torch.arange(max_seq_len, device=top.device).view(1, -1, 1).expand(batch_size, -1, -1)
    # Create a mask for values greater than or equal to top
    top_mask = range_tensor >= top.view(batch_size, 1, -1)

    # Create a mask for values less than or equal to bottom
    bottom_mask = range_tensor <= bottom.view(batch_size, 1, -1)
    
    # Combine the masks and convert to long
    mask = (top_mask & bottom_mask)
    
    return mask

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

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
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
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=attn_mask is None)
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

class DenseInject(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layer = config.n_layer
        # initialize to bias injection towards earlier layers mostly.
        self.inject_weights = torch.nn.Parameter(torch.arange(self.n_layer, 0, -1).float()/self.n_layer, requires_grad=True)
        
    def forward(self, dense: torch.Tensor, layer_num: int):
        return torch.relu(self.inject_weights)[layer_num] * dense

class DenseCapture(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layer = config.n_layer
        self.capture_weights = torch.nn.Parameter(
            # initialize to bias capture towards later layers mostly.
            torch.arange(self.n_layer).float() / (self.n_layer // 6),
            requires_grad=True
        )

    def forward(self, activations: torch.Tensor, layer_num: int):
        return torch.softmax(self.capture_weights, dim=0)[layer_num] * activations

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
    n_layer: int = 2
    n_head: int = 8
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    causal: bool = True
    
    n_embd_proj: int = 768 * 4 # intermediate dimensionality
    rotary_pct: int = 0.25 
    use_parallel_residual: bool = True
    layer_norm_eps = 1e-5
    origin: str = None
    
    def __str__(self):
        return (f"GPTConfig(block_size={self.block_size}, vocab_size={self.vocab_size}, "
                f"n_layer={self.n_layer}, n_head={self.n_head}, n_embd={self.n_embd}, "
                f"dropout={self.dropout}, bias={self.bias}, causal={self.causal}, "
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
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
       
        self.dense_capture = DenseCapture(config)
        self.dense_inject = DenseInject(config)

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

    def forward(self, idx, dense=None, targets=None, attn_mask=None, pos_mask=None, attn_mask_bound_top=None, attn_mask_bound_bottom=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        
        if attn_mask_bound_top is not None and attn_mask_bound_bottom is not None:
            attn_mask = bounds_to_mask(attn_mask_bound_top, attn_mask_bound_bottom, t)
        
        if attn_mask is not None:
            # Add an extra dimension for the number of heads (1 in this case)
            attn_mask = attn_mask.unsqueeze(1)

            # Expand the mask along the new dimension without duplicating data in memory
            attn_mask = attn_mask.expand(-1, self.config.n_head , -1, -1)

        if pos_mask is None:
            pos_mask = torch.arange(t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        
        sin, cos = apply_rotary_mask(pos_mask, self.sin_cached, self.cos_cached) 

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        
        dense_out = torch.zeros_like(x)
        
        for i, block in enumerate(self.transformer.h):
            if dense is not None:
                # This can be done more efficiently by doing it one by one likely, but this is easier to read for now.
                x = block(x + self.dense_inject(dense, i), sin, cos, attn_mask)
                dense_out = dense_out + self.dense_capture(x, i)
            else:
                x = block(x, sin, cos, attn_mask)
    
        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1) if targets is not None else None
        
        return logits, dense_out, loss
    
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

        sd['dense_inject.inject_weights'] = DenseInject(config).state_dict()['inject_weights']
        sd['dense_capture.capture_weights'] = DenseCapture(config).state_dict()['capture_weights']
        
        return sd

    def configure_optimizers(self, weight_decay, learning_rate, betas, device):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Special parameters that require higher learning rate
        special_params_names = ['dense_inject.inject_weights', 'dense_capture.capture_weights']
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

    def rearrange_outputs(self, outputs, outputs_positions):
        return torch.gather(outputs, 1, outputs_positions.unsqueeze(-1).expand_as(outputs))

    def create_dense_inputs(self, idx, dense_outputs=None, dense_outputs_positions=None):
        """
        Takes a new set of input tokens, and the dense outputs from the previous forward pass,
        then creates the expected dense inputs for the next forward pass that line up wiht the given input tokens.
        
        idx: (batch_size, seq_len)
        dense_outputs: (batch_size, seq_len-1, hidden_size)
        
        output: (batch_size, seq_len, hidden_size)
        """
        # Dense outputs are offset by one (as they match up with the targets) AND they're full, not masked yet.
        if dense_outputs is None:
            # we're on the first pass, so we don't have any dense outputs yet
            return torch.zeros((idx.shape[0], idx.shape[1], self.config.n_embd), device=idx.device)
        
        if dense_outputs_positions is not None:
            dense_outputs = self.rearrange_outputs(dense_outputs, dense_outputs_positions)

        if dense_outputs.shape[1] == idx.shape[1]:
            # we're too long, so just trim off the last token
            dense_outputs = dense_outputs[:, :-1, :]

        dense_unmasked = torch.cat([
            torch.zeros_like(dense_outputs[:, :1, :]),
            dense_outputs,
        ], dim=1)

        mask = idx != THOUGHT_TOKEN_ID
        
        return dense_unmasked.masked_fill(mask.unsqueeze(-1), 0.0)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, dense_input=None, temperature=0.7, top_p=0.9):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        if dense_input is None:
            dense_input = self.create_dense_inputs(idx)

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            logits, dense_out, _ = self(idx, dense=dense_input)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :]
            # sample from the top-p distribution
            idx_next = self.sample_top_p(logits, temperature, top_p)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            
            dense_input = self.create_dense_inputs(idx, dense_out)

        return idx

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
