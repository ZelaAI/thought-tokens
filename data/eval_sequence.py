from dataclasses import dataclass
import torch
from typing import List

THOUGHT_TOKEN_ID = -1
from data.sequence import Sequence

@dataclass
class EvalSequence(Sequence):
    id: int
    context: torch.Tensor
    max_new_tokens: int
    completions: List[torch.Tensor]

    def __post_init__(self):
        self.length = len(self.context) + sum([len(c) for c in self.completions])
        self.loglikelihoods = [torch.tensor([-float('inf')]) for c in self.completions]

    def __repr__(self) -> str:
        return f"<EvalSequence id={self.id} length={self.length} max_new={self.max_new_tokens} context={self.context} completions={self.completions} loglikelihoods={self.loglikelihoods}>"
        
    # We want to prioritize the context with the most forward passes remaining
    def __lt__(self, other):
        return self.max_new_tokens >= other.max_new_tokens

    def get_attn_mask_bounds(self, causal=False):
        top = torch.arange(self.length, dtype=torch.long)
        bottom = torch.ones(self.length, dtype=torch.long) * (self.length - 1)
        
        if not causal:
            top[:len(self.context)] = 0
            
        i = len(self.context)
        for completion_len in [len(c) for c in self.completions]:
            bottom[i:i+completion_len] = i + completion_len - 1
            i += completion_len
        
        return top, bottom

    def get_pos_mask(self):
        context_mask = [torch.arange(len(self.context))] # context
        completion_masks = [torch.arange(len(self.context), len(self.context)+len(completion)) for completion in self.completions]
        
        return torch.cat(context_mask+completion_masks)
        
    def get_inputs(self):
        return torch.cat([self.context] + self.completions)
    
    def get_targets(self, train=False):
        targets = self.get_inputs()
        targets[:len(self.context)] = -1
        targets = targets.roll(-1, dims=0)
        
        if not train:
            # If an item is dense token, we want to mask it out of loss
            targets[torch.eq(targets, THOUGHT_TOKEN_ID)] = -1
        
        return targets
    
    def get_target_pos_mask(self):
        # this mask allows us to share logits between the final context token and the first target token for multiple completions
        mask = torch.arange(self.length)
        completion_lengths = [len(c) for c in self.completions]
        
        i = len(self.context) - 1
        for completion_len in completion_lengths:
            mask[i] = len(self.context) - 1
            i += completion_len
        
        return mask

    def get_generate_positions(self) -> torch.Tensor:
        indexes = []
        
        completion_lengths = [len(c) for c in self.completions]
        
        if self.max_new_tokens > 0:
            i = len(self.context)
            for completion_len in completion_lengths:
                indexes.append(i + completion_len - 1)
                i += completion_len
            
        return torch.tensor(indexes, dtype=torch.long)
    
    def max_dense_tokens(self) -> int:
        # compute maximum depth we need to go to ensure all dense tokens are filled in...
        return max(
            [
                torch.sum(torch.eq(completion, THOUGHT_TOKEN_ID)) for completion in self.completions
            ]
        ).item()
    
    def get_measured_completion_lengths(self) -> List[int]:
        return [
            len(completion) - torch.sum(torch.eq(completion, THOUGHT_TOKEN_ID)) for completion in self.completions
        ]
        
    
    @torch.no_grad()
    def report_logits(self, sampled_tokens: torch.Tensor, loglikelihoods: torch.Tensor, sampled_tokens_loglikelihoods: torch.Tensor) -> bool:
        # Pre-store lengths since we may be mutating the completions
        completion_lengths = [len(c) for c in self.completions]
        has_generated = False
        
        loglikelihoods = loglikelihoods.to(torch.float32)
        sampled_tokens_loglikelihoods = sampled_tokens_loglikelihoods.to(torch.float32)
        
        if self.max_new_tokens > 0:
            for i in range(len(self.completions)):
                self.completions[i] = torch.cat([self.completions[i], sampled_tokens[i].unsqueeze(0)])
                self.length += 1

            self.max_new_tokens -= 1
            has_generated = True

        if self.max_new_tokens == 0:
            
            measured_completion_lengths = self.get_measured_completion_lengths()
            
            i = len(self.context)
            for c, completion_len in enumerate(completion_lengths):
                loglikelihood_total = torch.sum(loglikelihoods[ i - 1 :i - 1 + completion_len ])
                
                if has_generated:
                    loglikelihood_total += sampled_tokens_loglikelihoods[c]

                self.loglikelihoods[c] = loglikelihood_total / measured_completion_lengths[c]

                i += completion_len
    
        # Return True if we're done with this sequence
        return self.max_new_tokens == 0 

@dataclass
class EvalBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    attn_mask_bound_top: torch.Tensor
    attn_mask_bound_bottom: torch.Tensor
    pos_mask: torch.Tensor
    target_pos_mask: torch.Tensor
    generate_positions: torch.Tensor
    packs: List[List[EvalSequence]]
    max_dense_tokens: int

    def to(self, device):
        if 'cpu' not in device:
            for attr in ('inputs', 'targets', 'attn_mask_bound_top', 'attn_mask_bound_bottom', 'pos_mask', 'target_pos_mask', 'generate_positions'):
                setattr(self, attr, getattr(self, attr).pin_memory().to(device, non_blocking=True))
        return self

    def __str__(self) -> str:
        packs_count = len(self.packs)
        sequence_count = sum([len(p) for p in self.packs])
        return f'<Batch packs_count={packs_count} sequence_count={sequence_count} shape={self.inputs.shape} max_dense_tokens={self.max_dense_tokens}>'
    
    @classmethod
    def packs_to_batch_factory(cls, train, causal, max_seq_len, batch_size):
        def packs_to_batch(packs: List[List[EvalSequence]]):

            bs, T = batch_size, max_seq_len
            inputs = torch.zeros(bs, T, dtype=torch.long) # Assuming padding token is 0
            targets = torch.ones(bs, T, dtype=torch.long) * -1

            pos_mask = torch.zeros(bs, T, dtype=torch.long)
            target_pos_mask = torch.zeros(bs, T, dtype=torch.long)
            
            generate_positions = torch.tensor([], dtype=torch.long)
            
            max_dense_tokens = 0
            
            attn_mask_bound_top = torch.arange(T, dtype=torch.long).repeat(bs, 1)
            attn_mask_bound_bottom = torch.arange(T, dtype=torch.long).repeat(bs, 1)

            for i, pack in enumerate(packs):
                pos = 0
                for seq in pack:
                    inputs[i, pos:pos+seq.length] = seq.get_inputs()                
                    targets[i, pos:pos+seq.length] = seq.get_targets(train)
                    pos_mask[i, pos:pos+seq.length] = seq.get_pos_mask()
                    target_pos_mask[i, pos:pos+seq.length] = seq.get_target_pos_mask() + pos # Add pos to offset everything properly
                    
                    generate_positions = torch.cat([
                        generate_positions,
                        seq.get_generate_positions() + pos + i * T # Add pos and i * T to offset everything properly as we flatten all the way down to a single dimension
                    ])
                    
                    seq_max_dense_tokens = seq.max_dense_tokens()
                    if seq_max_dense_tokens > max_dense_tokens:
                        max_dense_tokens = seq_max_dense_tokens

                    top, bottom = seq.get_attn_mask_bounds(causal=causal)

                    attn_mask_bound_top[i, pos:pos+seq.length] = top + pos
                    attn_mask_bound_bottom[i, pos:pos+seq.length] = bottom + pos
                
                    pos += seq.length
                    
            return EvalBatch(inputs, targets, attn_mask_bound_top, attn_mask_bound_bottom, pos_mask, target_pos_mask, generate_positions, packs, max_dense_tokens)

        return packs_to_batch

def report_logits(sampled_tokens: torch.Tensor, loglikelihoods: torch.Tensor, sampled_tokens_loglikelihoods: torch.Tensor, packs: List[List[EvalSequence]]):
    incomplete_sequences = []
    
    generate_index = 0
    
    for i, pack in enumerate(packs):
        seq_start = 0
        
        # Pre-store lengths since we'll be mutating the pack
        seq_lengths = [seq.length for seq in pack]
        
        for seq, seq_length in zip(pack, seq_lengths):
            generate_length = len(seq.completions) if seq.max_new_tokens > 0 else 0
            
            is_done = seq.report_logits(
                sampled_tokens[generate_index : generate_index + generate_length], 
                loglikelihoods[i, seq_start:seq_start+seq_length],
                sampled_tokens_loglikelihoods[generate_index : generate_index + generate_length]
            )
        
            if not is_done:
                incomplete_sequences.append(seq)
                    
            seq_start += seq_length
            generate_index += generate_length
        
    return incomplete_sequences
