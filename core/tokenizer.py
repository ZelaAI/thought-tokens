import torch
from transformers import GPTNeoXTokenizerFast
from transformers import EncodecModel, AutoProcessor

import functools

class Tokenizer:
    def __init__(self, name = 'EleutherAI/pythia-410m'):
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(name)

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.add_tokens(['<|start_text|>', '<|end_text|>','<|start_audio|>','<|end_audio|>','<|audio|>'])

        self.start_text_id = self.tokenizer.encode('<|start_text|>')[0]
        self.end_text_id = self.tokenizer.encode('<|end_text|>')[0]
        self.start_audio_id = self.tokenizer.encode('<|start_audio|>')[0]
        self.end_audio_id = self.tokenizer.encode('<|end_audio|>')[0]
        self.audio_id = self.tokenizer.encode('<|audio|>')[0]

    @functools.lru_cache(maxsize=None)
    def _cached_encode(self, value):
        return self.tokenizer.encode(value)

    def encode(self, value):
        cached_result = self._cached_encode(value)
        return torch.tensor(cached_result, dtype=torch.long)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)
    

class AudioTokenizer:
    def __init__(self):
        print("Loading Audio Encoder")
        self.model = EncodecModel.from_pretrained("facebook/encodec_24khz")
        self.processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")

        self.size = self.model.quantizer.codebook_size
        
        self.start_text_id = self.size + 1
        self.end_text_id = self.size + 2
        self.start_audio_id = self.size + 3
        self.end_audio_id = self.size + 4
        self.text_id = self.size + 5
        
        self.size += 5
        print("Loaded Audio Encoder, size:", self.size)
        
    def encode(self, audio_array):
        # expects an audio array already in 24khz
        inputs = self.processor(raw_audio=audio_array, sampling_rate=self.processor.sampling_rate, return_tensors="pt")
        encoder_outputs = self.model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=1.5)
        
        return encoder_outputs.audio_codes.squeeze(0).squeeze(0)
    
    def decode(self, audio_codes):
        audio_values = self.model.decode(audio_codes.unsqueeze(0).unsqueeze(0), [None])[0]
        return audio_values.squeeze(0).squeeze(0)
    