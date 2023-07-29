import pytest
from core.tokenizer import AudioTokenizer
from datasets import load_dataset, Audio as AudioHF
import torch

def test_audio_tokenizer():
    librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    librispeech_dummy = librispeech_dummy.cast_column("audio", AudioHF(sampling_rate=24000))

    audio_array = librispeech_dummy[0]["audio"]["array"]
    
    tokenizer = AudioTokenizer()
    
    tokenized = tokenizer.encode(audio_array)
    
    # expected shape 2, 440
    assert tokenized.shape == (2, 440)
    assert torch.all(tokenized[:, :8] == torch.tensor([[  62,  835,  835,  835,  835,  835,  835,  835],
        [1007, 1007, 1007,  544,  424,  424, 1007,  424]]))
    
    decoded = tokenizer.decode(tokenized)
    assert decoded.shape == (140800,)