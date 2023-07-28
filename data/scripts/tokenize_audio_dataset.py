import sys
sys.path.append('.')
from datasets import load_dataset, Audio as AudioHF
from data.packer import Packer
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import random
from datasets import Dataset
from transformers import GPTNeoXTokenizerFast
from core.tokenizer import Tokenizer, AudioTokenizer

max_seq_len = 2048 + 1 # account for input/target offset
num_workers = 4
source_dataset_repo_id = "hf-internal-testing/librispeech_asr_dummy"
source_dataset_split = 'clean'
destination_dataset_repo_id = "ZelaAI/librispeech_tiny_2048"

print("Loading tokenizer...")
tokenizer = Tokenizer()
audio_tokenizer = AudioTokenizer()
print("Loading dataset...")
dataset = load_dataset(source_dataset_repo_id, source_dataset_split)
dataset = dataset.cast_column("audio", AudioHF(sampling_rate=24000))

def tokenize(example):
    text_tokenized = tokenizer.encode(example['text'].lower()).numpy()
    audio_tokenized = audio_tokenizer.encode(example['audio']['array']).numpy()

    text_length = text_tokenized.shape[-1]
    audio_length = audio_tokenized.shape[-1]

    assert audio_tokenized.shape[0] == 2 # we're expecting two channel audio for now

    construct_text_text_tokens = np.concatenate([np.array([tokenizer.start_text_id]), text_tokenized, np.array([tokenizer.end_text_id])])
    construct_audio_text_tokens = np.concatenate([np.array([tokenizer.start_audio_id]), np.array([tokenizer.audio_id] * audio_length), np.array([tokenizer.end_audio_id])])

    construct_text_audio_tokens = np.concatenate([np.array([audio_tokenizer.start_text_id]), np.array([audio_tokenizer.text_id] * text_length), np.array([audio_tokenizer.end_text_id])])
    construct_audio_audio_tokens_1 = np.concatenate([np.array([audio_tokenizer.start_audio_id]), audio_tokenized[0], np.array([audio_tokenizer.end_audio_id])])
    construct_audio_audio_tokens_2 = np.concatenate([np.array([audio_tokenizer.start_audio_id]), audio_tokenized[1], np.array([audio_tokenizer.end_audio_id])])


    audio_first = random.random() < 0.5

    if audio_first:
        text_tokens = np.concatenate([construct_audio_text_tokens, construct_text_text_tokens])
        audio_tokens_1 = np.concatenate([construct_audio_audio_tokens_1, construct_text_audio_tokens])
        audio_tokens_2 = np.concatenate([construct_audio_audio_tokens_2, construct_text_audio_tokens])
    else:
        text_tokens = np.concatenate([construct_text_text_tokens, construct_audio_text_tokens])
        audio_tokens_1 = np.concatenate([construct_text_audio_tokens, construct_audio_audio_tokens_1])
        audio_tokens_2 = np.concatenate([construct_text_audio_tokens, construct_audio_audio_tokens_2])
        
    return {
        'text_tokens': text_tokens,
        'audio_tokens_1': audio_tokens_1,
        'audio_tokens_2': audio_tokens_2,
    }


print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize)
tokenized_dataset.set_format(type='numpy', columns=['text_tokens', 'audio_tokens_1', 'audio_tokens_2'])

def length(example):
    return {
        'length': len(example['text_tokens']),
    }

with_lengths = tokenized_dataset.map(length)
filtered = with_lengths.filter(lambda example: example['length'] <= max_seq_len)

## Packing the dataset
# We don't naively just concat examples together, as that leads to an unnecessary amount of examples sitting on boundaries
# Packing is much more effective.

@dataclass
class TrainingSequence:
    id: int
    length: int
    text_tokens: np.array
    audio_tokens_1: np.array
    audio_tokens_2: np.array

    def __lt__(self, other):
        # No-op
        return False

print("Preparing dataset for packing...")
training_sequences = []

for i, example in tqdm(enumerate(filtered['train'])):
    training_sequences.append(TrainingSequence(
        id=i,
        length=example['length'],
        text_tokens=example['text_tokens'],
        audio_tokens_1=example['audio_tokens_1'],
        audio_tokens_2=example['audio_tokens_2'],
    ))

print("Packing dataset...")
packer = Packer(max_seq_len, training_sequences)
packed_sequences = packer.to_list()
print(packer)

print("Shuffling dataset...")
random.shuffle(packed_sequences)

packed_text_tokens = []
packed_audio_tokens_1 = []
packed_audio_tokens_2 = []

print("Concatenating dataset...")
for packed_sequence in tqdm(packed_sequences):
    packed_text_tokens.append(np.concatenate([seq.text_tokens for seq in packed_sequence]))
    packed_audio_tokens_1.append(np.concatenate([seq.audio_tokens_1 for seq in packed_sequence]))
    packed_audio_tokens_2.append(np.concatenate([seq.audio_tokens_2 for seq in packed_sequence]))

print("Pushing dataset to HuggingFace Hub...")
final_dataset = Dataset.from_dict({
    'text_tokens': packed_text_tokens,
    'audio_tokens_1': packed_audio_tokens_1,
    'audio_tokens_2': packed_audio_tokens_2,
})

final_dataset.push_to_hub(destination_dataset_repo_id)