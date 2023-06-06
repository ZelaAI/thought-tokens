import sys
sys.path.append('.')
from datasets import load_dataset
from data.packer import Packer
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
import random
from datasets import Dataset
from transformers import GPTNeoXTokenizerFast

max_seq_len = 512 + 1 # account for input/target offset
num_workers = 4
source_dataset_repo_id = "JeanKaddour/minipile"
destination_dataset_repo_id = "ZelaAI/minipile_512"

print("Loading tokenizer...")
tokenizer = GPTNeoXTokenizerFast.from_pretrained('EleutherAI/pythia-410m')
print("Loading dataset...")
dataset = load_dataset(source_dataset_repo_id)

def tokenize(example):
    text = example['text'] + tokenizer.eos_token
    tokens = tokenizer(text, return_tensors='np')['input_ids'][0]
    return {
        'tokens': tokens,
    }

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize, num_proc=num_workers)
tokenized_dataset.set_format(type='numpy', columns=['tokens'])

## Packing the dataset
# We don't naively just concat examples together, as that leads to an unnecessary amount of examples sitting on boundaries
# Packing is much more effective.

@dataclass
class TrainingSequence:
    id: int
    length: int
    tokens: np.array

    def __lt__(self, other):
        # No-op
        return False

print("Preparing dataset for packing...")
training_sequences = []
for i, unsplit_tokens in tqdm(enumerate(tokenized_dataset['train']['tokens'])):
    # takes a single example and splits it into multiple examples
    examples = [unsplit_tokens[i:i+max_seq_len] for i in range(0, len(unsplit_tokens), max_seq_len)]
    
    # Remove really small examples entirely. Would occur if truncated at just the end of a long sequence
    if len(examples[-1]) < 30:
        examples = examples[:-1]
    
    lengths = [len(example) for example in examples]
    
    for length, tokens in zip(lengths, examples):
        training_sequences.append(TrainingSequence(
            id=i,
            length=length,
            tokens=tokens,
        ))

print("Packing dataset...")
packer = Packer(max_seq_len, training_sequences)
packed_sequences = packer.to_list()
print(packer)

print("Shuffling dataset...")
random.shuffle(packed_sequences)

packed_tokens = []

print("Concatenating dataset...")
for packed_sequence in tqdm(packed_sequences):
    packed_tokens_item = np.concatenate([
        seq.tokens for seq in packed_sequence
    ])
    
    packed_tokens.append(packed_tokens_item)

print("Pushing dataset to HuggingFace Hub...")
final_dataset = Dataset.from_dict({
    'tokens': packed_tokens,
})

final_dataset.push_to_hub(destination_dataset_repo_id)