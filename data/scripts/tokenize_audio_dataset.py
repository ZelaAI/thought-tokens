import sys
from data.scripts.librispeech import get_dataset
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
from multiprocessing import Pool, cpu_count

max_seq_len = 2048 + 1 # account for input/target offset
num_workers = 4
destination_dataset_repo_id = "ZelaAI/lj_speech_2048"

def tokenize(example, tokenizer, audio_tokenizer):
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


    audio_first = False#random.random() < 0.5

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
        'length': len(text_tokens),
    }

def chunked_map(func, data, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()
    print(f"Using {num_processes} processes out of {cpu_count()} available")

    chunk_size = len(data) // num_processes
    chunks = [data.select(range(i, min(i + chunk_size, len(data)))) for i in range(0, len(data), chunk_size)]
    print(f"Splitting data into {len(chunks)} chunks")
    print(f"Chunk sizes: {[len(chunk) for chunk in chunks]}")

    results = []
    with Pool(processes=num_processes) as pool:
        for result_chunk in tqdm(pool.imap_unordered(func, chunks), total=len(chunks)):
            results.extend(result_chunk)

    return results

def tokenize_chunk(chunk):
    print("Loading tokenizer...")
    tokenizer = Tokenizer()
    audio_tokenizer = AudioTokenizer()
    print("Loaded audio tokenizer")
    return [tokenize(example, tokenizer, audio_tokenizer) for example in tqdm(chunk)]


if __name__ == '__main__':
    print("Loading dataset...")
    dataset = load_dataset("lj_speech", split="train")
    
    dataset = dataset.cast_column("audio", AudioHF(sampling_rate=24000))

    print("Tokenizing dataset...")
    tokenized_data = chunked_map(tokenize_chunk, dataset, num_processes=4)

    tokenized_dataset = Dataset.from_list(tokenized_data)
    print(tokenized_dataset)

    tokenized_dataset.set_format(type='numpy', columns=['text_tokens', 'audio_tokens_1', 'audio_tokens_2'], output_all_columns=True)

    filtered = tokenized_dataset.filter(lambda example: example['length'] <= max_seq_len)

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

    for i, example in tqdm(enumerate(filtered)):
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