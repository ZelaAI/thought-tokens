from torch.utils.data import Dataset, IterableDataset
from utils import log_mel_spectrogram, load_audio, pad_or_trim, N_FRAMES
import torch
from dataclasses import dataclass
from typing import List
import feedparser
import requests
import os
from tqdm import tqdm

def download_url(url):
    save_path = 'dataset/' + os.path.basename(url)
    print(f'Downloading {url} to {save_path}')
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as file:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            file.write(chunk)
    return os.path.abspath(save_path)

"""
Simply returns the path to an audio file on disk. For real data, it might need to fetch the file from a remote server.
"""
class AudioDatasetFake(Dataset):
    def __init__(self):
        pass
    
    def __len__(self):
        return 5
    
    def __getitem__(self, idx):
        return 'audio.mp3'

class LexDataset(Dataset):
    def __init__(self):
        rss_url = 'https://lexfridman.com/feed/podcast/'
        feed = feedparser.parse(rss_url)
        self.mp3_urls = [entry.enclosures[0].href for entry in feed.entries if entry.enclosures and entry.enclosures[0].type == 'audio/mpeg']

    def __len__(self):
        return len(self.mp3_urls)
    
    def __getitem__(self, idx):
        return download_url(self.mp3_urls[idx])
    
class SpectrogramDataset(IterableDataset):
    def __init__(self, audio_dataset):
        self.audio_dataset = audio_dataset
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None: offset, increase = 0, 1
        else: offset, increase = worker_info.id, worker_info.num_workers

        global_index = offset
        
        while global_index < len(self.audio_dataset):
            audio_path = self.audio_dataset[global_index]
            
            spectrogram = log_mel_spectrogram(load_audio(audio_path))
            
            # split the spectrogram into 30 second chunks (N_FRAMES)
            for i in range(0, spectrogram.shape[1], N_FRAMES):
                spectrogram_chunk = spectrogram[:, i:i+N_FRAMES]
                yield SpectrogramChunk(inputs=pad_or_trim(spectrogram_chunk, N_FRAMES), id=global_index)

            global_index += increase

@dataclass
class SpectrogramChunk:
    inputs: torch.Tensor
    id: int

@dataclass
class Batch:
    inputs: torch.Tensor
    ids: List[int]
    max_dense_tokens: int

    def to(self, device):
        if 'cpu' not in device:
            self.inputs = self.inputs.to(device, non_blocking=True)
        return self

    @classmethod
    def collate_fn(cls, spectrogram_chunk: List[SpectrogramChunk]):
        inputs = torch.stack([seq.inputs for seq in spectrogram_chunk])
        ids = [seq.id for seq in spectrogram_chunk]

        return Batch(inputs, ids, 16)


