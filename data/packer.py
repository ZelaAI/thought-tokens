from torch.utils.data import IterableDataset
from collections import defaultdict
from heapq import heappush, heappop
from typing import List
from tqdm import tqdm

from data.sequence import Sequence

class Packer(IterableDataset):
    length_map = defaultdict(list)
    current_max_length = 0
    
    done = False
    
    stat_total_packs = 0
    stat_total_tokens = 0

    def __str__(self) -> str:
        return f"<Packer: {self.stat_total_packs} packs, {self.stat_total_tokens} tokens, {self.stat_total_tokens / (self.max_seq_len * self.stat_total_packs) * 100:.2f}% efficiency>"
    
    def __init__(self, max_seq_len, items = []):
        self.max_seq_len = max_seq_len
        self.add_to_queue(items)

    def add_to_queue(self, items: List[Sequence]):
        for item in items:
            heappush(self.length_map[item.length], item)
            self.current_max_length = max(self.current_max_length, item.length)
    
    def __iter__(self):
        return self

    def __next__(self) -> List[Sequence]:        
        pack = []
        pack_length = 0
        
        length = self.current_max_length

        while length > 0:
            if self.length_map[length]:
                item = heappop(self.length_map[length])
                
                pack.append(item)
                pack_length += item.length
                
                length = min(self.max_seq_len - pack_length, length)
            else:
                if length == self.current_max_length:
                    self.current_max_length -= 1
                length -= 1
        
        self.stat_total_packs += 1
        self.stat_total_tokens += pack_length
        
        # Sometimes we may have a few empty packs at end of batch, but we can't be sure we're done until add_to_queue is called again.
        if self.done and len(pack) == 0:
            raise StopIteration
        
        return pack
    
    def to_list(self):
        self.done = True
        return list(tqdm(self))
