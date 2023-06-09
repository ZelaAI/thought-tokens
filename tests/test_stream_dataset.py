import pytest
from torch.utils.data import DataLoader
from collections import Counter
from data.stream_dataset import HuggingfaceStreamDataset
from data.train_sequence import TrainBatch

huggingface_name = "ZelaAI/minipile_512_tiny_streamable"

# Merged all tests into one here, cause it's too slow to load the dataset multiple times
def test_stream_dataset():
    dataset = HuggingfaceStreamDataset(huggingface_name)
    loader = DataLoader(dataset, batch_size=1, num_workers=2, collate_fn=TrainBatch.collate_fn)

    seen_rows = Counter()
    num_rows = 0
    for row in loader:
        seen_rows[str(row)] += 1
        num_rows += 1
        if num_rows >= 2000:
            break

    assert num_rows >= 2000, "Failed to load 2000 rows"

    repeated_rows = [row for row, count in seen_rows.items() if count > 1]
    assert not repeated_rows, f"Repeated rows found: {repeated_rows}"
