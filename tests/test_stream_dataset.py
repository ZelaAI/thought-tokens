import pytest
from torch.utils.data import DataLoader
from collections import Counter
from data.stream_dataset import HuggingfaceStreamDataset

huggingface_name = "ZelaAI/minipile_512_tiny_streamable"

def test_load_rows():
    dataset = HuggingfaceStreamDataset(huggingface_name)
    loader = DataLoader(dataset, batch_size=1, num_workers=1)

    num_rows = 0
    for _ in loader:
        num_rows += 1
        if num_rows >= 2000:
            break

    assert num_rows >= 2000, "Failed to load 2000 rows"

def test_multithreaded_loading():
    dataset = HuggingfaceStreamDataset(huggingface_name)
    loader = DataLoader(dataset, batch_size=1, num_workers=2)

    num_rows = 0
    for _ in loader:
        num_rows += 1
        if num_rows >= 2000:
            break

    assert num_rows >= 2000, "Failed to load 2000 rows in multithreaded mode"

def test_no_repeated_rows():
    dataset = HuggingfaceStreamDataset(huggingface_name)
    loader = DataLoader(dataset, batch_size=1, num_workers=2)

    seen_rows = Counter()
    num_rows = 0
    for row in loader:
        seen_rows[str(row)] += 1
        num_rows += 1
        if num_rows >= 2000:
            break

    repeated_rows = [row for row, count in seen_rows.items() if count > 1]
    assert not repeated_rows, f"Repeated rows found: {repeated_rows}"
