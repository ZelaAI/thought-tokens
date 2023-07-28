import pytest
from data.evals import TestAllNano
from core.model import GPTConfig
from core.train import train
import torch

# Merged all tests into one here, cause it's too slow to load the dataset multiple times
def test_train_base():
    train(
        device='cpu',
        compile=False,
        dtype=torch.float32,
        batch_size=1,
        max_seq_len=2048,

        wandb_log=False,

        model_config=GPTConfig.from_pretrained('EleutherAI/pythia-70m'),
        load_from_huggingface='EleutherAI/pythia-70m',
        dataset_name = "ZelaAI/librispeech_tiny_2048_streamable",

        max_iters=3,
        log_interval=1,
        eval_interval=2,
        warmup_iters=0,

        TestAllClass=TestAllNano,
    )
