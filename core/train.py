import os
import sys
import time
import math
from contextlib import nullcontext
from core.utils import TokensPerSecondTimer, mint_names

from data.packer import Packer
from data.stream_dataset import HuggingfaceStreamDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from torch.utils.data import DataLoader
from huggingface_hub import HfApi
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import json

from tqdm import tqdm
from data.evals import ModelTester, TestAll
from data.eval_sequence import EvalSequence, EvalBatch, report_logits
from data.train_sequence import TrainSequence, TrainBatch

from multiprocessing import Process

from core.model import GPTConfig, GPT, Tokenizer
import random

def train(
    # -----------------------------------------------------------------------------
    eval_only = False, # if True, script exits right after the first eval
    train_only = False,

    # I/O
    eval_interval = 125,
    log_interval = 1,

    # Checkpointing
    out_dir = 'out',
    checkpoint_interval = 125,
    upload_checkpoint_interval = 250,
    repo_id = "alexedw/gptx-default",

    # wandb logging
    wandb_log = True, # disabled by default
    wandb_project = 'gptx',
    wandb_run_name = 'default-run-name', # 'run' + str(time.time())
    wandb_run_group = None,

    # data
    dataset_name = 'ZelaAI/minipile_512_streamable',

    gradient_accumulation_steps = 8, # used to simulate larger batch sizes
    batch_size = 24, # if gradient_accumulation_steps > 1, this is the micro-batch size
    max_seq_len = 512,
    TestAllClass=TestAll,

    # evals
    tokenizer_name = 'EleutherAI/pythia-410m',

    # adamw optimizer
    max_iters = 10000,
    learning_rate = 1e-5,
    weight_decay = 0.1,
    beta1 = 0.9,
    beta2 = 0.95,
    grad_clip = 1.0, # disable if == 0.0

    # learning rate decay settings
    decay_lr = True, # whether to decay the learning rate
    warmup_iters = 1000, # how many steps to warm up for
    lr_decay_iters = 10000, # should be ~= max_iters per Chinchilla
    min_lr = 1e-6, # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # Model State
    iter_num = 0,
    model_config = GPTConfig.from_pretrained('EleutherAI/pythia-410m'),
    load_from_huggingface = 'EleutherAI/pythia-410m',
    load_from_huggingface_revision = 'main',
    load_from_checkpoint = None,
    load_from_checkpoint_local = False,

    temperature = 0.7,
    top_p = 0.9,

    # DDP settings
    backend = 'nccl', # 'nccl', 'gloo', etc.
    # system
    device = 'cuda',
    dtype = torch.float16,
    compile = True,

    insert_dense_tokens = 12,

    **_kwargs
):
  
    # mark all params as globals
    # -----------------------------------------------------------------------------
    config_keys = [k for k, v in locals().items() if not k.startswith('_')]
    local_items = locals()
    config = {k: str(local_items[k]) for k in config_keys} # will be useful for logging
    print('Running with Config:')
    for k, v in config.items():
        print(f'  {k}: {v}')
    
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE']) # total number of training processes
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        world_size = 1
        master_process = True
        seed_offset = 0

    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(1337 + seed_offset)
    torch.cuda.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    ctx = torch.amp.autocast(device_type='cuda', dtype=dtype) if 'cuda' in device else nullcontext()

    model = GPT(model_config)
    model.to(device)

    tokenizer = Tokenizer(tokenizer_name)

    if load_from_huggingface is not None:
        state_dict = GPT.state_dict_from_huggingface(load_from_huggingface, revision=load_from_huggingface_revision)
        model.load_state_dict(state_dict)
        state_dict = None

    if compile:
        unoptimized_model = model
        model = torch.compile(unoptimized_model) # pytorch 2.0

    if not train_only:
        model_tester = ModelTester(tokenizer, append_dense_tokens=insert_dense_tokens > 0, max_seq_len=max_seq_len)

        test_all = TestAllClass(model_tester)
        
        # make sure metrics.jsonl exists
        with open(f'{out_dir}/metrics.jsonl', 'a') as f:
            pass

    if not eval_only:
        train_dataset = HuggingfaceStreamDataset(dataset_name, skip_to=batch_size * iter_num * gradient_accumulation_steps)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=TrainBatch.collate_fn, num_workers=2, prefetch_factor=2)
        train_dataloader_iter = iter(train_dataloader)

        def get_batch():
            return next(train_dataloader_iter).to(device)

        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler_enabled = dtype == torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled, init_scale=2**12, growth_interval=1000)

        # optimizer
        optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)

    if load_from_checkpoint is not None:        
        state_dict = torch.load(hf_hub_download(load_from_checkpoint, "model_state.pt", revision=str(iter_num)), map_location=device)
        model.load_state_dict(state_dict)
        state_dict = None
        
        if not eval_only:
            optimizer_state_dict = torch.load(hf_hub_download(load_from_checkpoint, "optimizer_state.pt", revision=str(iter_num)), map_location=device)
            optimizer.load_state_dict(optimizer_state_dict)
            optimizer_state_dict = None
        # prevent immediate re-upload of checkpoint
        iter_num += 1
        
    if load_from_checkpoint_local:
        model.load_state_dict(torch.load(f"{out_dir}/{iter_num}/model_state.pt", map_location=device))
        if not eval_only:
                optimizer.load_state_dict(torch.load(f"{out_dir}/{iter_num}/optimizer_state.pt", map_location=device))
        iter_num += 1

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    @torch.no_grad()
    def run_evals():
        model.eval()
        
        packer = Packer(max_seq_len, model_tester.gather_dataset(test_all))
        
        collate_fn = EvalBatch.packs_to_batch_factory(
            train=False,
            causal=True,
            max_seq_len=max_seq_len,
            batch_size=batch_size
        )
        
        dataloader = DataLoader(packer, batch_size=batch_size, collate_fn=collate_fn)
        dataloader_iter = iter(dataloader)

        st, ll, stl, prev_batch = None, None, None, EvalBatch(None, None, None, None, None, None, None, [[]], None)
        batch = next(dataloader_iter).to(device)
    
        eval_tokens_per_second_timer = TokensPerSecondTimer(batch_size * max_seq_len)

        while len(batch.packs[0]) > 0 or len(prev_batch.packs[0]) > 0:
            with ctx:
                logits, _ = model(batch.inputs, targets=batch.targets)

            todo = report_logits(st, ll, stl, prev_batch.packs)
            packer.add_to_queue(todo)

            prev_batch = batch
            batch = next(dataloader_iter).to(device)

            st, stl = model.sample_top_p_selective(logits, prev_batch.generate_positions, temperature, top_p)
            ll = model.loglikelihood_selective(logits, prev_batch.targets, prev_batch.target_pos_mask)
            
            st, ll, stl = st.cpu(), ll.cpu(), stl.cpu()
            logits = None
            
            tokens_per_second = eval_tokens_per_second_timer()
            # print(f'eval: {tokens_per_second:.0f} tokens/s, packer: {packer}, batch: {batch}', end='\r', flush=True)
            print(f'eval: {tokens_per_second:.0f} tokens/s  ', end='\r', flush=True)
        print()
        
        model.train()
        return test_all()         

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    def upload_folder_to_hf(api, folder_path, repo_id, revision):
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
        )

    # logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config, group=wandb_run_group)
        wandb.save(f"{out_dir}/metrics.jsonl", policy="live")

    checkpoints = []
    # training loop
    batch = get_batch() if not eval_only else None # fetch the very first batch
    t0 = time.time()
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    metrics = None
    tokens_per_second_timer = TokensPerSecondTimer(batch_size * world_size * gradient_accumulation_steps * max_seq_len)
    
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        if not eval_only:
            for param_group in optimizer.param_groups:
                if param_group['name'] == 'special_params':
                    param_group['lr'] = lr * 5
                else:
                    param_group['lr'] = lr

        if (iter_num % eval_interval == 0 and master_process and not train_only) or eval_only:
            metrics = {
                "iter": iter_num,
                "lr": lr,
                **run_evals(),
            }
            
            with open(f"{out_dir}/metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")
            
            print(f"step {iter_num}")
            for k, v in metrics.items():
                print(f"- {k}: {v}")
            if wandb_log:
                wandb.log(metrics)
            if eval_only:
                break

        if iter_num % checkpoint_interval == 0 and master_process:
            checkpoint_dir = f'{out_dir}/{iter_num}'
            os.makedirs(checkpoint_dir, exist_ok=True)

            torch.save(model.state_dict(), f"{checkpoint_dir}/model_state.pt")
            torch.save(optimizer.state_dict(), f"{checkpoint_dir}/optimizer_state.pt")
            with open(f"{checkpoint_dir}/config.py", "w") as f:
                f.write(str(config))
                f.write(f"\niter_num = {iter_num}")

            checkpoints.append(iter_num)

            if len(checkpoints) > 3:
                # delete oldest checkpoint
                checkpoint = checkpoints.pop(0)
                os.remove(f"{out_dir}/{checkpoint}/model_state.pt")
                os.remove(f"{out_dir}/{checkpoint}/optimizer_state.pt")

            print(f"Checkpoint saved at {checkpoint_dir}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "checkpoint_saved": iter_num,
                })
            if iter_num > 0 and repo_id is not None and iter_num % upload_checkpoint_interval == 0:
                api = HfApi()
                api.create_repo(repo_id=repo_id, private=True, exist_ok=True, repo_type='model')
                api.create_branch(repo_id=repo_id, branch=str(iter_num), exist_ok=True, repo_type='model')
                
                # Start a new process for the upload
                upload_process = Process(target=upload_folder_to_hf, args=(api, checkpoint_dir, repo_id, str(iter_num)))
                upload_process.start()

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                _, loss = model(batch.inputs, targets=batch.targets)
                
                loss = loss / gradient_accumulation_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            batch = get_batch()
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Loss is {loss}, stopping training, attempt rollback")

            if len(checkpoints) == 0:
                raise Exception("No checkpoints found - crashing :(")
            # get most recent checkpoint - pop from list so we don't use it again endlessly
            checkpoint = checkpoints.pop()
            iter_num = checkpoint + 1

            if wandb_log and master_process:
                wandb.log({"checkpoint_restored": checkpoint, "iter": iter_num})

            model.load_state_dict(torch.load(f"{out_dir}/{checkpoint}/model_state.pt", map_location=device))
            optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device)
            optimizer.load_state_dict(torch.load(f"{out_dir}/{checkpoint}/optimizer_state.pt", map_location=device))
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled, init_scale=2**12, growth_interval=1000) # reset scaler
            continue

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        tokens_per_second = tokens_per_second_timer()
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps # loss as float. note: this is a CPU-GPU sync point

            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, tokens per second {tokens_per_second:.2f}", flush=True)
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "batch/loss": lossf,
                    "lr": lr,
                    "tps": tokens_per_second, # convert to percentage
                })

        iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

    if ddp:
        destroy_process_group()
    
    # clean up memory
    model = None
    raw_model = None
    unoptimized_model = None
    optimizer = None
    torch.cuda.empty_cache()

if __name__ == "__main__":
    repo_id, wandb_run_name, wandb_run_group = mint_names()

    train(
        repo_id=repo_id,
        wandb_run_name=wandb_run_name,
        wandb_run_group=wandb_run_group,
    )