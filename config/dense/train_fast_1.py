"""
This test is going to just be a baseline mostly. I've got a bunch of tests I've done in an old repo
and I'd like to confirm that all the code I've written here performs roughly the same as the code.
"""
from core.train import train, mint_names

repo_id, wandb_run_name, wandb_run_group, wandb_log = mint_names()

learning_rate = 1e-5
min_lr = 1e-6

insert_dense_tokens = 12

gradient_accumulation_steps = 8
batch_size = 24

eval_interval = 125
checkpoint_interval = 125
upload_checkpoint_interval = 250

max_iters = 10000
warmup_iters = 1000
lr_decay_iters = 10000

####################  RESUME ####################
# iter_num = 0
# load_from_checkpoint = 'alexedw/dense-train-dense-9'
# load_from_huggingface = None
#################### ###### #####################

train(**globals())
