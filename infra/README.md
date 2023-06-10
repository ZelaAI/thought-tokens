# Running Infra
This directory contains a collection of useful scripts designed to help with running instances on [RunPod](https://runpod.io).

While not primarily designed for reuse, feel free to adapt the scripts to your needs. The main functionality involves connecting to a spot instance on RunPod, setting up this repository, and executing a Python script.

## Setup

For this to work, a `runpod.ignore.txt` file is required in this directory, adhering to the following format:

```
<RUNPOD_API_KEY>
<JUPYTER_PASSWORD>
<SSH_PUBLIC_KEY>
<HUGGINGFACE_TOKEN>
<WANDB_TOKEN>
```

For instance:
```
1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
hunter2
ssh-rsa AAAAB3NzaC1yc2E
hf_1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
wandb_1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
```

## Usage

To utilize these scripts, run `runpod.py` with the following arguments to:
1. Initiate a new instance,
2. Connect to and set up the instance,
3. Execute a Python script on the instance,
4. Automatically terminate the instance once the script execution completes or encounters an error (this can be disabled using the `debug` flag).

Execute the following command:
```bash
python infra/runpod.py
```

Or to enable debug mode:
```bash
python infra/runpod.py debug
```