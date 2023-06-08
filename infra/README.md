# Running Infra
This directory contains a collection of useful scripts designed to help with running instances on [RunPod](https://runpod.io).

While not primarily designed for reuse, feel free to adapt the scripts to your needs. The main functionality involves connecting to a spot instance on RunPod, setting up this repository, and executing a Python script.

## Setup

For this to work, a `runpod.ignore.txt` file is required in this directory, adhering to the following format:

```
<RUNPOD_API_KEY>
<JUPYTER_PASSWORD>
<SSH_PUBLIC_KEY>
```

For instance:

```
1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
hunter2
ssh-rsa AAAAB3NzaC1yc2E
```

Similarly, a `remote.ignore.txt` file is also expected within this directory. This file serves as the template for the commands that will be automatically executed upon connecting to the remote instance. A sample template could look like this:

```
cd /workspace
git clone https://github.com/ZelaAI/thought-tokens.git
cd thought-tokens
git checkout {branch}
pip install -r requirements.txt
huggingface-cli login --token hf_token_goes_here
wandb login wandb_token_goes_here

python {script_path}
```

In this template, `{branch}` and `{script_path}` are placeholders that will be replaced with the actual branch and script path passed as arguments to `runpod.py`.

## Usage

To utilize these scripts, run `runpod.py` with the following arguments to:
1. Initiate a new instance,
2. Connect to and set up the instance,
3. Execute a Python script on the instance,
4. Automatically terminate the instance once the script execution completes or encounters an error (this can be disabled using the `debug` flag).

Execute the following command:
```bash
python infra/runpod.py <branch> <python_script_path> [debug]
```

For example:
```bash
python infra/runpod.py main config/evals.py
```

Or to enable debug mode:
```bash
python infra/runpod.py main config/evals.py debug
```