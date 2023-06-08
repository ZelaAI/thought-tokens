# Running Infra
This just contains some helpful scripts to run instances on [RunPod](https://runpod.io).

Not intended to be super reusable, but feel free to use. Connects to a spot instance on RunPod, sets up this repository, and runs a python script.

## Setup

Expects a `runpod.ignore.txt` file within this directory with the following format:

```
<RUNPOD_API_KEY>
<JUPYTER_PASSWORD>
<SSH_PUBLIC_KEY>
```

For example

```
1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef
hunter2
ssh-rsa AAAAB3NzaC1yc2E
```

Also expects a `remote.ignore.txt` file within this directory, which is the template for the commands we'll execute automatically on connection to the remote instance. My template looks like this:

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

{branch} and {script_path} will be replaced with the branch and script path you pass as arguments to `runpod.py`.

## Usage

Run the `runpod.py` script with the following arguments to:
1. Start a new instance
2. Connect and setup the instance
3. Run a python script on the instance
4. Automatically terminate the instance when the script finishes or errors (can be disabled with the `debug` flag)

```bash
python infra/runpod.py <branch> <python_script_path> [debug]
```

For example...
```bash
python infra/runpod.py main config/evals.py
```

or with debug mode enabled...
```bash
python infra/runpod.py main config/evals.py debug
```