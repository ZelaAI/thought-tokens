import time
import subprocess
import os
import sys

class TokensPerSecondTimer:
    def __init__(self, tokens_per_call: int = 1):
        self.last_time = time.time()
        self.call_count = 0
        self.running_average = -1
        self.tokens_per_call = tokens_per_call

    def __call__(self):
        gap = time.time() - self.last_time
        tokens_per_second = self.tokens_per_call / gap
        if self.call_count > 3:
            # Rolling average
            self.running_average = self.running_average * 0.9 + tokens_per_second * 0.1
        elif self.call_count == 3:
            # Set baseline
            self.running_average = tokens_per_second
        
        self.last_time = time.time()
        self.call_count += 1
        
        return self.running_average        



def get_current_git_branch():
    result = subprocess.run(["git", "branch"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    lines = result.stdout.splitlines()
    for line in lines:
        if line.startswith('*'):
            return line[2:]
    return None

def mint_names():
    path = os.path.abspath(sys.modules['__main__'].__file__)
    folder = os.path.dirname(path).split('/')[-1]
    file = os.path.basename(path).split('.')[0]
    
    repo_id = f'alexedw/{folder.replace("_", "-")}-{file.replace("_", "-")}'
    wandb_run_name = f'{folder.replace("_", " ").title()}: {file.replace("_", " ").title()}'
    wandb_run_group = folder.replace("_", " ").title()
    
    return repo_id, wandb_run_name, wandb_run_group, True

