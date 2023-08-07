import json
import requests
import time
import sys
import subprocess

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

remote_script = """
git clone https://github.com/ZelaAI/thought-tokens.git
cd thought-tokens
git checkout {branch}
pip install -r requirements.txt
huggingface-cli login --token {huggingface_key}
wandb login {wandb_key}
python -m core.train
"""

dev_script = """
git clone https://github.com/ZelaAI/thought-tokens.git
cd thought-tokens
git checkout {branch}
pip install -r requirements.txt
huggingface-cli login --token {huggingface_key}
wandb login {wandb_key}
"""

with open('infra/runpod.ignore.txt', 'r') as file:
    lines = file.read().strip().split('\n')

    api_key, jupyter_password, ssh_public_key, huggingface_key, wandb_key = lines
    
def send_query(query: str):
    # Set the headers, URL, and GraphQL query
    headers = {'Content-Type': 'application/json'}
    url = f'https://api.runpod.io/graphql?api_key={api_key}'
    
    response = requests.post(url, headers=headers, json={
        'query': query,
    })
    
    if response.status_code == 200:
        return response.json()['data']
    else:
        print(response, query)
        print(response.text)
        raise Exception(f'Request failed with status code {response.status_code}')

def get_gpu_types():
    return send_query("""
            query GpuTypes {
                gpuTypes {
                    id
                    displayName
                    memoryInGb
                }
            }
        """)['gpuTypes']

def get_gpu_by_keyword(keyword):
    gpus = get_gpu_types()
    
    gpu_id = None
    for gpu in gpus:
        if keyword in gpu['id'] or keyword in gpu['displayName']:
            gpu_id = gpu['id']
            break
        
    if gpu_id is None:
        raise Exception(f'No GPU found with keyword "{keyword}"')
    
    return send_query(f"""
            query GpuTypes {{
              gpuTypes(input: {{id: "{gpu_id}"}}) {{
                id
                displayName
                memoryInGb
                secureCloud
                communityCloud
                securePrice
                communityPrice
                oneMonthPrice
                threeMonthPrice
                oneWeekPrice
                communitySpotPrice
                secureSpotPrice
              }}
            }}
        """)['gpuTypes']

def create_spot_pod(gpu_id, bidPerGpu):
    return send_query(f"""
        mutation {{
          podRentInterruptable(
            input: {{
              bidPerGpu: {bidPerGpu}
              cloudType: ALL
              gpuCount: 1
              volumeInGb: 30
              containerDiskInGb: 30
              minVcpuCount: 2
              minMemoryInGb: 15
              gpuTypeId: "{gpu_id}"
              name: "RunPod Pytorch {gpu_id}"
              imageName: "runpod/pytorch:3.10-2.0.1-120-devel"
              dockerArgs: ""
              ports: "22/tcp"
              volumeMountPath: "/workspace"
              startSsh: true
              supportPublicIp: true
              env: [{{ 
                key: "JUPYTER_PASSWORD", value: "{jupyter_password}",
              }}, {{
                key: "PUBLIC_KEY", value: "{ssh_public_key}",
              }}]
            }}
          ) {{
            id
            imageName
            env
            machineId
            machine {{
              podHostId
            }}
          }}
        }}
    """)['podRentInterruptable']

def create_normal_pod(gpu_id):
    return send_query(f"""
        mutation {{
          podFindAndDeployOnDemand(
            input: {{
              cloudType: ALL
              gpuCount: 1
              volumeInGb: 30
              containerDiskInGb: 30
              minVcpuCount: 2
              minMemoryInGb: 15
              gpuTypeId: "{gpu_id}"
              name: "RunPod Pytorch {gpu_id}"
              imageName: "runpod/pytorch:3.10-2.0.1-120-devel"
              dockerArgs: ""
              ports: "22/tcp"
              volumeMountPath: "/workspace"
              supportPublicIp: true
              startSsh: true
              env: [{{ 
                key: "JUPYTER_PASSWORD", value: "{jupyter_password}",
              }}, {{
                key: "PUBLIC_KEY", value: "{ssh_public_key}",
              }}]
            }}
          ) {{
            id
            imageName
            env
            machineId
            machine {{
              podHostId
            }}
          }}
        }}
    """)['podFindAndDeployOnDemand']

def get_pods():   
    return send_query("""
        query Pods {
            myself {
                pods {
                id
                name
                runtime {
                    uptimeInSeconds
                    ports {
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                    }
                    gpus {
                        id
                        gpuUtilPercent
                        memoryUtilPercent
                    }
                    container {
                    cpuPercent
                        memoryPercent
                    }
                }
                }
            }
        }
        """)['myself']['pods']
    
def poll_for_pod(id):
    pods = get_pods()
    desired_pods = [pod for pod in pods if pod['id'] == id]
    
    while len(desired_pods) == 0 or desired_pods[0]['runtime'] is None:
        print('.', end='')
        sys.stdout.flush()
        time.sleep(5)
        pods = get_pods()
        desired_pods = [pod for pod in pods if pod['id'] == id]
    
    print(' Found!')
    
    return desired_pods[0]

def pod_to_ip_port(pod):
    print(pod)
    for port in pod['runtime']['ports']:
        if port['type'] == 'tcp' and port['isIpPublic']:
            return port['ip'], port['publicPort']
    return None, None

def stop_pod(id):
    return send_query(f"""
        mutation {{
          podStop(input: {{podId: "{id}"}}) {{
            id
            desiredStatus
          }}
        }}
    """)['podStop']

def terminate_pod(id):
    return send_query(f"""
        mutation {{
          podTerminate(input: {{podId: "{id}"}})
        }}
    """)['podTerminate']
    

def run_job(
    # gpu = 'H100 PCIe',
    # spot=True,
    # gpu = 'A100 SXM',
    # gpu = 'A100',
    # spot=True,
    gpu = '4090',
    spot=False,
    script = 'touch test.txt',
    debug = False,
    **kwargs
):
    print("Running job with the following parameters:")
    print(f"GPU: {gpu}")
    print(f"Script:\n'''\n{script}\n'''")
    print(f"Debug: {debug}")
    
    gpu_details = get_gpu_by_keyword(gpu)
    gpu_id = gpu_details[0]['id']
    bidPerGpuHourly = gpu_details[0]['secureSpotPrice'] or gpu_details[0]['communitySpotPrice']
    bidPerGpuHourlyNormal = gpu_details[0]['securePrice'] or gpu_details[0]['communityPrice']

    print(f'Got GPU details for {gpu_id}, price: ${bidPerGpuHourly}/h SPOT ${bidPerGpuHourlyNormal}/h NORMAL')
    start = time.time()
    
    new_pod = None
    if spot:
        new_pod = create_spot_pod(gpu_id, bidPerGpuHourly or 1.0)
    if new_pod is None:
        print('Failed to create spot pod, trying normal pod.')
        bidPerGpuHourly = bidPerGpuHourlyNormal
        new_pod = create_normal_pod(gpu_id)
    
    print(f'Created new pod with ID:{new_pod["id"]} for ${bidPerGpuHourly}/h ... waiting to connect.')
    pod = poll_for_pod(new_pod['id'])
    ip, port = pod_to_ip_port(pod)
    print('Connecting to pod @', ip, ':', port)

    if debug:
        # Write ssh config file to allow for easy ssh access
        with open('infra/ssh_config.ignore.txt', 'w') as file:
            ssh_key_kind = ssh_public_key.split(' ')[0]
            file.write(f"Host {ip}\n  HostName {ip}\n  User root\n  Port {port}\n  IdentityFile ~/.ssh/{ssh_key_kind}")

    # Danger: Not great to SSH like this, but I already consider RunPod instances to be insecure as basic security hygiene.
    command = f'ssh -o StrictHostKeyChecking=no root@{ip} -p {port} bash -s'

    try:
        subprocess.run(command, shell=True, input=script, text=True)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, cancelling job.")

    if debug:
        print("Debug mode enabled, awaiting user input before terminating pod.")
        print(f"Connect to machine with: ssh root@{ip} -p {port}")
        inp = input("Please type 'kill' to terminate pod: ")
        while inp != 'kill':
            inp = input("Please type 'kill' to terminate pod: ")
        
    print("Finished. Stopping pod.")
    stop_pod(new_pod['id'])
    print("Terminating pod.")
    terminate_pod(new_pod['id'])
    pods = get_pods()
    print('Done. Found', len(pods), 'pods still alive.')
    
    total_time = (time.time() - start) / 60 # minutes
    print(f'Total time: {total_time:.2f} minutes', f'Estimated cost: ${bidPerGpuHourly * total_time / 60:.2f}')

if __name__ == '__main__':
    dev = True if 'dev' in sys.argv else False
    debug = True if 'debug' in sys.argv or dev else False

    branch = get_current_git_branch()
    
    script = dev_script if dev else remote_script
    script = script.format(branch=branch, wandb_key=wandb_key, huggingface_key=huggingface_key)
    
    run_job(script=script, debug=debug)