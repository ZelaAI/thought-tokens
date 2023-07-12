from tinygrad.tensor import Tensor
import numpy as np
from core.model import GPTConfig
from core.model_tiny import GPT
from tinygrad.state import load_state_dict, get_parameters
from tinygrad.helpers import dtypes
from tinygrad.nn.optim import AdamW

from transformers import GPTNeoXTokenizerFast
tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/pythia-70m")

config = GPTConfig.from_pretrained("EleutherAI/pythia-70m")

tiny_model = GPT(config)
tiny_state_dict = GPT.state_dict_from_huggingface("EleutherAI/pythia-70m")
load_state_dict(tiny_model, tiny_state_dict, strict=False)

import random
from operator import add, sub, mul, floordiv

def generate_equation():
    ops = {
        "+": add,
        "-": sub,
        "×": mul,
        "÷": floordiv,
    }

    op, operation = random.choice(list(ops.items()))

    if op == '/':
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        a = a * b # ensure a/b is an integer
    else:
        a = random.randint(1, 25)
        b = random.randint(1, 25)

    equation = f"{a} {op} {b} = {operation(a,b)}"
    return equation

print(generate_equation())


def get_batch(batch_size):
    tokenizer.pad_token = tokenizer.eos_token
    batch = []
    for _ in range(batch_size):
        batch.append(generate_equation() + tokenizer.eos_token)
        
    inputs = tokenizer(batch, padding=True, return_tensors="np")['input_ids']
    targets = np.roll(inputs.copy(), -1, axis=1)
    targets[:, -1] = -100
    return inputs, targets

def attempt_answer(question):
    ids = Tensor([tokenizer.encode(question)], dtype=dtypes.int32)
    for i in range(2):
        output = tiny_model(ids)
        output_token = Tensor([[output.softmax(axis=-1)[0,-1].numpy().argmax()]], dtype=dtypes.int32)
        
        ids = ids.cat(output_token, dim=1)
    
    return tokenizer.decode(ids.numpy()[0], skip_special_tokens=True)


def test_model(num_tests=100):
    correct = 0
    for _ in range(num_tests):
        question = generate_equation()
        incomplete_question = question.split("=")[0] + "="
        answer = attempt_answer(incomplete_question)

        if answer.strip() == question.strip():
            print(f"\033[92m {answer} \033[0m")  # output in green
            correct += 1
        else:
            print(f"\033[91m {answer} \033[0m")  # output in red
            
    print(f"Accuracy: {correct} out of {num_tests} = {correct/num_tests:.2f}")


Tensor.training = True
BS = 128
optim = AdamW(get_parameters(tiny_model), lr=1e-4)

def loss_fn(logits, targets):
    num_classes = logits.shape[-1]
    targets_onehot = targets.reshape(list(targets.shape)+[1]).repeat([1]*len(targets.shape)+[num_classes]).eq(Tensor.arange(num_classes, dtype=targets.dtype))
    return -1 * logits.log_softmax().mul(targets_onehot).sum() / targets_onehot.sum()



rolling_loss = 3

for i in range(1000):
    inputs, targets = get_batch(BS)
    logits = tiny_model(Tensor(inputs))
    loss = loss_fn(logits, Tensor(targets))

    loss.backward()
    optim.step()
    optim.zero_grad()
    
    rolling_loss = rolling_loss * 0.95 + loss.numpy() * 0.05
    print(f"step {i} rolling loss {rolling_loss:.3f} loss {loss.numpy():.3f}", end="\r")
    
    if i % 100 == 0:
        print()
        test_model()
        print()