import torch
from collections import defaultdict
import re

import datasets
import math
from collections import defaultdict
import torch
from typing import List
from data.eval_sequence import EvalSequence
from torch.utils.data import Dataset

class ModelTester:
    def __init__(self, tokenizer, append_dense_tokens=False, max_seq_len=512):
        self.call_id = 0
        self.dataset = []
        self.is_recording = False
        self.tokenizer = tokenizer
        self.append_dense_tokens = append_dense_tokens
        self.max_seq_len = max_seq_len
    
    def encode(self, text):
        return self.tokenizer.encode(text)
    
    def decode(self, tokens):
        return self.tokenizer.decode(tokens).replace("<|dense|>", "")
    
    def gather_dataset(self, function) -> List[EvalSequence]:
        self.call_id = 0
        self.is_recording = True
        self.dataset = []
        function()
        self.is_recording = False
        
        lengths = [seq.length for seq in self.dataset]
        # totals = [seq.max_new_tokens * (seq.length + (len(seq.completions) * seq.max_new_tokens // 2)) for seq in self.dataset)]
        print(f"Dataset: {len(self.dataset)} sequences, {sum(lengths)} tokens, {sum(lengths)/len(self.dataset)} avg length, {max(lengths)} max length")
            
        return self.dataset

    def rewind(self):
        self.call_id = 0

    def __call__(self, context: str, completions=[""], max_new_tokens: int = 0):
        if self.is_recording:
            if self.append_dense_tokens:
                completions = ["<|dense|>" + completion for completion in completions]
            response = EvalSequence(
                id=len(self.dataset),
                context=self.encode(context),
                completions=[self.encode(completion) for completion in completions],
                max_new_tokens=max_new_tokens,
            )

            if response.length + response.max_new_tokens * len(response.completions) > self.max_seq_len:
                raise ValueError(f"EvalSequence {response.id} is too long ({response.length + response.max_new_tokens * len(response.completions)} > {self.max_seq_len})")
            
            self.dataset.append(response)
            return response
        else:
            response = self.dataset[self.call_id]
            self.call_id += 1
            return response

class TestDataset:
    dataset: Dataset    
    name: str
    
    def __call__(self):
        results = defaultdict(float)

        total = 0
        total_skipped = 0
        
        for example in self.dataset:
            skipped, result = self.process_example(example)
            if not skipped:
                for k, v in result.items():
                    results[k] += v
                total += 1
            else:
                total_skipped += 1
        
        print('Ran', self.name, 'skipped', total_skipped)
        
        results_averaged = {
            k: v / total for k, v in results.items()
        }
        
        if 'ppl' in results_averaged:
            results_averaged['ppl'] = math.exp(results_averaged['ppl'])
        
        results_renamed = {
            f'{self.name}_{k}': v for k, v in results_averaged.items()
        }
        
        return results_renamed
    
    def process_example(self, example) -> tuple[bool, dict]:
        raise NotImplementedError
    
    def accuracy(self, loglikelihoods):
        """
        Expects a list of loglikelihoods, where the first is the correct answer.
        """
        correct, *rest = loglikelihoods
        softmaxed = torch.softmax(torch.tensor(loglikelihoods), dim=0)
        
        return {
            'accuracy': int(correct >= max(rest)),
            'softmaxed': softmaxed[0].item(),
        }
    
def compute_summative(metrics):
    summative_stats = {
        'add': {
            'sciq_softmaxed': -0.6077238103280823,
            'piqa_softmaxed': -0.5276557352961982,
            'lambada_loss': 3.038320721490364,
            'drop_@12': -0.14849986444558244,
            'hellaswag_softmaxed': -0.28309290260363207,
            'arc_easy_softmaxed': -0.33496170335209846,
            'arc_challenge_softmaxed': -0.24616402675676508,
            'winogrande_softmaxed': -0.5052074411779374
        },
        'multiply': {
            'sciq_softmaxed': 16.94857980239787,
            'piqa_softmaxed': 72.11052775569894,
            'lambada_loss': 0.8516359725254284,
            'drop_@12': 20.542073088951764,
            'hellaswag_softmaxed': 114.12281001842679,
            'arc_easy_softmaxed': 24.425137653486736,
            'arc_challenge_softmaxed': 75.00453479098687,
            'winogrande_softmaxed': 196.8258440124666
        }
    }
    
    summative = 23.5
    for key in metrics:
        if key in summative_stats['add']:
            summative += (metrics[key] + summative_stats['add'][key]) * summative_stats['multiply'][key]
    
    return summative

class TestArcEasy(TestDataset):   
    def __init__(self, model_tester: ModelTester):
        self.model_tester = model_tester
        self.dataset = datasets.load_dataset('alexedw/arc_easy', split='test')
        self.name = 'arc_easy'
        
    def process_example(self, example):
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        example["answerKey"] = num_to_letter.get(example["answerKey"], example["answerKey"])

        # options_text = '\n'.join([f"{str(i+1)}. {option}" for i, option in enumerate(example['choices']['text'])])
        context = f"Question: {example['question']}\nAnswer:"

        correct_text = example['choices']['text'][ord(example['answerKey']) - ord('A')]
        distractors_text = [" " + choice for choice in example['choices']['text'] if choice != correct_text]

        probs = self.model_tester(context, [" " + correct_text] + distractors_text).loglikelihoods
        
        return False, self.accuracy(probs)


class TestArcChallenge(TestArcEasy):
    def __init__(self, model_tester: ModelTester):
        self.model_tester = model_tester
        self.dataset = datasets.load_dataset('alexedw/arc_challenge', split='test')
        self.name = 'arc_challenge'

one_shot_maths = 'Passage: According to the 2009-2013 American Community Survey, the median income for a household in Saratoga County was $69,826, and the median income for a family was $87,058. Males had a median income of $59,636 versus $44,743 for females. The per capita income for the county was $35,176.  About 4.0% of families and 6.5% of the population were below the poverty line, including 7.4% of those under age 18 and 6.1% of those age 65 or over.\nQuestion: How many more dollars a year does a family bring in a year than a household?\nAnswer: 17232'

# # Code used to generate few shot examples
# train = datasets.load_dataset('drop', split='train').shuffle(seed=42).select(range(500))
# items = [f"Passage: {example['passage']}\nQuestion: {example['question']}\nAnswer: {example['answers_spans']['spans'][0]}" for example in train]
# items = [(item, len(model_tester.encode(item))) for item in items if len(model_tester.encode(item)) <= 150]


print("WARNING: USING SHORTER MAX LENGTH FOR DROP")
class TestDrop(TestDataset):
    max_new_tokens = 3
    num_to_generate = 12
    max_length = 500 
    
    def __init__(self, model_tester: ModelTester):
        self.model_tester = model_tester
        self.dataset = datasets.load_dataset('drop', split='validation').shuffle(seed=42).select(range(2000))
        self.name = 'drop'

    def process_example(self, example):
        context = f"{one_shot_maths}\nPassage: {example['passage']}\nQuestion: {example['question']}\nAnswer:"
        
        length = len(self.model_tester.encode(context)) + self.num_to_generate * self.max_new_tokens
        if length > self.max_length:
            return True, None
        
        gold_answers = example['answers_spans']['spans']
        
        # We pass in the first gold answer, and the model_pipeline is responsible for deciding how large of a generated answer to return
        generated_answer = self.model_tester(context, completions=[""]*self.num_to_generate, max_new_tokens=self.max_new_tokens)
        # generated_answer = self.model_tester(context, completions=[""], max_new_tokens=4)
        generated_answers = [self.model_tester.decode(c) for c in generated_answer.completions]        
        generated_answers = [a.split("\n")[0].strip() for a in generated_answers]
        
        # if generated_answers[0] != "":
        #     print(context, generated_answers, gold_answers)
        
        correct = [any([gold_answer.strip().lower() == a.lower() for gold_answer in gold_answers]) for a in generated_answers]
        return False, {
            '@1': int(correct[0]),
            '@12': int(any(correct)),
        }


class TestHellaSwag(TestDataset):
    def __init__(self, model_tester: ModelTester):
        self.model_tester = model_tester
        self.dataset = datasets.load_dataset('hellaswag', split='validation')
        self.name = 'hellaswag'
    
    def preprocess(self, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def process_example(self, example):
        context = example["activity_label"] + ": " + example["ctx_a"]
        if example["ctx_b"].strip() != "":
            context = context + " " + example["ctx_b"].capitalize()
        context = self.preprocess(context)

        correct_label = int(example['label'])
        correct_ending = ' ' + self.preprocess(example['endings'][correct_label])
        
        distractor_endings = [' ' + self.preprocess(ending) for i, ending in enumerate(example['endings']) if i != correct_label]

        probs = self.model_tester(context, [correct_ending] + distractor_endings).loglikelihoods

        return False, self.accuracy(probs)


class TestLambada(TestDataset):
    def __init__(self, model_tester: ModelTester):
        self.model_tester = model_tester
        self.dataset = datasets.load_dataset('EleutherAI/lambada_openai', split='test')
        self.name = 'lambada'
        
    def process_example(self, example):
        context = example["text"].rsplit(" ", 1)[0]
        completion = " " + example["text"].rsplit(" ", 1)[1]

        model_call = self.model_tester(context, [completion])
        exact_loss = model_call.loglikelihoods[0].item()
        
        length = model_call.get_measured_completion_lengths()[0]
        
        return False, {
            'loss': exact_loss,
            'ppl': -exact_loss * length
        }


class TestPiqa(TestDataset):
    def __init__(self, model_tester: ModelTester):
        self.model_tester = model_tester
        self.dataset = datasets.load_dataset('piqa', split='validation')
        self.name = 'piqa'
                
    def process_example(self, example):
        correct_solution = example['sol1'] if example['label'] == 0 else example['sol2']
        distractor_solution = example['sol2'] if example['label'] == 0 else example['sol1']

        context = f"Question: {example['goal']} Answer:"
        
        probs = self.model_tester(context, [f" {correct_solution}", f" {distractor_solution}"]).loglikelihoods

        return False, self.accuracy(probs)


class TestSciq(TestDataset):
    def __init__(self, model_tester: ModelTester):
        self.model_tester = model_tester
        self.dataset = datasets.load_dataset('sciq', split='validation')
        self.name = 'sciq'

    def process_example(self, example):
        context = f"{example['support'][:2160]}\nQuestion: {example['question']}\nAnswer:"
        
        completions = [
            ' ' + example['correct_answer'],
            ' ' + example['distractor1'],
            ' ' + example['distractor2'],
            ' ' + example['distractor3']
        ]
        
        probs = self.model_tester(context, completions).loglikelihoods

        return False, self.accuracy(probs)


class TestWinogrande(TestDataset):
    def __init__(self, model_tester: ModelTester):
        self.model_tester = model_tester
        self.dataset = datasets.load_dataset('winogrande', 'winogrande_xl', split='validation')
        self.name = 'winogrande'
                
    def partial_context(self, example):
        pronoun_loc = example["sentence"].index("_")
        return example["sentence"][:pronoun_loc]

    def partial_target(self, example):
        pronoun_loc = example["sentence"].index("_") + 1
        return " " + example["sentence"][pronoun_loc:].strip()

    def process_example(self, example):
        target = self.partial_target(example)
        context = self.partial_context(example)
        
        prob1 = self.model_tester(context + example['option1'], [target]).loglikelihoods[0] 
        prob2 = self.model_tester(context + example['option2'], [target]).loglikelihoods[0]
        
        correct_prob = prob1 if example['answer'] == '1' else prob2
        distractor_prob = prob2 if example['answer'] == '1' else prob1
        
        return False, self.accuracy([correct_prob, distractor_prob])
    
class TestWinograndeNano(TestWinogrande):
    def __init__(self, model_tester: ModelTester):
        self.model_tester = model_tester
        self.dataset = datasets.load_dataset('winogrande', 'winogrande_xl', split='validation').select(range(50))
        self.name = 'winogrande'

class TestAll:
    def __init__(self, model_tester: ModelTester):
        self.tests = [
            TestArcEasy(model_tester),
            TestArcChallenge(model_tester),
            TestSciq(model_tester),
            TestPiqa(model_tester),
            TestLambada(model_tester),
            TestDrop(model_tester),
            TestHellaSwag(model_tester),
            TestWinogrande(model_tester),
        ]
        
    def __call__(self):
        results = {}
        
        for test in self.tests:
            results.update(test())
        
        results['summative'] = compute_summative(results)
        
        return results
    
class TestAllNano(TestAll):
    def __init__(self, model_tester: ModelTester):
        self.tests = [
            TestWinograndeNano(model_tester),
        ]