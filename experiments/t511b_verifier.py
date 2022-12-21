import pickle as pkl

import torch
from tqdm import trange
from collections import OrderedDict
from transformers import T5ForConditionalGeneration, T5Tokenizer
from gadgets.util import parallelize_across_device


device = 'cuda' if torch.cuda.is_available() else 'cpu'
BLOCK_SIZE = 32

def truncate_string(s, stop_strs):
    for stop_str in stop_strs:
        if stop_str in s:
            s = s[:s.index(stop_str)]
    return s

def remove_prefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    return s


def sample_batched(
    model_tokenizer, prompts, temperature=0.001, n=1, bsize=8, 
    max_source_length=1024, max_target_length=3, save_score_tok_idx=None, verbose=True, stop_strs=None):
    model, tokenizer = model_tokenizer
    prompts_inflated = []

    if stop_strs is None:
        stop_strs = []
    for prompt in prompts:
        prompts_inflated.extend([prompt] * n)
    all_completions, all_first_scores = [], []

    if save_score_tok_idx is None:
        save_score_tok_idx = [150, 4273]

    with torch.no_grad():
        model.eval()
        num_batches = (len(prompts_inflated) - 1) // bsize + 1
        if verbose:
            pbar = trange(num_batches)
            pbar.set_description('inference')
        else:
            pbar = range(num_batches)
        for batch_idx in pbar:
            input_prompts = prompts_inflated[batch_idx * bsize: (batch_idx + 1) * bsize]
            inputs = tokenizer(input_prompts, 
                                return_tensors="pt",
                                padding="longest",
                                max_length=max_source_length,
                                truncation=True,
                                ).to(device)
            generation_result = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_target_length,
                return_dict_in_generate=True,
                output_scores=True
            )
            decoded_strs = tokenizer.batch_decode(
                generation_result.sequences,
                skip_special_tokens=True,
                return_dict_in_generate=True,
                clean_up_tokenization_spaces=False
            )

            all_completions.extend(decoded_strs)
            all_first_scores.extend(generation_result.scores[0].detach().cpu().numpy().tolist())
    return all_completions, all_first_scores


class Verifier:

    def __init__(self, size='xxl'):
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-%s" % size)
        self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-%s" % size)
        parallelize_across_device(self.model)
        self.model_tokenizer = (self.model, self.tokenizer)
        self.ind_verifier_template = open('models/templates/t5_ind_verifier.txt', 'r').read()
        self.cmp_verifier_template = open('models/templates/t5_cmp_verifier.txt', 'r').read()
    
    def verify_dicts(self, input_dicts):
        all_prompts = []
        for i, input_dict in enumerate(input_dicts):
            if input_dict['type'] == 'ind':
                hypothesis, text = input_dict['hypothesis'], input_dict['text']
                all_prompts.append(self.ind_verifier_template.format(hypothesis=hypothesis, text=text))
            else:
                text_A, text_B, hypothesis = input_dict['text_A'], input_dict['text_B'], input_dict['hypothesis']
                all_prompts.append(self.cmp_verifier_template.format(hypothesis=hypothesis, text_A=text_A, text_B=text_B))
                all_prompts.append(self.cmp_verifier_template.format(hypothesis=hypothesis, text_A=text_B, text_B=text_A))
        
        all_completions = []
        next_return_idx, next_finish_threshold = 0, 1 if input_dicts[0]['type'] == 'ind' else 2
        for start_idx in range(0, len(all_prompts), BLOCK_SIZE):
            prompts = all_prompts[start_idx: start_idx + BLOCK_SIZE]
            completions, scores = sample_batched(self.model_tokenizer, prompts, verbose=False)
            all_completions.extend(completions)

            while next_finish_threshold <= len(all_completions):
                if input_dicts[next_return_idx]['type'] == 'ind':
                    yield 'yes' in all_completions[next_finish_threshold - 1].lower()
                else:
                    if 'yes' in all_completions[next_finish_threshold - 2].lower() and 'no' in all_completions[next_finish_threshold - 1].lower():
                        yield 'A'
                    elif 'no' in all_completions[next_finish_threshold - 2].lower() and 'yes' in all_completions[next_finish_threshold - 1].lower():
                        yield 'B'
                    else:
                        yield 'unsure'
                
                next_return_idx += 1
                if next_return_idx == len(input_dicts):
                    return
                next_finish_threshold += 1 if input_dicts[next_return_idx]['type'] == 'ind' else 2

if __name__ == '__main__':
    input_dicts = [
        {'type': 'ind', 'hypothesis': 'is a positive review', 'text': 'I like this movie.'},
        {'type': 'cmp', 'hypothesis': 'is more positive', 'text_A': 'I like this movie.', 'text_B': 'I hate this movie.'}
    ]
    input_dicts = input_dicts * 30
    verifier = Verifier()
    for result in verifier.verify_dicts(input_dicts):
        print(result)
