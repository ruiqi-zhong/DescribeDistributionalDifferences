import os
import sys
sys.path.append('./')

from models.create_data_for_ft import construct_proposer_prompt, construct_verifier_w_examples_prompt, construct_paired_verifier_prompt
import torch
from tqdm import trange
from collections import OrderedDict
import random
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mount_dir = 'mount/'
debug_prompt = False
eps = 1e-8

def clamp(l, min_v, max_v):
    return [min(max_v, max(min_v, x)) for x in l]

def sm(l):
    l = clamp(l, -10.0, 10.0)
    l = np.array(l)
    l = l - l.max()
    l += eps * np.random.randn(len(l))
    return np.exp(l) / np.sum(np.exp(l))

def truncate_string(s, stop_strs):
    for stop_str in stop_strs:
        if stop_str in s:
            s = s[:s.index(stop_str)]
    s = s.strip()
    if len(s) != 0 and s[-1] == '.':
        s = s[:-1]
    s = s.strip()
    return s

def remove_prefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    return s


def sample_batched(
    model_tokenizer, prompts, temperature=0.8, bsize=8, 
    max_source_length=1024, max_target_length=512, save_score_tok_idx=None, verbose=False, stop_strs=None):
    model, tokenizer = model_tokenizer

    if stop_strs is None:
        stop_strs = []

    all_completions, all_first_scores = [], []

    if save_score_tok_idx is None:
        save_score_tok_idx = [150, 4273]

    with torch.no_grad():
        model.eval()
        num_batches = (len(prompts) - 1) // bsize + 1
        if verbose:
            pbar = trange(num_batches)
            pbar.set_description('inference')
        else:
            pbar = range(num_batches)
        for batch_idx in pbar:
            input_prompts = prompts[batch_idx * bsize: (batch_idx + 1) * bsize]
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

    results = []
    for idx, prompt in enumerate(prompts):
        results.append({
            'lm_postprocess': truncate_string(remove_prefix(all_completions[idx].replace(tokenizer.pad_token, ''), prompt), stop_strs), 
            'scores': sm([all_first_scores[idx][j] for j in save_score_tok_idx]), 
            'full_generated': all_completions[idx],
            'prompt': prompt
        })
    return results


class Engine:

    def __init__(
        self,
        model_tokenizer
    ):
        self.model_tokenizer = model_tokenizer
        self.prompt2result = {}
    
    def propose_hypotheses(self, ds, verbose=False):
        prompts = [construct_proposer_prompt(d['pos_sents'], d['neg_sents']) for d in ds]
        if debug_prompt:
            print('example proposer hypothesis prompt')
            print(prompts[0])
        if verbose:
            print('proposing hypotheses')
        return [d['lm_postprocess'] for d in sample_batched(self.model_tokenizer, prompts, verbose=verbose)]
    
    def verify_w_examples(self, ds):
        prompts = [construct_verifier_w_examples_prompt(d['positive_sample'], d['negative_sample'], d['hypothesis'], d['target_sample']) for d in ds]
        if debug_prompt:
            print('example verifier with example prompt')
            print(prompts[0])
        return [d['scores'][1] for d in sample_batched(self.model_tokenizer, prompts, max_target_length=2)]

    def verify_paired(self, ds):
        prompts = [construct_paired_verifier_prompt(d['positive_sample'], d['negative_sample'], d['hypothesis']) for d in ds]
        if debug_prompt:
            print('example verifier paired prompt')
            print(prompts[0])
        return [d['scores'][1] for d in sample_batched(self.model_tokenizer, prompts, max_target_length=2)]


    def sample_batched(
        self, prompts, temperature=0.8, n=1, bsize=8, 
        max_source_length=1024, max_target_length=512, save_score_tok_idx=None, verbose=False, stop_strs=None):

        uncomputed_prompts = list({prompt for prompt in prompts if prompt not in self.prompt2result})
        new_result = sample_batched(
            self.model_tokenizer, uncomputed_prompts, temperature=temperature, n=n, bsize=bsize, 
            max_source_length=max_source_length, max_target_length=max_target_length, 
            save_score_tok_idx=save_score_tok_idx, verbose=verbose, stop_strs=stop_strs)
        
        assert len(new_result) == len(uncomputed_prompts)
        for prompt, result in zip(uncomputed_prompts, new_result):
            self.prompt2result[prompt] = result
        return [self.prompt2result[prompt] for prompt in prompts]


if __name__ == '__main__':
    class ID_Dict(dict):
        def __getitem__(self, key):
            return 'example_' + key + str(random.randint(0, 1000000))

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = 't5-small'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    id_dict = ID_Dict()

    engine = Engine((model, tokenizer))
    ds = [id_dict for _ in range(10)]
    print(engine.propose_hypotheses([{'pos_sents': ['a', 'b'], 'neg_sents': ['c', 'd']}]))
    print(engine.verify_w_examples(ds))
    print(engine.verify_paired(ds))



