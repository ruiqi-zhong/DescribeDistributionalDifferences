import os
from models import construct_proposer_prompt, construct_verifier_w_examples_prompt, construct_paired_verifier_prompt
import torch
from tqdm import trange
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mount_dir = 'mount/'

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
    model_tokenizer, prompts, temperature=0.8, n=1, bsize=8, 
    max_source_length=1024, max_target_length=512, save_score_tok_idx=None, verbose=False, stop_strs=None):
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
    return_dict = OrderedDict()
    for i, prompt in enumerate(prompts):
        return_dict[prompt] = [
            {
                'lm_postprocess': truncate_string(remove_prefix(all_completions[idx].replace(tokenizer.pad_token, ''), prompt), stop_strs), 
                'scores': [all_first_scores[idx][j] for j in save_score_tok_idx], 
                'full_generated': all_completions[idx],
                'prompt': prompt
            }
            for idx in range(i * n, (i + 1) * n)
        ]
    return return_dict


class Engine:

    def __init__(
        self,
        model_tokenizer
    ):
        self.model_tokenizer = model_tokenizer
    
    def propose_hypotheses(self, ds):
        prompts = [construct_proposer_prompt(d['pos_sents'], d['neg_sents']) for d in ds]
        return sample_batched(self.model_tokenizer, prompts)
    
    def verify_w_examples(self, ds):
        prompts = [construct_verifier_w_examples_prompt(d['positive_sample'], d['negative_sample'], d['hypothesis'], d['target_sample']) for d in ds]
        return sample_batched(self.model_tokenizer, prompts)


