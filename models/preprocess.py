import json
from transformers import AutoTokenizer
import numpy as np
import os

tok_path = '/mount/models/t5-small-cp_tokenizer/'
if os.path.exists(tok_path):
    t5tok = AutoTokenizer.from_pretrained(tok_path)
else:
    t5tok = AutoTokenizer.from_pretrained('t5-small')


def tok_subspan(s, subspan_token_max_len=80, tok=t5tok):
    toks = t5tok.tokenize(s)
    total_length = len(toks)
    if total_length <= subspan_token_max_len:
        return s
    random_subspan = np.random.randint(0, total_length - subspan_token_max_len)
    subspan_toks = toks[random_subspan:random_subspan + subspan_token_max_len]
    return t5tok.convert_tokens_to_string(subspan_toks)


def prefix_subspan(s, prefix_token_max_len=80, tok=t5tok):
    toks = t5tok.tokenize(s)
    total_length = len(toks)
    if total_length <= prefix_token_max_len:
        return s
    subspan_toks = toks[:prefix_token_max_len]
    return t5tok.convert_tokens_to_string(subspan_toks) + '...'


def id(x):
    return x

def construct_blocks(pos_sentences, neg_sentences, num_incontext_samples=4, truncate=True):
    A_subsampled_sentences = np.random.choice(pos_sentences, min(num_incontext_samples, len(pos_sentences)), replace=False)
    truncate = id if not truncate else prefix_subspan
    A_subsampled_sentences_subspan = [truncate(x) for x in A_subsampled_sentences]
    A_block = ''.join(['Group A: ' + s + '\n' for s in A_subsampled_sentences_subspan])

    B_subsampled_sentences = np.random.choice(neg_sentences, min(num_incontext_samples, len(neg_sentences)), replace=False)
    B_subsampled_sentences_subspan = [truncate(x) for x in B_subsampled_sentences]
    B_block = ''.join(['Group B: ' + s + '\n' for s in B_subsampled_sentences_subspan])

    return {
        'A_block': A_block,
        'B_block': B_block,
        'A_subsampled_sentences_subspan': A_subsampled_sentences_subspan,
        'B_subsampled_sentences_subspan': B_subsampled_sentences_subspan
    }