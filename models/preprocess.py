import json
from transformers import AutoTokenizer
import numpy as np
import os

tok_path = '/mount/models/t5-small-cp_tokenizer/'
if os.path.exists(tok_path):
    t5tok = AutoTokenizer.from_pretrained(tok_path)
else:
    t5tok = AutoTokenizer.from_pretrained('t5-small')

def tok_subspan(s, subspan_token_max_len=80):
    toks = t5tok.tokenize(s)
    total_length = len(toks)
    if total_length <= subspan_token_max_len:
        return s
    random_subspan = np.random.randint(0, total_length - subspan_token_max_len)
    subspan_toks = toks[random_subspan:random_subspan + subspan_token_max_len]
    return t5tok.convert_tokens_to_string(subspan_toks)