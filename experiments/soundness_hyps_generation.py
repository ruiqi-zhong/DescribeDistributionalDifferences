import pickle as pkl
import sys
sys.path.append('./')
sys.path.append('../')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
from models.preprocess import construct_blocks, prefix_subspan
from transformers import AutoTokenizer
import openai
from gadgets.util import gpt3wrapper, convert_cmp_hs
import tqdm
import json
from collections import defaultdict
import pandas as pd
from copy import deepcopy
import numpy as np


openai.api_key = os.environ['openai_key']
engine_name = 'text-davinci-003'

applications = pkl.load(open('data/benchmark_applications_1stdraft.pkl', 'rb'))
tok = AutoTokenizer.from_pretrained('gpt2-medium')
SINGLE_SAMPLE_MAX_LENGTH = 256
proposer_template = open('models/templates/1217proposer_template_w_context.txt').read()
MAX_PROMPT_LENGTH = 3500
query_count = 0
K = 100

top_p2output_paths = {
    top_p: 'experiments/data/soundness_hypotheses_%f.pkl' % top_p for top_p in [0.2, 1.0]
}
for top_p, output_path in top_p2output_paths.items():
    logs = []
    for j, application in enumerate(tqdm.tqdm(applications)):
        if not application['benchmark']:
            continue
        pair_id = application['pair_id']
        application_id_in_all = application['application_id_in_all']
        assert application_id_in_all == j
        dataset_description = application['dataset_description']
        generation = application['generation']
        positive_description = application['pos_desc']
        negative_description = application['neg_desc']

        pairs = pkl.load(open('experiments/pair_extreme/results/v1-pair_id%d/result.pkl' % pair_id, 'rb'))['pairs']
        pos_samples, neg_samples = application['pos_samples'], application['neg_samples']
        if top_p < 0.99:
            for pair in pairs:
                if np.abs(pair['top_p'] - top_p) < 1e-3:
                    pos_samples = pair['pos']
                    neg_samples = pair['neg']
                    break

        if len(pos_samples) > K:
            pos_samples = random.sample(pos_samples, K)
        if len(neg_samples) > K:
            neg_samples = random.sample(neg_samples, K)

        A_sentences = [prefix_subspan(x, SINGLE_SAMPLE_MAX_LENGTH, tok) for x in pos_samples]
        B_sentences = [prefix_subspan(x, SINGLE_SAMPLE_MAX_LENGTH, tok) for x in neg_samples]
        target = application['target']
        user = application['user']

        num_incontext_samples = 25
        prompt = None
        while num_incontext_samples > 1:
            sent_subset = construct_blocks(A_sentences, B_sentences, num_incontext_samples=num_incontext_samples)
            
            A_block, B_block = sent_subset['A_block'], sent_subset['B_block']

            arg_dict = {
                'dataset_description': dataset_description,
                'generation': generation,
                'positive_description': positive_description,
                'negative_description': negative_description,
                'user': user,
                'target': target,
                'A_block': A_block,
                'B_block': B_block
            }
            prompt = proposer_template.format(**arg_dict)

            prompt_length = len(tok.encode(prompt))
            if prompt_length < MAX_PROMPT_LENGTH:
                break
            else:
                num_incontext_samples -= 1
                print('prompt too long, reducing num_incontext_samples to %d' % num_incontext_samples)

        query_args = {
            'engine': engine_name,
            'prompt': prompt,
            'temperature': 0.,
            'max_tokens': 512,
            'top_p': 1,
            'n': 1
        }
        query_count += 1
        print(prompt)
        input()
        continue

        result = gpt3wrapper(
            tag='proposer',
            **query_args
        )

        save_result = {
            'application_id_in_all': application_id_in_all,
            'result': result,
            'sent_subset': sent_subset,
            'query_args': query_args
        }
        logs.append(save_result)
        pkl.dump(logs, open(output_path, 'wb'))


random.seed(0)
debug = False
K = 10000 if not debug else 5

all_meta_data = []
for application in applications:
    meta_data = {
        'user': application['user'],
        'target': application['target'],
        'dataset_description': application['dataset_description'],
        'generation': application['generation'],
        'positive_description': application['pos_desc'],
        'negative_description': application['neg_desc']
    }
    all_meta_data.append(meta_data)

rep_hypotheses = pkl.load(open(top_p2output_paths[0.2], 'rb'))
non_rep_hypotheses = pkl.load(open(top_p2output_paths[1.0], 'rb'))

rep_hypotheses = rep_hypotheses[:K]
non_rep_hypotheses = non_rep_hypotheses[:K]

all_data_for_eval = []
hyp_id2hyp_dict = {}
for i, l in enumerate(rep_hypotheses):
    return_text = l['result']['choices'][0]['text']
    hyp_dicts = []
    hyps = [h.replace('"', '').strip() for h in l['result']['choices'][0]['text'].split('\n\n')[0].split('\n-')]
    for j, h in enumerate(hyps):
        hyp_dict = {'orig_text': h, 'rep': True, 'index': j, 'application_id_in_all': l['application_id_in_all']}
        hyp_dicts.append(hyp_dict)
    
    l = non_rep_hypotheses[i]
    return_text = l['result']['choices'][0]['text']
    hyps = [h.replace('"', '').strip() for h in l['result']['choices'][0]['text'].split('\n\n')[0].split('\n-')]
    for j, h in enumerate(hyps):
        hyp_dict = {'orig_text': h, 'rep': False, 'index': j, 'application_id_in_all': l['application_id_in_all']}
        if 'group b' not in h.lower():
            hyp_dicts.append(hyp_dict)

    for j, hyp_dict in enumerate(hyp_dicts):
        hyp_dict['hyp_id'] = random.randint(0, 1000000000)
        hyp_id2hyp_dict[hyp_dict['hyp_id']] = hyp_dict
    
    d = {
        'meta_data': all_meta_data[l['application_id_in_all']],
        'hyp_dicts': hyp_dicts
    }
    all_data_for_eval.append(d)

all_hs = [hyp_dict['orig_text'] for d in all_data_for_eval for hyp_dict in d['hyp_dicts']]
new_hs, _, _ = convert_cmp_hs(all_hs)
assert len(all_hs) == len(new_hs)
old_hs2new_hs = {old_h: new_h for old_h, new_h in zip(all_hs, new_hs)}
for d in all_data_for_eval:
    for hyp_dict in d['hyp_dicts']:
        if 'orig_text' in hyp_dict:
            hyp_dict['processed_text'] = old_hs2new_hs[hyp_dict['orig_text']]

pkl.dump(old_hs2new_hs, open('experiments/data/soundness_old_hs2new_hs.pkl', 'wb'))
pkl.dump(all_data_for_eval, open('experiments/data/soundness_all_hyps_for_internal_eval.pkl', 'wb'))
pkl.dump(hyp_id2hyp_dict, open('experiments/data/soundness_hyp_id2hyp_dict.pkl', 'wb'))
