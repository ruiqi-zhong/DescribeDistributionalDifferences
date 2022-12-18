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
from gadgets.util import gpt3wrapper, convert_cmp_hs, classify_cmp
import tqdm
import json
from collections import defaultdict
import pandas as pd
from copy import deepcopy


openai.api_key = os.environ['openai_key']
engine_name = 'text-davinci-003'

applications = pkl.load(open('data/benchmark_applications_1stdraft.pkl', 'rb'))
tok = AutoTokenizer.from_pretrained('gpt2-medium')
SINGLE_SAMPLE_MAX_LENGTH = 256
proposer_templates = {
    'w-context': open('models/templates/1217proposer_template_w_context.txt').read(),
    'wo-context': open('models/templates/1217proposer_template_wo_context.txt').read()
}
MAX_PROMPT_LENGTH = 3200
query_count = 0
K = 100
application_id2sent_subset = {}
for name, template in proposer_templates.items():
    output_path = 'experiments/data/insightful_hypotheses_%s.pkl' % name
    logs = []

    for j, application in enumerate(tqdm.tqdm(applications)):
        if not application['benchmark_insightfulness']:
            continue
        application_id_in_all = application['application_id_in_all']
        assert application_id_in_all == j
        dataset_description = application['dataset_description']
        generation = application['generation']
        positive_description = application['pos_desc']
        negative_description = application['neg_desc']
        
        target = application['target']
        user = application['user']

        num_incontext_samples = 25
        prompt = None

        arg_dict = {
            'dataset_description': dataset_description,
            'generation': generation,
            'positive_description': positive_description,
            'negative_description': negative_description,
            'user': user,
            'target': target
        }

        if application_id2sent_subset.get(application_id_in_all) is None:
            while num_incontext_samples > 1:
                pos_samples, neg_samples = application['pos_samples'], application['neg_samples']
                if len(pos_samples) > K:
                    pos_samples = random.sample(pos_samples, K)
                if len(neg_samples) > K:
                    neg_samples = random.sample(neg_samples, K)

                A_sentences = [prefix_subspan(x, SINGLE_SAMPLE_MAX_LENGTH, tok) for x in pos_samples]
                B_sentences = [prefix_subspan(x, SINGLE_SAMPLE_MAX_LENGTH, tok) for x in neg_samples]

                sent_subset = construct_blocks(A_sentences, B_sentences, num_incontext_samples=num_incontext_samples)
                
                A_block, B_block = sent_subset['A_block'], sent_subset['B_block']
                tmp_arg_dict = deepcopy(arg_dict)
                tmp_arg_dict['A_block'] = A_block
                tmp_arg_dict['B_block'] = B_block
                prompt = template.format(**tmp_arg_dict)
                prompt_length = len(tok.encode(prompt))
                if prompt_length < MAX_PROMPT_LENGTH:
                    break
                else:
                    num_incontext_samples -= 1
                    print('prompt too long, reducing num_incontext_samples to %d' % num_incontext_samples)
            application_id2sent_subset[application_id_in_all] = sent_subset

        sent_subset = application_id2sent_subset[application_id_in_all]
        arg_dict['A_block'] = sent_subset['A_block']
        arg_dict['B_block'] = sent_subset['B_block']

        query_args = {
            'engine': engine_name,
            'prompt': prompt,
            'temperature': 0.,
            'max_tokens': 512,
            'top_p': 1,
            'n': 1
        }
        query_count += 1

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
print(query_count)

random.seed(0)
debug = False
K = 10000 if not debug else 5

insightful_hypotheses = pkl.load(open('experiments/data/insightful_hypotheses_w-context.pkl', 'rb'))
non_insightful_hypotheses = pkl.load(open('experiments/data/insightful_hypotheses_wo-context.pkl', 'rb'))

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


insightful_hypotheses = insightful_hypotheses[:K]
non_insightful_hypotheses = non_insightful_hypotheses[:K]

all_data_for_eval = []
hyp_id2hyp_dict = {}
for i, l in enumerate(insightful_hypotheses):
    return_text = l['result']['choices'][0]['text']
    hyp_dicts = []
    hyps = [h.replace('"', '').strip() for h in l['result']['choices'][0]['text'].split('\n\n')[0].split('\n-')]
    hyps = [h for h in hyps if 'group b' not in h.lower() and 'group a' not in h.lower()]
    for j, h in enumerate(hyps):
        hyp_dict = {'orig_text': h, 'w_context': True, 'index': j, 'application_id_in_all': l['application_id_in_all'], 'is_comparison': classify_cmp(h)}
        hyp_dicts.append(hyp_dict)
    
    l = non_insightful_hypotheses[i]
    return_text = l['result']['choices'][0]['text']
    hyps = [h.replace('"', '').strip() for h in l['result']['choices'][0]['text'].split('\n\n')[0].split('\n-')]
    hyps = [h for h in hyps if 'group b' not in h.lower() and 'group a' not in h.lower()]
    
    for j, h in enumerate(hyps):
        hyp_dict = {'orig_text': h, 'w_context': False, 'index': j, 'application_id_in_all': l['application_id_in_all'], 'is_comparison': classify_cmp(h)}
        if 'group b' not in h.lower():
            hyp_dicts.append(hyp_dict)

    for j, hyp_dict in enumerate(hyp_dicts):
        hyp_dict['hyp_id'] = random.randint(0, 1000000000)
        hyp_id2hyp_dict[hyp_dict['hyp_id']] = hyp_dict
    
    d = {
        'meta_data': all_meta_data[l['application_id_in_all']],
        'hyp_dicts': hyp_dicts,
        'sent_subset': l['sent_subset']
    }
    all_data_for_eval.append(d)

pkl.dump(all_data_for_eval, open('experiments/data/all_hyps_for_internal_eval.pkl', 'wb'))
pkl.dump(hyp_id2hyp_dict, open('experiments/data/hyp_id2hyp_dict.pkl', 'wb'))

for d in all_data_for_eval:
    new_hyp_dicts = []
    for hyp_dict in d['hyp_dicts']:
        if hyp_dict['index'] < 3:
            new_hyp_dicts.append(hyp_dict)
            del hyp_dict['w_context']
            del hyp_dict['index']
    random.shuffle(new_hyp_dicts)
    d['hyp_dicts'] = new_hyp_dicts

all_rows = []
for d in all_data_for_eval:
    row = {}
    row.update(d['meta_data'])
    for hyp_dict in d['hyp_dicts']:
        new_row = deepcopy(row)
        new_row.update(hyp_dict)
        all_rows.append(new_row)
df = pd.DataFrame(all_rows)
df.to_csv('experiments/data/all_hyps_for_internal_eval_blinded.csv', index=False)
