import pickle as pkl
import random
import sys
sys.path.append('./')
sys.path.append('../')
from gadgets.util import convert_cmp_hs
import pandas as pd
from copy import deepcopy

random.seed(0)
debug = False
K = 10000 if not debug else 5

applications = pkl.load(open('data/benchmark_applications_1stdraft.pkl', 'rb'))

insightful_hypotheses = pkl.load(open('experiments/insightful_hypotheses_w-context.pkl', 'rb'))
non_insightful_hypotheses = pkl.load(open('experiments/insightful_hypotheses_wo-context.pkl', 'rb'))


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
    for j, h in enumerate(hyps):
        hyp_dict = {'orig_text': h, 'w_context': True, 'index': j, 'application_id_in_all': l['application_id_in_all']}
        hyp_dicts.append(hyp_dict)
    
    l = non_insightful_hypotheses[i]
    return_text = l['result']['choices'][0]['text']
    hyps = [h.replace('"', '').strip() for h in l['result']['choices'][0]['text'].split('\n\n')[0].split('\n-')]
    for j, h in enumerate(hyps):
        hyp_dict = {'orig_text': h, 'w_context': False, 'index': j, 'application_id_in_all': l['application_id_in_all']}
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

pkl.dump(old_hs2new_hs, open('experiments/insightful_old_hs2new_hs.pkl', 'wb'))
pkl.dump(all_data_for_eval, open('experiments/all_hyps_for_internal_eval.pkl', 'wb'))
pkl.dump(hyp_id2hyp_dict, open('experiments/hyp_id2hyp_dict.pkl', 'wb'))

for d in all_data_for_eval:
    new_hyp_dicts = []
    for hyp_dict in d['hyp_dicts']:
        if hyp_dict['index'] < 3:
            new_hyp_dicts.append(hyp_dict)
            del hyp_dict['w_context']
            del hyp_dict['index']
    random.shuffle(new_hyp_dicts)
    d['hyp_dicts'] = new_hyp_dicts

# pkl.dump(all_data_for_eval, open('scratch/all_hyps_for_internal_eval_blinded.pkl', 'wb'))
all_rows = []
for d in all_data_for_eval:
    row = {}
    row.update(d['meta_data'])
    for hyp_dict in d['hyp_dicts']:
        new_row = deepcopy(row)
        new_row.update(hyp_dict)
        all_rows.append(new_row)
df = pd.DataFrame(all_rows)
df.to_csv('experiments/all_hyps_for_internal_eval_blinded.csv', index=False)
