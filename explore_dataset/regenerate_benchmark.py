import pickle as pkl
from collections import defaultdict
from copy import deepcopy
import random

random.seed(0)
benchmark = pkl.load(open('benchmark_1212.pkl', 'rb'))
config2full_application_dict = defaultdict(list)
individual_full_application_dicts = []

for pair_idx, pair in enumerate(benchmark):
    pair['pair_id'] = pair_idx
    dataset_desc = pair['dataset_description']
    generation = pair['generation']
    positive_description = pair['pos_desc']
    negative_description = pair['neg_desc']
    for application_idx, application in enumerate(pair['applications']):
        full_application_dict = deepcopy(pair)
        del full_application_dict['applications']
        application = deepcopy(application)

        application['purely_exploratory'] = not application['targeted']
        del application['targeted']
        full_application_dict.update(application)
        
        user, target = full_application_dict['user'], full_application_dict['target']
        full_application_dict['application_idx_in_pair'] = application_idx
        full_application_dict['application_id_in_all'] = len(individual_full_application_dicts)
        individual_full_application_dicts.append(full_application_dict)

        config = (dataset_desc, generation, user, target, pair['pair_type'])
        config2full_application_dict[config].append(full_application_dict)

for config, full_application_dicts in config2full_application_dict.items():
    i = random.randint(0, len(full_application_dicts)-1)
    full_application_dict = full_application_dicts[i]
    if not full_application_dict['purely_exploratory']:
        full_application_dict['benchmark_insightfulness'] = True
    full_application_dict['benchmark'] = True

for full_application_dict in individual_full_application_dicts:
    if 'benchmark_insightfulness' not in full_application_dict:
        full_application_dict['benchmark_insightfulness'] = False
    if 'benchmark' not in full_application_dict:
        full_application_dict['benchmark'] = False

print(len([x for x in individual_full_application_dicts if x['benchmark_insightfulness']]))
pkl.dump(individual_full_application_dicts, open('benchmark_applications_1stdraft.pkl', 'wb'))