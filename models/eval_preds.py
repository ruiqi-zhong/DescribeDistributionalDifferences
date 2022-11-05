import json
from collections import defaultdict
import json
from nltk import PorterStemmer
import numpy as np
from collections import Counter
from itertools import product

removed_words = ['is', 'a', 'about', 'contain', 'describ', 'the', 'mention', 'an', 'of', 'to', 'use', 'or', 'in', 'express', 'experi', 'be', 'sound', 'peopl', 'someth', 'on', 'discuss', 'someon', 'movi', 'histor', 'action', 'like']
id2ground_truth = json.load(open('data/id2ground_truth.json', 'r'))
train_hyps = set(d['completion'] for d in json.load(open('data/ai2_1102data.json', 'r'))['train'] if d['name'] == 'proposer')
train_hyps.remove('is written in the first person')
def extract_words(h):
    h = h.split('.')[0].lower()
    return {PorterStemmer().stem(s) for s in h.split() if s not in removed_words}

train_hyps = {h: extract_words(h) for h in train_hyps}
def overlap(set1, set2):
    return len(set1 & set2) / len(set1 | set2)

def binary(logit):
    if logit[0] > logit[1]:
        return 'no'
    else:
        return 'yes'
def eval_proposer(model_preds):
    id2results = {}
    id2models_preds = defaultdict(list)
    for d in model_preds:
        id2models_preds[str(d['orig_d']['id'])].append(d)
    
    for id in sorted(id2ground_truth):
        golds = id2ground_truth[str(id)]
        model_preds = id2models_preds[str(id)]
        max_memo_similarity = 0
        max_memo_h = None
        memo_similarities = []
        max_correct_similarity = 0
        max_correct_h = None
        for d in model_preds:
            for generation in d['generations']:
                target = generation['lm_postprocess'].split('.')[0]

                target_hyps = extract_words(target)
                for h in train_hyps:
                    similarity = overlap(target_hyps, train_hyps[h])
                    memo_similarities.append(similarity)
                    if similarity > max_memo_similarity:
                        max_memo_similarity = similarity
                        max_memo_h = target
                
                for gold in golds:
                    gold_hyps = extract_words(gold)
                    similarity = overlap(target_hyps, gold_hyps)
                    if similarity > max_correct_similarity:
                        max_correct_similarity = similarity
                        max_correct_h = target

        result = {'max_memo_similarity': max_memo_similarity, 'max_memo_h': max_memo_h, 'max_correct_similarity': max_correct_similarity, 'max_correct_h': max_correct_h, 'memo_95_percentile': np.percentile(memo_similarities, 95)}
        id2results[id] = result

    avg_memo_similarity = np.mean([r['max_memo_similarity'] for r in id2results.values()])
    avg_correct_similarity = np.mean([r['max_correct_similarity'] for r in id2results.values()])
    avg_memo_95_percentile = np.mean([r['memo_95_percentile'] for r in id2results.values()])
    metrics = {'avg_memo_similarity': avg_memo_similarity, 'avg_correct_similarity': avg_correct_similarity, 'avg_memo_95_percentile': avg_memo_95_percentile}
    return metrics


def eval_paired_verifier(model_preds):
    paired_verifier_model_preds = [d for d in model_preds if d['orig_d']['name'] == 'paired_verifier']
    paired_verifier_model_preds_by_id = defaultdict(list)
    for d in paired_verifier_model_preds:
        paired_verifier_model_preds_by_id[d['orig_d']['id']].append(d)
    
    all_accs = []
    for id in paired_verifier_model_preds_by_id:
        accs = []
        for d in paired_verifier_model_preds_by_id[id]:
            accs.append(binary(d['generations'][0]['scores']) == d['demonstration'].strip())
        all_accs.append(np.mean(accs))
    return np.mean(all_accs)

def eval_verifier_w_examples(model_preds):
    verifier_w_example_model_preds_by_id = defaultdict(list)
    for d in model_preds:
        verifier_w_example_model_preds_by_id[d['orig_d']['id']].append(d)

    all_accs = []
    for id, model_preds in verifier_w_example_model_preds_by_id.items():
        acc = []
        gold_positive_preds_scores = [d['generations'][0]['scores'][1] - d['generations'][0]['scores'][0] for d in model_preds if d['demonstration'].strip() == 'yes']
        gold_negative_preds_scores = [d['generations'][0]['scores'][1] - d['generations'][0]['scores'][0] for d in model_preds if d['demonstration'].strip() == 'no']
        for pos_score, neg_score in product(gold_positive_preds_scores, gold_negative_preds_scores):
            if pos_score > neg_score:
                acc.append(1)
            else:
                acc.append(0)
        all_accs.append(np.mean(acc))
    return np.mean(all_accs)



def eval_target(path):
    model_preds = json.load(open(path, 'r'))
    metrics = eval_proposer([d for d in model_preds if d['orig_d']['name'] == 'proposer'])

    paired_verifier_model_preds = [d for d in model_preds if d['orig_d']['name'] == 'paired_verifier']
    metrics['paired_verifier_acc'] = eval_paired_verifier(paired_verifier_model_preds)

    verifier_w_example_model_preds = [d for d in model_preds if d['orig_d']['name'] == 'verifier_w_examples']
    metrics['verifier_w_examples_acc'] = eval_verifier_w_examples(verifier_w_example_model_preds)
    return metrics


if __name__ == '__main__':
    import pandas as pd

    model_preds = json.load(open('../../model_preds/tune_prompts/eval_generations-temperature=0.01_n=1_step=3000.json', 'r'))
    model_preds_grouped_by_templated_name = defaultdict(list)
    for d in model_preds:
        model_preds_grouped_by_templated_name[d['orig_d']['template']].append(d)
    # for templated_name in model_preds_grouped_by_templated_name:
    #     model_preds = model_preds_grouped_by_templated_name[templated_name]
    #     print(len(model_preds))
    #     print(model_preds[0]['prompt'])
    #     input()
    #     print(templated_name, eval_verifier_w_examples(model_preds_grouped_by_templated_name[templated_name]))
    # exit(0)

    ds = []
    # eval_path = 'data/ai2_1102data_dummy_perfect.json'
    for step in [0, 100, 200, 300, 400, 500]:
        print(step)
        # eval_path = '../../model_preds/unified_1102_tk/temperature=0.80_n=1_step=%d.json' % step
        eval_path = '../../unified_1104tk/temperature=0.80_n=1_step=%d.json' % step
        metrics = eval_target(eval_path)
        metrics['step'] = step
        metrics['name'] = 'orig'
        ds.append(metrics)

    # for step in [0, 100, 200, 300, 400, 500, 600, 700]:
    #     print(step)
    #     eval_path = '../../model_preds/unified_1102_tk_small_bsize/temperature=0.80_n=1_step=%d.json' % step
    #     metrics = eval_target(eval_path)
    #     metrics['step'] = step
    #     metrics['name'] = 'small_bsize'
    #     ds.append(metrics)
    df = pd.DataFrame(ds)
    print(df)


