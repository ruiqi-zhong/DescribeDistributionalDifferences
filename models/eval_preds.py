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


def eval_target(path):
    model_preds = json.load(open(path, 'r'))
    metrics = eval_proposer([d for d in model_preds if d['orig_d']['name'] == 'proposer'])

    # metrics['paired_verifier_acc'] = np.mean([binary(d['generations'][0]['scores']) == d['demonstration'].strip() for d in model_preds if d['orig_d']['name'] == 'paired_verifier'])

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
    metrics['paired_verifier_acc'] = np.mean(all_accs)
    verifier_w_example_model_preds = [d for d in model_preds if d['orig_d']['name'] == 'verifier_w_examples']
    verifier_w_example_model_preds_by_id = defaultdict(list)
    for d in verifier_w_example_model_preds:
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
    metrics['verifier_w_examples_acc'] = np.mean(all_accs)
    return metrics
            
    # return metrics

if __name__ == '__main__':
    import pandas as pd
    ds = []
    # eval_path = 'data/ai2_1102data_dummy_perfect.json'
    for step in [0, 100, 200, 300, 400, 500, 600, 700]:
        print(step)
        eval_path = '../../model_preds/temperature=0.80_n=1_step=%d.json' % step
        metrics = eval_target(eval_path)
        ds.append(metrics)
    df = pd.DataFrame(ds)
    print(df)
