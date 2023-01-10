import json
import sys
sys.path.append('./')
from experiments.t511b_verifier import Verifier
import random
import os
from gadgets.util import classify_cmp
import numpy as np
from argparse import ArgumentParser
from experiments.t511b_proposer import T5Proposer
from tqdm import tqdm, trange


sources = {'stock_news_20222012_094121_clusters_sqrtsize', 'un_debates_20222012_095924_clusters_sqrtsize', 'wikitext_20222012_094906_clusters_sqrtsize', 'open_review_20222012_092958_clusters_sqrtsize', 'reuters_authorship_20222012_093302_clusters_sqrtsize'}

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--proposer_model_path', type=str, default=None)
    parser.add_argument('--verifier_model_paths', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_eval', action='store_true')
    parser.add_argument('--no_propose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--source_limited', action='store_true')

    args = parser.parse_args()

    DEBUG = args.debug
    train_data_count = 2000 if not DEBUG else 5
    test_data_count = 10000 if not DEBUG else 10

    if args.proposer_model_path is None:
        args.proposer_model_path = 'google/flan-t5-xxl'
    if args.verifier_model_paths is None:
        args.verifier_model_paths = 'models/ckpts/best_verifier'
    
    source_limited = '' if not args.source_limited else '_source_limited'
    proposer_model_path = args.proposer_model_path
    verifier_model_paths = args.verifier_model_paths.split(',')
    
    if args.proposer_model_path[-1] == '/':
        proposer_model_path = args.proposer_model_path[:-1]
    for i in range(len(verifier_model_paths)):
        if verifier_model_paths[i][-1] == '/':
            verifier_model_paths[i] = verifier_model_paths[i][:-1]

    proposer_model_name = os.path.basename(str(proposer_model_path))
    
    random.seed(args.seed)

    data_name2dicts = {
        'train': json.load(open('experiments/clusters/train_data.json', 'r')),
        'test': json.load(open('experiments/clusters/test_data.json', 'r'))[:test_data_count]
    }
    if args.source_limited:
        data_name2dicts['train'] = [d for d in data_name2dicts['train'] if d['source'] in sources]
    save_path_template = 'experiments/clusters/propose_and_verify_{proposer_model_name}_{verifier_model_name}_{seed}_{split}_{iteration}_{num_samples}_{source_limited}.jsonl'

    def get_verifier_logs(split_name, idxes=None, tag=''):
        print('iteration', iteration)
        print('loading proposer from %s' % proposer_model_path)
        if idxes is None:
            idxes = list(range(len(data_name2dicts[split_name])))
        data_dicts = data_name2dicts[split_name]
        pbar = tqdm(idxes)

        saved_proposer_outputs = []
        proposer = T5Proposer(model_path=proposer_model_path)
        for i in pbar:
            pbar.set_description('proposer, tag %s' % (tag))
            d = data_dicts[i]
            input_dicts = [d] * args.num_samples
            hypotheses = proposer.propose(input_dicts)
            save_d = {'train_id': i, 'hypotheses': hypotheses}
            saved_proposer_outputs.append(save_d)
        del proposer

        for verifier_model_path in verifier_model_paths:
            print('loading verifier from %s' % verifier_model_path)
            verifier = Verifier(model_path=verifier_model_path)
            save_path = save_path_template.format(proposer_model_name=proposer_model_name, verifier_model_name=os.path.basename(verifier_model_path), seed=args.seed, split=split_name, iteration=iteration, num_samples=args.num_samples, source_limited=source_limited)
            print('saving to %s' % save_path)
            with open(save_path, 'a') as f:
                pbar = trange(len(saved_proposer_outputs))
                for i in pbar:
                    pbar.set_description('verifier, tag %s' % (tag))
                    d =  saved_proposer_outputs[i]
                    train_id, hypotheses = d['train_id'], d['hypotheses']
                    pos_sents, neg_sents = data_dicts[train_id]['pos_sents'], data_dicts[train_id]['neg_sents']

                    h2result = {}
                    for h in hypotheses:
                        result = {}
                        h_type = 'cmp' if classify_cmp(h) else 'ind'
                        if h_type == 'cmp':
                            continue
                        result['type'] = h_type

                        if h_type == 'cmp':
                            dicts = []
                            for pos_sent in pos_sents:
                                for neg_sent in neg_sents:
                                    dicts.append({'text_A': pos_sent, 'text_B': neg_sent, 'hypothesis': h, 'type': 'cmp'})
                            score = np.mean(np.array([x == 'A' for x in verifier.verify_dicts(dicts)]))
                            result['score'] = score
                        else:
                            dicts = []
                            for pos_sent in pos_sents:
                                dicts.append({'text': pos_sent, 'hypothesis': h, 'type': 'ind'})
                            for neg_sent in neg_sents:
                                dicts.append({'text': neg_sent, 'hypothesis': h, 'type': 'ind'})
                            results = list(verifier.verify_dicts(dicts))

                            frac_A = np.mean(np.array(results[:len(pos_sents)]))
                            frac_B = np.mean(np.array(results[len(pos_sents):]))
                            score = frac_A - frac_B
                            result['score'] = score
                        h2result[h] = result

                    save_d = {'train_id': train_id, 'hypotheses': h2result}
                    f.write(json.dumps(save_d) + '\n')

    iteration = 0
    if not args.no_eval:
        get_verifier_logs('test', tag='test')
    if not args.no_propose:
        while True:
            print('Iteration %d' % iteration)
            random_order = list(range(len(data_name2dicts['train'])))
            random.shuffle(random_order)
            idxes = random_order[:train_data_count]
            get_verifier_logs('train', idxes=idxes, tag='train')
            iteration += 1
