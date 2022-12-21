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


if __name__ == '__main__':
    data_count = 2000

    parser = ArgumentParser()

    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    random.seed(args.seed)

    size = 'xxl'

    iteration = 0
    save_path = 'experiments/clusters/verify_%s_%d.jsonl' % ((os.path.basename(str(args.model_path)), args.seed))
    while True:
        print('Iteration %d' % iteration)
        train_data = json.load(open('experiments/clusters/train_data.json', 'r'))
        random_order = list(range(len(train_data)))
        random.shuffle(random_order)
        if args.model_path is None:
            proposer = T5Proposer(size)
        else:
            proposer = T5Proposer(args.model_path)
        
        saved_proposer_outputs = []
        pbar = tqdm(random_order[:data_count])
        for i in pbar:
            pbar.set_description('proposer, iteration %d' % (iteration))
            d = train_data[i]
            input_dicts = [d] * 4
            hypotheses = proposer.propose(input_dicts)
            save_d = {'train_id': i, 'hypotheses': hypotheses}
            saved_proposer_outputs.append(save_d)
        del proposer
        verifier = Verifier(size)

        with open(save_path, 'a') as f:
            pbar = trange(len(saved_proposer_outputs))
            for i in pbar:
                pbar.set_description('verifier, iteration %d' % (iteration))
                d =  saved_proposer_outputs[i]
                train_id, hypotheses = d['train_id'], d['hypotheses']
                pos_sents, neg_sents = train_data[train_id]['pos_sents'], train_data[train_id]['neg_sents']

                h2result = {}
                for h in hypotheses:
                    result = {}
                    h_type = 'cmp' if classify_cmp(h) else 'ind'
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
        iteration += 1