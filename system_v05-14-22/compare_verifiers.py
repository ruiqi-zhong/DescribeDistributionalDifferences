import os
import argparse
import pickle as pkl
from get_extreme_w_highlight import return_extreme_values
from proposer_wrapper import init_proposer
from verifier_wrapper import init_verifier
import json
from typing import List
from datetime import datetime
import tqdm

def describe(pos: List[str], # a list of text samples from D_1
             neg: List[str], # a list of text samples from D_0
             note: str='', # a note about this distribution, for logging purposes
             proposer_name: str='t5ruiqi-zhong/t5-small', # the name of the proposer. the name starts with either t5 or gpt3, followed by the directory/model-name/engine name. change argument to "t5t5-small" to debug
             verifier_name: str='ruiqi-zhong/t5verifier_0514', # the name of the verifier, with options detailed in verifier_wrapper.py. change argument to "dummy" to debug
             save_folder=None):

    # saving the initial arguments
    if save_folder is None:
        save_folder = 'jobs/compare-verifiers-' + datetime.now().strftime("%m%d_%H%M")
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    else:
        print('Folder %s exists' % save_folder)
    print('results will be saved to %s' % save_folder)

    spec = {
        'note': note,
        'pos': pos,
        'neg': neg,
        'proposer_name': proposer_name,
        'verifier_name': verifier_name 
    }

    for k in ['note', 'proposer_name', 'verifier_name']:
        print(k, spec[k])
    pkl.dump(
        spec, open(os.path.join(save_folder, 'spec.pkl'), 'wb')
    )
    
    # get samples that are representative of the differences between two distributions
    extreme_vals = return_extreme_values(pos, neg)
    pkl.dump(extreme_vals, open(os.path.join(save_folder, 'get_extreme_result.pkl'), 'wb'))
    
    # propose hypotheses
    pos2score, neg2score = extreme_vals['pos2score'], extreme_vals['neg2score']
    proposer = init_proposer(proposer_name)
    proposed_hypotheses = proposer.propose_hypothesis(pos2score, neg2score)
    
    pkl.dump(proposed_hypotheses, open(os.path.join(save_folder, 'proposed_hypotheses.pkl'), 'wb'))
    
    # verify the hypotheses
    verifier = init_verifier(verifier_name)
    h2result_standard = {}
    for h in set(proposed_hypotheses):
        h2result_standard[h] = verifier.return_verification(h, pos, neg, 500)

    # verify the hypotheses
    h2result_active = verifier.return_verification_active(proposed_hypotheses, pos, neg)

    print(h2result_standard)
    print(h2result_active)

    pkl.dump(h2result_standard, open(os.path.join(save_folder, 'standard_scored_hypotheses.pkl'), 'wb'))
    pkl.dump(h2result_active, open(os.path.join(save_folder, 'active_scored_hypotheses.pkl'), 'wb'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",action='store_true')
    args = parser.parse_args()

    proposer_name = 't5-small' if args.debug else 't5ruiqi-zhong/t5-small' 
    verifier_name = 'dummy' if args.debug else 'ruiqi-zhong/t5verifier_0514' 

    distribution_pairs = json.load(open('../benchmark_sec_4/benchmark.json'))

    for i, d in enumerate(tqdm.tqdm(distribution_pairs)):

        describe(pos=d['positive_samples'], 
                neg=d['negative_samples'],
                proposer_name=proposer_name,
                verifier_name=verifier_name,
                note='benchmark %d; can be anything, for logging purpose only' % i)
