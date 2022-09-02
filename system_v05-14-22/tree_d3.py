import os
import random
import pickle as pkl
from get_extreme_w_highlight import return_extreme_values
from proposer_wrapper import init_proposer
from verifier_wrapper import init_verifier
import json
from typing import List

class D3TreeSystem:

    def __init__(self,
                proposer_name: str='t5ruiqi-zhong/t5proposer_0514', # the name of the proposer. the name starts with either t5 or gpt3, followed by the directory/model-name/engine name. change argument to "t5t5-small" to debug
                verifier_name: str='ruiqi-zhong/t5verifier_0514'):# the name of the verifier, with options detailed in verifier_wrapper.py. change argument to "dummy" to debug

        self.root = None
        self.proposer_name = proposer_name
        self.proposer = init_proposer(proposer_name)
        self.verifier_name = verifier_name
        self.verifier = init_verifier(verifier_name)

    def fit(self,
            pos: List[str], # a list of text samples from D_1
            neg: List[str], # a list of text samples from D_0
            pair: str='', # name of the distribution
            depth = 0,
            save_folder=None):
        
        self.root = Tree(
            self.proposer_name,
            self.proposer,
            self.verifier_name,
            self.verifier
        )
        self.root.fit(pos, neg, pair, depth, save_folder)

class Tree:

    def __init__(self,
        proposer_name: str,
        proposer,
        verifier_name: str,
        verifier):
        
        self.left = None # left branch
        self.right = None # right branch
        self.top_hypothesis = None # best hypothesis
        self.proposer_name = proposer_name
        self.proposer = proposer
        self.verifier_name = verifier_name
        self.verifier = verifier

    def fit(self,
            pos: List[str], # a list of text samples from D_1
            neg: List[str], # a list of text samples from D_0
            pair: str='', # name of the distribution
            depth = 0,
            save_folder=None):

        print(f'{len(pos + neg)} samples')
        # saving the initial arguments
        if save_folder is None:
            save_folder = f'end2end_jobs/{pair}+{str(random.randint(0, 100000))}'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        else:
            print('Folder %s exists' % save_folder)
        print('results will be saved to %s' % save_folder)
        spec = {
            'pair': pair,
            'pos': pos,
            'neg': neg,
            'proposer_name': self.proposer_name,
            'verifier_name': self.verifier_name 
        }
        for k in ['pair', 'proposer_name', 'verifier_name']:
            print(k, spec[k])
        pkl.dump(
            spec, open(os.path.join(save_folder, 'spec.pkl'), 'wb')
        )
        
        # get samples that are representative of the differences between two distributions
        extreme_vals = return_extreme_values(pos, neg)
        pkl.dump(extreme_vals, open(os.path.join(save_folder, 'get_extreme_result.pkl'), 'wb'))
        
        # propose hypotheses
        pos2score, neg2score = extreme_vals['pos2score'], extreme_vals['neg2score']
        proposed_hypotheses = self.proposer.propose_hypothesis(pos2score, neg2score)
        
        pkl.dump(proposed_hypotheses, open(os.path.join(save_folder, 'proposed_hypotheses.pkl'), 'wb'))
        
        # verify the hypotheses
        h2result = {}
        for h in set(proposed_hypotheses):
            h2result[h] = self.verifier.return_verification(h, pos, neg, 400)
        
        pkl.dump(h2result, open(os.path.join(save_folder, 'scored_hypotheses.pkl'), 'wb'))
        

        # split by top hypothesis
        print([(h, h2result[h]['h_score']) for h in proposed_hypotheses])
        top_h = max(h2result, key=lambda h: h2result[h]['h_score'])
        print(top_h)


        if depth == 0:
            return

        self.top_hypothesis = top_h
        splitresult = self.verifier.return_split(top_h, pos, neg)
        
        left_branch = Tree(
            self.proposer_name,
            self.proposer,
            self.verifier_name,
            self.verifier
        )
        left_branch.fit(splitresult['pos_neg'],
                        splitresult['neg_neg'],
                        pair,
                        depth-1,
                        save_folder)
        self.left = left_branch

        right_branch = Tree(
            self.proposer_name,
            self.proposer,
            self.verifier_name,
            self.verifier
        )
        right_branch.fit(splitresult['pos_pos'],
                        splitresult['neg_pos'],
                        pair,
                        depth-1,
                        save_folder)
        self.right = right_branch