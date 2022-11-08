from copy import deepcopy
from collections import defaultdict
from itertools import product
import math
import numpy as np
from models.engine import Engine
import random
import tqdm
import pickle as pkl
import json
from typing import Dict, List, Tuple
from gadgets.lexical_diversity import lexical_diversity
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
import os

DEBUG = False

VERIFY_HYP_BLOCK_SIZE = 32


def r_to_z(r):
    return math.log((1 + r) / (1 - r)) / 2.0

def z_to_r(z):
    e = math.exp(2 * z)
    return((e - 1) / (e + 1))

def r_confidence_interval(r, alpha, n):
    z = r_to_z(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha/2)  # 2-tailed z critical value

    lo = z - z_crit * se
    hi = z + z_crit * se

    # Return a sequence
    return (z_to_r(lo), z_to_r(hi))


#  calculate a high confidence interval of the pearsonr correlation between x and y
def calculate_corr(x, y, alpha=1e-5):
    x, y = np.array(x), np.array(y)
    n = len(x)
    corr, p = stats.pearsonr(x, y)
    lo, hi = r_confidence_interval(corr, alpha, n)
    result = {'corr': corr, 'p': p, 'lo': lo, 'hi': hi, 'n': n}
    return result


def search_clf_w_sparse_positive_weights(X, Y, K):
    initial_C = 0.01
    X = np.array(X)

    # weaker and weaker regularization
    all_Cs = [initial_C * (2 ** i) for i in range(13)]
    print('Searching for the correct regularization strength...')
    pbar = tqdm.tqdm(enumerate(all_Cs))
    for clf_round, C in pbar:
        pbar.set_description(f'Strength: {C}')
        clf = LogisticRegression(C=C, solver='liblinear', max_iter=1000, penalty='l1')
        clf.fit(X, Y)
        if np.sum(clf.coef_ > 0) >= K:
            break

    # return positive idxes
    coef = clf.coef_[0]
    dim_sorted = sorted(range(len(coef)), key=lambda i: coef[i], reverse=True)
    top_K_dims = dim_sorted[:K]
    selected_dims = [i for i in top_K_dims if coef[i] > 0]
    if len(selected_dims) == 0:
        return [0]
    return selected_dims


class DistributionPairInstance:

    def __init__(
        self, 
        pos2scores, 
        neg2scores, 
        engine, 
        top_fraction=None,
        num_hyps=10,
        max_round=2,
    ):
        self.orig_pos2scores = pos2scores
        self.orig_neg2scores = neg2scores
        self.orig_sent2membership = {}
        for sent in pos2scores:
            self.orig_sent2membership[sent] = 1.
        for sent in neg2scores:
            self.orig_sent2membership[sent] = 0.

        self.engine = engine

        self.current_sent2residual = None
        self.current_pos2representative = None
        self.current_neg2representative = None

        self.current_round = 0
        self.all_hypotheses = []

        self.top_fraction = top_fraction
        if top_fraction is None:
            self.top_fraction = [0.05, 0.2, 1.0]
        self.num_hyps = num_hyps
        self.max_round = max_round
        self.logs = []
    
    # this function sets the representativeness based on residual
    # if the residual is None, then it sets the representativeness to be the original scores
    def set_current_representative_for_proposal(self):
        self.current_pos2representative = deepcopy(self.orig_pos2scores)
        self.current_neg2representative = deepcopy(self.orig_neg2scores)
        if self.current_sent2residual is not None: 
            for sent in self.current_pos2representative:
                self.current_pos2representative[sent] = self.current_sent2residual[sent]
            for sent in self.current_neg2representative:
                self.current_neg2representative[sent] = -self.current_sent2residual[sent]
        else:
            self.current_sent2residual = {}
            for sent in self.current_pos2representative:
                self.current_sent2residual[sent] = 1.0
            for sent in self.current_neg2representative:
                self.current_sent2residual[sent] = 0.0

    def get_hypotheses(self):
        proposed_hypotheses = []
        proposer_args = []
        for p in self.top_fraction:
            for _ in range(self.num_hyps):
                sorted_pos = sorted(self.current_pos2representative, key=self.current_pos2representative.get, reverse=True)
                sorted_neg = sorted(self.current_neg2representative, key=self.current_neg2representative.get, reverse=True)
                pos, neg = lexical_diversity(sorted_pos, sorted_neg, top_p=p)
                proposer_args.append({'pos_sents': pos, 'neg_sents': neg})
        raw_nl_hyps = self.engine.propose_hypotheses(proposer_args, verbose=True)
        raw_nl_hyps = [h for h in raw_nl_hyps if h is not None]
        if DEBUG:
            raw_nl_hyps = ['hyp %d' % random.randint(0, 10) for _ in range(len(raw_nl_hyps))]

        print('Get in-context example demonstrations of the hypotheses')
        for hypothesis, proposer_arg in tqdm.tqdm(zip(raw_nl_hyps, proposer_args)):
            pos_sents, neg_sents = proposer_arg['pos_sents'], proposer_arg['neg_sents']
            paired_verifier_args = []
            for (pos_sent, neg_sent) in product(pos_sents, neg_sents):
                arg = {'positive_sample': pos_sent, 'negative_sample': neg_sent, 'hypothesis': hypothesis, 'expected_polarity': 1}
                paired_verifier_args.append(arg)
                arg = {'positive_sample': neg_sent, 'negative_sample': pos_sent, 'hypothesis': hypothesis, 'expected_polarity': -1}
                paired_verifier_args.append(arg)
            all_scores = self.engine.verify_paired(paired_verifier_args)
            assert len(all_scores) == len(paired_verifier_args)
            for d, s in zip(paired_verifier_args, all_scores):
                d['credit'] = d['expected_polarity'] * s
            
            pos_sent2credit = {pos_sent: np.mean([d['credit'] for d in paired_verifier_args if d['positive_sample'] == pos_sent]) for pos_sent in pos_sents}
            best_pos_sent = max(pos_sent2credit, key=pos_sent2credit.get)
            neg_sent2credit = {neg_sent: np.mean([d['credit'] for d in paired_verifier_args if d['negative_sample'] == neg_sent]) for neg_sent in neg_sents}
            best_neg_sent = max(neg_sent2credit, key=neg_sent2credit.get)
            hypothesis_confidence = (np.mean(list(pos_sent2credit.values())) + np.mean(list(neg_sent2credit.values()))) / 2
            hypothesis_dict = {
                'hypothesis': hypothesis, 
                'pos_sent': best_pos_sent, 
                'neg_sent': best_neg_sent, 
                'confidence': hypothesis_confidence, 
                'sent2score': {}, 
                'fullly_computed': False, 
                '+': True,
                'provenance': proposer_arg
            }
            proposed_hypotheses.append(hypothesis_dict)
        self.all_hypotheses.extend(proposed_hypotheses)
        return proposed_hypotheses


    # calculate the correlation between the residual and the hypothesis on existing data
    # NOTE: we will first calculate its correlation with the original 0/1 binary label
    # and flip it such that it is positively correlated
    def get_correlation_info(self, hypothesis):
        ordered_text = sorted(hypothesis['sent2score'], key=hypothesis['sent2score'].get)
        gold = [self.current_sent2residual[sent] for sent in ordered_text]
        pred = [hypothesis['sent2score'][sent] for sent in ordered_text]
        orig = [self.orig_sent2membership[sent] for sent in ordered_text]
        orig_corr = calculate_corr(orig, pred)

        # if the hypothesis is negatively correlated with the orig label
        # flip its value to calculate its correlation with the residual
        # notice that the values hypothesis['sent2score'] will never be flipped
        # the score will only be flipped when we calculate its correlation
        # and when we use it to run a logistic regression
        if orig_corr['corr'] < 0:
            hypothesis['+'] = False
            pred = [-x for x in pred]

        residual_corr = calculate_corr(gold, pred)
        return residual_corr
    
    # rule out weak hypotheses based on the current residual
    # this is the place where @Peter can improve the algorithm
    def filter_weak_hypotheses(self, hypotheses, K=5):
        corr_info = [self.get_correlation_info(h) for h in hypotheses]

        lower_bounds = [c['lo'] for c in corr_info]
        top_K_lower_bounds = sorted(lower_bounds, reverse=True)[:K]

        new_hypotheses = []
        for h, c in zip(hypotheses, corr_info):
            if c['hi'] < top_K_lower_bounds[-1]:
                continue
            new_hypotheses.append(h)
        return new_hypotheses


    def get_best_hypotheses_active(self, target_hypotheses):
        random_sent_order = list(self.current_sent2residual.keys())
        random.shuffle(random_sent_order)

        # competitive_hypotheses is a list of hypotheses that can still be in the top-5
        competitive_hypotheses = list(target_hypotheses)
        cur_pointer = 0

        print('Filtering out weak hypotheses')

        # enumerate the sentences in random order
        with tqdm.tqdm(total=len(random_sent_order)) as pbar:
            while cur_pointer < len(random_sent_order):

                # take a batch of sentences, and compute a score for every competitive hypotheses
                sents = random_sent_order[cur_pointer:cur_pointer+VERIFY_HYP_BLOCK_SIZE]
                cur_pointer += VERIFY_HYP_BLOCK_SIZE

                # construct the verifier_w_example arguments
                verifier_w_example_args = []
                for sent in sents:
                    for hypothesis in competitive_hypotheses:
                        verifier_arg = {'positive_sample': hypothesis['pos_sent'], 'negative_sample': hypothesis['neg_sent'], 'hypothesis': hypothesis['hypothesis'], 'target_sample': sent, 'orig_h': hypothesis}
                        verifier_w_example_args.append(verifier_arg)
                
                # run the verifier 
                all_scores = self.engine.verify_w_examples(verifier_w_example_args)
                assert len(all_scores) == len(verifier_w_example_args)
                for d, s in zip(verifier_w_example_args, all_scores):
                    d['orig_h']['sent2score'][d['target_sample']] = s
                
                # filter out weaker hypotheses based on UCB
                # TODO: Peter can probably improve here. 
                competitive_hypotheses = self.filter_weak_hypotheses(competitive_hypotheses)
                pbar.update(len(sents))
                pbar.set_description('Num hypotheses: %d' % len(competitive_hypotheses))
        for h in competitive_hypotheses:
            assert len(h['sent2score']) == len(self.current_sent2residual)
            h['fullly_computed'] = True
        return competitive_hypotheses


    def calculate_residual(self):
        print('Calculating residual at round %d' % self.current_round)
        hypotheses = []
        for h in self.all_hypotheses:
            if not h['fullly_computed']:
                continue
            hypotheses.append(h)
        
        sents = list(self.orig_sent2membership)
        Y = np.array([self.orig_sent2membership[sent] for sent in sents])
        X = np.array([[h['sent2score'][sent] * (1 if h['+'] else -1) for h in hypotheses] for sent in sents])
        
        print('Selecting features')
        selected_feature_dims = search_clf_w_sparse_positive_weights(X, Y, self.current_round + 1)
        selected_X = X[:, selected_feature_dims]
        clf = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
        clf.fit(selected_X, Y)
        Y_hat = clf.predict_proba(selected_X)[:,1]
        self.current_sent2residual = {sent: Y[i] - Y_hat[i] for i, sent in enumerate(sents)}

        self.current_selected_hypotheses_weight = []
        for i in range(len(selected_feature_dims)):
            hypothesis = deepcopy(hypotheses[selected_feature_dims[i]])
            hypothesis['weight'] = clf.coef_[0][i]
            self.current_selected_hypotheses_weight.append(hypothesis)
        
    
    def one_step(self):
        log_this_round = {}
        # we will sort the sentences from both groups based on their representativeness scores
        # and will show the proposer models the representative examples
        self.set_current_representative_for_proposal()

        # log the current residual and representativeness score
        log_this_round['start_residual'] = deepcopy(self.current_sent2residual)
        log_this_round['start_representativeness'] = {'pos': deepcopy(self.current_pos2representative), 'neg': deepcopy(self.current_neg2representative)}
        
        # propose the hypotheses
        hypotheses = self.get_hypotheses()

        # run the verifier, and adaptively deciding which hypothesis we want to keep
        self.get_best_hypotheses_active(hypotheses)

        self.calculate_residual()

        # log the current residual and representativeness score
        log_this_round['end_residual'] = deepcopy(self.current_sent2residual)
        log_this_round['end_representativeness'] = {'pos': deepcopy(self.current_pos2representative), 'neg': deepcopy(self.current_neg2representative)}
        
        log_this_round['current_h_weight'] = deepcopy(self.current_selected_hypotheses_weight)

        self.current_round += 1
        self.logs.append(log_this_round)
    
    def run(self):
        # proceed multiple rounds
        while self.current_round < self.max_round:
            self.one_step()
        return {
            'logs': self.logs,
            'hypotheses': self.all_hypotheses
        }



def subsample(sent2score: Dict[str, float], subsample_size=200) -> Dict[str, float]:
    if len(sent2score) <= subsample_size:
        return sent2score
    all_sents = list(sent2score.keys())
    random.shuffle(all_sents)
    return {sent: sent2score[sent] for sent in all_sents[:subsample_size]}


if __name__ == '__main__':
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    # toy test case will run on toy examples with t5-small to check whether code is executable
    # realistic test case will run on the 54 benchmark datasets
    test_case = 'toy'
    if test_case == 'realistic':
        from gadgets.util import parallelize_across_device

        benchmark = json.load(open('benchmark_sec_4/benchmark.json'))

        # the model path; just let me know if you cannot access it
        ckpt_path = '/scratch/users/ruiqi-zhong/describe_difference/DescribeDistributionalDifferences/models/ckpts/checkpoint-500/'
        if not os.path.exists(ckpt_path):
            ckpt_path = 'models/checkpoint-200/'
        model = AutoModelForSeq2SeqLM.from_pretrained()

        model.eval()

        # parallelize the model across device if you are running on small GPUs
        # no need if using saruman
        parallelize_across_device(model)

        # loading the tokenizer
        tok_path = '/mount/models/t5-small-cp_tokenizer/'
        if os.path.exists(tok_path):
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained('t5-small')
    
        print('Finished loading model')
        tokenizer.model_max_length = 1024
        model_tokenizer = (model, tokenizer)
        id2result = {}
        for distribution_idx in range(10):
            # pretty generic data loading, nothing interesting happening here
            distr = benchmark[distribution_idx]
            positive_samples, negative_samples = distr['positive_samples'], distr['negative_samples']
            pos2score = {s: 1 for s in positive_samples}
            neg2score = {s: 1 for s in negative_samples}

            pos2score = subsample(pos2score)
            neg2score = subsample(neg2score)

            # engine takes in a model and a tokenizer as the argument
            # it can propose hypothesis, compare two examples given a hypothesis, and classify whether a target example satisfies a hypothesis given an in-context example
            # you can see them in models/templates
            engine = Engine(model_tokenizer)

            # a class that learns a linear combination of natural language hypotheses
            # interesting things happen mostly in the .run function
            dp_instance = DistributionPairInstance(pos2score, neg2score, engine, max_round=2)
            result = dp_instance.run()
            id2result[distribution_idx] = result
            pkl.dump(id2result, open('debug/1105run.pkl', 'wb'))

    if test_case == 'toy':
        # a toy test case
        tok_path = '/mount/models/t5-small-cp_tokenizer/'
        if os.path.exists(tok_path):
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained('t5-small')

        model_path = '/mount/models/t5-small-cp_model/'
        if os.path.exists(model_path):
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        model_tokenizer = (model, tokenizer)
        engine = Engine(model_tokenizer)

        pos2scores = {
            'pos neg %.2f' % f: f for f in np.arange(0.5, 1.0, 0.002)
        }
        neg2scores = {
            'neg pos %.2f' % f: f for f in np.arange(0.5, 1.0, 0.002)
        }

        dp_instance = DistributionPairInstance(pos2scores, neg2scores, engine)
        result = dp_instance.run()
        pkl.dump(result, open('result.pkl', 'wb'))