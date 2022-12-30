from copy import deepcopy
import sys
sys.path.append('./')
from experiments.t511b_verifier import Verifier

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
from gadgets.util import gpt3wrapper, convert_cmp_hs, classify_cmp, convert_cmp_to_ind
from models.preprocess import construct_blocks, prefix_subspan
from transformers import AutoTokenizer
from argparse import ArgumentParser
import argparse


DEBUG = False
tok = AutoTokenizer.from_pretrained('gpt2-medium')
VERIFY_HYP_BLOCK_SIZE = 32
eps = 1e-5


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


class DistributionPairInstance1228:

    def __init__(
        self,
        application,
        verifier,
        proposer,
        top_fraction=None,
        num_hyps=10,
        max_round=2,
    ):
        pos2score, neg2score = application['pos2score'], application['neg2score']
        self.orig_pos2score, self.orig_neg2score = pos2score, neg2score
        self.orig_sent2membership = {}
        for sent in pos2score:
            self.orig_sent2membership[sent] = 1.
        for sent in neg2score:
            self.orig_sent2membership[sent] = 0.

        self.proposer, self.verifier = proposer, verifier

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
        self.current_pos2representative = deepcopy(self.orig_pos2score)
        self.current_neg2representative = deepcopy(self.orig_neg2score)
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
            h_count_w_p = 0
            sorted_pos = sorted(self.current_pos2representative, key=self.current_pos2representative.get, reverse=True)
            sorted_neg = sorted(self.current_neg2representative, key=self.current_neg2representative.get, reverse=True)
            for idx in range(3):
                pos, neg = lexical_diversity(sorted_pos, sorted_neg, top_p=p, num_sentences=25)
                hyps, provenance = self.proposer.propose_hypotheses(pos, neg)
                provenance['top_p'] = p
                provenance['round'] = self.current_round
                provenance['idx'] = idx
                h_count_w_p += len(hyps)
                for hyp in hyps:
                    h_dict = {
                        'hypothesis': hyp,
                        'sent2score': {}, 
                        'fullly_computed': False, 
                        '+': True,
                        'provenance': provenance,
                        'corr_w_membership': None
                    }
                    proposed_hypotheses.append(h_dict)
                    h_count_w_p += 1
                    if h_count_w_p >= self.num_hyps:
                        break
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
        hypothesis['corr_w_membership'] = orig_corr

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

                # construct the verifier dicts
                verifier_dicts = []
                for sent in sents:
                    for hypothesis in competitive_hypotheses:
                        verifier_dict = { 'hypothesis': hypothesis['hypothesis'], 'text': sent, 'type': 'ind', 'pointer': hypothesis}
                        verifier_dicts.append(verifier_dict)
                
                # run the verifier 
                all_scores = list(self.verifier.verify_ind_dicts_w_scores(verifier_dicts))
                assert len(all_scores) == len(verifier_dicts)
                for d, s in zip(verifier_dicts, all_scores):
                    d['pointer']['sent2score'][d['text']] = s + eps * random.random()
                
                # filter out weaker hypotheses based on UCB
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
        selected_feature_dims = search_clf_w_sparse_positive_weights(X, Y, max(self.current_round + 1, 4))
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

def subsample(sent2score: Dict[str, float], subsample_size=1000) -> Dict[str, float]:
    if len(sent2score) <= subsample_size:
        return sent2score
    all_sents = list(sent2score.keys())
    random.shuffle(all_sents)
    return {sent: sent2score[sent] for sent in all_sents[:subsample_size]}



DEFAULT_HYPOTHESES = [
    "talks about politics, such as presidential election.",
    "contains insulting language for immigrants.",
    "uses double negation, i.e., using two negations in a sentence."
]
SINGLE_SAMPLE_MAX_LENGTH = 256
MAX_PROMPT_LENGTH = 3200

class Proposer1228:

    def __init__(self, application, use_default_hypotheses=False, single_max_length=SINGLE_SAMPLE_MAX_LENGTH, engine_name='text-davinci-003', temperature=0.7):
        if use_default_hypotheses:
            self.example_hypotheses = DEFAULT_HYPOTHESES
        else:
            self.example_hypotheses = (application['example_hypotheses'] + DEFAULT_HYPOTHESES)[:3]
        
        self.application = application
        self.prompt_template = open('models/templates/1228_w_hypotheses.txt', 'r').read()
        self.single_max_length = single_max_length
        self.engine_name = engine_name
        self.temperature = temperature

    
    def propose_hypotheses(self, pos_sents, neg_sents):
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
        random.shuffle(self.example_hypotheses)
        for i, hypothesis in enumerate(self.example_hypotheses):
            arg_dict[f'example_hypothesis_{i+1}'] = hypothesis

        while num_incontext_samples > 1:
            pos_samples, neg_samples = pos_sents, neg_sents

            A_sentences = [prefix_subspan(x, self.single_max_length, tok) for x in pos_samples]
            B_sentences = [prefix_subspan(x, self.single_max_length, tok) for x in neg_samples]

            sent_subset = construct_blocks(A_sentences, B_sentences, num_incontext_samples=num_incontext_samples, truncate=False)
            
            A_block, B_block = sent_subset['A_block'], sent_subset['B_block']
            tmp_arg_dict = deepcopy(arg_dict)
            tmp_arg_dict['A_block'] = A_block
            tmp_arg_dict['B_block'] = B_block
            prompt = self.prompt_template.format(**tmp_arg_dict)
            prompt_length = len(tok.encode(prompt))
            if prompt_length < MAX_PROMPT_LENGTH:
                break
            else:
                num_incontext_samples -= 1
                print('prompt too long, reducing num_incontext_samples to %d' % num_incontext_samples)

        arg_dict['A_block'] = sent_subset['A_block']
        arg_dict['B_block'] = sent_subset['B_block']
        prompt = self.prompt_template.format(**arg_dict)

        query_args = {
            'engine': self.engine_name,
            'prompt': prompt,
            'temperature': self.temperature,
            'max_tokens': 512,
            'top_p': 1,
            'n': 1
        }

        result = gpt3wrapper(
            tag='proposer',
            **query_args
        )

        returned_text = result['choices'][0]['text']

        hs = []
        for h in returned_text.split('\n\n')[0].split('\n-'):
            h = convert_cmp_to_ind(h.replace('"', '').strip())
            if h is not None:
                hs.append(h)

        return hs, query_args


class DummyVerifier:

    def __init__(self):
        pass
    
    def verify_ind_dicts_w_scores(self, ind_dicts):
        for _ in range(len(ind_dicts)):
            yield random.random() < 0.5


def flip_application(application):
    application = deepcopy(application)
    application['pos_desc'], application['neg_desc'] = application['neg_desc'], application['pos_desc']
    application['pos_samples'], application['neg_samples'] = application['neg_samples'], application['pos_samples']
    application['split']['train']['pos_samples'], application['split']['train']['neg_samples'] = application['split']['train']['neg_samples'], application['split']['train']['pos_samples']
    application['split']['test']['pos_samples'], application['split']['test']['neg_samples'] = application['split']['test']['neg_samples'], application['split']['test']['pos_samples']
    if 'pos2score' in application:
        application['pos2score'], application['neg2score'] = application['neg2score'], application['pos2score']
    return application

if __name__ == '__main__':
    #applications = pkl.load(open('data/benchmark_applications_2nddraft.pkl', 'rb'))

    ### this snippets test the proposer on the benchmark applications
    test_proposer = False
    if test_proposer:
        applications = pkl.load(open('data/benchmark_applications_2nddraft.pkl', 'rb'))
        all_logs = []
        for application in applications:
            proposer = Proposer1228(application, use_default_hypotheses=False)
            application_id = application['v2_id']

            pos_samples, neg_samples = application['pos_samples'], application['neg_samples']
            random.shuffle(pos_samples)
            random.shuffle(neg_samples)
            pos_samples, neg_samples = pos_samples[:25], neg_samples[:25]

            result, provenance = proposer.propose_hypotheses(pos_samples, neg_samples)
            result['application_id'] = application_id
            all_logs.append(result)

            if application['flip']:
                result, provenance = proposer.propose_hypotheses(pos_samples, neg_samples)
                result['application_id'] = application_id
                all_logs.append(result)
            pkl.dump(all_logs, open('scratch/explore_proposer1228.pkl', 'wb'))
    
    test_algo = False

    if test_algo:
        applications = json.load(open('scratch/fake_icml_applications.json', 'r'))
        all_results = []
        for application in applications:
            proposer = Proposer1228(application, use_default_hypotheses=False)
            verifier = DummyVerifier()
            dpi = DistributionPairInstance1228(
                application,
                proposer=proposer, 
                verifier=verifier
            )
            results = dpi.run()
            all_results.append(results)
            pkl.dump(all_results, open('scratch/fake_icml_fake_verifier_results.pkl', 'wb'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='all')
    args = parser.parse_args()

    split = args.split

    applications = pkl.load(open('data/benchmark_applications_2nddraft.pkl', 'rb'))

    if split != 'all':
        applications = [x for x in applications if int(x['v1_id']) % 3 == int(split) and not x['purely_exploratory']]
        model_path = 'models/ckpts/verifier_mod_3_%d/' % int(split)
    else:
        applications = [x for x in applications if x['purely_exploratory']]
        model_path = 'models/ckpts/best_verifier/'
    
    for application in applications:
        pair_id = application['pair_id']
        extreme_results = pkl.load(open('experiments/pair_extreme_backup/results/v1-pair_id%d/result.pkl' % int(pair_id), 'rb'))
        application['pos2score'], application['neg2score'] = subsample(extreme_results['pos2score'], 1000), subsample(extreme_results['neg2score'], 1000)

    verifier = Verifier(model_path=model_path)

    application_l = []
    for application in applications:
        application['orientation'] = 'pos'
        application_l.append(application)
        if application['flip']:
            new_application = flip_application(application)
            new_application['orientation'] = 'neg'
            application_l.append(new_application)

    for application in application_l:
        print('===========================================================================================')
        proposer = Proposer1228(application)
        application_id = application['v2_id']
        save_path = 'application_results/proposer_results_%s_%s_%s.pkl' % (split, application_id, application['orientation'])
        pos2score, neg2score = extreme_results['pos2score'], extreme_results['neg2score']
        # pos2score, neg2score = subsample(pos2score, 1000), subsample(neg2score, 1000)

        dpi = DistributionPairInstance1228(application=application, proposer=proposer, verifier=verifier, max_round=1)
        result = dpi.run()
        result['application_id'] = application_id
        result['flip'] = application['flip']
        pkl.dump(result, open(save_path, 'wb'))

