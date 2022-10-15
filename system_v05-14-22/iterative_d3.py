from get_extreme_w_highlight import return_extreme_values
from proposer_wrapper import init_proposer
from verifier_wrapper import init_verifier, evaluate_single
from sklearn.linear_model import LinearRegression
import json
import pandas as pd
from typing import List

def iterative_d3(pos: List[str], # a list of text samples from D_1
                neg: List[str], # a list of text samples from D_0
                note: str='', # a note about this distribution, for logging purposes
                proposer_name: str='t5ruiqi-zhong/t5proposer_0514', # the name of the proposer. the name starts with either t5 or gpt3, followed by the directory/model-name/engine name. change argument to "t5t5-small" to debug
                verifier_name: str='ruiqi-zhong/t5verifier_0514', # the name of the verifier, with options detailed in verifier_wrapper.py. change argument to "dummy" to debug
                depth=3):

    """Returns a set of hypotheses that predict pos/neg representativeness for given depth"""
    
    extreme_vals = return_extreme_values(pos, neg)
    pos2score, neg2score = extreme_vals['pos2score'], extreme_vals['neg2score']

    proposer = init_proposer(proposer_name)
    verifier = init_verifier(verifier_name)

    predict_scores( pos,
                    neg,
                    pos2score,
                    neg2score,
                    depth,
                    proposer,
                    verifier)


def predict_scores(pos,
                   neg,
                   pos2score,
                   neg2score,
                   depth,
                   proposer,
                   verifier):
    """Finds the most predictive marginal score"""
    
    if not depth:
        return

    # propose a set of hypotheses
    proposed_hypotheses = proposer.propose_hypothesis(pos2score, neg2score)
    
    d_text2score = {}
    d_text2score.update(pos2score)
    d_text2score.update(neg2score)

    # identify best hypotheses
    h2result = verifier.return_verification_active(proposed_hypotheses, pos, neg)
    top_h = max(h2result, key=lambda h: h2result[h]['h_score'])

    print(top_h)

    all_samples = pos+neg

    # evaluate hypothesis on every sample
    evaluate_results = evaluate_single(top_h,all_samples,verifier)
    h_text2score = evaluate_results['h_text2score']
    
    # update pos2score nad neg2score
    df = pd.DataFrame({'text':all_samples})
    df['d_score'] = df['text'].apply(lambda t: d_text2score[t])
    df['h_score'] = df['text'].apply(lambda t: h_text2score[t])

    linreg = LinearRegression()
    linreg.fit(df[['h_score']], df['d_score'])
    df['prediction'] = linreg.predict(df[['h_score']])
    df['residual'] = df['d_score'] -  df['prediction']
    
    text2res = {}
    for text, res in zip(all_samples, df['residual'].tolist()):
        text2res[text] = res
    
    for pos_text in pos:
        pos2res = text2res[pos_text]

    for neg_text in neg:
        neg2res = text2res[neg_text]

    predict_scores(pos, neg, pos2res, neg2res, depth-1, proposer, verifier)

if __name__ == '__main__':
    import tqdm
    distribution_pairs = json.load(open('../benchmark_sec_4/benchmark.json'))[10:]

    all_h2score = []
    for i, d in enumerate(tqdm.tqdm(distribution_pairs)):
        print(distribution_pairs['pair'])
        h2score = iterative_d3(pos=d['positive_samples'], 
                           neg=d['negative_samples'], 
                           note='benchmark %d; can be anything, for logging purpose only' % i)
                           