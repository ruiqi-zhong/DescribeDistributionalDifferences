from get_extreme_w_highlight import return_extreme_values
from proposer_wrapper import init_proposer, Proposer
from verifier_wrapper import init_verifier, evaluate_single
from sklearn.linear_model import Lasso
import pandas as pd
from typing import List, Callable

SNIP_COL = 'snippet'
REP_COL = 'rep'
DIST_COL = 'dist'
HYP_COL = 'h_score_'
RES_COL = 'residual'

def lasso( data: pd.DataFrame,
                    alpha: float=1) -> pd.DataFrame:
    """Performs LASSO regression of representativeness on hypotheses"""
    X = data[[col for col in data.columns if HYP_COL in col]] # all features
    y = data[REP_COL] # rep score
    clf = Lasso(alpha=alpha)
    clf.fit(X, y)
    drop_hyp = [hyp for (hyp, coef) in zip(X.columns, clf.coef_) if not coef]
    data = data.drop(drop_hyp, axis=1) # drop features
    data[RES_COL] = y - clf.predict(X) # update residuals
    return data

def iterative_d3(   pos: List[str], # a list of text samples from D_1
                    neg: List[str], # a list of text samples from D_0
                    proposer_name: str='t5ruiqi-zhong/t5proposer_0514', # the name of the proposer. the name starts with either t5 or gpt3, followed by the directory/model-name/engine name. change argument to "ip-small" to debug
                    verifier_name: str='ruiqi-zhong/t5verifier_0514', # the name of the verifier, with options detailed in verifier_wrapper.py. change argument to "dummy" to debug
                    depth=3,
                    selector:Callable=lasso):
    """Returns a set of hypotheses that predict pos/neg representativeness for given depth"""
    
    # get representative samples
    extreme_vals = return_extreme_values(pos, neg)
    pos2score, neg2score = extreme_vals['pos2score'], extreme_vals['neg2score']

    proposer = init_proposer(proposer_name)
    verifier = init_verifier(verifier_name)

    # get text2score
    d_text2score = {}
    d_text2score.update(pos2score)
    d_text2score.update(neg2score)

    snippets = pos+neg
    dists = [1]*len(pos) + [0]*len(neg) # ground truth

    data = pd.DataFrame({SNIP_COL:snippets, DIST_COL:dists})
    data[REP_COL] = data[SNIP_COL].apply(d_text2score.get) # get rep scores
    data[RES_COL] = data[REP_COL] # residual before predictors

    # recursively predict scores
    predict_scores( data=data,
                    depth=depth,
                    proposer=proposer,
                    verifier=verifier,
                    selector=selector)

def predict_scores(data: pd.DataFrame,
                   depth: int,
                   proposer: Proposer,
                   verifier,
                   selector:Callable):
    """Finds the most predictive marginal score."""
    
    if not depth: # done recursing
        return

    pos_data, neg_data = data[data[DIST_COL] == 1],  data[data[DIST_COL] == 0]
    pos, neg = pos_data[SNIP_COL], neg_data[SNIP_COL]

    # key idea: we predict residual
    pos2score = dict(zip(pos_data[SNIP_COL], pos_data[RES_COL]))
    neg2score = dict(zip(neg_data[SNIP_COL], neg_data[RES_COL]))

    # propose a set of hypotheses
    proposed_hypotheses = proposer.propose_hypothesis(pos2score, neg2score)
    
    # get best hypothesis
    h2result = verifier.return_verification_active(proposed_hypotheses, pos, neg)
    top_h = max(h2result, key=lambda h: h2result[h]['h_score'])
    print(f'h_{depth}: {top_h}')

    snippets = data[SNIP_COL].tolist()

    # evaluate hypothesis on every sample
    evaluate_results = evaluate_single(top_h,snippets,verifier)
    h_text2score = evaluate_results['h_text2score']

    # update pos2score nad neg2score
    data[HYP_COL + depth] = data[SNIP_COL].apply(lambda t: h_text2score[t])

    predict_scores( data=data,
                    depth=depth-1,
                    proposer=proposer,
                    verifier=verifier,
                    selector=selector)