import pickle as pkl
from scipy.stats import pearsonr
import random
import numpy as np

algo_run_result = pkl.load(open('debug/1105run.pkl', 'rb'))



for j in range(8):
    result = algo_run_result[j]

    logs = result['logs']
    last_hs = logs[-1]['current_h_weight']
    for current_h, score in last_hs:
        print('+' if current_h['+'] else '-', current_h['hypothesis'], score)
    print('=====================')

for t in range(8):
    result = algo_run_result[t]

    hypotheses = result['hypotheses']
    hypotheses = random.sample(hypotheses, 10)

    corr_matrix = np.zeros((len(hypotheses), len(hypotheses)))
    diff = 0
    for i in range(len(hypotheses)):
        for j in range(len(hypotheses[i])):
            h1, h2 = hypotheses[i], hypotheses[j]
            sent2score_1, sent2score_2 = h1['sent2score'], h2['sent2score']
            intersection = list(set(sent2score_1.keys()).intersection(set(sent2score_2.keys())))
            diff += len(sent2score_1) + len(sent2score_2) - 2 * len(intersection)
            sent_score_1_list = [sent2score_1[sent] for sent in intersection]
            sent_score_2_list = [sent2score_2[sent] for sent in intersection]
            corr, p = pearsonr(sent_score_1_list, sent_score_2_list)

            corr_matrix[i, j] = corr
    all_corrs = corr_matrix[np.triu_indices(len(hypotheses), k=1)]
    print(diff)
    for percentage in [85, 90, 95, 99]:
        print('percentage', percentage, 'corr', np.percentile(all_corrs, percentage))

