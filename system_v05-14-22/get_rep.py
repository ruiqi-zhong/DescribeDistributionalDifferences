import json
import os
import pickle as pkl
from typing import List

import tqdm

from get_extreme_w_highlight import return_extreme_values

def get_rep(pos: List[str], # a list of text samples from D_1
             neg: List[str], # a list of text samples from D_0
             pair: str='',
             save_folder=None):
    
    if save_folder is None:
        save_folder = os.path.join('end2end_jobs', pair)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    else:
        print('Folder %s exists' % save_folder)

    # get samples that are representative of the differences between two distributions
    extreme_vals = return_extreme_values(pos, neg)
    pkl.dump(extreme_vals, open(os.path.join(save_folder, 'get_extreme_result.pkl'), 'wb')) 

if __name__ == '__main__':

    distribution_pairs = json.load(open('../benchmark_sec_4/benchmark_0709.json'))

    all_h2score = []
    for i, d in enumerate(tqdm.tqdm(distribution_pairs[85:])):
        
        print("pair: ", d['pair'], i)
        try: 
            get_rep(pos=d['positive_samples'], neg=d['negative_samples'], pair="asdf" + str(i))
        except: 
            continue