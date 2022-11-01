from iterative_d3 import iterative_d3
import json
import tqdm
import argparse

if __name__ == '__main__':
    distribution_pairs = json.load(open('../benchmark_sec_4/benchmark.json'))[17:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",action='store_true')
    args = parser.parse_args()

    proposer_name = 't5t5-small' if args.debug else 't5ruiqi-zhong/t5t5-small' 
    verifier_name = 'dummy' if args.debug else 'ruiqi-zhong/t5verifier_0514' 

    all_h2score = []
    for i, d in enumerate(tqdm.tqdm(distribution_pairs)):
        h2score = iterative_d3(pos=d['positive_samples'],
                            proposer_name=proposer_name,
                            verifier_name=verifier_name, 
                           neg=d['negative_samples'], 
                           note='benchmark %d; can be anything, for logging purpose only' % i)
                           