from iterative_d3 import iterative_d3
import json

if __name__ == '__main__':
    import tqdm
    distribution_pairs = json.load(open('../benchmark_sec_4/benchmark.json'))[17:]

    all_h2score = []
    for i, d in enumerate(tqdm.tqdm(distribution_pairs)):
        h2score = iterative_d3(pos=d['positive_samples'], 
                           neg=d['negative_samples'], 
                           note='benchmark %d; can be anything, for logging purpose only' % i)
                           