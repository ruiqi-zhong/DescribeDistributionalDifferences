from tree_d3 import D3TreeSystem
import json

if __name__ == '__main__':
    dists = json.load(open('../benchmark_sec_4/benchmark_ss.json'))
    
    d3sys = D3TreeSystem()
    for d in dists[22:25]:
        pos = d['positive_samples']
        neg = d['negative_samples']
        pair = d['pair']
        print(pair)
        d3sys.fit(pos,
            neg,
            pair,
            depth=1)