from tree_d3 import D3TreeSystem
import json

if __name__ == '__main__':
    d = json.load(open('../benchmark_sec_4/benchmark_ss.json'))[46]
    pos = d['positive_samples']
    neg = d['negative_samples']
    pair = d['pair']
    print(pair)
    d3sys = D3TreeSystem()
    d3sys.fit(pos,
          neg,
          pair,
          depth=1)