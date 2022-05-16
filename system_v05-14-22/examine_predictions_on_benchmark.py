import json


h2score_list = json.load(open('benchmark_h2score.json', 'r'))
ground_truth = [x['human_annotations'] for x in json.load(open('../benchmark_sec_4/benchmark.json'))]

TOP_K = 20
assert len(h2score_list) == len(ground_truth)
print('Printing %d descriptions for in total %d distribution pairs' % (TOP_K, len(h2score_list)))

for i in range(len(h2score_list)):
    print('============ Distribution pair %d ============' % i)
    human_anns = ground_truth[i]
    print('Human annotation: ')
    for h in human_anns:
        print('>', h)
    
    h2score = h2score_list[i]
    sorted_hs = sorted(h2score, key=lambda h: h2score[h], reverse=True)
    print('Top %d automatically generated descriptions: ' % TOP_K)
    for h in sorted_hs[:TOP_K]:
        print('>', h)
    print()
