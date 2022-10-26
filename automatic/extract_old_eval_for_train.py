import json
from collections import Counter

manual_evaluation = json.load(open('../benchmark_sec_4/manual_evaluation.json', 'r'))
old_ground_truth = [x['human_annotations'] for x in json.load(open('../benchmark_sec_4/benchmark.json', 'r'))]
num_distr_pairs = len(old_ground_truth)

data = []

for system_name, content in manual_evaluation.items():
    for i in range(num_distr_pairs):
        preds = content[i]
        for pred, grade in preds:
            d = {'all_gold': old_ground_truth[i], 'system_name': system_name, 'pred': pred, 'grade': grade}
            data.append(d)

print(data[0])