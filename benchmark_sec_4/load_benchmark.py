import json

benchmark = json.load(open('benchmark.json', 'r'))
print('number of tasks', len(benchmark))
# the task you want to examine
task_idx = 33
print('human annotations', benchmark[task_idx]['human_annotations'])
print('positive samples')
for sample in benchmark[task_idx]['positive_samples'][:5]:
    print('>', sample)
print('negative samples')
for sample in benchmark[task_idx]['negative_samples'][:5]:
    print('>', sample)