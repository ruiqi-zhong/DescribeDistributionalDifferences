import numpy as np
import json
import tqdm
from preprocess import tok_subspan


def clamp_in_range(ls, lowerbound=-2, upperbound=2):
    return [x if lowerbound <= x <= upperbound else (lowerbound if x < lowerbound else upperbound) for x in ls]

def update_ground_truth():
    updated_ground_truth = {
        45:  ['concerns visa application', 'is about getting a visa'],
        42:  ['denies climate change'],
        43: ['shows an environmental concern'],
        2: ['is a moview review', 'is a review'],
        1: ['is a plot summary', 'is a plot summary of a film']
    }

    data = json.load(open('old_data/test_complete_data.json'))

    for i, golds in updated_ground_truth.items():
        data[i]['demonstrations'] = golds
    for i in range(len(data)):
        data[i]['id'] = i
    json.dump(data, open('old_data/test_complete_data.json', 'w'))


proposer_template = open('models/templates/ai2proposer_full.txt').read()

def construct_proposer_prompt(pos_sentences, neg_sentences, num_incontext_samples=4):
    subsampled_sentences = np.random.choice(pos_sentences, min(num_incontext_samples, len(pos_sentences)), replace=False)
    subsampled_sentences = ['Group A: ' + tok_subspan(s) + '\n' for s in subsampled_sentences]
    A_block = ''.join(subsampled_sentences)

    subsampled_sentences = np.random.choice(neg_sentences, min(num_incontext_samples, len(neg_sentences)), replace=False)
    subsampled_sentences = ['Group B: ' + tok_subspan(s) + '\n' for s in subsampled_sentences]
    B_block = ''.join(subsampled_sentences)

    prompt = proposer_template.format(A_block=A_block, B_block=B_block)
    return prompt


def create_proposer_prompt_completion_from_dict(d, num_data=10, num_incontext_samples=4):
    target = d['demonstrations'][0]
    pos_sentences = d['pos']
    neg_sentences = d['neg']

    data = []
    for i in range(num_data):
        prompt = construct_proposer_prompt(pos_sentences, neg_sentences, num_incontext_samples=num_incontext_samples)
        completion = target

        d = {'prompt': prompt, 'completion': completion}
        data.append(d)
    
    return data


def create_proposer_data():
    test_dicts = json.load(open('old_data/test_complete_data.json', 'r'))
    all_test = []
    for d in tqdm.tqdm(test_dicts):
        for x in create_proposer_prompt_completion_from_dict(d):
            x['id'] = d['id']
            all_test.append(x)

    train_dicts = json.load(open('old_data/train_complete_data.json', 'r'))
    all_train = []
    for d in tqdm.tqdm(train_dicts):
        all_train.extend(create_proposer_prompt_completion_from_dict(d, num_data=2))

    all_data = {
        'train': all_train,
        'eval': all_test
    }
    return all_data

paired_verifier_template = open('models/templates/ai2paired_verifier_full.txt').read()
def construct_paired_verifier_prompt(sent_A, sent_B, hypothesis):
    prompt = paired_verifier_template.format(sent_A=sent_A, sent_B=sent_B, hypothesis=hypothesis)
    return prompt

def create_paired_verifier_prompt_completion_from_dict(d, num_data=10):
    hypothesis = d['demonstrations'][0]
    pos_sentences = d['pos']
    neg_sentences = d['neg']

    if len(pos_sentences) == 0 or len(neg_sentences) == 0:
        return []

    data = []
    existing_data = set()
    for i in range(num_data):
        answer_yes = np.random.choice([True, False])
        sent_A = np.random.choice(pos_sentences if answer_yes else neg_sentences)
        sent_B = np.random.choice(neg_sentences if answer_yes else pos_sentences)
        prompt = construct_paired_verifier_prompt(sent_A, sent_B, hypothesis)
        completion = 'yes' if answer_yes else 'no'
        d = {'prompt': prompt, 'completion': completion}
        if prompt not in existing_data:
            data.append(d)
            existing_data.add(prompt)
    return data


    

def create_paired_verifier_data():
    train_dicts = json.load(open('old_data/train_complete_data.json', 'r'))
    all_train = []

    for d in tqdm.tqdm(train_dicts):
        all_train.extend(create_paired_verifier_prompt_completion_from_dict(d, num_data=5))
    test_dicts = json.load(open('old_data/test_complete_data.json', 'r'))
    all_test = []
    for d in tqdm.tqdm(test_dicts):
        for x in create_paired_verifier_prompt_completion_from_dict(d, num_data=10):
            x['id'] = d['id']
            all_test.append(x)
    
    all_data = {
        'train': all_train,
        'eval': all_test
    }
    return all_data


verifier_w_examples_template = open('models/templates/ai2verifier_w_examples.txt').read()
def construct_verifier_w_examples_prompt(positive_sample, negative_sample, hypothesis, target_sample):
    prompt = verifier_w_examples_template.format(positive_sample=positive_sample, negative_sample=negative_sample, hypothesis=hypothesis, target_sample=target_sample)
    return prompt

def create_verifier_w_examples_prompt_completion_from_dict(d, num_data=10):
    hypothesis = d['demonstrations'][0]
    pos_sentences = d['pos']
    neg_sentences = d['neg']

    if len(pos_sentences) <=1 or len(neg_sentences) <= 1:
        return []

    data = []
    existing_data = set()
    for i in range(num_data):
        answer_yes = np.random.choice([True, False])
        positive_sample = np.random.choice(pos_sentences)
        negative_sample = np.random.choice(neg_sentences)

        target_sample = np.random.choice(pos_sentences if answer_yes else neg_sentences)
        if target_sample == positive_sample or target_sample == negative_sample:
            continue

        prompt = construct_verifier_w_examples_prompt(positive_sample, negative_sample, hypothesis, target_sample)
        completion = 'yes' if answer_yes else 'no'
        d = {'prompt': prompt, 'completion': completion}
        if prompt not in existing_data:
            data.append(d)
            existing_data.add(prompt)

    return data


def create_verifier_w_examples_data():
    train_dicts = json.load(open('old_data/train_complete_data.json', 'r'))
    all_train = []
    for d in tqdm.tqdm(train_dicts):
        all_train.extend(create_verifier_w_examples_prompt_completion_from_dict(d, num_data=5))
    test_dicts = json.load(open('old_data/test_complete_data.json', 'r'))
    all_test = []
    for d in tqdm.tqdm(test_dicts):
        for x in create_verifier_w_examples_prompt_completion_from_dict(d, num_data=10):
            x['id'] = d['id']
            all_test.append(x)
    
    all_data = {
        'train': all_train,
        'eval': all_test
    }
    return all_data

def create_perfect_dummy(data_path):
    all_data = json.load(open(data_path, 'r'))

    dummy_perfect = []
    for d in all_data['eval']:
        x = {}
        x['prompt'] = d['prompt']
        x['demonstration'] = d['completion']
        x['orig_d'] = d
        x['generations'] = [{'lm_postprocess': d['completion']}]
        dummy_perfect.append(x)

    return dummy_perfect



if __name__ == '__main__':
    # update_ground_truth()

    name2data = {
        'proposer': create_proposer_data(),
        'paired_verifier': create_paired_verifier_data(),
        'verifier_w_examples': create_verifier_w_examples_data()
    }

    all_data = {}
    for key in ['train', 'eval']:
        key_data = []
        for name, data in name2data.items():
            for d in data[key]:
                d['name'] = name
                key_data.append(d)
        all_data[key] = key_data
    # json.dump(all_data, open('models/data/ai2_1102data.json', 'w'))

    # data_path = 'models/data/ai2_1102data.json'
    # data = create_perfect_dummy(data_path)
    # json.dump(data, open('data/ai2_1102data_dummy_perfect.json', 'w'))
