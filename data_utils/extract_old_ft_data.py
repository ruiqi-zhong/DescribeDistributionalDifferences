import json
from transformers import AutoTokenizer
import numpy as np

t5tok = AutoTokenizer.from_pretrained("t5-small")

full_template = open('new_templates/ai2proposer_full.txt', 'r').read()


def tok_subspan(s, subspan_token_max_len=80):
    toks = t5tok.tokenize(s)
    total_length = len(toks)
    if total_length <= subspan_token_max_len:
        return s
    random_subspan = np.random.randint(0, total_length - subspan_token_max_len)
    subspan_toks = toks[random_subspan:random_subspan + subspan_token_max_len]
    return t5tok.convert_tokens_to_string(subspan_toks)

def create_prompt_completion_from_dict(d, num_data=5, num_incontext_samples=4):
    target = d['demonstrations'][0]
    pos_sentences = d['pos']
    neg_sentences = d['neg']

    data = []
    for i in range(num_data):
        subsampled_sentences = np.random.choice(pos_sentences, min(num_incontext_samples, len(pos_sentences)), replace=False)
        subsampled_sentences = ['Group A: ' + tok_subspan(s) + '\n' for s in subsampled_sentences]
        A_block = ''.join(subsampled_sentences)

        subsampled_sentences = np.random.choice(neg_sentences, min(num_incontext_samples, len(neg_sentences)), replace=False)
        subsampled_sentences = ['Group B: ' + tok_subspan(s) + '\n' for s in subsampled_sentences]
        B_block = ''.join(subsampled_sentences)

        prompt = full_template.format(A_block=A_block, B_block=B_block)
        completion = target

        d = {'prompt': prompt, 'completion': completion}
        data.append(d)
    
    return data


if __name__ == '__main__':
    test_dicts = json.load(open('old_data/test_complete_data.json', 'r'))
    all_test = []
    for d in test_dicts:
        all_test.extend(create_prompt_completion_from_dict(d))

    train_dicts = json.load(open('old_data/train_complete_data.json', 'r'))
    all_train = []
    for d in train_dicts:
        all_train.extend(create_prompt_completion_from_dict(d, num_data=2))

    all_data = {
        'train': all_train,
        'eval': all_test
    }
    json.dump(all_data, open('data/icml22_ai2_full_proposal_prompt_data.json', 'w'))








