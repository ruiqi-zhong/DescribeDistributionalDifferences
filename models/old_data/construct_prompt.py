import pickle as pkl
import os
from collections import defaultdict
import random
import json
from transformers import AutoTokenizer

t5tok = AutoTokenizer.from_pretrained('t5-small')

# number of in-context examples for each group
K = 4
max_length = 80 # at most 80 T5 tokens per sample


# trim individual samples so that the samples can fit in the prompt
def trim(sent, max_length):
    toks_A = t5tok(sent)['input_ids'][:-1]
    if max_length == float('inf'):
        s = t5tok.decode(toks_A)
    else:
        s = t5tok.decode(toks_A[:max_length])
    if len(toks_A) > max_length:
        s += ' ...'
    return s


def sample_sentences(xs, k, group_id, t5_maxlen=float('inf')):
    random.shuffle(xs)
    return '\n'.join(['Group %s: %s' % (group_id, trim(x, t5_maxlen)) for x in xs[:k]])


# given a list of positive samples (distribution 0/group A), a list of negative samples (distribution 1/group B), and the number of in-context samples for the proposer
# choose a random set of samples for each distribution and return a prompt for the proposer
def create_prompt_from_pos_neg_sentences(positive_examples, negative_examples, k=K):
    group_A_text = sample_sentences(positive_examples, k=k, group_id='A')
    group_B_text = sample_sentences(negative_examples, k=k, group_id='B')

    prompt = group_A_text + '\n\n' + group_B_text + '\n\n'
    prompt += 'Compared to sentences from Group B, each sentence from Group A'
    return prompt


# this load the 
train_complete_data = json.load(open('train_complete_data.json', 'r'))
d = train_complete_data[0]
pos, neg = d['pos'], d['neg']
prompt = create_prompt_from_pos_neg_sentences(pos, neg)
print(prompt)
print(d['demonstrations'])

# here is a SHORT example you can use as in-context demonstration according to the AI2 format
pos = [
    'I love this movie.',
    'Really exciting.',
    'Definitely watch it.',
    'Highly recommend.'
]
neg = [
    'I hate this movie.',
    'Boring.',
    'Waste of time.',
    'Not worth the ticket.'
]
prompt = create_prompt_from_pos_neg_sentences(pos, neg)
print(prompt)
completion = 'is a positive movie review.'
print(completion)