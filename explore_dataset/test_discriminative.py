from nltk.tokenize import word_tokenize
import json
from sklearn.metrics import roc_auc_score
from collections import defaultdict
from itertools import chain
import numpy as np
import random


def extract_feature(s):
    words = word_tokenize(s)
    word_count = len(words)
    number_count = len([w for w in s.split() if w.isdigit()])
    capital_initial = False
    if len(s) > 0:
        capital_initial = s[0].isupper()
    features = {
        'word_count': word_count,
        'number_count': number_count,
        'capital_initial': capital_initial
    }
    return features

# takes in a pair of text groups
# uses a list of shallow features to discriminate them
# generally, we should avoid using text groups that can be easily discriminated by shallow features
def discriminated_w_simple_feature(positive_samples, negative_samples):
    
    # subsample 1000
    if len(positive_samples) > 1000:
        positive_samples = random.sample(positive_samples, 1000)
    if len(negative_samples) > 1000:
        negative_samples = random.sample(negative_samples, 1000)

    positive_word_occurence_count = defaultdict(int)
    for sample in positive_samples:
        for word in set(word_tokenize(sample)):
            positive_word_occurence_count[word] += 1
    positive_word2freq = {
        word: count / len(positive_samples)
        for word, count in positive_word_occurence_count.items()
    }
    positive_word2freq = defaultdict(lambda: 0, positive_word2freq)
    
    negative_word_occurence_count = defaultdict(int)
    for sample in negative_samples:
        for word in set(word_tokenize(sample)):
            negative_word_occurence_count[word] += 1
    
    negative_word2freq = {
        word: count / len(negative_samples)
        for word, count in negative_word_occurence_count.items()
    }
    negative_word2freq = defaultdict(lambda: 0, negative_word2freq)

    word2diff = {
        (word, np.abs(positive_word2freq[word] - negative_word2freq[word]))
        for word in chain(positive_word_occurence_count, negative_word_occurence_count)
    }
    
    discriminating_words = sorted(word2diff, key=lambda x: x[1], reverse=True)[:20]

    positive_features = [extract_feature(s) for s in positive_samples]
    negative_features = [extract_feature(s) for s in negative_samples]

    for w, f in discriminating_words:
        for i in range(len(positive_features)):
            positive_features[i][w] = w in word_tokenize(positive_samples[i])
    
    for w, f in discriminating_words:
        for i in range(len(negative_features)):
            negative_features[i][w] = w in word_tokenize(negative_samples[i])

    # test the discriminative power of the features
    positive_labels = [1] * len(positive_samples)
    negative_labels = [0] * len(negative_samples)
    all_features = positive_features + negative_features
    all_labels = positive_labels + negative_labels

    max_discriminnative_auc_roc = 0
    for feature_name in positive_features[0].keys():
        feature_values = [f[feature_name] for f in all_features]
        auc = roc_auc_score(all_labels, feature_values)
        auc = max(auc, 1 - auc)
        if auc > max_discriminnative_auc_roc:
            max_discriminnative_auc_roc = auc
    return max_discriminnative_auc_roc


if __name__ == '__main__':

    pairs = json.load(open('error_analyses_1126.json'))
    for pair in pairs:
        positive_samples = pair['positive_samples']
        negative_samples = pair['negative_samples']
        s = discriminated_w_simple_feature(positive_samples, negative_samples)
        print(s)
        print(pair['description'])
    exit(0)


    benchmark1028 = json.load(open('benchmark_1018.json'))
    all_types = set(pair['type'] for pair in benchmark1028)

    for pair_id, pair in enumerate(benchmark1028):
        if 'cluster' not in pair['type']:
            continue
        print('=============')
        positive_samples = pair['positive_samples']
        negative_samples = pair['negative_samples']
        s = discriminated_w_simple_feature(positive_samples, negative_samples)
        print(pair_id, pair['type'], s)