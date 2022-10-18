import re
import json
import string
import math
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from collections import OrderedDict

from random import choice


def total_words(A, B):
    """
    Gets total frequency of non-stop words of all sentences after stemming in 
    groups A and B. Ex. if "bat" appears 10 times out of 100 words then 
    the entry for "bat" would be 0.1
    """
    stops = set(stopwords.words("english"))
    ps = PorterStemmer()
    word_map = dict()

    def loop(sentences):
        for key in sentences:
            key = key.translate(str.maketrans('', '', string.punctuation))
            words = word_tokenize(key)

            for w in words: 
                if not w in stops:
                    word_map[w] = word_map.get(ps.stem(w), 0) + 1
        
    loop(A)
    loop(B) 

    total = 0
    for key in word_map:
        total += word_map[key]

    out = OrderedDict() 

    for k, v in sorted(
            word_map.items(), key=lambda item: item[1], reverse=True):
        out[k] = v / total
    
    return out

def lexical_diversity(A, B, top_p = 0.2):
    """
    Uses weighted probabilty distributions to determine whether to pick a candidate or not.
    1. Grabs top p sentences in groups A and B 
    2. loops through until a and b are at least length 5. 
        for each iteration:
            a. grab a random pair of sentences from A and B 
            b. remove punctuation and tokenize 
            c. find the most frequent shared word bewteen the two sentences.
               If the sentnces don't share any words then add them both regardless. 
            d. Grab the frequency f from step c. and run a weighted probability random choice
               distribution, where we add the sentence with probability 1 / (1 + sqrt(f))
               The intution behind this is if f is a very frequent word, f would be higher 
               and would make 1 / (1 + sqrt(f)) smaller. We take the square root 
               of the frequency to prevent underflow as the frequencies can get very small. 
    """
    a_candidates = [] 
    b_candidates = [] 

    A = [key for i, key in enumerate(A.keys()) if i <= int(top_p * len(A.keys()))]
    B = [key for i, key in enumerate(B.keys()) if i <= int(top_p * len(B.keys()))]
    word_freq = total_words(A, B)
    
    while len(a_candidates) < 5 or len(b_candidates) < 5:
        if len(A) == 0 or len(B) == 0: 
            break
        a = choice(A)
        b = choice(B)

        a_no_punc = a.translate(str.maketrans('', '', string.punctuation))
        b_no_punc = b.translate(str.maketrans('', '', string.punctuation))

        a_words = word_tokenize(a_no_punc)
        b_words = word_tokenize(b_no_punc) 

        max_freq = -1
        for word in a_words:
            if word in b_words and word in word_freq:
                max_freq = max(max_freq, word_freq[word])
        
        if max_freq != -1:
            p = 1 / (1 + math.sqrt(max_freq))
            draw = np.random.choice([1, 0], 1, [p, 1 - p])

            if draw == 1:
                a_candidates.append(a)
                b_candidates.append(b) 
                A.remove(a)
                B.remove(b)
        else:
            a_candidates.append(a)
            b_candidates.append(b) 
            A.remove(a)
            B.remove(b)

    return a_candidates, b_candidates

if __name__ == "__main__":
    ps = PorterStemmer()
    out = []

    for i in range(10):
        data = json.load(open(f"shap_results/shap_result_{i}.json"))
        pos_sorted = OrderedDict()
        neg_sorted = OrderedDict()
        for k, v in sorted(
            data["pos2score"].items(), key=lambda item: item[1], reverse=True):
            pos_sorted[k] = v

        for k, v in sorted(
                data["neg2score"].items(), key=lambda item: item[1], reverse=True
            ):
            neg_sorted[k] = v

        pos, neg = lexical_diversity(pos_sorted, neg_sorted)

        output_json = {
            "A": [sentence for sentence in pos],
            "A_Highlight": [data["pos2highlight"][sentence] for sentence in pos],
            "B": [sentence for sentence in neg],
            "B_Highlight": [data["neg2highlight"][sentence] for sentence in neg],
        }

        out.append(output_json)

    with open("shap_result_groups.json", "w") as f:
        out = json.dumps(out, indent=4)
        f.write(out)
