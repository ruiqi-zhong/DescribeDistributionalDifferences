import re
import json
import string
import math
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from collections import OrderedDict
from collections import defaultdict
from random import choice


def lexical_diversity(A, B, top_p = 0.2):
    """
    Gets two random sentences from A and B from the top p sentences from each 
        1. for each word in sentence A, check if it exists in sentence B 
            a. if the word exists in both sentences, check if the 
               counts between our current total set of words is > 2 --> if yes, we don't add
            b. if the word does not exist in both sentences, check if we 
               already have more than 2 occurences of the word in our total count
               for a sentences --> if yes, we don't add
    """
    a_candidates = [] 
    b_candidates = [] 
    stops = set(stopwords.words("english"))
    ps = PorterStemmer()

    A = [key for i, key in enumerate(A.keys()) if i <= int(top_p * len(A.keys()))]
    B = [key for i, key in enumerate(B.keys()) if i <= int(top_p * len(B.keys()))]
    a_words_count = defaultdict(int)
    b_words_count = defaultdict(int)
    while len(a_candidates) < 5 or len(b_candidates) < 5:
        a = choice(A)
        b = choice(B) 
        a_no_punc = a.translate(str.maketrans('', '', string.punctuation))
        b_no_punc = b.translate(str.maketrans('', '', string.punctuation))

        a_words = {ps.stem(word) for word in word_tokenize(a_no_punc) if word not in stops}
        b_words = {ps.stem(word) for word in word_tokenize(b_no_punc) if word not in stops} 

        no_add = False
        for word in a_words:
            if word in b_words:
                diff = a_words_count[word] - b_words_count[word]

                if abs(diff) > 2:
                    no_add = True
                    break
            else:
                if a_words_count[word] > 2:
                    no_add = True 
                    break
        
        if not no_add:
            for word in a_words:
                a_words_count[word] += 1
            for word in b_words:
                b_words_count[word] += 1

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
