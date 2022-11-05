import re
import json
import string
import math
import numpy as np
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from collections import OrderedDict
from collections import defaultdict
from random import choice
from typing import List, Set

nltk.download("stopwords")

stops = set(stopwords.words("english"))
ps = PorterStemmer()

# reorder the list
# now the list is going to be a random shuffle of sorted_l[:top_p fraction] + a random shuffle of sorted_l[top_p fraction:]
def re_order(sorted_l: List[str], top_p: float) -> List[str]:
    part1 = sorted_l[: int(len(sorted_l) * top_p)]
    part2 = sorted_l[int(len(sorted_l) * top_p) :]
    np.random.shuffle(part1)
    np.random.shuffle(part2)
    return part1 + part2


def get_word_sent_of_sent(sent: str) -> Set[str]:
    sent_no_punc = sent.translate(str.maketrans("", "", string.punctuation))
    word_set = {
        ps.stem(word) for word in word_tokenize(sent_no_punc) if word not in stops
    }
    return word_set


# A is a sorted list of sentences for D_1, with more representative sentences at the beginning
# B is a sorted list of sentences for D_2, with more representative sentences at the beginning
# top_p is the fraction of sentences we want to focus on, but if we run out of sentences, we will use the rest of the sentences
# num_sentences is the number of sentences we want to return for each group
def lexical_diversity(
    sorted_A: List[str], sorted_B: List[str], top_p: float = 0.2, num_sentences: int = 5
):
    a_candidates = []
    b_candidates = []

    reordered_A = re_order(sorted_A, top_p)
    reordered_B = re_order(sorted_B, top_p)

    a_words_count, b_words_count = defaultdict(int), defaultdict(int)

    # enumerate through the sentences
    # keeps track of how many sentences we have examined
    cur_A_pointer, cur_B_pointer = 0, 0

    # we add 1 sentence for group A and group B alternatively, until we have num_sentences sentences
    for _ in range(num_sentences):
        # add a sentence for group A
        # enumerate the sentence from the reordered_A list until we find a legitimate sentence
        while cur_A_pointer < len(reordered_A):

            # get the sentence
            sent_A = reordered_A[cur_A_pointer]
            cur_A_pointer += 1

            # get the set of words of the sentence
            word_set_A = get_word_sent_of_sent(sent_A)

            # decide whether to add the sentence
            add_A_flg = True
            for word in word_set_A:
                if a_words_count[word] - b_words_count[word] >= 2:
                    add_A_flg = False
                    break

            # if we decide to add the sentence, add it to the candidate list and then stop the while loop
            if add_A_flg:
                a_candidates.append(sent_A)
                for word in word_set_A:
                    a_words_count[word] += 1
                break

        # same as above, but for group B instead of group A
        while cur_B_pointer < len(reordered_B):
            sent_B = reordered_B[cur_B_pointer]
            cur_B_pointer += 1
            word_set_B = get_word_sent_of_sent(sent_B)

            add_B_flg = True
            for word in word_set_B:
                if b_words_count[word] - a_words_count[word] >= 2:
                    add_B_flg = False

            if add_B_flg:
                b_candidates.append(sent_B)
                for word in word_set_B:
                    b_words_count[word] += 1
                break

    return a_candidates, b_candidates


if __name__ == "__main__":

    def one_sample_generator(word_freq):
        s = ""
        for w, f in word_freq.items():
            if np.random.rand() < f:
                s += w + " "
        return s

    A_word_freq = {
        "one": 1,
        "two": 0.8,
        "three": 0.2,
        "four": 0.2,
        "five": 0.2,
        "six": 0.2,
    }
    As = [one_sample_generator(A_word_freq) for _ in range(10)]
    print(As)
    As = sorted(As, key=lambda x: (1 if "two" in x else 0) + 0.1 * len(x), reverse=True)
    print(As)
    B_word_freq = {
        "one": 1,
        "seven": 0.8,
        "eight": 0.2,
        "nine": 0.2,
        "ten": 0.2,
        "eleven": 0.2,
    }
    Bs = [one_sample_generator(B_word_freq) for _ in range(10)]
    Bs = sorted(
        Bs, key=lambda x: (1 if "seven" in x else 0) + 0.1 * len(x), reverse=True
    )
    print(Bs)
    print(lexical_diversity(As, Bs, top_p=0.2, num_sentences=5))
