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
from collections import Counter

from random import choice
from typing import List, Set

import os

nltk.download("stopwords")

stops = set(stopwords.words("english"))
ps = PorterStemmer()

# reorder the list
# now the list is going to be a random shuffle of sorted_l[:top_p fraction] + a random shuffle of sorted_l[top_p fraction:]


def re_order(sorted_l: List[str], top_p: float) -> List[str]:
    part1 = sorted_l[: int(len(sorted_l) * top_p)]
    part2 = sorted_l[int(len(sorted_l) * top_p):]
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
    distribution_pairs = json.load(
        open("../benchmark_sec_4/benchmark_1018.json"))
    indexes = defaultdict(list)
    print("counting...")
    for i, p in enumerate(distribution_pairs):
        indexes[p["type"]].append(i)

    # print(indexes)
    # print("counting...")
    # c = Counter([p["type"] for p in distribution_pairs])
    tasks = ["category_error", "task_error"]
    out = {}
    for task in tasks:
        task_dict = {}
        for i in range(10):
            save_folder = f"end2end_jobs_benchmark_1018/benchmark_1018-all-{task}-{i}"
            if not os.path.exists(save_folder):
                continue

            data = json.load(
                open(os.path.join(save_folder, "shap_result.json")))
            pos_data = list(data["pos2score"])
            data_type = task
            data_index = -1
            for j in indexes[task]:
                if pos_data[0] in distribution_pairs[j]["positive_samples"]:
                    data_index = j
            i_dict = {}

            i_dict["average_roc"] = sum(data["auc_roc"]) / len(data["auc_roc"])
            i_dict["auc_roc"] = data["auc_roc"]
            # average = sum(data["auc_roc"]) / len(data["auc_roc"])
            i_dict["num_pos"] = len(data["pos2score"])
            i_dict["num_neg"] = len(data["neg2score"])

            task_dict[str(data_index)] = i_dict
            # for p in [0.05, 0.2, 1.0]:

            # pos_sorted = OrderedDict()
            # neg_sorted = OrderedDict()
            # for k, v in sorted(
            #     data["pos2score"].items(), key=lambda item: item[1], reverse=True
            # ):
            #     pos_sorted[k] = v

            # for k, v in sorted(
            #     data["neg2score"].items(), key=lambda item: item[1], reverse=True
            # ):
            #     neg_sorted[k] = v

            # As = [k for k in pos_sorted]
            # Bs = [k for k in neg_sorted]
            # # print(As[:3])

            # # print(Bs[:3])
            # pos, neg = lexical_diversity(As, Bs, top_p=p, num_sentences=5)

            # output_json = {
            #     "A": [sentence for sentence in pos],
            #     "A_Highlight": [
            #         data["pos2highlight"][sentence] for sentence in pos
            #     ],
            #     "B": [sentence for sentence in neg],
            #     "B_Highlight": [
            #         data["neg2highlight"][sentence] for sentence in neg
            #     ],
            #     "auc_roc": data["auc_roc"]
            # }
            # p_dict[str(p)] = output_json
            # task_dict[str(data_index)] = p_dict

        out[task] = task_dict

    with open("stats.json", "w") as f:
        out = json.dumps(out, indent=4)
        f.write(out)
    # with open("shap_results_all_1.0.json", "w") as f:
    #     out = json.dumps(out, indent=4)
    #     f.write(out)
