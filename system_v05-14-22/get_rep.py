from collections import Counter
from collections import defaultdict
import json
import os
import pickle as pkl
from typing import List
import random
import tqdm
import argparse
from get_extreme_w_highlight import return_extreme_values, eval_only


def get_rep(
    pos: List[str],  # a list of text samples from D_1
    neg: List[str],  # a list of text samples from D_0
    pair: str = "",
    save_folder=None,
):

    if save_folder is None:
        save_folder = os.path.join("end2end_jobs_1127_filtered_clusters", pair)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    else:
        print("Folder %s exists" % save_folder)

    # get samples that are representative of the differences between two distributions
    # extreme_vals = eval_only(pos, neg, True)
    extreme_vals = return_extreme_values(pos, neg)
    # pkl.dump(
    #     extreme_vals, open(os.path.join(save_folder, "get_extreme_result.pkl"), "wb")
    # )
    print("Saving to json...")
    with open(os.path.join(save_folder, "shap_result.json"), "w") as f:
        out = json.dumps(extreme_vals, indent=4)
        f.write(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Trains and evaluates describe difference model")

    parser.add_argument("-i", "--index", type=int, default=0)

    args = parser.parse_args()
    index = args.index
    print(index)
    distribution_pairs = json.load(
        open("../benchmark_sec_4/1127_filtered_clusters.json"))

    pair = distribution_pairs[index]
    type = pair["type"]
    print(f"Running get_rep on category type {type} at index {index}")
    get_rep(
        pos=pair["positive_samples"],
        neg=pair["negative_samples"],
        pair="1127_filtered_clusters" + "-" + type + "-" + str(index),
    )
    # for i, pair in enumerate(tqdm.tqdm(distribution_pairs)):
    #     get_rep(
    #         pos=pair["positive_samples"],
    #         neg=pair["negative_samples"],
    #         pair="benchmark_1018-old-" + str(i),
    #         top_p=top_p,
    #     )
    # distribution_pairs = json.load(open("../benchmark_sec_4/benchmark_1026.json"))
    # indexes = defaultdict(list)
    # # print("counting...")
    # for i, p in enumerate(distribution_pairs):
    #     indexes[p["type"]].append(i)
    # # print("counting...")
    # # c = Counter([p["type"] for p in distribution_pairs])
    # # print(c)
    # for type in indexes:
    #     if len(indexes[type]) > 10:
    #         pairs = random.sample(indexes[type], 10)
    #     else:
    #         pairs = indexes[type]

    #     s = ""
    #     for p in pairs:
    #         s += str(p) + " "

    #     print(f"{type}=({s[:len(s)-1]})")
    # for type in indexes:
    #     if len(indexes[type]) > 10:
    #         pairs = random.sample(indexes[type], 10)
    #     else:
    #         pairs = indexes[type]

    #     for i, d in enumerate(tqdm.tqdm(pairs)):
    #         if type == "category_error" and i not in [1, 2, 5, 6, 7, 8, 9]:
    #             continue
    #         print(type, "index: ", i, "json_index:", d)
    #         pair = distribution_pairs[d]
    #         # try:
    #         get_rep(
    #             pos=pair["positive_samples"],
    #             neg=pair["negative_samples"],
    #             pair="benchmark_1018-all-" + type + "-" + str(i),
    #             top_p=top_p,
    #         )
