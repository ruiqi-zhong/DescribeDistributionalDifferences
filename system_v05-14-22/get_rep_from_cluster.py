import json
import os
import pickle as pkl
from typing import List

import numpy as np

import tqdm

from get_extreme_w_highlight import return_extreme_values


def get_rep(
    pos: List[str],  # a list of text samples from D_1
    neg: List[str],  # a list of text samples from D_0
    pair: str = "",
    save_folder=None,
):
    if save_folder is None:
        save_folder = os.path.join("end2end_jobs", pair)
        print(save_folder)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    else:
        print("Folder %s exists" % save_folder)

    # get samples that are representative of the differences between two distributions
    extreme_vals = return_extreme_values(pos, neg, False)
    pkl.dump(
        extreme_vals, open(os.path.join(save_folder, "get_extreme_result.pkl"), "wb")
    )


if __name__ == "__main__":
    with open("cluster.json", "r") as f:
        clusters = json.load(f)

    indexes = [f"{i}" for i in range(64)]

    pairs = np.random.choice(indexes, size=(10, 2), replace=False)

    for (i, pair) in enumerate(tqdm.tqdm(pairs)):
        print("pair:", i)
        d0_index, d1_index = pair[0], pair[1]

        d0, d1 = clusters[d0_index], clusters[d1_index]
        try:
            get_rep(pos=d0, neg=d1, pair=str())
        except:
            continue
